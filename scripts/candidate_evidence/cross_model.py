"""Compare frozen M2/M3 candidates using actual same-protein residue unions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from scripts.paths import table_path, output_directory

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

FROZEN = Path('results/catalogues/GvpMiner_M2_M3')
THRESHOLDS = (0.3, 0.5, 0.7)


def safe_output_root(source_root, output_root):
    return output_directory(source_root, output_root)


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def residue_unions(occurrences):
    grouped = defaultdict(list)
    for protein, start, end in occurrences:
        if pd.isna(protein) or not str(protein) or int(start) != start or int(end) != end or start < 0 or end <= start:
            raise ValueError(f'Invalid half-open occurrence: {(protein, start, end)}')
        grouped[str(protein)].append((int(start), int(end)))
    result = {}
    for protein, intervals in grouped.items():
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        result[protein] = merged
    return result


def residue_count(unions):
    return sum(end - start for intervals in unions.values() for start, end in intervals)


def interval_intersection(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def compare_unions(left, right):
    nl, nr = residue_count(left), residue_count(right)
    common = left.keys() & right.keys()
    overlap = sum(interval_intersection(left[p], right[p]) for p in common)
    return {'left_residues': nl, 'right_residues': nr, 'intersection_residues': overlap,
            'residue_jaccard': overlap / (nl + nr - overlap) if nl + nr else 0.,
            'left_coverage': overlap / nl if nl else None,
            'right_coverage': overlap / nr if nr else None,
            'left_proteins': len(left), 'right_proteins': len(right),
            'intersection_proteins': len(common),
            'protein_jaccard': len(common) / len(left.keys() | right.keys()) if left or right else 0.}


def hungarian_matches(matrix, threshold):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return set()
    # Invalid edges and dummy vertices have zero utility. They cannot steal a valid edge.
    n, m = matrix.shape
    utility = np.zeros((n + m, n + m))
    utility[:n, :m] = np.where(matrix >= threshold, matrix, 0.)
    rows, cols = linear_sum_assignment(utility, maximize=True)
    return {(int(i), int(j)) for i, j in zip(rows, cols) if i < n and j < m and matrix[i, j] >= threshold}


def match_family(family, left, right):
    luids, ruids = sorted(left), sorted(right)
    matrix = np.zeros((len(luids), len(ruids)))
    all_pairs = {}
    for i, luid in enumerate(luids):
        for j, ruid in enumerate(ruids):
            score = compare_unions(left[luid], right[ruid])
            matrix[i, j] = score['residue_jaccard']
            all_pairs[i, j] = {'family': family, 'm2_uid': luid, 'm3_uid': ruid, **score}
    assignments = {t: hungarian_matches(matrix, t) for t in THRESHOLDS}
    # Lexicographic UID tie breaking is deterministic and documented; ties are also counted.
    best_l = {i: int(np.argmax(matrix[i])) for i in range(len(luids)) if len(ruids) and matrix[i].max() > 0}
    best_r = {j: int(np.argmax(matrix[:, j])) for j in range(len(ruids)) if len(luids) and matrix[:, j].max() > 0}
    candidates = []
    for model, own, other, uids in [('M2', left, right, luids), ('M3', right, left, ruids)]:
        for k, uid in enumerate(uids):
            scores = matrix[k] if model == 'M2' else matrix[:, k]
            partner_idx = (best_l if model == 'M2' else best_r).get(k)
            pair = (all_pairs[k, partner_idx] if model == 'M2' else all_pairs[partner_idx, k]) if partner_idx is not None else None
            r = {'family': family, 'method': model, 'motif_uid': uid,
                 'n_proteins': len(own[uid]), 'n_union_residues': residue_count(own[uid]),
                 'status': 'evaluated' if pair else ('empty_occurrences' if not own[uid] else 'no_residue_overlap'),
                 'best_match_uid': (ruids if model == 'M2' else luids)[partner_idx] if pair else None,
                 'best_residue_jaccard': float(scores.max()) if scores.size else 0.,
                 'best_match_tie_count': int(np.sum(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))) if pair else 0,
                 'best_protein_jaccard': pair['protein_jaccard'] if pair else None,
                 'best_own_coverage': pair['left_coverage' if model == 'M2' else 'right_coverage'] if pair else None,
                 'best_partner_coverage': pair['right_coverage' if model == 'M2' else 'left_coverage'] if pair else None,
                 'reciprocal_best': bool(pair and (best_r.get(partner_idx) == k if model == 'M2' else best_l.get(partner_idx) == k))}
            for threshold in THRESHOLDS:
                suffix = f'j{round(threshold * 100):03d}'
                r['partner_count_' + suffix] = int(np.sum(scores >= threshold))
                r['has_match_' + suffix] = bool(np.any(scores >= threshold))
                r['multiple_partners_' + suffix] = r['partner_count_' + suffix] > 1
                selected = [(i, j) for i, j in assignments[threshold] if (i if model == 'M2' else j) == k]
                r['hungarian_match_' + suffix] = bool(selected)
                r['hungarian_partner_' + suffix] = (ruids[selected[0][1]] if model == 'M2' else luids[selected[0][0]]) if selected else None
                r['reciprocal_best_' + suffix] = bool(r['reciprocal_best'] and r['best_residue_jaccard'] >= threshold)
            # Complements Jaccard graph degrees with directional containment fragments.
            directional = []
            for op in range(len(other)):
                p = all_pairs[k, op] if model == 'M2' else all_pairs[op, k]
                cover = p['right_coverage' if model == 'M2' else 'left_coverage']
                if p['intersection_residues'] and cover is not None and cover >= .5:
                    directional.append((ruids if model == 'M2' else luids)[op])
            r['n_partners_half_contained'] = len(directional)
            r['half_contained_partner_uids'] = json.dumps(directional)
            candidates.append(r)
    pairs = []
    for (i, j), pair in all_pairs.items():
        pair['reciprocal_best'] = best_l.get(i) == j and best_r.get(j) == i
        for threshold in THRESHOLDS:
            suffix = f'j{round(threshold * 100):03d}'
            pair['qualifies_' + suffix] = pair['residue_jaccard'] >= threshold
            pair['hungarian_match_' + suffix] = (i, j) in assignments[threshold]
        # Keep all nonzero residue pairs and protein-only overlaps, including negatives for regional overlap.
        if pair['intersection_proteins'] > 0:
            pairs.append(pair)
    return candidates, pairs


def load_frozen(source_root, methods=('M2', 'M3')):
    root = Path(source_root) / FROZEN
    families = sorted(p.name for p in root.iterdir() if p.is_dir())
    data, inputs = {}, []
    for family in families:
        for method in methods:
            catalog_path = root / family / f'motif_catalog_{method}.csv'
            occurrence_path = root / family / f'motif_occurrences_{method}.csv'
            catalog, occurrences = pd.read_csv(catalog_path), pd.read_csv(table_path(occurrence_path))
            inputs.extend([file_record(catalog_path), file_record(occurrence_path)])
            if catalog.motif_uid.duplicated().any() or occurrences.duplicated(['motif_uid', 'parent_id', 'occ_start', 'occ_end']).any():
                raise ValueError(f'Duplicate keys in {family}/{method}')
            if not set(occurrences.motif_uid) <= set(catalog.motif_uid):
                raise ValueError(f'Orphan occurrences in {family}/{method}')
            if set(catalog.family) - {family} or set(occurrences.family) - {family}:
                raise ValueError(f'Wrong family in {family}/{method}')
            unions = {str(uid): {} for uid in catalog.motif_uid}
            for uid, group in occurrences.groupby('motif_uid', sort=False):
                unions[str(uid)] = residue_unions(group[['parent_id', 'occ_start', 'occ_end']].itertuples(index=False, name=None))
            data[family, method] = (catalog, occurrences, unions)
    return families, data, inputs


def summarize_candidates(frame, families=None):
    result = []
    keys = [(family, method) for family in families for method in ('M2', 'M3')] if families else sorted(set(zip(frame.family, frame.method)))
    for family, method in keys:
        group = frame.loc[(frame.family == family) & (frame.method == method)]
        row = {'family': family, 'method': method, 'n_candidates': len(group),
               'n_with_residue_overlap': int((group.status == 'evaluated').sum()),
               'median_best_residue_jaccard': float(group.best_residue_jaccard.median()) if len(group) else None,
               'n_multiple_half_contained_partners': int((group.n_partners_half_contained > 1).sum())}
        for t in THRESHOLDS:
            suffix = f'j{round(t * 100):03d}'
            for name in ['has_match', 'hungarian_match', 'reciprocal_best', 'multiple_partners']:
                row['n_' + name + '_' + suffix] = int(group[name + '_' + suffix].sum())
        result.append(row)
    return pd.DataFrame(result)


def run(source_root, output_root):
    output = safe_output_root(source_root, output_root)
    families, data, inputs = load_frozen(source_root)
    candidates, pairs = [], []
    for family in families:
        c, p = match_family(family, data[family, 'M2'][2], data[family, 'M3'][2])
        candidates.extend(c)
        pairs.extend(p)
    frame, pair_frame = pd.DataFrame(candidates), pd.DataFrame(pairs)
    family_summary = summarize_candidates(frame, families=families)
    overall = summarize_candidates(frame.assign(family='ALL'))
    frame.to_csv(output / 'candidate_matches.csv', index=False)
    (output / 'candidate_matches.json').write_text(frame.to_json(orient='records', indent=2, force_ascii=False), encoding='utf-8')
    pair_frame.to_csv(output / 'pair_matches.csv', index=False)
    family_summary.to_csv(output / 'family_summary.csv', index=False)
    overall.to_csv(output / 'overall_summary.csv', index=False)
    summary = {'primary_threshold': .5, 'sensitivity_thresholds': [.3, .7],
               'n_families': len(families), 'n_candidate_bearing_families': int(frame.family.nunique()),
               'overall': json.loads(overall.to_json(orient='records', double_precision=15)),
               'families': json.loads(family_summary.to_json(orient='records', double_precision=15))}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps({'module': 'cross_model', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(Path(source_root).resolve()), 'inputs': inputs, 'parameters': {'coordinate_convention': '0-based half-open',
        'thresholds': list(THRESHOLDS), 'primary_threshold': .5, 'hungarian_objective': 'maximum sum of eligible residue Jaccards with null dummy vertices'}}, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'methods.md').write_text('''# Cross-model regional correspondence

The frozen 20/15 Stage C M2 and M3 catalogs are compared within each family. Cluster labels and numeric motif ranks are never used as evidence of correspondence. For each candidate, actual 0-based half-open occurrence intervals are merged per protein. Residues on different proteins are distinct. Jaccard is the summed same-protein intersection length divided by total union length. Directional coverage divides the intersection by each candidate's total union-residue length. Protein Jaccard uses protein membership independently of residue overlap. Source occurrence_id suffixes reset within proteins, so uniqueness is checked on motif_uid, parent_id, occ_start, occ_end rather than occurrence_id alone.

The primary Jaccard threshold is 0.5; sensitivity thresholds are 0.3 and 0.7. Best partners maximize residue Jaccard; exact ties (within 1e-12 for reported tie counts) are broken by lexicographic full UID. Reciprocal best matches require mutual selection. Hungarian assignment maximizes total Jaccard over eligible edges with dummy null vertices, separately at each threshold. Each candidate is retained, including unmatched or empty candidates. Fractions use all candidates of that method as denominator.

Alternative partners are retained in pair_matches.csv for every shared-protein pair; pairs sharing no protein have zero scores and are omitted from that table. Degrees greater than one in each thresholded graph describe possible splits (M2 to multiple M3) or merges (multiple M2 to M3). The separately reported partners half-contained in a candidate have intersection/partner-residues >=0.5 and positive intersection; their individual Jaccards may be small. This directional containment is a descriptive fragmentation measure and is not the primary correspondence criterion. Regional correspondence provides no independent experimental or functional validation.
''', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.output_root), indent=2))


if __name__ == '__main__':
    main()
