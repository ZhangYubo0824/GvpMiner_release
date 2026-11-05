"""Independently recompute absolute MSA columns, every interval and all summaries.

Scientific outputs are read-only; writes only validation.json.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from scripts.paths import table_path

import numpy as np
import pandas as pd

FROZEN = 'results/catalogues/GvpMiner_M2_M3'
AA = 'ACDEFGHIKLMNPQRSTVWY'
FORBIDDEN = ('background', 'difference', 'p_value', 'q_value', 'rank', 'percentile', 'weighted', 'paired')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fasta(path):
    result, key = {}, None
    with Path(path).open(encoding='ascii') as handle:
        for line in handle:
            if line.startswith('>'):
                key = line[1:].split()[0]
                assert key not in result
                result[key] = ''
            elif line.strip():
                result[key] += line.strip().upper()
    return result


def close(left, right):
    if left is None or pd.isna(left):
        assert right is None or pd.isna(right), (left, right)
    else:
        assert right is not None and not pd.isna(right) and np.isclose(left, right, rtol=1e-10, atol=1e-12), (left, right)


def schema(frame):
    assert not any(term in column for column in frame.columns for term in FORBIDDEN), frame.columns


def verify(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert summary['analysis'] == 'absolute_descriptive_unweighted_v2'
    assert len(summary['inputs']) == 33
    for record in summary['inputs']:
        assert Path(record['path']).is_relative_to(source)
        assert digest(record['path']) == record['sha256']
    metadata = pd.read_csv(source/'data/reference/preprocessed/cleaned_gvp_dataset.csv', keep_default_na=False)
    candidates = pd.read_csv(output/'candidate_conservation.csv')
    schema(candidates)
    assert len(candidates) == 522 and candidates.n_occurrences.sum() == 84012
    assert not candidates.duplicated(['family', 'motif_uid']).any()
    expected = pd.concat([pd.read_csv(p) for p in (source/FROZEN).glob('*/motif_catalog_M2.csv')], ignore_index=True)
    assert candidates[['family', 'motif_uid']].to_records(index=False).tolist() == expected[['family', 'motif_uid']].sort_values(['family', 'motif_uid'], kind='stable').to_records(index=False).tolist()
    family_checks, statuses, n_sequences, n_occurrences, protein_count, n_columns = [], Counter(), 0, 0, 0, 0
    for family, group in candidates.groupby('family', sort=True):
        folder = output/family
        manifest = json.loads((folder/'alignment_manifest.json').read_text(encoding='utf-8'))
        cache = Path(manifest['cache_directory'])
        assert manifest['input_sha256'] == digest(cache/'input.fasta')
        assert manifest['alignment_sha256'] == digest(cache/'alignment.fasta')
        inputs, alignment = fasta(cache/'input.fasta'), fasta(cache/'alignment.fasta')
        identifiers = pd.read_csv(cache/'identifiers.csv', keep_default_na=False)
        by_protein = dict(zip(identifiers.protein_id, identifiers.alignment_id))
        fm = metadata[metadata.family == family]
        assert set(fm.protein_id) == set(by_protein) and len(fm) == len(inputs)
        assert set(inputs) == set(alignment) and len({len(x) for x in alignment.values()}) == 1
        for row in fm.itertuples(index=False):
            key = by_protein[row.protein_id]
            assert inputs[key] == row.sequence.upper() == alignment[key].replace('-', '')
        n_sequences += len(inputs)
        # Independent counting and entropy implementation, no production score helper.
        matrix = np.array([list(x) for x in alignment.values()])
        counts = np.stack([np.count_nonzero(matrix == a, axis=0) for a in AA])
        totals = counts.sum(axis=0)
        occupancy = totals / len(matrix)
        col_scores = np.full(matrix.shape[1], np.nan)
        for j in range(matrix.shape[1]):
            if totals[j] and occupancy[j] >= .5:
                proportions = counts[:, j][counts[:, j] > 0] / totals[j]
                entropy = -sum(float(p)*math.log(float(p)) for p in proportions)
                col_scores[j] = min(1, max(0, 1-entropy/math.log(20)))
        saved_columns = pd.read_csv(folder/'column_scores.csv.gz')
        schema(saved_columns)
        assert saved_columns.alignment_column.tolist() == list(range(len(col_scores)))
        assert np.allclose(saved_columns.conservation, col_scores, equal_nan=True, atol=1e-12)
        assert np.allclose(saved_columns.canonical_occupancy, occupancy, atol=1e-12)
        n_columns += len(col_scores)
        residue = {protein: col_scores[np.flatnonzero(np.array(list(alignment[key])) != '-')]
                   for protein, key in by_protein.items()}
        original = pd.read_csv(table_path(source/FROZEN/family/'motif_occurrences_M2.csv'))
        occ = pd.read_csv(folder/'occurrence_conservation.csv.gz')
        schema(occ)
        key_fields = ['motif_uid', 'parent_id', 'occ_start', 'occ_end']
        assert set(map(tuple, original[key_fields].values)) == set(map(tuple, occ[key_fields].values))
        assert len(original) == len(occ) and not occ.duplicated(key_fields).any()
        for row in occ.itertuples(index=False):
            assert row.n_residues == max(0, row.occ_end-row.occ_start)
            if row.parent_id not in residue:
                assert row.status == 'missing_sequence' and pd.isna(row.candidate_score)
                continue
            values = residue[row.parent_id]
            if row.occ_start < 0 or row.occ_end <= row.occ_start or row.occ_end > len(values):
                assert row.status == 'invalid_coordinates' and pd.isna(row.candidate_score)
                continue
            window = values[row.occ_start:row.occ_end]
            valid = window[np.isfinite(window)]
            close(len(valid)/len(window), row.valid_position_coverage)
            assert row.n_valid_positions == len(valid)
            if len(valid)*2 >= len(window):
                assert row.status == 'scored'
                close(float(sum(valid)/len(valid)), row.candidate_score)
            else:
                assert row.status == 'inadequate_valid_positions' and pd.isna(row.candidate_score)
        statuses.update(occ.status)
        n_occurrences += len(occ)
        protein = pd.read_csv(folder/'protein_conservation.csv.gz')
        schema(protein)
        assert not protein.duplicated(['motif_uid', 'protein_id']).any()
        expected_keys = set(map(tuple, occ[['motif_uid','parent_id']].values))
        assert set(map(tuple, protein[['motif_uid','protein_id']].values)) == expected_keys
        protein_index = protein.set_index(['motif_uid','protein_id'])
        for (uid, pid), rows in occ.groupby(['motif_uid','parent_id']):
            scored = rows.loc[rows.status == 'scored', 'candidate_score']
            row = protein_index.loc[(uid,pid)]
            close(float(scored.median()) if len(scored) else None, row.median_conservation)
            assert row.n_occurrences == len(rows) and row.n_valid_occurrences == len(scored)
            close(float(rows.valid_position_coverage.mean()), row.mean_valid_position_coverage)
            assert row.status == ('no_valid_occurrences' if not len(scored) else 'scored' if len(scored) == len(rows) else 'partially_scored')
        protein_count += len(protein)
        for row in group.itertuples(index=False):
            selected = occ[occ.motif_uid == row.motif_uid]
            values = protein.loc[protein.motif_uid == row.motif_uid, 'median_conservation'].dropna()
            assert row.n_occurrences == len(selected)
            assert row.n_valid_occurrences == int((selected.status == 'scored').sum())
            assert row.n_total_proteins == selected.parent_id.nunique() and row.n_valid_proteins == len(values)
            close(len(values)/row.n_total_proteins if row.n_total_proteins else None, row.valid_protein_fraction)
            close(float(selected.valid_position_coverage.mean()) if len(selected) else None, row.mean_valid_position_coverage)
            stats = {'mean_conservation': values.mean(), 'median_conservation': values.median(),
                     'sd_conservation': values.std(ddof=1), 'min_conservation': values.min(), 'max_conservation': values.max()}
            for field, value in stats.items():
                close(value, getattr(row, field))
            assert row.status == ('no_occurrences' if selected.empty else 'no_valid_occurrences' if values.empty else
                                  'scored' if (selected.status == 'scored').all() else 'partially_scored')
        saved_family = pd.read_csv(folder/'candidate_conservation.csv')
        pd.testing.assert_frame_equal(group.reset_index(drop=True), saved_family)
        family_checks.append({'family': family, 'n_sequences': len(inputs), 'n_candidates': len(group),
                              'n_occurrences': len(occ), 'n_alignment_columns': len(col_scores),
                              'alignment_sha256': manifest['alignment_sha256']})
        print(f'{family}: all columns, {len(occ)} occurrence scores, proteins and candidates verified', flush=True)
    distribution = summary['distribution']
    values = candidates.mean_conservation.dropna()
    assert distribution['n_candidates_with_score'] == len(values)
    for field, value in [('mean_candidate_conservation', values.mean()), ('median_candidate_conservation', values.median()),
                         ('min_candidate_conservation', values.min()), ('max_candidate_conservation', values.max())]:
        close(float(value), distribution[field])
    saved_families = pd.read_csv(output/'family_summary.csv')
    for family, frame in candidates.groupby('family'):
        row = saved_families[saved_families.family == family].iloc[0]
        values = frame.mean_conservation.dropna()
        assert row.n_candidates == len(frame) and row.n_candidates_with_score == len(values)
        assert row.n_occurrences == frame.n_occurrences.sum() and row.n_valid_occurrences == frame.n_valid_occurrences.sum()
        for field, value in [('mean_candidate_conservation', values.mean()), ('median_candidate_conservation', values.median()),
                             ('min_candidate_conservation', values.min()), ('max_candidate_conservation', values.max())]:
            close(value, row[field])
    assert dict(statuses) == summary['occurrence_status_counts']
    assert summary['n_aligned_sequences'] == n_sequences == 26492
    assert n_occurrences == 84012 and len(family_checks) == 15
    qa = {'all_checks_passed': True, 'analysis': summary['analysis'], 'n_input_files_hashed': 33,
          'all_522_candidate_keys_retained': True, 'all_84012_actual_occurrences_verified': True,
          'all_input_hashes_unchanged': True, 'all_full_length_sequences_validated': n_sequences,
          'all_alignment_columns_recomputed': n_columns, 'all_occurrence_scores_recomputed': n_occurrences,
          'all_protein_summaries_recomputed': protein_count, 'all_candidate_summaries_recomputed': len(candidates),
          'descriptive_only_schema_verified': True, 'occurrence_status_counts': dict(statuses),
          'candidate_status_counts': dict(Counter(candidates.status)), 'family_checks': family_checks}
    (output/'validation.json').write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in qa.items() if k != 'family_checks'}, indent=2), flush=True)
    return qa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    verify(args.source_root, args.output_root)
