"""Deterministic region-candidate accounting, independent of GPU inference."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from adapter_common import (CATALOG_FIELDS, OCCURRENCE_FIELDS, canonical_sequence_id,
                            normalize_occurrence, write_rows)

FILTER_VERSION = 2
CATALOG_COLUMNS = CATALOG_FIELDS + ['n_proteins', 'rank_within_method']
DECISION_COLUMNS = ['family', 'method', 'motif_uid', 'n_proteins', 'n_occurrences',
                    'score', 'status', 'representative_uid', 'protein_jaccard',
                    'reciprocal_interval_coverage']


def canonical_protein_id(value):
    token = str(value or '').strip().split()
    if not token:
        return ''
    fields = token[0].split('|')
    if len(fields) > 1 and fields[0] in {'sp', 'tr', 'ref', 'gb', 'emb', 'dbj'}:
        if not fields[1]:
            raise ValueError('Protein header has no accession')
        return fields[1]
    return canonical_sequence_id(token[0])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, name):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f'{name} must be a finite integer')
    return int(number)


def _validated_rows(rows):
    unique = {}
    peptides_by_region = {}
    count = 0
    for source in rows:
        count += 1
        # Recognize standard UniProt headers before the shared generic alias
        # helper can collapse sp|ACCESSION|NAME to the non-unique prefix "sp".
        prepared = dict(source)
        for field in ('sequence_id', 'parent_id'):
            prepared[field] = canonical_protein_id(prepared.get(field))
        row = normalize_occurrence(prepared)
        for field in ['family', 'method', 'motif_uid', 'parent_id', 'sequence_id']:
            if not row[field]:
                raise ValueError(f'Occurrence {count}: missing {field}')
        start = _integer(row['occ_start'], 'occ_start')
        end = _integer(row['occ_end'], 'occ_end')
        score = float(row['score'])
        peptide = row['motif_sequence']
        if start < 0 or not 5 <= end-start <= 30:
            raise ValueError(f'Occurrence {count}: expected valid 5-30 residue coordinates')
        if not math.isfinite(score) or score <= 0:
            raise ValueError(f'Occurrence {count}: peptide activation must be finite and positive')
        if len(peptide) != end-start or not peptide.isascii() or not peptide.isalpha():
            raise ValueError(f'Occurrence {count}: peptide sequence must match coordinate length')
        row.update(occ_start=str(start), occ_end=str(end), score=format(score, '.17g'))
        # Protein support is counted at canonical parent level, not by aliases.
        row['parent_id'] = canonical_protein_id(row['parent_id'])
        key = (row['family'], row['method'], row['motif_uid'], row['parent_id'], start, end)
        region = (row['family'], row['parent_id'], start, end)
        if region in peptides_by_region and peptides_by_region[region] != peptide:
            raise ValueError(f'Conflicting peptides for identical parent interval {region}')
        peptides_by_region[region] = peptide
        if key in unique and unique[key]['motif_sequence'] != peptide:
            raise ValueError(f'Conflicting peptides for identical occurrence {key}')
        # Tie-break the complete row so input order cannot change exports.
        if key not in unique or (score, json.dumps(row, sort_keys=True)) > (
                float(unique[key]['score']), json.dumps(unique[key], sort_keys=True)):
            unique[key] = row
    return [unique[key] for key in sorted(unique)], count


def process_candidates(rows):
    """Validate and account for every activated feature without local filtering."""
    raw, input_count = _validated_rows(rows)
    groups = defaultdict(list)
    for row in raw:
        groups[(row['family'], row['method'], row['motif_uid'])].append(row)
    profiles = {}
    for key, members in groups.items():
        protein_scores = defaultdict(list)
        for row in members:
            protein = row['parent_id']
            protein_scores[protein].append(float(row['score']))
        profiles[key] = dict(proteins=set(protein_scores),
                             score=statistics.median(max(v) for v in protein_scores.values()))
    rank = sorted(groups, key=lambda k: (k[:2], -len(profiles[k]['proteins']),
                                        -profiles[k]['score'], k[2]))
    decisions = []
    retained = []
    for key in rank:
        profile = profiles[key]
        decision = dict(family=key[0], method=key[1], motif_uid=key[2],
                        n_proteins=len(profile['proteins']), n_occurrences=len(groups[key]),
                        score=profile['score'], status='retained', representative_uid=key[2],
                        protein_jaccard='', reciprocal_interval_coverage='')
        retained.append(key)
        decisions.append(decision)
    ranks = {}
    catalog = []
    raw_catalog = []
    for key in rank:
        first = groups[key][0]
        item = {field: first.get(field, '') for field in CATALOG_FIELDS}
        item.update(n_occurrences=len(groups[key]), n_proteins=len(profiles[key]['proteins']),
                    score=profiles[key]['score'], rank_within_method='')
        raw_catalog.append(dict(item))
        if key in retained:
            ranks[key[:2]] = ranks.get(key[:2], 0)+1
            item['rank_within_method'] = ranks[key[:2]]
            catalog.append(item)
    counts = dict(input_occurrences=input_count, unique_raw_occurrences=len(raw),
                  duplicate_occurrences_removed=input_count-len(raw), raw_features=len(groups),
                  below_min_proteins=sum(d['status']=='below_min_proteins' for d in decisions),
                  redundant_features=sum(d['status']=='redundant_region' for d in decisions),
                  capped_features=sum(d['status']=='explicit_cap' for d in decisions),
                  retained_features=len(retained),
                  retained_occurrences=sum(len(groups[k]) for k in retained))
    assert counts['raw_features'] == sum(counts[k] for k in [
        'below_min_proteins', 'redundant_features', 'capped_features', 'retained_features'])
    return dict(occurrences=[r for k in retained for r in groups[k]], catalog=catalog,
                raw_occurrences=raw, raw_catalog=raw_catalog, decisions=decisions, counts=counts,
                parameters=dict(min_proteins=1, top_features=0,
                                deduplicate=False, protein_jaccard=.9,
                                interval_iou=.8, matched_fraction=.9))


def export_results(output_dir, method, result, provenance):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f'motif_occurrences_{method}.csv': (result['occurrences'], OCCURRENCE_FIELDS),
        f'motif_catalog_{method}.csv': (result['catalog'], CATALOG_COLUMNS),
        f'motif_occurrences_{method}_raw.csv': (result['raw_occurrences'], OCCURRENCE_FIELDS),
        f'motif_catalog_{method}_raw.csv': (result['raw_catalog'], CATALOG_COLUMNS),
        'motifae_feature_decisions.csv': (result['decisions'], DECISION_COLUMNS),
    }
    # Prepare all files before replacing any old completed output; manifest is last.
    with tempfile.TemporaryDirectory(dir=output_dir, prefix='.motifae-') as temporary:
        staging = Path(temporary)
        for name, (rows, columns) in outputs.items():
            write_rows(staging/name, rows, columns)
        manifest = dict(filter_version=FILTER_VERSION, counts=result['counts'],
                        parameters=result['parameters'], provenance=provenance,
                        output_sha256={name: sha256(staging/name) for name in outputs},
                        analysis_scope='feature_inventory',
                        motif_discovery_status='activated_peptides_only_pssm_not_built',
                        interpretation='Feature-level activated peptides; motif sequence patterns require GibbsCluster alignment and PSSMs.')
        (staging/'motifae_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        (output_dir/'motifae_manifest.json').unlink(missing_ok=True)
        for name in outputs:
            os.replace(staging/name, output_dir/name)
        os.replace(staging/'motifae_manifest.json', output_dir/'motifae_manifest.json')
    return manifest
