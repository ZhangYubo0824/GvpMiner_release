"""Summarise a completed MotifAE run using the current experiment definitions.

Read-only for all experiment inputs. Does not rerun enrichment,
change q values, filter candidates, or modify experiment inputs.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'work/motifae'
OUT = Path(__file__).resolve().parent
RUN = 'current'
PRIMARY = ('contact_density', 'relative_SASA', 'long_range_contact_count')
HASHES = {}
CHECKS = []


def sha(path):
    path = Path(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    HASHES[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=digest)
    return digest


def load(path):
    sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path, record_hash=True):
    if record_hash:
        sha(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def write(name, records):
    records = list(records)
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def number(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def ratio(a, b):
    return a / b if b else None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def positions(value):
    return tuple(int(p) for p in json.loads(value) if p is not None)


def summarise(label, records):
    n = len(records)
    total = dict(family=label, method='MotifAE', candidates=n)
    for name in ('structural_summarized', 'structural_testable', 'structural_supported',
                 'strict_structural_supported', 'annotation_has_output', 'annotation_supported',
                 'annotation_historical_eligible', 'annotation_historical_supported',
                 'strict_structure_and_annotation', 'strict_structure_only',
                 'annotation_only', 'neither_strict_structure_nor_annotation'):
        total[name] = sum(r[name] for r in records)
    total.update(
        structural_no_summary=n-total['structural_summarized'],
        structural_summarized_untestable=total['structural_summarized']-total['structural_testable'],
        structural_tested_not_supported=total['structural_testable']-total['structural_supported'],
        annotation_no_output=n-total['annotation_has_output'],
        annotation_output_not_supported=total['annotation_has_output']-total['annotation_supported'],
        structural_support_per_testable=ratio(total['structural_supported'], total['structural_testable']),
        strict_structural_support_per_input=ratio(total['strict_structural_supported'], n),
        annotation_support_per_output=ratio(total['annotation_supported'], total['annotation_has_output']),
        annotation_support_per_input=ratio(total['annotation_supported'], n),
        annotation_output_per_input=ratio(total['annotation_has_output'], n),
        historical_annotation_support_per_eligible=ratio(total['annotation_historical_supported'], total['annotation_historical_eligible']),
        dual_support_per_input=ratio(total['strict_structure_and_annotation'], n),
        input_occurrences=sum(r['input_occurrences'] for r in records),
        structurally_analyzed_occurrences=sum(r['structural_instances'] for r in records),
        median_member_proteins=median(r['member_proteins'] for r in records),
        median_candidate_mean_plddt=median(r['mean_plddt'] for r in records),
        median_composition_control_coverage=median(r['composition_control_coverage'] for r in records),
    )
    total['structure_occurrence_coverage'] = ratio(total['structurally_analyzed_occurrences'], total['input_occurrences'])
    return total


def main():
    begin = time.monotonic()
    handoff = SOURCE / 'handoff' / RUN
    h = load(handoff / 'handoff_manifest.json')
    families = sorted(h['families'])
    assert len(families) == 16 and 'GvpH' in families
    for relative, expected in h['files_sha256'].items():
        p = (handoff / relative).resolve()
        assert p.is_relative_to(handoff.resolve())
        assert sha(p) == expected, f'Handoff hash mismatch: {relative}'
    CHECKS.append('All handoff manifest output hashes match downloaded files.')
    sequences = {r['protein_id']: r['sequence'] for r in rows(handoff / 'sequence_index.csv', False)}
    exp5 = SOURCE / 'exp5' / RUN
    preparation = load(exp5 / 'exp5_preparation_manifest.json')
    assert preparation['handoff_manifest_sha256'] == sha(handoff / 'handoff_manifest.json')
    progress = load(exp5 / 'work_existing_annotations/batch_progress.json')
    assert progress['state'] == 'complete' and progress['failed_batches'] == progress['pending_batches'] == 0
    output_counts = Counter()
    minp = {}
    annotation_foreground = {}
    annotation_universe = {}
    annotation_genes = defaultdict(set)
    enrichment_path = exp5 / 'clusterprofiler_enrichment_existing_annotations.csv'
    enrichment_rows = 0
    for row in rows(enrichment_path):
        assert row['method'] == 'MotifAE'
        key = row['family'], row['motif_cluster']
        output_counts[key] += 1
        enrichment_rows += 1
        p = number(row['p.adjust'])
        if p is not None:
            assert 0 <= p <= 1
            minp[key] = min(minp.get(key, 1), p)
        annotation_foreground[key] = int(row['foreground_total'])
        annotation_universe[key] = int(row['universe_total'])
        annotation_genes[key].update(row['geneID'].split('/'))
    all_records = []
    alignment_counts = []
    feature_validation = []
    endpoint_records = []
    for family in families:
        print(f'[{family}] verifying coordinates and calculating metrics...', flush=True)
        catalog = list(rows(handoff / family / 'MotifAE/motif_catalog_MotifAE.csv', False))
        cats = {r['motif_uid']: r for r in catalog}
        assert len(cats) == len(catalog) == h['families'][family]['motifs']
        alignment = SOURCE / 'alignment' / family / RUN / 'guarded'
        batch = load(alignment / 'batch_manifest.json')
        assert sha(alignment / 'batch_manifest.json') == h['families'][family]['batch_manifest_sha256']
        assert batch['status'] == 'alignment_samples_parsed' and batch['failed_groups'] == 0
        features = list(rows(alignment / 'feature_summary.csv'))
        patterns = {r['motif_uid'] for r in features if r['pattern_built'] == 'True'}
        assert patterns == set(cats)
        assert sum(int(r['mapped_instances']) for r in features) == h['families'][family]['occurrences']
        alignment_counts.append(dict(family=family, groups=len(features),
            retained_patterns=len(patterns), all_trash_groups=sum(r['status']=='all_sequences_trash' for r in features),
            candidate_instances=sum(int(r['total_candidate_instances']) for r in features),
            retained_instances=sum(int(r['mapped_instances']) for r in features),
            trash_instances=sum(int(r['trash_instances']) for r in features),
            unassessed_instances=sum(int(r['unassessed_instances']) for r in features),
            background_proteins=h['families'][family]['background_proteins'],
            foreground_proteins=h['families'][family]['foreground_proteins']))
        fg = defaultdict(set)
        core_spans = defaultdict(list)
        core_lengths = defaultdict(list)
        expected = Counter()
        input_counts = Counter()
        for row in rows(handoff / family / 'MotifAE/motif_occurrences_MotifAE.csv', False):
            uid = row['motif_uid']; pid = row['parent_id']
            assert uid in cats and row['localization_scope'] == 'selected_top1000_alignment_instances'
            pos = positions(row['core_residue_positions_0based'])
            core_spans[uid].append(max(pos)-min(pos)+1)
            core_lengths[uid].append(len(pos))
            core = ''.join(sequences[pid][p] for p in pos)
            assert core == row['aligned_core'].replace('-', '')
            expected[(uid, pid, pos, core)] += 1
            fg[uid].add(pid); input_counts[uid] += 1
        for uid, c in cats.items():
            assert len(fg[uid]) == int(c['n_proteins'])
            assert input_counts[uid] == int(c['n_occurrences'])
        terms = defaultdict(set)
        term_path = exp5 / 'work_existing_annotations/family_term2gene' / f'term2gene_{family}.csv'
        for row in rows(term_path):
            terms[row['term_id']].add(row['protein_id'])
        eligible_sets = [v for v in terms.values() if 10 <= len(v) <= 500]
        eligible_proteins = set().union(*eligible_sets) if eligible_sets else set()
        folder = SOURCE / 'exp4' / RUN / family / 'MotifAE'
        summary = load(folder / 'analysis_summary.json')
        progress4 = load(folder / 'analysis_progress.json')
        assert progress4['complete'] is True and progress4['residue_semantics_version'] == 'exact-core-residues-v1'
        assert summary['motif_instances_requested'] == sum(input_counts.values())
        clusters = {r['motif_cluster']: r for r in rows(folder / 'motif_cluster_summary.csv')}
        assert set(clusters) <= set(cats) and len(clusters) == summary['motif_cluster_summaries']
        primary = defaultdict(dict)
        for row in rows(folder / 'structural_enrichment_results.csv'):
            if row['control_type'] == 'composition_matched' and row['feature'] in PRIMARY:
                uid = row['motif_cluster']; feat = row['feature']
                assert uid in cats and feat not in primary[uid]
                primary[uid][feat] = row
        coverage = {}
        for row in rows(folder / 'control_diagnostics.csv'):
            if row['control_type'] == 'composition_matched':
                assert row['motif_cluster'] not in coverage
                coverage[row['motif_cluster']] = float(row['instance_coverage_fraction'])
        actual = Counter()
        plddt = defaultdict(list)
        long_range_max = 0
        for row in rows(folder / 'motif_structure_features.csv'):
            pos = positions(row['residue_positions_0based'])
            token = row['motif_cluster'], row['sequence_id'], pos, row['motif_sequence']
            assert expected[token] > 0, f'Unexpected structural occurrence: {token}'
            expected[token] -= 1
            actual[row['motif_cluster']] += 1
            long_range_max = max(long_range_max, int(row['long_range_contact_count']))
            p = number(row['motif_mean_pLDDT'])
            if p is not None:
                plddt[row['motif_cluster']].append(p)
        assert sum(actual.values()) == summary['motif_instances_analyzed']
        assert sum(expected.values()) == summary['skipped_motif_instances']
        assert sum(int(c['n_instances']) for c in clusters.values()) == sum(actual.values())
        feature_validation.append(dict(family=family, input_occurrences=sum(input_counts.values()),
            exact_core_structural_occurrences=sum(actual.values()), unrepresented_occurrences=sum(expected.values()),
            unexpected_structural_occurrences=0,
            max_input_core_span=max(max(v) for v in core_spans.values()),
            max_observed_long_range_contact_count=long_range_max))
        for uid, c in cats.items():
            key = family, uid; s = clusters.get(uid)
            t = primary[uid]
            qs = {feature: number(t.get(feature, {}).get('primary_q_value')) for feature in PRIMARY}
            testable = s is not None and all(q is not None for q in qs.values())
            significant = testable and any(q < .05 for q in qs.values())
            confidence = number(s['mean_motif_pLDDT']) if s else None
            cov = coverage.get(uid)
            if s:
                assert actual[uid] == int(s['n_instances'])
                if plddt[uid]:
                    assert math.isclose(statistics.mean(plddt[uid]), confidence, abs_tol=.001)
            strict = bool(significant and confidence is not None and confidence >= 70 and cov is not None and cov >= .9)
            has_output = output_counts[key] > 0
            supported = minp.get(key, 1) <= .05
            if has_output:
                assert annotation_foreground[key] == len(fg[uid])
                assert annotation_universe[key] == h['families'][family]['background_proteins']
                assert annotation_genes[key] <= fg[uid]
            eligible_n = len(fg[uid] & eligible_proteins)
            historical = bool(eligible_sets) and eligible_n >= 3
            record = dict(family=family, method='MotifAE', motif_uid=uid,
                input_occurrences=input_counts[uid], member_proteins=len(fg[uid]),
                min_observed_core_length=min(core_lengths[uid]), max_observed_core_length=max(core_lengths[uid]),
                max_core_coordinate_span=max(core_spans[uid]),
                structural_summarized=s is not None, structural_testable=testable,
                structural_supported=bool(significant), strict_structural_supported=strict,
                mean_plddt=confidence, composition_control_coverage=cov,
                structural_instances=actual[uid], annotation_has_output=has_output,
                annotation_supported=supported, annotation_rows=output_counts[key],
                minimum_annotation_p_adjust=minp.get(key), eligible_annotated_foreground=eligible_n,
                annotation_historical_eligible=historical,
                annotation_historical_supported=historical and supported,
                strict_structure_and_annotation=strict and supported,
                strict_structure_only=strict and not supported,
                annotation_only=supported and not strict,
                neither_strict_structure_nor_annotation=not strict and not supported,
                structural_status='supported' if significant else 'tested_not_supported' if testable else 'incomplete_primary_tests' if s else 'no_summary',
                annotation_status='supported' if supported else 'output_not_supported' if has_output else 'no_output')
            for feature in PRIMARY:
                record[f'{feature}_primary_q'] = qs[feature]
                record[f'{feature}_effect'] = number(t.get(feature, {}).get('effect_median_difference'))
            all_records.append(record)
        print(f'[{family}] {len(cats)} candidates; {sum(actual.values()):,} exact structural occurrences verified', flush=True)
    assert set(output_counts) <= {(r['family'], r['motif_uid']) for r in all_records}
    CHECKS.extend(['All 16 family catalogues match retained alignment pattern identifiers and counts.',
                   'All analysed Exp4 instances match handoff protein IDs, exact core positions and residue sequences.',
                   'Structural occurrence counts and candidate mean pLDDT reconcile with exported summaries.',
                   'Enrichment candidate keys and foreground/universe sizes reconcile with handoff membership.'])
    total = summarise('ALL_16_FAMILIES', all_records)
    family_summary = [summarise(f, [r for r in all_records if r['family']==f]) for f in families]
    for key in ('candidates', 'structural_testable', 'structural_supported', 'strict_structural_supported',
                'annotation_has_output', 'annotation_supported', 'strict_structure_and_annotation', 'input_occurrences'):
        assert total[key] == sum(r[key] for r in family_summary)
    for family in ['ALL_16_FAMILIES', *families]:
        selected = [r for r in all_records if family == 'ALL_16_FAMILIES' or r['family']==family]
        eligible = [r for r in selected if r['structural_testable']]
        for feature in PRIMARY:
            sig = [r for r in eligible if r[f'{feature}_primary_q'] < .05]
            effects = [r[f'{feature}_effect'] for r in sig if r[f'{feature}_effect'] is not None]
            qtiles = statistics.quantiles(effects, n=4, method='inclusive') if len(effects)>1 else [None]*3
            endpoint_records.append(dict(family=family, endpoint=feature, testable=len(eligible),
                significant=len(sig), significant_per_testable=ratio(len(sig), len(eligible)),
                positive_effect=sum(v>0 for v in effects), negative_effect=sum(v<0 for v in effects),
                zero_median_effect=sum(v==0 for v in effects),
                median_effect=median(effects), effect_q25=qtiles[0], effect_q75=qtiles[2]))
    write('overall_summary.csv', [total])
    write('family_summary.csv', family_summary)
    write('motif_level_metrics.csv', all_records)
    write('structural_endpoint_summary.csv', endpoint_records)
    write('alignment_family_summary.csv', alignment_counts)
    write('structure_coordinate_checks.csv', feature_validation)
    a_total = dict(family='ALL_16_FAMILIES', **{k: sum(r[k] for r in alignment_counts) for k in alignment_counts[0] if k!='family'})
    assert a_total['candidate_instances'] == a_total['retained_instances']+a_total['trash_instances']+a_total['unassessed_instances']
    assert a_total['retained_patterns'] == total['candidates']
    assert a_total['retained_instances'] == total['input_occurrences']
    write('alignment_overall_summary.csv', [a_total])
    manifest = dict(verification_status='ANALYZED', analysis='recomputed summaries from downloaded outputs; no experimental rerun',
        source=str(SOURCE), run=RUN, evaluation_protocol='current candidate-level structure and annotation definitions',
        scope='all 16 families including GvpH; selected retained Top1000 alignment instances',
        structural_rule='all three composition-matched primary_q_value finite; any <0.05',
        strict_structural_rule='structural support AND candidate mean pLDDT >=70 AND composition control instance coverage >=0.90',
        annotation_rule='any enrichment row = has output; any original within-candidate BH p.adjust <=0.05 = supported',
        aggregation='pooled candidate counts, not mean family percentages',
        historical_annotation_rule='additional eligible annotation foreground >=3 shown separately, not main result',
        tests_rerun=False, raw_inputs_modified=False, checks=CHECKS, overall=total,
        alignment=a_total, enrichment_rows=enrichment_rows, sources=HASHES,
        elapsed_seconds=time.monotonic()-begin)
    (OUT/'calculation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(overall=total, alignment=a_total, elapsed_seconds=time.monotonic()-begin), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True,
                        help='Completed MotifAE run workspace containing handoff, alignment, exp4 and exp5.')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2]/'work/motifae_metrics')
    args = parser.parse_args()
    ROOT = args.run_root.resolve()
    SOURCE = ROOT
    RUN = args.run_name
    OUT = args.output_dir.resolve()
    if OUT == ROOT or ROOT in OUT.parents:
        parser.error('Output must be outside the read-only research source directory.')
    OUT.mkdir(parents=True, exist_ok=True)
    main()
