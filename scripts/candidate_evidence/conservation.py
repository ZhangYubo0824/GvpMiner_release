"""Absolute, descriptive MSA conservation for all frozen Stage C 20/15/M2 candidates.

No ranking, percentile, reference-window comparison, hypothesis test or p/q value.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from scripts.paths import table_path
import sys

import numpy as np
import pandas as pd

from .msa_alignment import align_family, residue_to_column, sha256, write_json, write_csv, atomic_write_bytes, atomic_write_text

AA = 'ACDEFGHIKLMNPQRSTVWY'
FROZEN = 'results/catalogues/GvpMiner_M2_M3'
ANALYSIS = 'absolute_descriptive_unweighted_v2'


def write_compressed_csv(frame, path):
    """Publish a complete gzip without truncating an existing result in place."""
    path = Path(path)
    payload = gzip.compress(frame.to_csv(index=False).encode('utf-8'), mtime=0)
    atomic_write_bytes(path, payload)


def column_conservation(alignment, minimum_occupancy=.5):
    """Unweighted entropy among canonical residues, with explicit occupancy QC."""
    matrix = np.array([list(s.upper()) for s in alignment])
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError('A nonempty rectangular alignment is required')
    counts = np.array([(matrix == a).sum(axis=0) for a in AA], dtype=float)
    total = counts.sum(axis=0)
    occupancy = total / len(matrix)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    scores = np.clip(1 + (p * logp).sum(axis=0) / np.log(20), 0, 1)
    scores[(occupancy < minimum_occupancy) | (total == 0)] = np.nan
    return scores, occupancy


def window_score(scores, start, end):
    values = scores[start:end]
    valid = np.isfinite(values)
    coverage = float(valid.mean()) if len(values) else 0.0
    return (float(np.mean(values[valid])) if coverage >= .5 else None), coverage


def protein_scores(rows):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get('candidate_score')
        if row['status'] == 'scored' and value is not None and np.isfinite(value):
            grouped[str(row['parent_id'])].append(float(value))
    return {protein: float(np.median(values)) for protein, values in grouped.items()}


def candidate_statistics(rows):
    scores = np.array(list(protein_scores(rows).values()), dtype=float)
    n = len(scores)
    n_valid = sum(row['status'] == 'scored' for row in rows)
    total_proteins = len({str(r['parent_id']) for r in rows})
    return {
        'n_occurrences': len(rows), 'n_valid_occurrences': n_valid,
        'n_total_proteins': total_proteins, 'n_valid_proteins': n,
        'valid_protein_fraction': n / total_proteins if total_proteins else None,
        'mean_valid_position_coverage': float(np.mean([r['valid_position_coverage'] for r in rows])) if rows else None,
        'mean_conservation': float(scores.mean()) if n else None,
        'sd_conservation': float(scores.std(ddof=1)) if n > 1 else None,
        'median_conservation': float(np.median(scores)) if n else None,
        'min_conservation': float(scores.min()) if n else None,
        'max_conservation': float(scores.max()) if n else None,
        'status': ('no_occurrences' if not rows else 'no_valid_occurrences' if not n else
                   'scored' if n_valid == len(rows) else 'partially_scored'),
    }


def evaluate_family(family, metadata, catalog, occurrences, folder, cache, mafft, threads):
    folder.mkdir(parents=True, exist_ok=True)
    alignment, manifest = align_family(family, metadata, cache, mafft, threads)
    ids = metadata.protein_id.astype(str).tolist()
    scores, occupancy = column_conservation([alignment[p] for p in ids])
    write_compressed_csv(pd.DataFrame({'alignment_column': np.arange(len(scores)),
                                      'conservation': scores, 'canonical_occupancy': occupancy}), folder/'column_scores.csv.gz')
    write_json(folder/'alignment_manifest.json', {**manifest, 'cache_directory': str(cache.resolve())})
    residue_scores = {p: scores[residue_to_column(alignment[p])] for p in ids}
    all_rows, candidates, proteins = [], [], []
    for candidate in catalog.itertuples(index=False):
        selected = occurrences[occurrences.motif_uid == candidate.motif_uid]
        rows = []
        for occ in selected.itertuples(index=False):
            protein, start, end = str(occ.parent_id), int(occ.occ_start), int(occ.occ_end)
            row = {'family': family, 'motif_uid': candidate.motif_uid,
                   'occurrence_id': occ.occurrence_id, 'parent_id': protein,
                   'occ_start': start, 'occ_end': end, 'n_residues': max(0, end-start),
                   'n_valid_positions': 0, 'candidate_score': None,
                   'valid_position_coverage': 0.0, 'status': 'missing_sequence'}
            if protein in residue_scores:
                values = residue_scores[protein]
                if start < 0 or end <= start or end > len(values):
                    row['status'] = 'invalid_coordinates'
                else:
                    score, coverage = window_score(values, start, end)
                    row.update(candidate_score=score, valid_position_coverage=coverage,
                               n_valid_positions=int(np.isfinite(values[start:end]).sum()),
                               status='scored' if score is not None else 'inadequate_valid_positions')
            rows.append(row)
        all_rows.extend(rows)
        candidates.append({'family': family, 'motif_uid': candidate.motif_uid, **candidate_statistics(rows)})
        values = protein_scores(rows)
        for protein in dict.fromkeys(str(r['parent_id']) for r in rows):
            pr = [r for r in rows if str(r['parent_id']) == protein]
            value = values.get(protein)
            nvalid = sum(r['status'] == 'scored' for r in pr)
            proteins.append({'family': family, 'motif_uid': candidate.motif_uid, 'protein_id': protein,
                             'n_occurrences': len(pr), 'n_valid_occurrences': nvalid,
                             'mean_valid_position_coverage': float(np.mean([r['valid_position_coverage'] for r in pr])),
                             'median_conservation': value,
                             'status': 'no_valid_occurrences' if value is None else
                                       'scored' if nvalid == len(pr) else 'partially_scored'})
    write_compressed_csv(pd.DataFrame(all_rows), folder/'occurrence_conservation.csv.gz')
    write_compressed_csv(pd.DataFrame(proteins, columns=['family', 'motif_uid', 'protein_id', 'n_occurrences',
                                                       'n_valid_occurrences', 'mean_valid_position_coverage',
                                                       'median_conservation', 'status']), folder/'protein_conservation.csv.gz')
    write_csv(folder/'candidate_conservation.csv', pd.DataFrame(candidates))
    summary = {'family': family, 'n_candidates': len(candidates), 'n_occurrences': len(all_rows),
               'n_valid_occurrences': sum(r['status'] == 'scored' for r in all_rows),
               'n_candidates_with_score': sum(r['mean_conservation'] is not None for r in candidates),
               'occurrence_status_counts': dict(Counter(r['status'] for r in all_rows)), 'alignment': manifest}
    write_json(folder/'family_summary.json', summary)
    print(f'{family}: {len(candidates)} candidates; {summary["n_candidates_with_score"]} with scores', flush=True)
    return candidates, summary


def distribution(frame):
    values = frame.mean_conservation.dropna()
    return {'n_candidates': len(frame), 'n_candidates_with_score': len(values),
            'mean_candidate_conservation': float(values.mean()) if len(values) else None,
            'median_candidate_conservation': float(values.median()) if len(values) else None,
            'min_candidate_conservation': float(values.min()) if len(values) else None,
            'max_candidate_conservation': float(values.max()) if len(values) else None,
            'n_valid_occurrences': int(frame.n_valid_occurrences.sum()),
            'n_occurrences': int(frame.n_occurrences.sum())}


def write_descriptive_report(output, result, summary):
    d = summary['distribution']
    text = f'''# 522个候选的MSA绝对保守性分析

本分析描述冻结Stage C 20/15/M2候选本身的序列保守程度，不排名、不计算蛋白内百分位、不与背景比较，不进行显著性检验或高/中/低分级。全部候选按家族和原编号排列；空白分数表示不可评价，不表示零分。

## 实施流程

同家族清洗后完整蛋白 → 已验证的MAFFT多序列比对 → 不加权归一化熵位点分数 → 映射每个候选真实出现区间 → 同蛋白重复出现取中位数 → 候选内等权蛋白描述统计。

共有{summary['n_aligned_families']}个有候选家族、{summary['n_aligned_sequences']:,}条完整蛋白序列完成比对身份与哈希核对。GvpH没有当前候选，不参与比对。已有比对直接复用，不重新聚类、不改变522个候选成员。

位点分数C=1-H/ln(20)，越接近1表示有效标准氨基酸分布越集中。缺口和非标准氨基酸不进入频率。保留50%列占位率及50%片段有效位点门槛；它们是质量控制条件，不是保守性高低阈值。相较参考方法，使用每条蛋白的实际候选坐标，并明确加入列占位率门槛及蛋白级聚合。

每个候选报告蛋白级得分的均值、样本标准差、中位数和范围，并列出总/有效蛋白数、总/有效出现次数和平均有效位点覆盖率。只有一条有效蛋白时标准差为空；没有有效蛋白时分数全部为空。

## 整体结果

- 保留候选：{len(result)}；有分数：{d['n_candidates_with_score']}；无分数：{len(result)-d['n_candidates_with_score']}。
- 实际出现区间：{d['n_occurrences']:,}；可评分：{d['n_valid_occurrences']:,}。
- 对每个候选的蛋白等权平均分再做候选等权汇总：均值{d['mean_candidate_conservation']:.6f}，中位数{d['median_candidate_conservation']:.6f}，范围{d['min_candidate_conservation']:.6f}–{d['max_candidate_conservation']:.6f}。

上述均值不是将全部残基或全部蛋白混合计算的均值。候选和同源蛋白之间存在相关性，这里仅作描述，不将其视为独立生物学重复。

## 家族汇总

以下按家族名称排列，不按得分排序；均值/中位数/范围均针对家族内候选的mean_conservation。

|家族|候选数|有分数|候选均值的均值|候选均值的中位数|候选均值范围|有效出现/全部出现|
|---|---:|---:|---:|---:|---|---:|
'''
    for row in summary['family_distributions']:
        fmt = lambda v: f'{v:.4f}' if v is not None else '不可评价'
        text += (f"|{row['family']}|{row['n_candidates']}|{row['n_candidates_with_score']}|"
                 f"{fmt(row['mean_candidate_conservation'])}|{fmt(row['median_candidate_conservation'])}|"
                 f"{fmt(row['min_candidate_conservation'])}–{fmt(row['max_candidate_conservation'])}|"
                 f"{row['n_valid_occurrences']}/{row['n_occurrences']}|\n")
    text += '\n## 缺失情况与解释范围\n\n'
    text += '出现区间状态：' + '；'.join(f'{k}={v:,}' for k,v in summary['occurrence_status_counts'].items()) + '。\n\n'
    missing = result[result.mean_conservation.isna()]
    text += '无分数候选：' + ('、'.join(missing.motif_uid) if len(missing) else '无') + '。\n\n'
    text += ('分数描述候选位置在整个家族MSA中的氨基酸一致程度，不是仅在该候选成员内重新比对。'
             '分数不表示候选优于其他区域，也不是功能验证；家族采样、重复序列、对齐质量和占位率门槛会影响分数。'
             '复现方法见methods.md；逐候选表见candidate_conservation.csv；家族表见family_summary.csv；'
             '逐区间和逐蛋白压缩CSV位于对应家族目录；独立复算记录见validation.json。\n')
    atomic_write_text(output/'report_zh.md', text)


METHODS = '''# Descriptive absolute MSA conservation

The frozen Stage C 20-aa/15-aa-stride/M2 catalogue is unchanged. All cleaned full-length sequences are aligned separately within each candidate-bearing family using MAFFT7.526 FFT-NS-1 (--amino --retree 1 --maxiterate 0 --thread 4). Saved alignments are reused only after input/alignment SHA256 checks and exact ungapped-sequence validation. Input/alignment hashes and scoring invocation provenance are retained.

Only unweighted canonical-amino-acid frequencies are used. C=1-H/ln(20), with gaps/noncanonical residues excluded. A column requires >=50% canonical occupancy across all family input sequences. This occupancy QC is an explicit adaptation beyond excluding entirely invalid columns in the reference description; scores describe the evaluable portion, and sampling/phylogenetic dependence remain limitations.

Actual zero-based half-open occurrence coordinates map through each protein's nongap columns. Each occurrence score is the arithmetic mean of valid column scores, requiring >=50% of its residues to be valid. Missing sequences, invalid coordinates and inadequate valid-position coverage remain explicit rows. The same candidate's repeated occurrences within one protein are reduced to one median score. Candidate mean, sample SD(ddof=1), median, minimum and maximum describe those equally weighted protein scores. SD is missing for one valid protein; all score summaries are missing with no valid proteins. Occurrence coverage averages include unscorable occurrences. All522 candidate rows are retained without rankings, percentiles, background windows, hypothesis tests, p/q values or high/medium/low categories.

Family/global distributions summarize the candidate means with equal candidate weight; family tables are alphabetical and candidate rows retain catalogue order. High absolute conservation does not establish relative enrichment, specificity, experimental function or independent evolutionary replication.

Run from the repository root after installing requirements.txt:

    python -m scripts.candidate_evidence.conservation --source-root . --output-root work/conservation
    python -m scripts.candidate_evidence.verify_conservation --source-root . --output-root work/conservation

All scientific score outputs are independently recomputed from retained full alignments by verify_conservation.py. Version, input hashes, configuration and command arguments are recorded in summary.json.
'''


def output_paths(source, output, cache):
    """Keep derivations outside source data and alignments outside score output."""
    source, output, cache = (Path(p).resolve() for p in (source, output, cache))
    for path in (output, cache):
        if path == source or source.is_relative_to(path) or (path.is_relative_to(source) and not path.is_relative_to(source/"work")):
            raise ValueError('Output and alignment cache must be outside the source tree')
    if cache.is_relative_to(output) or output.is_relative_to(cache):
        raise ValueError('Alignment cache and score output must be separate folders')
    return source, output, cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[2]/'work/conservation')
    parser.add_argument('--alignment-cache', type=Path)
    parser.add_argument('--mafft', default='mafft')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    source, output = args.source_root.resolve(), args.output_root.resolve()
    default_cache = source/'work/msa_alignments'
    source, output, cache = output_paths(source, output, args.alignment_cache or default_cache)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = source/'data/reference/preprocessed/cleaned_gvp_dataset.csv'
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    all_candidates, families, inputs, empty = [], [], [], []
    inputs.append({'path': str(metadata_path), 'sha256': sha256(metadata_path)})
    for folder in sorted((source/FROZEN).iterdir()):
        paths = [table_path(folder/'motif_catalog_M2.csv'), table_path(folder/'motif_occurrences_M2.csv')]
        if not paths[0].exists():
            continue
        catalog, occurrences = [pd.read_csv(table_path(p)) for p in paths]
        inputs.extend({'path': str(p), 'sha256': sha256(p)} for p in paths)
        if catalog.empty:
            empty.append({'family': folder.name, 'n_candidates': 0, 'n_occurrences': len(occurrences)})
            continue
        rows, summary = evaluate_family(folder.name, metadata[metadata.family == folder.name].reset_index(drop=True),
                                        catalog, occurrences, output/folder.name, cache/folder.name,
                                        args.mafft, args.threads)
        all_candidates.extend(rows)
        families.append(summary)
    result = pd.DataFrame(all_candidates)
    if len(result) != 522 or result.n_occurrences.sum() != 84012 or result.motif_uid.duplicated().any():
        raise ValueError('Frozen catalogue count or identity invariant failed')
    write_csv(output/'candidate_conservation.csv', result)
    family_distributions = [{'family': f, **distribution(g)} for f,g in result.groupby('family', sort=True)]
    write_csv(output/'family_summary.csv', pd.DataFrame(family_distributions))
    occurrence_counts = Counter()
    for f in families:
        occurrence_counts.update(f['occurrence_status_counts'])
    summary = {'analysis': ANALYSIS, 'n_candidates': len(result), 'n_occurrences': int(result.n_occurrences.sum()),
               'n_aligned_families': len(families), 'n_aligned_sequences': sum(f['alignment']['n_sequences'] for f in families),
               'distribution': distribution(result), 'family_distributions': family_distributions,
               'candidate_status_counts': dict(Counter(result.status)),
               'occurrence_status_counts': dict(occurrence_counts), 'families': families,
               'inputs': inputs, 'zero_candidate_families': empty,
               'configuration': {'weighting': 'unit_protein_weights', 'minimum_column_occupancy': .5,
                                 'minimum_occurrence_valid_coverage': .5,
                                 'protein_aggregation': 'median_of_valid_occurrence_means',
                                 'candidate_aggregation': 'equal_protein_mean_sd_median_min_max',
                                 'analysis_type': 'descriptive_only',
                                 'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__},
                                 'argv': sys.argv}}
    write_json(output/'summary.json', summary)
    atomic_write_text(output/'methods.md', METHODS)
    write_descriptive_report(output, result, summary)
    print(json.dumps({k: summary[k] for k in ['analysis','n_candidates','n_occurrences','distribution','candidate_status_counts','occurrence_status_counts']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
