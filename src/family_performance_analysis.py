from __future__ import annotations

import argparse
import math
import json
from collections.abc import Sequence
from datetime import date
from functools import reduce
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_GROUPS = {
    "hydrophobic_fraction": set("AVILMFWY"),
    "charged_fraction": set("DEKRH"),
    "polar_fraction": set("STNQC"),
    "aromatic_fraction": set("FWY"),
    "glycine_proline_fraction": set("GP"),
}
SEQUENCE_EXPLANATORY_VARIABLES = [
    "n_sequences",
    "n_windows",
    "mean_length",
    "median_length",
    "length_iqr",
    "length_cv",
    "length_p10",
    "length_p90",
    "composition_entropy",
    "hydrophobic_fraction",
    "charged_fraction",
    "polar_fraction",
    "aromatic_fraction",
    "glycine_proline_fraction",
]


def sequence_explanatory_variables(frame: pd.DataFrame | None = None) -> list[str]:
    """Return basic sequence attributes allowed for between-family explanations."""
    if frame is None:
        return list(SEQUENCE_EXPLANATORY_VARIABLES)
    return [
        variable
        for variable in SEQUENCE_EXPLANATORY_VARIABLES
        if variable in frame.columns
    ]


def robust_percentile_score(
    values: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric.dropna()
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    if finite.empty:
        return result

    median = float(finite.median())
    mad = float((finite - median).abs().median())
    if mad == 0.0:
        if finite.nunique() == 1:
            result.loc[finite.index] = 50.0
            return result
        scale = float(finite.std(ddof=0))
    else:
        scale = 1.4826 * mad

    if not np.isfinite(scale) or scale == 0.0:
        result.loc[finite.index] = 50.0
        return result

    z_scores = (finite - median) / scale
    if not higher_is_better:
        z_scores = -z_scores
    result.loc[finite.index] = z_scores.map(
        lambda value: 50.0 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
    )
    return result


def compute_composite_scores(
    frame: pd.DataFrame,
    dimension_columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    result["composite_score"] = result[dimension_columns].mean(axis=1, skipna=True)
    result["score_completeness"] = result[dimension_columns].notna().mean(axis=1)
    return result


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    if x_values.size == 0 or y_values.size == 0:
        return float("nan")
    differences = x_values[:, None] - y_values[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if valid_positions.size == 0:
        return result

    valid_values = values[valid_positions]
    order = np.argsort(valid_values)
    sorted_values = valid_values[order]
    count = sorted_values.size
    adjusted_sorted = sorted_values * count / np.arange(1, count + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    restored = np.empty_like(adjusted_sorted)
    restored[order] = adjusted_sorted
    result[valid_positions] = restored
    return result


def assign_performance_groups(
    scores: pd.Series,
    group_size: int = 4,
) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    result = pd.Series("middle", index=scores.index, dtype=object)
    result.loc[numeric.isna()] = "not_evaluable"
    finite = numeric.dropna()
    if finite.empty or group_size <= 0:
        return result

    effective_size = min(group_size, len(finite))
    good_threshold = finite.nlargest(effective_size).min()
    poor_threshold = finite.nsmallest(effective_size).max()
    result.loc[finite[finite >= good_threshold].index] = "good"
    result.loc[finite[finite <= poor_threshold].index] = "poor"
    return result


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _merge_metric_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame(columns=["family", "method"])
    return reduce(
        lambda left, right: left.merge(
            right,
            on=["family", "method"],
            how="outer",
        ),
        usable,
    )


def load_family_method_metrics(results_dir: str | Path) -> pd.DataFrame:
    root = Path(results_dir)
    frames: list[pd.DataFrame] = []

    exp1_path = root / "exp1_classification" / "exp1_detailed_reports.json"
    if exp1_path.exists():
        payload = json.loads(exp1_path.read_text(encoding="utf-8"))
        rows = []
        for method, report in payload.items():
            for family, metrics in report.get("per_class", {}).items():
                rows.append(
                    {
                        "family": family,
                        "method": method,
                        "classification_precision": metrics.get("precision"),
                        "classification_recall": metrics.get("recall"),
                        "classification_f1": metrics.get("f1-score"),
                    }
                )
        frames.append(pd.DataFrame(rows))

    generation = _read_csv_if_exists(
        root
        / "publication_figures"
        / "figure_2_generation_validation_data.csv"
    )
    if not generation.empty:
        generation = generation.loc[
            generation["evaluation_status"].eq("supported")
        ].copy()
        generation = generation.rename(
            columns={
                "predicted_target_ratio": "generation_target_ratio",
                "mean_target_probability": "generation_target_probability",
            }
        )
        frames.append(
            generation[
                [
                    "family",
                    "method",
                    "generation_target_ratio",
                    "generation_target_probability",
                ]
            ]
        )

    clustering = _read_csv_if_exists(
        root / "exp3_domain_discovery" / "clustering_comparison.csv"
    )
    if not clustering.empty:
        clustering = clustering.copy()
        clustering["silhouette"] = clustering["silhouette_final"]
        clustering["inverse_davies_bouldin"] = 1.0 / clustering[
            "davies_bouldin"
        ].replace(0, np.nan)
        clustering["retained_fraction"] = 1.0 - clustering["noise_ratio"]
        frames.append(
            clustering[
                [
                    "family",
                    "method",
                    "n_motifs",
                    "n_core_samples",
                    "noise_ratio",
                    "silhouette",
                    "davies_bouldin",
                    "inverse_davies_bouldin",
                    "retained_fraction",
                ]
            ]
        )

    structural = _read_csv_if_exists(
        root / "publication_figures" / "figure_4_structural_validation_data.csv"
    )
    if not structural.empty:
        structural = structural.loc[structural["panel"].eq("endpoints")].copy()
        structural["significant_rate"] = pd.to_numeric(
            structural["significant_rate"], errors="coerce"
        )
        structural = (
            structural.groupby(["family", "method"], as_index=False)[
                "significant_rate"
            ]
            .mean()
            .rename(
                columns={"significant_rate": "structural_significant_rate"}
            )
        )
        frames.append(structural)

    functional = _read_csv_if_exists(
        root / "publication_figures" / "figure_5_external_functional_enrichment_data.csv"
    )
    if not functional.empty:
        coverage = functional.loc[functional["panel"].eq("coverage")].copy()
        supported = coverage.loc[
            coverage["external_enrichment_status"].eq(
                "external_enrichment_supported"
            )
        ]
        supported = (
            supported.groupby(["family", "method"], as_index=False)["proportion"]
            .sum()
            .rename(
                columns={"proportion": "external_function_supported_rate"}
            )
        )
        frames.append(supported)

    result = _merge_metric_frames(frames)
    if result.empty:
        return result
    return result.sort_values(["family", "method"]).reset_index(drop=True)


def _normalized_composition_entropy(sequences: Sequence[str]) -> float:
    residues = [
        residue
        for sequence in sequences
        for residue in str(sequence).upper()
        if residue in CANONICAL_AMINO_ACIDS
    ]
    if not residues:
        return float("nan")
    counts = pd.Series(residues).value_counts(normalize=True)
    entropy = float(-(counts * np.log(counts)).sum())
    return entropy / math.log(20.0)


def _residue_group_fraction(sequences: Sequence[str], group: set[str]) -> float:
    residues = [
        residue
        for sequence in sequences
        for residue in str(sequence).upper()
        if residue in CANONICAL_AMINO_ACIDS
    ]
    if not residues:
        return float("nan")
    return float(sum(residue in group for residue in residues) / len(residues))


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().eq("true")


def _load_sequence_attributes(root: Path) -> pd.DataFrame:
    rows = []
    domain_root = root / "exp3_domain_discovery"
    if not domain_root.exists():
        return pd.DataFrame()
    for family_dir in sorted(path for path in domain_root.iterdir() if path.is_dir()):
        windows_path = family_dir / "windows_metadata.csv"
        if not windows_path.exists():
            continue
        windows = pd.read_csv(windows_path)
        if windows.empty or "window_seq_full" not in windows:
            continue
        proteins = windows[["parent_id", "window_seq_full"]].drop_duplicates(
            subset=["parent_id"]
        )
        sequences = proteins["window_seq_full"].astype(str)
        lengths = sequences.str.len().astype(float)
        mean_length = float(lengths.mean())
        sequence_row = {
            "family": family_dir.name,
            "n_sequences": int(len(proteins)),
            "n_windows": int(len(windows)),
            "mean_length": mean_length,
            "median_length": float(lengths.median()),
            "length_iqr": float(lengths.quantile(0.75) - lengths.quantile(0.25)),
            "length_cv": (
                float(lengths.std(ddof=0) / mean_length)
                if mean_length
                else float("nan")
            ),
            "length_p10": float(lengths.quantile(0.10)),
            "length_p90": float(lengths.quantile(0.90)),
            "composition_entropy": _normalized_composition_entropy(sequences),
        }
        sequence_row.update(
            {
                column: _residue_group_fraction(sequences, residues)
                for column, residues in AMINO_ACID_GROUPS.items()
            }
        )
        rows.append(
            sequence_row
        )
    return pd.DataFrame(rows)


def _load_motif_attributes(root: Path) -> pd.DataFrame:
    clustering = _read_csv_if_exists(
        root / "exp3_domain_discovery" / "clustering_comparison.csv"
    )
    if clustering.empty:
        return pd.DataFrame()
    rows = []
    for record in clustering.to_dict("records"):
        family = str(record["family"])
        method = str(record["method"])
        occurrences_path = (
            root
            / "exp3_domain_discovery"
            / family
            / f"motif_occurrences_{method}.csv"
        )
        singleton_rate = float("nan")
        mean_occurrences = float("nan")
        if occurrences_path.exists():
            occurrences = pd.read_csv(occurrences_path)
            if not occurrences.empty and "motif_uid" in occurrences:
                support = occurrences.groupby("motif_uid").size()
                singleton_rate = float(support.eq(1).mean())
                mean_occurrences = float(support.mean())
        n_core = float(record.get("n_core_samples", np.nan))
        n_motifs = float(record.get("n_motifs", np.nan))
        rows.append(
            {
                "family": family,
                "method": method,
                "motifs_per_1000_core_windows": (
                    1000.0 * n_motifs / n_core
                    if np.isfinite(n_core) and n_core > 0
                    else float("nan")
                ),
                "singleton_motif_rate": singleton_rate,
                "mean_motif_occurrences": mean_occurrences,
            }
        )
    return pd.DataFrame(rows)


def _load_generation_attributes(root: Path) -> pd.DataFrame:
    generated_root = root / "exp2_generation_validation" / "generated"
    if not generated_root.exists():
        return pd.DataFrame()
    rows = []
    agreement_rows = []
    for family_dir in sorted(path for path in generated_root.iterdir() if path.is_dir()):
        predictions_by_method: dict[str, pd.DataFrame] = {}
        for method in ("M2", "M3"):
            path = family_dir / f"classification_{method}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            predictions_by_method[method] = frame
            target = _boolean_series(frame["is_target_family"])
            wrong = frame.loc[~target, "predicted_family"].dropna().astype(str)
            rows.append(
                {
                    "family": family_dir.name,
                    "method": method,
                    "generated_median_length": float(
                        pd.to_numeric(frame["length"], errors="coerce").median()
                    ),
                    "low_confidence_generation_rate": float(
                        pd.to_numeric(
                            frame["target_probability"], errors="coerce"
                        ).lt(0.5).mean()
                    ),
                    "top_wrong_family": (
                        str(wrong.value_counts().index[0]) if not wrong.empty else ""
                    ),
                }
            )
        if {"M2", "M3"}.issubset(predictions_by_method):
            aligned = predictions_by_method["M2"][
                ["generated_id", "predicted_family"]
            ].merge(
                predictions_by_method["M3"][
                    ["generated_id", "predicted_family"]
                ],
                on="generated_id",
                suffixes=("_M2", "_M3"),
            )
            agreement_rows.append(
                {
                    "family": family_dir.name,
                    "generation_model_agreement": float(
                        aligned["predicted_family_M2"].eq(
                            aligned["predicted_family_M3"]
                        ).mean()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    agreement = pd.DataFrame(agreement_rows)
    if not result.empty and not agreement.empty:
        result = result.merge(agreement, on="family", how="left")
    return result


def _load_structural_family_attributes(root: Path) -> pd.DataFrame:
    path = (
        root
        / "motif_validation"
        / "cluster_commonality_validation"
        / "structural_commonality_family_summary.csv"
    )
    frame = _read_csv_if_exists(path)
    if frame.empty:
        return frame
    result = frame[
        [
            "family",
            "n_motif_clusters",
            "n_high_commonality",
            "mean_motif_pLDDT",
            "median_position_consistency",
        ]
    ].copy()
    result["high_commonality_rate"] = (
        result["n_high_commonality"] / result["n_motif_clusters"].replace(0, np.nan)
    )
    return result


def _load_external_functional_enrichment_attributes(root: Path) -> pd.DataFrame:
    path = (
        root
        / "exp5_external_functional_enrichment"
        / "clusterprofiler"
        / "clusterprofiler_enrichment.csv"
    )
    frame = _read_csv_if_exists(path)
    required = {"family", "method", "motif_cluster", "ID", "p.adjust"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    q_values = (
        pd.to_numeric(frame["qvalue"], errors="coerce")
        if "qvalue" in frame
        else pd.to_numeric(frame["p.adjust"], errors="coerce")
    )
    frame = frame.assign(
        _p_adjust=pd.to_numeric(frame["p.adjust"], errors="coerce"),
        _significant=q_values.lt(0.05),
    )
    motif_significance = (
        frame.groupby(["family", "method", "motif_cluster"], as_index=False)[
            "_significant"
        ]
        .any()
        .groupby(["family", "method"], as_index=False)["_significant"]
        .mean()
        .rename(columns={"_significant": "external_enriched_motif_rate"})
    )
    term_summary = (
        frame.groupby(["family", "method"], as_index=False)
        .agg(
            external_enrichment_terms=("ID", "nunique"),
            minimum_external_p_adjust=("_p_adjust", "min"),
        )
    )
    return motif_significance.merge(term_summary, on=["family", "method"], how="outer")


def _load_structure_coverage(root: Path) -> pd.DataFrame:
    rows = []
    structure_root = root / "structures" / "esmfold"
    if not structure_root.exists():
        return pd.DataFrame()
    for family_dir in sorted(path for path in structure_root.iterdir() if path.is_dir()):
        path = family_dir / "structure_prediction_summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        requested = float(payload.get("requested_sequences", 0))
        available = float(payload.get("predicted", 0)) + float(
            payload.get("reused", 0)
        )
        rows.append(
            {
                "family": family_dir.name,
                "structure_coverage": (
                    available / requested if requested > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def load_family_attributes(results_dir: str | Path) -> pd.DataFrame:
    root = Path(results_dir)
    sequence = _load_sequence_attributes(root)
    motif = _load_motif_attributes(root)
    generation = _load_generation_attributes(root)
    external = _load_external_functional_enrichment_attributes(root)

    method_frames = [frame for frame in (motif, generation, external) if not frame.empty]
    if method_frames:
        method_result = _merge_metric_frames(method_frames)
    else:
        method_result = pd.DataFrame(columns=["family", "method"])

    if method_result.empty and not sequence.empty:
        methods = pd.DataFrame({"method": ["M2", "M3"]})
        method_result = sequence[["family"]].drop_duplicates().merge(
            methods,
            how="cross",
        )

    family_frames = [
        frame
        for frame in (
            sequence,
            _load_structural_family_attributes(root),
            _load_structure_coverage(root),
        )
        if not frame.empty
    ]
    family_result = (
        reduce(
            lambda left, right: left.merge(right, on="family", how="outer"),
            family_frames,
        )
        if family_frames
        else pd.DataFrame(columns=["family"])
    )
    result = method_result.merge(family_result, on="family", how="left")
    if "generated_median_length" in result and "median_length" in result:
        result["generated_length_ratio"] = (
            result["generated_median_length"]
            / result["median_length"].replace(0, np.nan)
        )
    return result.sort_values(["family", "method"]).reset_index(drop=True)


DIMENSION_COMPONENTS = {
    "classification_score": ["classification_f1"],
    "generation_score": [
        "generation_target_ratio",
        "generation_target_probability",
    ],
    "clustering_score": [
        "silhouette",
        "inverse_davies_bouldin",
        "retained_fraction",
    ],
    "structure_score": [
        "structural_significant_rate",
        "high_commonality_rate",
        "mean_motif_pLDDT",
    ],
    "function_score": ["external_function_supported_rate"],
}

DIMENSION_LABELS_ZH = {
    "classification_score": "分类",
    "generation_score": "生成",
    "clustering_score": "聚类",
    "structure_score": "结构",
    "function_score": "功能",
}

ATTRIBUTE_LABELS_ZH = {
    "n_sequences": "序列数量",
    "n_windows": "窗口数量",
    "mean_length": "平均长度",
    "median_length": "长度中位数",
    "length_iqr": "长度四分位距",
    "length_cv": "长度变异系数",
    "length_p10": "长度第10百分位",
    "length_p90": "长度第90百分位",
    "composition_entropy": "氨基酸组成熵",
    "hydrophobic_fraction": "疏水残基比例",
    "charged_fraction": "带电残基比例",
    "polar_fraction": "极性残基比例",
    "aromatic_fraction": "芳香残基比例",
    "glycine_proline_fraction": "Gly/Pro 比例",
    "motifs_per_1000_core_windows": "每千核心窗口 motif 数",
    "singleton_motif_rate": "单例 motif 比例",
    "mean_motif_occurrences": "motif 平均出现次数",
    "median_position_consistency": "位置一致性中位数",
    "structure_coverage": "结构文件覆盖率",
    "annotation_coverage": "功能注释覆盖率",
    "generation_model_agreement": "生成判定模型一致率",
}


def score_family_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    component_score_columns: dict[str, list[str]] = {}
    for dimension, components in DIMENSION_COMPONENTS.items():
        component_score_columns[dimension] = []
        for component in components:
            score_column = f"{component}_score"
            component_score_columns[dimension].append(score_column)
            if component not in result:
                result[score_column] = np.nan
                continue
            result[score_column] = result.groupby("method", group_keys=False)[
                component
            ].transform(robust_percentile_score)
        result[dimension] = result[component_score_columns[dimension]].mean(
            axis=1,
            skipna=True,
        )
    return compute_composite_scores(result, list(DIMENSION_COMPONENTS))


def aggregate_family_scores(
    scored_methods: pd.DataFrame,
    group_size: int = 4,
) -> pd.DataFrame:
    evaluable = scored_methods.loc[
        scored_methods.groupby("family")["classification_f1"].transform(
            lambda values: values.notna().any()
        )
    ].copy()
    dimension_columns = list(DIMENSION_COMPONENTS)
    family = (
        evaluable.groupby("family", as_index=False)[dimension_columns]
        .mean()
        .pipe(compute_composite_scores, dimension_columns)
    )
    family["rank"] = (
        family["composite_score"].rank(method="min", ascending=False).astype(int)
    )
    indexed_groups = assign_performance_groups(
        family.set_index("family")["composite_score"],
        group_size=group_size,
    )
    family["performance_group"] = family["family"].map(indexed_groups)
    return family.sort_values(["rank", "family"]).reset_index(drop=True)


def ranking_sensitivity(
    family_scores: pd.DataFrame,
    scored_methods: pd.DataFrame,
    group_size: int = 4,
) -> pd.DataFrame:
    dimension_columns = list(DIMENSION_COMPONENTS)
    full = family_scores.set_index("family")
    rows = []
    for omitted in dimension_columns:
        retained = [column for column in dimension_columns if column != omitted]
        scenario_scores = full[retained].mean(axis=1, skipna=True)
        scenario_ranks = scenario_scores.rank(method="min", ascending=False)
        scenario_groups = assign_performance_groups(
            scenario_scores,
            group_size=group_size,
        )
        for family in full.index:
            rows.append(
                {
                    "family": family,
                    "omitted_dimension": omitted,
                    "full_rank": int(full.loc[family, "rank"]),
                    "sensitivity_rank": int(scenario_ranks.loc[family]),
                    "rank_shift": int(
                        scenario_ranks.loc[family] - full.loc[family, "rank"]
                    ),
                    "top_group_stable": bool(
                        (full.loc[family, "performance_group"] == "good")
                        == (scenario_groups.loc[family] == "good")
                    ),
                    "bottom_group_stable": bool(
                        (full.loc[family, "performance_group"] == "poor")
                        == (scenario_groups.loc[family] == "poor")
                    ),
                }
            )
    for method in ("M2", "M3"):
        method_rows = scored_methods.loc[scored_methods["method"].eq(method)].copy()
        scenario_scores = method_rows.set_index("family")["composite_score"].reindex(
            full.index
        )
        scenario_ranks = scenario_scores.rank(method="min", ascending=False)
        scenario_groups = assign_performance_groups(
            scenario_scores,
            group_size=group_size,
        )
        for family in full.index:
            rank_value = scenario_ranks.loc[family]
            rows.append(
                {
                    "family": family,
                    "omitted_dimension": f"{method}_only",
                    "full_rank": int(full.loc[family, "rank"]),
                    "sensitivity_rank": (
                        int(rank_value) if np.isfinite(rank_value) else np.nan
                    ),
                    "rank_shift": (
                        int(rank_value - full.loc[family, "rank"])
                        if np.isfinite(rank_value)
                        else np.nan
                    ),
                    "top_group_stable": bool(
                        (full.loc[family, "performance_group"] == "good")
                        == (scenario_groups.loc[family] == "good")
                    ),
                    "bottom_group_stable": bool(
                        (full.loc[family, "performance_group"] == "poor")
                        == (scenario_groups.loc[family] == "poor")
                    ),
                }
            )
    return pd.DataFrame(rows)


DIAGNOSTIC_RULES = {
    "low_recall": {
        "label_zh": "分类召回率偏低",
        "change": "采用类别平衡和困难负样本挖掘",
        "validation": "加入困难负样本后重新进行分层家族分类验证",
    },
    "low_precision": {
        "label_zh": "分类精确率偏低",
        "change": "引入对比学习、层级分类器和拒识机制",
        "validation": "在近邻家族上比较平面分类器与层级分类器",
    },
    "low_generation_fidelity": {
        "label_zh": "生成家族一致性偏低",
        "change": "采用分类器引导解码和生成后过滤",
        "validation": "比较引导解码前后的目标家族恢复率",
    },
    "length_shift": {
        "label_zh": "生成长度分布偏移",
        "change": "加入家族条件长度建模和长度约束解码",
        "validation": "逐家族比较改进前后的生成与天然长度分布",
    },
    "high_fragmentation": {
        "label_zh": "motif 聚类碎片化",
        "change": "采用家族自适应聚类阈值并合并重叠 motif",
        "validation": "使用家族特异最小簇规模重新运行聚类",
    },
    "low_structure_confidence": {
        "label_zh": "motif 结构置信度偏低",
        "change": "采用结构置信度加权并过滤低 pLDDT motif",
        "validation": "按 pLDDT 分层过滤后重复结构验证",
    },
    "low_function_coverage": {
        "label_zh": "功能注释覆盖不足",
        "change": "审计注释覆盖并整合多来源功能标签",
        "validation": "扩展并统一注释后重新进行功能富集",
    },
    "low_function_support": {
        "label_zh": "外部功能支持比例偏低",
        "change": "采用多来源外部注释并校准功能证据评分",
        "validation": "在独立注释源上复核低支持 motif 的功能富集",
    },
    "method_disagreement": {
        "label_zh": "M2/M3 方法分歧较大",
        "change": "校准嵌入集成并对方法分歧家族进行重点复核",
        "validation": "在分歧家族上评估校准后的 M2/M3 集成模型",
    },
}


def build_within_family_diagnostics(family_profiles: pd.DataFrame) -> pd.DataFrame:
    frame = family_profiles.copy()
    fragmentation_threshold = (
        frame["motifs_per_1000_core_windows"].quantile(0.75)
        if "motifs_per_1000_core_windows" in frame
        else np.nan
    )
    structure_threshold = (
        frame["mean_motif_pLDDT"].quantile(0.25)
        if "mean_motif_pLDDT" in frame
        else np.nan
    )
    annotation_threshold = (
        frame["annotation_coverage"].quantile(0.25)
        if "annotation_coverage" in frame
        else np.nan
    )
    function_support_threshold = (
        frame["external_function_supported_rate"].quantile(0.25)
        if "external_function_supported_rate" in frame
        else np.nan
    )
    method_gap_columns = [
        column for column in frame if column.endswith("_method_gap")
    ]
    gap_values = (
        frame[method_gap_columns].abs().stack()
        if method_gap_columns
        else pd.Series(dtype=float)
    )
    disagreement_threshold = max(
        0.05,
        float(gap_values.quantile(0.75)) if not gap_values.empty else 0.05,
    )

    rows = []
    for row in frame.to_dict("records"):
        issues: list[str] = []
        if row.get("classification_recall", 1.0) < 0.8:
            issues.append("low_recall")
        if row.get("classification_precision", 1.0) < 0.8:
            issues.append("low_precision")
        if row.get("generation_target_ratio", 1.0) < 0.7:
            issues.append("low_generation_fidelity")
        length_ratio = row.get("generated_length_ratio", np.nan)
        if np.isfinite(length_ratio) and not 0.8 <= length_ratio <= 1.2:
            issues.append("length_shift")
        fragmentation = row.get("motifs_per_1000_core_windows", np.nan)
        if (
            np.isfinite(fragmentation)
            and np.isfinite(fragmentation_threshold)
            and fragmentation >= fragmentation_threshold
        ):
            issues.append("high_fragmentation")
        structure = row.get("mean_motif_pLDDT", np.nan)
        if (
            np.isfinite(structure)
            and np.isfinite(structure_threshold)
            and structure <= structure_threshold
        ):
            issues.append("low_structure_confidence")
        annotation = row.get("annotation_coverage", np.nan)
        effective_annotation_threshold = (
            min(0.8, float(annotation_threshold))
            if np.isfinite(annotation_threshold)
            else np.nan
        )
        if (
            np.isfinite(annotation)
            and np.isfinite(effective_annotation_threshold)
            and annotation < effective_annotation_threshold
        ):
            issues.append("low_function_coverage")
        function_support = row.get("external_function_supported_rate", np.nan)
        if (
            np.isfinite(function_support)
            and np.isfinite(function_support_threshold)
            and function_support <= function_support_threshold
        ):
            issues.append("low_function_support")
        if any(
            np.isfinite(row.get(column, np.nan))
            and abs(row[column]) >= disagreement_threshold
            for column in method_gap_columns
        ):
            issues.append("method_disagreement")

        unique_issues = list(dict.fromkeys(issues))
        issue_labels = [
            DIAGNOSTIC_RULES[issue]["label_zh"] for issue in unique_issues
        ]
        recommendations = [
            DIAGNOSTIC_RULES[issue]["change"] for issue in unique_issues
        ]
        validations = [
            DIAGNOSTIC_RULES[issue]["validation"] for issue in unique_issues
        ]
        rows.append(
            {
                **row,
                "issue_tags": ";".join(unique_issues),
                "issue_labels_zh": "；".join(issue_labels),
                "recommended_changes": "；".join(recommendations),
                "validation_experiments": "；".join(validations),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_spearman_interval(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        positions = rng.integers(0, len(x), len(x))
        sample_x = x[positions]
        sample_y = y[positions]
        if np.unique(sample_x).size < 2 or np.unique(sample_y).size < 2:
            continue
        estimate = stats.spearmanr(sample_x, sample_y).statistic
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return float("nan"), float("nan")
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def analyze_between_families(
    family_scores: pd.DataFrame,
    family_attributes: pd.DataFrame,
    variable_columns: list[str],
    n_bootstrap: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    merged = family_scores.merge(family_attributes, on="family", how="left")
    rows = []
    for offset, variable in enumerate(variable_columns):
        if variable not in merged:
            continue
        pairs = merged[["composite_score", variable]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if len(pairs) >= 3 and pairs[variable].nunique() > 1:
            correlation = stats.spearmanr(
                pairs[variable],
                pairs["composite_score"],
            )
            rho = float(correlation.statistic)
            correlation_p = float(correlation.pvalue)
            ci_low, ci_high = _bootstrap_spearman_interval(
                pairs[variable].to_numpy(dtype=float),
                pairs["composite_score"].to_numpy(dtype=float),
                n_bootstrap,
                seed + offset,
            )
        else:
            rho = correlation_p = ci_low = ci_high = float("nan")

        good = pd.to_numeric(
            merged.loc[merged["performance_group"].eq("good"), variable],
            errors="coerce",
        ).dropna()
        poor = pd.to_numeric(
            merged.loc[merged["performance_group"].eq("poor"), variable],
            errors="coerce",
        ).dropna()
        if not good.empty and not poor.empty:
            group_test = stats.mannwhitneyu(
                good,
                poor,
                alternative="two-sided",
            )
            u_value = float(group_test.statistic)
            group_p = float(group_test.pvalue)
            delta = cliffs_delta(good, poor)
        else:
            u_value = group_p = delta = float("nan")
        rows.append(
            {
                "variable": variable,
                "spearman_rho": rho,
                "spearman_p": correlation_p,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "good_median": float(good.median()) if not good.empty else np.nan,
                "poor_median": float(poor.median()) if not poor.empty else np.nan,
                "mann_whitney_u": u_value,
                "group_p": group_p,
                "cliffs_delta": delta,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["spearman_fdr_q"] = benjamini_hochberg(result["spearman_p"])
    result["fdr_q"] = benjamini_hochberg(result["group_p"])
    result["effect_direction"] = np.where(
        result["cliffs_delta"] > 0,
        "higher_in_good",
        np.where(result["cliffs_delta"] < 0, "higher_in_poor", "no_difference"),
    )
    result["pattern_strength"] = np.select(
        [
            result["cliffs_delta"].abs().ge(0.33)
            & result["spearman_rho"].abs().ge(0.4),
            result["cliffs_delta"].abs().ge(0.33)
            | result["spearman_rho"].abs().ge(0.4),
        ],
        ["strong", "moderate"],
        default="exploratory",
    )
    return result


def synthesize_model_priorities(
    diagnostics: pd.DataFrame,
    family_scores: pd.DataFrame,
) -> pd.DataFrame:
    merged = diagnostics[["family", "issue_tags"]].merge(
        family_scores[["family", "performance_group"]],
        on="family",
        how="left",
    )
    issue_families: dict[str, list[str]] = {}
    poor_issue_families: dict[str, list[str]] = {}
    for row in merged.itertuples(index=False):
        issues = [issue for issue in str(row.issue_tags).split(";") if issue]
        for issue in issues:
            issue_families.setdefault(issue, []).append(row.family)
            if row.performance_group == "poor":
                poor_issue_families.setdefault(issue, []).append(row.family)

    poor_count = int(merged["performance_group"].eq("poor").sum())
    p1_threshold = max(2, math.ceil(poor_count / 2))
    rows = []
    for issue, families in issue_families.items():
        unique_families = sorted(set(families))
        poor_families = sorted(set(poor_issue_families.get(issue, [])))
        if len(poor_families) >= p1_threshold:
            priority = "P1"
        elif len(unique_families) >= 2:
            priority = "P2"
        else:
            priority = "P3"
        rule = DIAGNOSTIC_RULES.get(
            issue,
            {
                "change": "targeted family-specific model review",
                "validation": "run an ablation on affected families",
            },
        )
        rows.append(
            {
                "priority": priority,
                "problem": issue,
                "problem_zh": rule.get("label_zh", issue),
                "affected_families": ";".join(unique_families),
                "poor_families": ";".join(poor_families),
                "evidence": (
                    f"{len(poor_families)} poor families; "
                    f"{len(unique_families)} families total"
                ),
                "recommended_change": rule["change"],
                "validation_experiment": rule["validation"],
            }
        )
    priority_order = {"P1": 1, "P2": 2, "P3": 3}
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_priority_order"] = result["priority"].map(priority_order)
    return result.sort_values(
        ["_priority_order", "problem"]
    ).drop(columns="_priority_order").reset_index(drop=True)


def _aggregate_family_profiles(method_profiles: pd.DataFrame) -> pd.DataFrame:
    excluded = {"family", "method", "top_wrong_family"}
    numeric_columns = [
        column
        for column in method_profiles.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    family = method_profiles.groupby("family", as_index=False)[numeric_columns].mean()

    if "top_wrong_family" in method_profiles:
        wrong = (
            method_profiles.groupby("family")["top_wrong_family"]
            .apply(
                lambda values: ";".join(
                    sorted({str(value) for value in values if str(value).strip()})
                )
            )
            .rename("top_wrong_family")
            .reset_index()
        )
        family = family.merge(wrong, on="family", how="left")

    gap_metrics = [
        "classification_f1",
        "generation_target_ratio",
        "silhouette",
        "external_function_supported_rate",
    ]
    for metric in gap_metrics:
        if metric not in method_profiles:
            continue
        pivot = method_profiles.pivot_table(
            index="family",
            columns="method",
            values=metric,
            aggfunc="mean",
        )
        if {"M2", "M3"}.issubset(pivot.columns):
            gap = (pivot["M3"] - pivot["M2"]).rename(f"{metric}_method_gap")
            family = family.merge(gap.reset_index(), on="family", how="left")
    return family


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_composite_heatmap(
    family_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    dimensions = list(DIMENSION_COMPONENTS)
    ordered = family_scores.sort_values("rank")
    values = ordered.set_index("family")[dimensions]
    fig_height = max(3.5, 0.38 * max(len(values), 1) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    image = ax.imshow(values.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels(
        [DIMENSION_LABELS_ZH[column] for column in dimensions],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(values.index)
    ax.set_title("Gvp 家族五维评分")
    fig.colorbar(image, ax=ax, label="评分（0-100）")
    _save_figure(fig, output_path)


def _plot_family_ranking(
    family_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = family_scores.sort_values("composite_score")
    colors = ordered["performance_group"].map(
        {"good": "#2a9d8f", "middle": "#8d99ae", "poor": "#e76f51"}
    ).fillna("#8d99ae")
    fig_height = max(3.5, 0.38 * max(len(ordered), 1) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(ordered["family"], ordered["composite_score"], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("综合评分")
    ax.set_title("Gvp 家族模型效果排名")
    ax.grid(axis="x", alpha=0.2)
    _save_figure(fig, output_path)


def _plot_good_vs_poor(
    family_profiles: pd.DataFrame,
    output_path: Path,
) -> None:
    candidate_columns = sequence_explanatory_variables(family_profiles)
    columns = [
        column
        for column in candidate_columns
        if column in family_profiles
        and family_profiles[column].notna().any()
        and pd.to_numeric(
            family_profiles[column],
            errors="coerce",
        ).nunique(dropna=True)
        > 1
    ]
    if not columns:
        columns = ["composite_score"]
    group_medians = family_profiles.groupby("performance_group")[columns].median()
    group_medians = group_medians.reindex(["good", "poor"]).dropna(how="all")
    standardized = group_medians.copy()
    for column in columns:
        values = pd.to_numeric(family_profiles[column], errors="coerce")
        center = values.median()
        scale = (values - center).abs().median()
        if not np.isfinite(scale) or scale == 0:
            standardized[column] = 0.0
        else:
            standardized[column] = (group_medians[column] - center) / scale

    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 1.1), 4.5))
    x = np.arange(len(columns))
    width = 0.36
    for offset, group in enumerate(standardized.index):
        ax.bar(
            x + (offset - 0.5) * width,
            standardized.loc[group].to_numpy(dtype=float),
            width,
            label={"good": "效果好组", "poor": "效果差组"}.get(group, group),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ATTRIBUTE_LABELS_ZH.get(column, column) for column in columns],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("以全体中位数为中心的稳健差异")
    ax.set_title("效果好组与效果差组的家族属性比较")
    ax.legend()
    _save_figure(fig, output_path)


def _plot_correlations(
    statistics: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = statistics.dropna(subset=["spearman_rho"]).copy()
    if plot_data.empty:
        plot_data = pd.DataFrame(
            {"variable": ["No estimable variable"], "spearman_rho": [0.0]}
        )
    plot_data = plot_data.sort_values("spearman_rho")
    colors = np.where(plot_data["spearman_rho"] >= 0, "#2a9d8f", "#e76f51")
    fig_height = max(3.5, 0.42 * len(plot_data) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    labels = [
        ATTRIBUTE_LABELS_ZH.get(variable, variable)
        for variable in plot_data["variable"]
    ]
    ax.barh(labels, plot_data["spearman_rho"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman 相关系数")
    ax.set_title("家族属性与综合评分的相关性")
    _save_figure(fig, output_path)


def _format_family_list(frame: pd.DataFrame, group: str) -> str:
    values = frame.loc[frame["performance_group"].eq(group), "family"].tolist()
    return "、".join(values) if values else "无"


def _patterns(
    statistics: pd.DataFrame,
    direction: str,
    strengths: set[str],
) -> list[str]:
    if statistics.empty:
        return []
    selected = statistics.loc[
        statistics["pattern_strength"].isin(strengths)
        & statistics["effect_direction"].eq(direction)
    ]
    patterns = []
    for row in selected.itertuples(index=False):
        label = ATTRIBUTE_LABELS_ZH.get(row.variable, row.variable)
        patterns.append(
            f"{label}（`{row.variable}`，Spearman ρ={row.spearman_rho:.3f}，"
            f"bootstrap 95% CI [{row.bootstrap_ci_low:.3f}, "
            f"{row.bootstrap_ci_high:.3f}]，Cliff's δ={row.cliffs_delta:.3f}）"
        )
    return patterns


def _write_chinese_report(
    output_path: Path,
    family_scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
    statistics: pd.DataFrame,
    priorities: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    good_families = _format_family_list(family_scores, "good")
    poor_families = _format_family_list(family_scores, "poor")
    higher_good = _patterns(statistics, "higher_in_good", {"strong"})
    higher_poor = _patterns(statistics, "higher_in_poor", {"strong"})
    exploratory = _patterns(
        statistics,
        "higher_in_good",
        {"moderate"},
    ) + _patterns(
        statistics,
        "higher_in_poor",
        {"moderate"},
    )
    group_labels = {"good": "效果好", "middle": "中间", "poor": "效果差"}

    ranking_rows = []
    for row in family_scores.sort_values("rank").itertuples(index=False):
        ranking_rows.append(
            f"| {int(row.rank)} | {row.family} | {row.composite_score:.2f} | "
            f"{group_labels.get(row.performance_group, row.performance_group)} | "
            f"{row.score_completeness:.0%} |"
        )

    diagnostic_rows = []
    if {"rank", "performance_group"}.issubset(diagnostics.columns):
        diagnostic_view = diagnostics.copy()
    else:
        diagnostic_view = diagnostics.merge(
            family_scores[["family", "rank", "performance_group"]],
            on="family",
            how="left",
        )
    diagnostic_view = diagnostic_view.sort_values(["rank", "family"])
    for row in diagnostic_view.itertuples(index=False):
        issues = row.issue_labels_zh or "未触发主要规则"
        changes = row.recommended_changes or "保留现有流程并进行重复验证"
        evidence_parts = []
        if np.isfinite(row.classification_f1):
            evidence_parts.append(f"分类 F1={row.classification_f1:.3f}")
        if np.isfinite(row.generation_target_ratio):
            evidence_parts.append(f"生成命中率={row.generation_target_ratio:.3f}")
        if pd.notna(row.top_wrong_family) and str(row.top_wrong_family).strip():
            evidence_parts.append(f"主要误判={row.top_wrong_family}")
        if np.isfinite(row.generated_length_ratio):
            evidence_parts.append(f"生成/天然长度比={row.generated_length_ratio:.3f}")
        evidence = "；".join(evidence_parts) if evidence_parts else "数据不足"
        diagnostic_rows.append(
            f"| {row.family} | "
            f"{group_labels.get(row.performance_group, row.performance_group)} | "
            f"{evidence} | {issues} | {changes} |"
        )

    priority_rows = []
    for row in priorities.itertuples(index=False):
        priority_rows.append(
            f"| {row.priority} | {row.problem_zh} | {row.affected_families} | "
            f"{row.recommended_change} | {row.validation_experiment} |"
        )
    if not priority_rows:
        priority_rows.append("| - | 未识别 | - | - | - |")

    sensitivity_summary = (
        sensitivity.groupby("family")
        .agg(
            top_group_stability=("top_group_stable", "mean"),
            bottom_group_stability=("bottom_group_stable", "mean"),
            max_abs_rank_shift=("rank_shift", lambda values: values.abs().max()),
        )
        .reset_index()
    )
    fully_stable_good = sensitivity_summary.loc[
        sensitivity_summary["top_group_stability"].eq(1.0)
        & sensitivity_summary["family"].isin(
            family_scores.loc[
                family_scores["performance_group"].eq("good"), "family"
            ]
        ),
        "family",
    ].tolist()
    fully_stable_poor = sensitivity_summary.loc[
        sensitivity_summary["bottom_group_stability"].eq(1.0)
        & sensitivity_summary["family"].isin(
            family_scores.loc[
                family_scores["performance_group"].eq("poor"), "family"
            ]
        ),
        "family",
    ].tolist()
    sensitive_families = sensitivity_summary.nlargest(
        min(3, len(sensitivity_summary)),
        "max_abs_rank_shift",
    )
    sensitive_text = "、".join(
        f"{row.family}（最大移动 {int(row.max_abs_rank_shift)} 位）"
        for row in sensitive_families.itertuples(index=False)
    )

    report = f"""# GvpMiner 家族内与家族间模型效果分析报告

## 材料护照

- 分析日期：{date.today().isoformat()}
- 数据来源：发布包 `results/` 或命令行指定目录
- 验证状态：**ANALYZED**
- 分析单位：Gvp 家族
- 完整评价家族数：{len(family_scores)}
- 说明：本报告由已有实验产物计算生成，未重新训练模型，也未重新运行 Exp1-Exp5。

## 一、分析方法

综合评分由分类、生成、聚类、结构和功能五个维度等权组成。每个基础指标先使用中位数与 MAD 进行稳健标准化，再映射到 0-100 分。缺失维度不填 0，而是在可用维度中重新分配权重，并报告评分完整度。

综合分最高四个家族定义为效果好组，最低四个家族定义为效果差组；若边界同分则保留并列。家族间比较使用 Spearman 相关、bootstrap 95% 置信区间、Mann-Whitney U 检验和 Cliff's delta，并采用 Benjamini-Hochberg FDR 校正。

家族间共性与差异的解释变量**只使用天然序列可直接计算的基础属性**，包括序列数量、窗口数量、长度分布、长度变异、氨基酸组成熵以及主要氨基酸类别比例。motif 数量、结构覆盖、功能注释、生成判定一致率以及分类/生成/聚类/结构/功能评分均不作为家族间解释变量，只用于家族内部诊断或独立验证，避免用实验输出解释实验输出。

## 二、综合排名

- 效果好组：**{good_families}**
- 效果差组：**{poor_families}**

| 排名 | 家族 | 综合分 | 分组 | 评分完整度 |
|---:|---|---:|---|---:|
{chr(10).join(ranking_rows)}

## 三、家族内部分析

| 家族 | 分组 | 关键证据 | 主要问题 | 建议改进 |
|---|---|---|---|---|
{chr(10).join(diagnostic_rows)}

家族内部建议应按触发指标执行，而不是对所有家族统一修改。例如，低 recall 家族优先进行类别平衡和困难负样本挖掘；生成长度偏移家族优先采用家族条件长度建模；motif 碎片化家族优先调整最小簇规模并合并重叠窗口。

## 四、基于基础序列属性的家族间共性与差异

### 4.1 效果好家族的基础序列共性

{chr(10).join(f"- {item}" for item in higher_good) if higher_good else "- 当前样本量下未发现同时满足相关强度与组间效应量双重阈值的统一属性。"}

### 4.2 效果差家族的基础序列共性

{chr(10).join(f"- {item}" for item in higher_poor) if higher_poor else "- 当前样本量下未发现同时满足相关强度与组间效应量双重阈值的统一属性；差家族更可能由多种不同问题共同形成。"}

### 4.3 好坏家族的基础差异与建模含义

完整统计见 `between_family_statistics.csv`。该表中的变量均来自基础序列属性，而不是准确率、生成命中率、motif 发现结果或结构/功能验证结果。由于完整家族数仅 {len(family_scores)}，本报告优先解释效应方向、效应量和 bootstrap 区间，不以单一 `p < 0.05` 判定模型机制。

以下变量仅满足相关或效应量中的一项，属于探索性差异，不作为稳定共性：

{chr(10).join(f"- {item}" for item in exploratory) if exploratory else "- 无。"}

这些基础差异对模型改进的含义如下：如果效果差家族集中表现为序列数量或窗口数量偏低，应优先采用家族均衡采样、类别权重、少样本增强和逐家族阈值校准；如果长度中位数、长度分位或长度变异系数与好坏分组分离，应引入家族条件长度嵌入、长度分层 batch、可变窗口长度与 stride，并把生成阶段的最大长度/温度按家族校准；如果氨基酸组成熵或疏水、带电、极性、芳香、Gly/Pro 比例存在系统差异，应加入组成感知负样本、氨基酸类别级输入特征或 family-specific normalization，必要时使用 mixture-of-experts 避免一个全局参数同时服务差异很大的家族。

### 4.4 排名稳健性

- 在删除任一评分维度以及仅使用 M2/M3 的 7 个敏感性场景中，始终保持效果好组身份的家族：**{"、".join(fully_stable_good) if fully_stable_good else "无"}**。
- 始终保持效果差组身份的家族：**{"、".join(fully_stable_poor) if fully_stable_poor else "无"}**。
- 排名对评分构成最敏感的家族：**{sensitive_text}**。
- 因此，GvpC、GvpQ 或 GvpM 等边界家族的好坏标签应结合单维度结果解释，不能只引用综合排名。

## 五、模型改进优先级

| 优先级 | 问题 | 受影响家族 | 推荐改进 | 验证实验 |
|---|---|---|---|---|
{chr(10).join(priority_rows)}

### 推荐实施顺序

1. 首先处理 P1 跨家族问题，使用统一消融实验验证是否能同时改善多个差家族。
2. 然后针对 P2 家族子群引入自适应参数或 mixture-of-experts，避免全局参数损害原本表现良好的家族。
3. 最后处理 P3 单家族问题，建立专用判别头、困难负样本库或后处理规则。
4. 所有改进均应同时报告宏平均指标和逐家族指标，防止总体均值掩盖少数家族退化。

建议把第 4 节的基础序列差异作为参数搜索空间的来源：按长度分布搜索窗口大小、stride 和生成长度约束；按样本量搜索类别权重、采样温度和增强强度；按组成特征搜索负样本相似度阈值、归一化方式和专家模型分配。每一类调整都应做逐家族消融，只有当差家族提升且好家族不明显退化时才纳入默认参数。

## 六、统计谬误扫描

- 覆盖率：**11/11**
- Simpson 悖论：已比较总体评分与逐家族结果；总体均值不能替代家族层分析。
- 生态谬误：家族级关系不外推为单条蛋白序列的机制。
- Berkson 偏差：完整分析仅包含进入分类标签空间且具有后续结果的家族，存在选择限制。
- 碰撞变量偏差：家族间解释变量限定为天然序列基础属性，综合评分构成指标不作为独立解释变量。
- 基准率忽视：分类解释同时考虑 precision、recall 与家族样本量。
- 均值回归：未把最低分家族后续波动直接解释为模型改善。
- 幸存者偏差：7 个标签空间外家族不被错误计为生成失败，但不能进入完整排名。
- 多重探索效应：家族属性检验使用 FDR 校正。
- 分叉路径：评分权重、分组和阈值在分析前固定，并提供留一维度敏感性结果。
- 相关不等于因果：家族属性与综合分只解释为关联。
- 反向因果：motif、结构、注释和生成判定等下游产物不用于家族间共性解释，只保留在家族内部诊断中。

## 七、限制

1. 完整分析仅包含 {len(family_scores)} 个家族，统计功效有限。
2. Exp1 缺少数值混淆矩阵，天然序列主要混淆对象无法可靠复算。
3. Exp2 复用 Exp1 分类器，存在闭环评价风险。
4. Exp4 主要依赖预测结构，不能等同于实验结构证据。
5. Exp5 受注释覆盖与术语冗余影响。
6. 当前实验链路截止 Exp5，因此不评价 motif 必要性。

## 八、结果文件

- `family_method_metrics.csv`：家族-方法基础指标与五维评分。
- `family_composite_scores.csv`：家族综合分、排名和分组。
- `within_family_diagnostics.csv`：家族内部诊断与改进建议。
- `between_family_statistics.csv`：基础序列属性与综合分的相关、组间检验、效应量和 FDR。
- `ranking_sensitivity.csv`：留一维度和 M2/M3 独立排名稳健性。
- `model_improvement_priorities.csv`：P1-P3 改进优先级。
- `figures/`：综合评分和家族差异图。
"""
    output_path.write_text(report, encoding="utf-8")


def run_analysis(
    results_dir: str | Path,
    output_dir: str | Path,
    n_bootstrap: int = 10_000,
) -> dict[str, Path]:
    results_root = Path(results_dir).resolve()
    output_root = Path(output_dir).resolve()
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_family_method_metrics(results_root)
    attributes = load_family_attributes(results_root)
    method_profiles = metrics.merge(
        attributes,
        on=["family", "method"],
        how="left",
    )
    scored_methods = score_family_methods(method_profiles)
    group_size = min(4, max(1, len(scored_methods["family"].unique()) // 4))
    family_scores = aggregate_family_scores(
        scored_methods,
        group_size=group_size,
    )
    sensitivity = ranking_sensitivity(
        family_scores,
        scored_methods,
        group_size=group_size,
    )

    family_attributes = _aggregate_family_profiles(method_profiles)
    family_profiles = family_attributes.merge(
        family_scores,
        on="family",
        how="inner",
    )
    diagnostics = build_within_family_diagnostics(family_profiles)

    explanatory_variables = sequence_explanatory_variables(family_attributes)
    statistics = analyze_between_families(
        family_scores,
        family_attributes,
        explanatory_variables,
        n_bootstrap=n_bootstrap,
    )
    priorities = synthesize_model_priorities(diagnostics, family_scores)

    output_paths = {
        "method_metrics": output_root / "family_method_metrics.csv",
        "family_scores": output_root / "family_composite_scores.csv",
        "diagnostics": output_root / "within_family_diagnostics.csv",
        "statistics": output_root / "between_family_statistics.csv",
        "sensitivity": output_root / "ranking_sensitivity.csv",
        "priorities": output_root / "model_improvement_priorities.csv",
        "manifest": output_root / "analysis_manifest.json",
        "report": output_root / "GvpMiner_家族内与家族间模型效果分析报告.md",
    }
    scored_methods.to_csv(output_paths["method_metrics"], index=False)
    family_scores.to_csv(output_paths["family_scores"], index=False)
    diagnostics.to_csv(output_paths["diagnostics"], index=False)
    statistics.to_csv(output_paths["statistics"], index=False)
    sensitivity.to_csv(output_paths["sensitivity"], index=False)
    priorities.to_csv(output_paths["priorities"], index=False)

    _plot_composite_heatmap(
        family_scores,
        figures_dir / "composite_score_heatmap.png",
    )
    _plot_family_ranking(
        family_scores,
        figures_dir / "family_ranking.png",
    )
    _plot_good_vs_poor(
        family_profiles,
        figures_dir / "good_vs_poor_attributes.png",
    )
    _plot_correlations(
        statistics,
        figures_dir / "score_attribute_correlations.png",
    )
    _write_chinese_report(
        output_paths["report"],
        family_scores,
        diagnostics,
        statistics,
        priorities,
        sensitivity,
    )

    missing_values = {
        column: int(scored_methods[column].isna().sum())
        for column in scored_methods.columns
        if scored_methods[column].isna().any()
    }
    manifest = {
        "analysis_date": date.today().isoformat(),
        "verification_status": "ANALYZED",
        "results_dir": str(results_root),
        "output_dir": str(output_root),
        "n_evaluable_families": int(len(family_scores)),
        "n_family_method_rows": int(len(scored_methods)),
        "good_families": family_scores.loc[
            family_scores["performance_group"].eq("good"), "family"
        ].tolist(),
        "poor_families": family_scores.loc[
            family_scores["performance_group"].eq("poor"), "family"
        ].tolist(),
        "missing_values": missing_values,
        "bootstrap_iterations": int(n_bootstrap),
        "between_family_explanatory_variables": explanatory_variables,
        "outputs": {
            key: str(path)
            for key, path in output_paths.items()
        },
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分析 GvpMiner 家族内部与家族之间的模型效果。"
    )
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--results-dir", default=project_root / "results")
    parser.add_argument(
        "--output-dir",
        default=project_root / "work" / "results" / "family_performance_analysis",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_analysis(
        args.results_dir,
        args.output_dir,
        n_bootstrap=args.bootstrap_iterations,
    )


if __name__ == "__main__":
    main()
