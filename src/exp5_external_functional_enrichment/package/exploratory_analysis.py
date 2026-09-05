from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .support_analysis import derived_seed


METHOD_ORDER = ("GLAM2", "M2", "M3", "MEME", "MotifAE", "STREME")
REQUIRED_COLUMNS = (
    "family",
    "method",
    "motif_cluster",
    "motif_rank",
    "rank_source",
    "protein_count",
    "annotation_coverage",
    "protein_size_bin",
    "annotation_coverage_bin",
    "is_testable",
    "is_supported",
    "status",
)
ALLOWED_SUPPORT_STATUSES = (
    "tested_supported",
    "tested_not_significant",
    "unevaluable_no_family_annotations",
    "unevaluable_no_eligible_gene_sets",
    "unevaluable_insufficient_annotated_foreground",
)
STATUS_COLUMNS = {
    "tested_supported": "supported_motifs",
    "tested_not_significant": "tested_nonsignificant_motifs",
    "unevaluable_no_family_annotations": "no_annotation_family_motifs",
    "unevaluable_no_eligible_gene_sets": "no_eligible_term_motifs",
    "unevaluable_insufficient_annotated_foreground": (
        "insufficient_family_background_motifs"
    ),
}


def _coerce_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def validate_motif_level_table(rows: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"motif-level table missing columns: {missing}")
    if rows.empty:
        raise ValueError("motif-level table is empty")
    keys = ["family", "method", "motif_cluster"]
    if rows.duplicated(keys).any():
        raise ValueError("duplicate motif keys in family/method/motif_cluster")
    statuses = set(rows["status"].astype(str))
    unknown = sorted(statuses - set(ALLOWED_SUPPORT_STATUSES))
    if unknown:
        raise ValueError(f"unknown support statuses: {unknown}")

    testable = rows["is_testable"].map(_coerce_bool)
    supported = rows["is_supported"].map(_coerce_bool)
    expected_testable = rows["status"].isin(
        ["tested_supported", "tested_not_significant"]
    )
    expected_supported = rows["status"].eq("tested_supported")
    if not testable.equals(expected_testable) or not supported.equals(
        expected_supported
    ):
        raise ValueError("status/boolean mismatch in motif-level table")
    if (supported & ~testable).any():
        raise ValueError("supported motifs must be testable")

    protein_count = pd.to_numeric(rows["protein_count"], errors="coerce")
    coverage = pd.to_numeric(rows["annotation_coverage"], errors="coerce")
    rank = pd.to_numeric(rows["motif_rank"], errors="coerce")
    if protein_count.isna().any() or (~np.isfinite(protein_count)).any():
        raise ValueError("protein_count must be finite")
    if (protein_count <= 0).any():
        raise ValueError("protein_count must be positive")
    if coverage.isna().any() or (~np.isfinite(coverage)).any():
        raise ValueError("annotation_coverage must be finite")
    if ((coverage < 0) | (coverage > 1)).any():
        raise ValueError("annotation_coverage must be within [0, 1]")
    if rank.isna().any() or (~np.isfinite(rank)).any():
        raise ValueError("motif_rank must be finite")
    if (rank <= 0).any():
        raise ValueError("motif_rank must be positive")


def _method_sort_key(method: str) -> tuple[int, str]:
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


def summarise_method_yield(rows: pd.DataFrame) -> pd.DataFrame:
    validate_motif_level_table(rows)
    records: list[dict[str, object]] = []
    for method, group in rows.groupby("method", sort=False):
        counts = group["status"].value_counts()
        record: dict[str, object] = {
            "method": str(method),
            "candidate_motifs": int(len(group)),
            "testable_motifs": int(
                counts.get("tested_supported", 0)
                + counts.get("tested_not_significant", 0)
            ),
            "families_present": int(group["family"].nunique()),
        }
        for status, column in STATUS_COLUMNS.items():
            record[column] = int(counts.get(status, 0))
        total = int(record["candidate_motifs"])
        testable = int(record["testable_motifs"])
        supported = int(record["supported_motifs"])
        decomposition = sum(int(record[column]) for column in STATUS_COLUMNS.values())
        if decomposition != total:
            raise ValueError(
                f"status decomposition failed for {method}: {decomposition} != {total}"
            )
        record["testable_fraction"] = testable / total
        record["supported_fraction_all"] = supported / total
        record["supported_fraction_testable"] = (
            supported / testable if testable else math.nan
        )
        records.append(record)
    records.sort(key=lambda item: _method_sort_key(str(item["method"])))
    return pd.DataFrame(records)


def add_normalized_rank(rows: pd.DataFrame) -> pd.DataFrame:
    validate_motif_level_table(rows)
    ranked_groups: list[pd.DataFrame] = []
    for _, group in rows.groupby(["family", "method"], sort=True):
        ranked = group.sort_values(
            ["motif_rank", "motif_cluster"], kind="mergesort"
        ).copy()
        size = len(ranked)
        ranked["normalized_rank"] = (
            0.0 if size == 1 else np.arange(size, dtype=float) / (size - 1)
        )
        ranked_groups.append(ranked)
    return pd.concat(ranked_groups, ignore_index=True)


def family_cluster_bootstrap_ci(
    grouped_counts: pd.DataFrame,
    numerator_column: str,
    denominator_column: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    required = {"family", numerator_column, denominator_column}
    missing = sorted(required - set(grouped_counts.columns))
    if missing:
        raise ValueError(f"bootstrap counts missing columns: {missing}")
    family_counts = (
        grouped_counts.groupby("family", as_index=False)[
            [numerator_column, denominator_column]
        ]
        .sum()
        .sort_values("family")
    )
    if family_counts.empty:
        return {
            "ci_low": math.nan,
            "ci_high": math.nan,
            "requested_iterations": iterations,
            "valid_iterations": 0,
            "skipped_iterations": iterations,
            "bootstrap_seed": seed,
            "ci_status": "no_families",
        }
    numerators = family_counts[numerator_column].to_numpy(dtype=float)
    denominators = family_counts[denominator_column].to_numpy(dtype=float)
    generator = np.random.Generator(np.random.PCG64(seed))
    values: list[float] = []
    skipped = 0
    for _ in range(iterations):
        indices = generator.integers(0, len(family_counts), len(family_counts))
        denominator = float(denominators[indices].sum())
        if denominator <= 0:
            skipped += 1
            continue
        values.append(float(numerators[indices].sum()) / denominator)
    valid = len(values)
    stable = valid / iterations >= 0.9
    return {
        "ci_low": float(np.quantile(values, 0.025)) if values else math.nan,
        "ci_high": float(np.quantile(values, 0.975)) if values else math.nan,
        "requested_iterations": iterations,
        "valid_iterations": valid,
        "skipped_iterations": skipped,
        "bootstrap_seed": seed,
        "ci_status": "stable" if stable else "unstable",
    }


def _derived_bootstrap_seed(
    global_seed: int, method: str, top_k: int, metric: str
) -> int:
    import hashlib

    material = f"{global_seed}|{method}|{top_k}|{metric}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def summarise_topk_sensitivity(
    rows: pd.DataFrame,
    *,
    k_values: Sequence[int] = (1, 3, 5, 10, 20, 50),
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = 20_260_812,
) -> pd.DataFrame:
    ranked = add_normalized_rank(rows)
    records: list[dict[str, object]] = []
    for top_k in k_values:
        if top_k < 1:
            raise ValueError("all Top-K values must be positive")
        selected = (
            ranked.sort_values(
                ["family", "method", "motif_rank", "motif_cluster"],
                kind="mergesort",
            )
            .groupby(["family", "method"], sort=False, group_keys=False)
            .head(top_k)
            .copy()
        )
        selected["_selected"] = 1
        selected["_testable"] = selected["is_testable"].map(_coerce_bool).astype(int)
        selected["_supported"] = selected["is_supported"].map(_coerce_bool).astype(int)
        for method, group in selected.groupby("method", sort=False):
            family_counts = (
                group.groupby("family", as_index=False)[
                    ["_selected", "_testable", "_supported"]
                ]
                .sum()
                .sort_values("family")
            )
            selected_count = int(group["_selected"].sum())
            testable_count = int(group["_testable"].sum())
            supported_count = int(group["_supported"].sum())
            record: dict[str, object] = {
                "method": str(method),
                "top_k_requested": int(top_k),
                "selected_motif_count": selected_count,
                "testable_motif_count": testable_count,
                "supported_motif_count": supported_count,
                "supported_fraction_selected": supported_count / selected_count,
                "supported_fraction_testable": (
                    supported_count / testable_count if testable_count else math.nan
                ),
                "testable_fraction_selected": testable_count / selected_count,
                "families_present": int(group["family"].nunique()),
                "rank_source_set": ";".join(
                    sorted(set(group["rank_source"].astype(str)))
                ),
            }
            metrics = (
                (
                    "supported_fraction_selected",
                    "_supported",
                    "_selected",
                ),
                (
                    "supported_fraction_testable",
                    "_supported",
                    "_testable",
                ),
                (
                    "testable_fraction_selected",
                    "_testable",
                    "_selected",
                ),
            )
            for metric, numerator, denominator in metrics:
                ci = family_cluster_bootstrap_ci(
                    family_counts,
                    numerator,
                    denominator,
                    iterations=bootstrap_iterations,
                    seed=_derived_bootstrap_seed(
                        bootstrap_seed, str(method), int(top_k), metric
                    ),
                )
                for key, value in ci.items():
                    record[f"{metric}_{key}"] = value
            records.append(record)
    records.sort(
        key=lambda item: (
            _method_sort_key(str(item["method"])),
            int(item["top_k_requested"]),
        )
    )
    return pd.DataFrame(records)


def summarise_rank_deciles(
    rows: pd.DataFrame,
    *,
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = 20_260_812,
) -> pd.DataFrame:
    ranked = add_normalized_rank(rows)
    ranked["_testable"] = ranked["is_testable"].map(_coerce_bool)
    ranked["_supported"] = ranked["is_supported"].map(_coerce_bool).astype(int)
    testable = ranked[ranked["_testable"]].copy()
    testable["rank_decile"] = np.minimum(
        10, np.floor(testable["normalized_rank"].astype(float) * 10).astype(int) + 1
    )
    records: list[dict[str, object]] = []
    for method in sorted(ranked["method"].astype(str).unique(), key=_method_sort_key):
        method_rows = testable[testable["method"].astype(str) == method]
        for decile in range(1, 11):
            group = method_rows[method_rows["rank_decile"] == decile]
            testable_count = int(len(group))
            supported_count = int(group["_supported"].sum())
            record: dict[str, object] = {
                "method": method,
                "rank_decile": decile,
                "normalized_rank_low": (decile - 1) / 10,
                "normalized_rank_high": decile / 10,
                "testable_motif_count": testable_count,
                "supported_motif_count": supported_count,
                "supported_fraction_testable": (
                    supported_count / testable_count if testable_count else math.nan
                ),
                "family_count": int(group["family"].nunique()),
                "status": "observed" if testable_count else "no_observations",
            }
            if testable_count:
                family_counts = (
                    group.assign(_denominator=1)
                    .groupby("family", as_index=False)[["_supported", "_denominator"]]
                    .sum()
                )
                ci = family_cluster_bootstrap_ci(
                    family_counts,
                    "_supported",
                    "_denominator",
                    iterations=bootstrap_iterations,
                    seed=_derived_bootstrap_seed(
                        bootstrap_seed, method, decile, "rank_decile"
                    ),
                )
            else:
                ci = {
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "requested_iterations": bootstrap_iterations,
                    "valid_iterations": 0,
                    "skipped_iterations": bootstrap_iterations,
                    "bootstrap_seed": _derived_bootstrap_seed(
                        bootstrap_seed, method, decile, "rank_decile"
                    ),
                    "ci_status": "no_observations",
                }
            for key, value in ci.items():
                record[f"supported_fraction_testable_{key}"] = value
            records.append(record)
    return pd.DataFrame(records)


def benjamini_hochberg(
    p_values: Iterable[float | None],
) -> list[float]:
    values = list(p_values)
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    adjusted = [math.nan] * len(values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index, value = ordered[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, value * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _not_estimable_rank_result(
    method: str,
    reason: str,
    *,
    sample_size: int,
    family_count: int,
) -> dict[str, object]:
    return {
        "method": method,
        "status": "not_estimable",
        "reason": reason,
        "sample_size": sample_size,
        "family_cluster_count": family_count,
        "rank_coefficient": math.nan,
        "rank_robust_se": math.nan,
        "rank_odds_ratio": math.nan,
        "rank_ci_low": math.nan,
        "rank_ci_high": math.nan,
        "rank_p_value": math.nan,
        "rank_q_value": math.nan,
        "converged": False,
        "iteration_count": 0,
        "hessian_condition_number": math.nan,
        "standard_error_type": "family_cluster_robust",
    }


def fit_method_rank_model(rows: pd.DataFrame, method: str) -> dict[str, object]:
    ranked = add_normalized_rank(rows)
    method_rows = ranked[
        (ranked["method"].astype(str) == method)
        & ranked["is_testable"].map(_coerce_bool)
    ].copy()
    sample_size = len(method_rows)
    families = sorted(method_rows["family"].astype(str).unique())
    family_count = len(families)
    if sample_size == 0:
        return _not_estimable_rank_result(
            method, "no testable motifs", sample_size=0, family_count=0
        )
    rank_values = method_rows["normalized_rank"].to_numpy(dtype=float)
    if float(np.ptp(rank_values)) <= 1e-12:
        return _not_estimable_rank_result(
            method,
            "rank has no within-method variation",
            sample_size=sample_size,
            family_count=family_count,
        )
    if family_count < 4:
        return _not_estimable_rank_result(
            method,
            "fewer than four protein-family clusters",
            sample_size=sample_size,
            family_count=family_count,
        )
    coverage = method_rows["annotation_coverage"].to_numpy(dtype=float)
    log_protein = np.log2(method_rows["protein_count"].to_numpy(dtype=float))

    def standardize(values: np.ndarray, label: str) -> np.ndarray | None:
        scale = float(values.std(ddof=0))
        if not math.isfinite(scale) or scale <= 1e-12:
            return None
        return (values - float(values.mean())) / scale

    coverage_standardized = standardize(coverage, "annotation_coverage")
    protein_standardized = standardize(log_protein, "log2_protein_count")
    if coverage_standardized is None or protein_standardized is None:
        return _not_estimable_rank_result(
            method,
            "covariate has no variation",
            sample_size=sample_size,
            family_count=family_count,
        )
    family_values = method_rows["family"].astype(str).to_numpy()
    family_dummies = np.column_stack(
        [(family_values == family).astype(float) for family in families[1:]]
    )
    x = np.column_stack(
        [
            np.ones(sample_size),
            rank_values,
            coverage_standardized,
            protein_standardized,
            family_dummies,
        ]
    )
    y = method_rows["is_supported"].map(_coerce_bool).to_numpy(dtype=float)
    parameter_count = x.shape[1]
    if sample_size <= parameter_count:
        return _not_estimable_rank_result(
            method,
            "sample size does not exceed parameter count",
            sample_size=sample_size,
            family_count=family_count,
        )
    if np.linalg.matrix_rank(x) < parameter_count:
        return _not_estimable_rank_result(
            method,
            "rank-deficient design matrix",
            sample_size=sample_size,
            family_count=family_count,
        )
    beta = np.zeros(parameter_count, dtype=float)
    converged = False
    condition = math.nan
    iteration_count = 0
    for iteration in range(1, 101):
        eta = np.clip(x @ beta, -35.0, 35.0)
        probability = np.clip(1 / (1 + np.exp(-eta)), 1e-9, 1 - 1e-9)
        weights = probability * (1 - probability)
        hessian = x.T @ (weights[:, None] * x)
        condition = float(np.linalg.cond(hessian))
        if not math.isfinite(condition) or condition > 1e12:
            return _not_estimable_rank_result(
                method,
                "ill-conditioned Hessian",
                sample_size=sample_size,
                family_count=family_count,
            )
        score = x.T @ (y - probability)
        try:
            update = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            return _not_estimable_rank_result(
                method,
                "singular Hessian",
                sample_size=sample_size,
                family_count=family_count,
            )
        beta = beta + update
        iteration_count = iteration
        if float(np.max(np.abs(update))) < 1e-8:
            converged = True
            break
    if not converged or not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 20:
        return _not_estimable_rank_result(
            method,
            "non-convergence or separation",
            sample_size=sample_size,
            family_count=family_count,
        )

    eta = np.clip(x @ beta, -35.0, 35.0)
    probability = np.clip(1 / (1 + np.exp(-eta)), 1e-9, 1 - 1e-9)
    weights = probability * (1 - probability)
    bread_matrix = x.T @ (weights[:, None] * x)
    try:
        bread = np.linalg.solve(bread_matrix, np.eye(parameter_count))
    except np.linalg.LinAlgError:
        return _not_estimable_rank_result(
            method,
            "singular covariance bread",
            sample_size=sample_size,
            family_count=family_count,
        )
    residual = y - probability
    meat = np.zeros((parameter_count, parameter_count), dtype=float)
    for family in families:
        mask = family_values == family
        cluster_score = x[mask].T @ residual[mask]
        meat += np.outer(cluster_score, cluster_score)
    correction = (family_count / (family_count - 1)) * (
        (sample_size - 1) / (sample_size - parameter_count)
    )
    covariance = correction * bread @ meat @ bread
    variance = float(covariance[1, 1])
    if not math.isfinite(variance) or variance <= 0:
        return _not_estimable_rank_result(
            method,
            "non-finite cluster-robust rank variance",
            sample_size=sample_size,
            family_count=family_count,
        )
    coefficient = float(beta[1])
    robust_se = math.sqrt(variance)
    z_value = coefficient / robust_se
    p_value = math.erfc(abs(z_value) / math.sqrt(2))
    return {
        "method": method,
        "status": "estimable",
        "reason": "",
        "sample_size": sample_size,
        "family_cluster_count": family_count,
        "rank_coefficient": coefficient,
        "rank_robust_se": robust_se,
        "rank_odds_ratio": math.exp(coefficient),
        "rank_ci_low": coefficient - 1.96 * robust_se,
        "rank_ci_high": coefficient + 1.96 * robust_se,
        "rank_p_value": p_value,
        "rank_q_value": math.nan,
        "converged": True,
        "iteration_count": iteration_count,
        "hessian_condition_number": condition,
        "standard_error_type": "family_cluster_robust",
    }


def fit_all_method_rank_models(rows: pd.DataFrame) -> pd.DataFrame:
    validate_motif_level_table(rows)
    methods = sorted(rows["method"].astype(str).unique(), key=_method_sort_key)
    records = [fit_method_rank_model(rows, method) for method in methods]
    p_values = [
        float(record["rank_p_value"])
        if record["status"] == "estimable"
        else None
        for record in records
    ]
    for record, q_value in zip(records, benjamini_hochberg(p_values)):
        record["rank_q_value"] = q_value
    return pd.DataFrame(records)


def build_matched_random_pools(
    rows: pd.DataFrame, *, top_k: int = 10
) -> dict[tuple[str, str], dict[str, object]]:
    ranked = add_normalized_rank(rows)
    selected = (
        ranked.sort_values(
            ["family", "method", "motif_rank", "motif_cluster"], kind="mergesort"
        )
        .groupby(["family", "method"], sort=False, group_keys=False)
        .head(top_k)
    )
    output: dict[tuple[str, str], dict[str, object]] = {}
    for (family, method), group in ranked.groupby(["family", "method"], sort=True):
        observed = selected[
            (selected["family"] == family) & (selected["method"] == method)
        ]
        strata_needed = Counter(
            (int(row.protein_size_bin), int(row.annotation_coverage_bin))
            for row in observed.itertuples()
        )
        pools: dict[tuple[int, int], list[dict[str, object]]] = {}
        for stratum, population in group.groupby(
            ["protein_size_bin", "annotation_coverage_bin"], sort=True
        ):
            records = population.sort_values("motif_cluster", kind="mergesort").to_dict(
                orient="records"
            )
            pools[(int(stratum[0]), int(stratum[1]))] = records
        for stratum, count in strata_needed.items():
            available = len(pools.get(stratum, []))
            if available < count:
                raise ValueError(
                    f"cannot conserve matched draw for {family}/{method} stratum "
                    f"{stratum}: need {count}, have {available}"
                )
        output[(str(family), str(method))] = {
            "family": str(family),
            "method": str(method),
            "observed_rows": observed.sort_values(
                ["motif_rank", "motif_cluster"], kind="mergesort"
            ).to_dict(orient="records"),
            "strata_needed": strata_needed,
            "pools": pools,
        }
    return output


def draw_joint_matched_background(
    pools: dict[tuple[str, str], dict[str, object]],
    *,
    iteration: int,
    seed: int,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    draws: dict[tuple[str, str], list[dict[str, object]]] = {}
    for key in sorted(pools):
        group = pools[key]
        family = str(group["family"])
        method = str(group["method"])
        generator = random.Random(derived_seed(seed, family, method, iteration))
        selected: list[dict[str, object]] = []
        strata_needed = group["strata_needed"]
        group_pools = group["pools"]
        assert isinstance(strata_needed, Counter)
        assert isinstance(group_pools, dict)
        for stratum, count in sorted(strata_needed.items()):
            population = group_pools.get(stratum, [])
            if len(population) < count:
                raise ValueError(
                    f"cannot conserve matched draw for {family}/{method} stratum "
                    f"{stratum}: need {count}, have {len(population)}"
                )
            selected.extend(dict(row) for row in generator.sample(population, count))
        draws[key] = sorted(selected, key=lambda row: str(row["motif_cluster"]))
    return draws


def global_matched_background_test(
    rows: pd.DataFrame,
    *,
    iterations: int = 10_000,
    seed: int = 20_260_813,
    top_k: int = 10,
) -> pd.DataFrame:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    pools = build_matched_random_pools(rows, top_k=top_k)
    methods = sorted({method for _, method in pools}, key=_method_sort_key)
    observed: dict[str, dict[str, int]] = {}
    for method in methods:
        observed_rows = [
            row
            for (family, group_method), group in pools.items()
            if group_method == method
            for row in group["observed_rows"]
        ]
        observed[method] = {
            "selected": len(observed_rows),
            "testable": sum(_coerce_bool(row["is_testable"]) for row in observed_rows),
            "supported": sum(_coerce_bool(row["is_supported"]) for row in observed_rows),
        }
    null_supported = {
        method: np.zeros(iterations, dtype=np.int32) for method in methods
    }
    null_testable = {
        method: np.zeros(iterations, dtype=np.int32) for method in methods
    }
    for iteration in range(iterations):
        draws = draw_joint_matched_background(pools, iteration=iteration, seed=seed)
        for (_, method), selected in draws.items():
            null_supported[method][iteration] += sum(
                _coerce_bool(row["is_supported"]) for row in selected
            )
            null_testable[method][iteration] += sum(
                _coerce_bool(row["is_testable"]) for row in selected
            )
    records: list[dict[str, object]] = []
    for method in methods:
        values = null_supported[method]
        observed_count = observed[method]["supported"]
        selected_count = observed[method]["selected"]
        null_mean = float(values.mean())
        records.append(
            {
                "method": method,
                "top_k_requested": top_k,
                "observed_selected_count": selected_count,
                "observed_testable_count": observed[method]["testable"],
                "observed_supported_count": observed_count,
                "conserved_draw_count_per_iteration": selected_count,
                "null_testable_mean": float(null_testable[method].mean()),
                "null_supported_mean": null_mean,
                "null_supported_median": float(np.median(values)),
                "null_supported_q025": float(np.quantile(values, 0.025)),
                "null_supported_q975": float(np.quantile(values, 0.975)),
                "observed_minus_null_supported_count": observed_count - null_mean,
                "observed_minus_null_percentage_points": (
                    (observed_count - null_mean) / selected_count * 100
                    if selected_count
                    else math.nan
                ),
                "empirical_p_ge_observed": (
                    1 + int(np.count_nonzero(values >= observed_count))
                )
                / (iterations + 1),
                "adjusted_q_value": math.nan,
                "random_iterations": iterations,
                "global_seed": seed,
                "status": "valid",
                "sampling_contract": (
                    "without replacement; exact family-method, floor(log2(protein_count)) "
                    "and five annotation-coverage bins; observed Top-K remains eligible"
                ),
            }
        )
    q_values = benjamini_hochberg(
        [float(record["empirical_p_ge_observed"]) for record in records]
    )
    for record, q_value in zip(records, q_values):
        record["adjusted_q_value"] = q_value
    return pd.DataFrame(records)
