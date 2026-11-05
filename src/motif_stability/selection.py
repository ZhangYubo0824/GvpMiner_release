from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _finite_float(
    row: Mapping[str, object],
    key: str,
    *,
    default: float,
) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _count(row: Mapping[str, object], key: str) -> int:
    return int(_finite_float(row, key, default=0.0))


def _stability_floor(row: Mapping[str, object]) -> float:
    return min(
        _finite_float(row, "median_window_jaccard", default=0.0),
        _finite_float(row, "median_protein_jaccard", default=0.0),
        _finite_float(row, "median_occurrence_jaccard", default=0.0),
    )


def _fallback_key(row: Mapping[str, object]) -> tuple[float, ...]:
    return (
        _stability_floor(row),
        _finite_float(row, "stable_protein_coverage", default=0.0),
        _finite_float(row, "stable_window_coverage", default=0.0),
        -_finite_float(row, "noise_ratio", default=1.0),
        float(_count(row, "stable_cluster_number")),
        -_finite_float(row, "min_cluster_size", default=0.0),
        -_finite_float(row, "min_samples", default=0.0),
    )


def _plateau_key(row: Mapping[str, object]) -> tuple[float, ...]:
    return (
        _finite_float(row, "stable_window_coverage", default=0.0),
        -_finite_float(row, "noise_ratio", default=1.0),
        _stability_floor(row),
        -float(_count(row, "stable_cluster_number")),
        -_finite_float(row, "min_cluster_size", default=0.0),
        -_finite_float(row, "min_samples", default=0.0),
    )


def _rescue_key(row: Mapping[str, object]) -> tuple[float, ...]:
    return (
        _finite_float(row, "stable_protein_coverage", default=0.0),
        _finite_float(row, "stable_window_coverage", default=0.0),
        -_finite_float(row, "noise_ratio", default=1.0),
        _stability_floor(row),
        -float(_count(row, "stable_cluster_number")),
        -_finite_float(row, "min_cluster_size", default=0.0),
        -_finite_float(row, "min_samples", default=0.0),
    )


def select_parameter_setting(
    parameter_rows: Sequence[Mapping[str, object]],
    *,
    min_window_jaccard: float = 0.5,
    min_protein_jaccard: float = 0.7,
    min_occurrence_jaccard: float = 0.6,
    stability_tolerance: float = 0.02,
    protein_coverage_tolerance: float = 0.03,
    fragmentation_guard_min_clusters: int = 100,
    fragmentation_reduction_ratio: float = 1.5,
    low_coverage_threshold: float = 0.7,
    rescue_min_stability_floor: float = 0.9,
    rescue_min_protein_gain: float = 0.2,
) -> dict[str, object]:
    """Select a family-wise setting from a stability plateau.

    Stability is a qualification gate and plateau constraint rather than an
    objective to maximize at the expense of protein coverage.  The optional
    parsimony and coverage-rescue steps handle heavily fragmented catalogs and
    highly stable but biologically narrow settings, respectively.
    """
    if not parameter_rows:
        raise ValueError("parameter_rows cannot be empty")
    if stability_tolerance < 0 or protein_coverage_tolerance < 0:
        raise ValueError("selection tolerances must be non-negative")
    if fragmentation_guard_min_clusters < 1:
        raise ValueError("fragmentation_guard_min_clusters must be positive")
    if fragmentation_reduction_ratio <= 1:
        raise ValueError("fragmentation_reduction_ratio must be greater than 1")
    qualified = [
        row
        for row in parameter_rows
        if _count(row, "stable_cluster_number") > 0
        and _finite_float(row, "median_window_jaccard", default=0.0)
        >= min_window_jaccard
        and _finite_float(row, "median_protein_jaccard", default=0.0)
        >= min_protein_jaccard
        and _finite_float(row, "median_occurrence_jaccard", default=0.0)
        >= min_occurrence_jaccard
    ]
    if not qualified:
        selected = dict(max(parameter_rows, key=_fallback_key))
        selected["selection_mode"] = "fallback"
        selected["selection_strategy"] = "exploratory_fallback"
        selected["stability_floor"] = _stability_floor(selected)
        return selected

    best_stability = max(_stability_floor(row) for row in qualified)
    plateau = [
        row
        for row in qualified
        if _stability_floor(row) >= best_stability - stability_tolerance
    ]
    best_protein_coverage = max(
        _finite_float(row, "stable_protein_coverage", default=0.0)
        for row in plateau
    )
    coverage_band = [
        row
        for row in plateau
        if _finite_float(row, "stable_protein_coverage", default=0.0)
        >= best_protein_coverage - protein_coverage_tolerance
    ]
    chosen = max(coverage_band, key=_plateau_key)

    chosen_stable_clusters = _count(chosen, "stable_cluster_number")
    if chosen_stable_clusters >= fragmentation_guard_min_clusters:
        maximum_parsimonious_clusters = (
            chosen_stable_clusters / fragmentation_reduction_ratio
        )
        parsimonious = [
            row
            for row in coverage_band
            if _count(row, "stable_cluster_number")
            <= maximum_parsimonious_clusters
        ]
        if parsimonious:
            chosen = max(parsimonious, key=_plateau_key)

    strategy = "stability_plateau"
    chosen_protein_coverage = _finite_float(
        chosen, "stable_protein_coverage", default=0.0
    )
    if chosen_protein_coverage < low_coverage_threshold:
        rescue_candidates = [
            row
            for row in qualified
            if _stability_floor(row) >= rescue_min_stability_floor
            and _finite_float(row, "stable_protein_coverage", default=0.0)
            >= chosen_protein_coverage + rescue_min_protein_gain
        ]
        if rescue_candidates:
            chosen = max(rescue_candidates, key=_rescue_key)
            strategy = "coverage_rescue"

    selected = dict(chosen)
    selected["selection_mode"] = "qualified"
    selected["selection_strategy"] = strategy
    selected["stability_floor"] = _stability_floor(selected)
    return selected


def count_seed_recoveries(
    reference_clusters: Mapping[int, set[str]],
    seed_runs: Iterable[Mapping[int, set[str]]],
    *,
    min_jaccard: float = 0.5,
) -> dict[int, int]:
    """Count in how many independent seed runs each reference cluster is recovered."""
    from .matching import hungarian_match

    counts = {int(label): 0 for label in reference_clusters}
    for run in seed_runs:
        result = hungarian_match(
            reference_clusters,
            run,
            min_match_jaccard=min_jaccard,
            split_merge_jaccard=min_jaccard,
        )
        for match in result.matches:
            counts[match.reference_label] += 1
    return counts
