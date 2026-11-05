from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from .contracts import ParameterPair


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _qualified(row: Mapping[str, Any]) -> bool:
    mode = str(row.get("selection_mode", row.get("selection_strategy", ""))).lower()
    return mode in {"qualified", "stability_plateau", "coverage_rescue"}


def _floor(row: Mapping[str, Any]) -> float:
    explicit = row.get("stability_floor")
    if explicit not in (None, ""):
        return _number(explicit)
    return min(
        _number(row.get("median_window_jaccard")),
        _number(row.get("median_protein_jaccard")),
        _number(row.get("median_occurrence_jaccard")),
    )




def _family_medians(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family", "UNKNOWN"))].append(
            _floor(row) if key == "stability_floor" else _number(row.get(key))
        )
    return [median(values) for values in grouped.values()]


def summarize_candidate(
    pair: ParameterPair,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate rows cannot be empty")
    families = {str(row.get("family", "UNKNOWN")) for row in rows}
    qualified_families = {
        str(row.get("family", "UNKNOWN")) for row in rows if _qualified(row)
    }
    result: dict[str, Any] = {
        "pair": pair,
        "window_size": pair.window_size,
        "window_step": pair.window_step,
        "family_count": len(families),
        "qualified_family_count": len(qualified_families),
        "qualified_family_fraction": len(qualified_families) / len(families),
        "mcs_lower_boundary": any(int(_number(row.get("min_cluster_size"))) == 10
                                  for row in rows),
        "mcs_upper_boundary": any(int(_number(row.get("min_cluster_size"))) == 80
                                  for row in rows),
    }
    for key in (
        "stability_floor",
        "stable_protein_coverage",
        "stable_window_coverage",
        "noise_ratio",
        "stable_cluster_number",
        "runtime_seconds",
    ):
        values = _family_medians(rows, key)
        result[key] = median(values) if values else 0.0
    return result


def _family_metric_medians(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _floor(row) if key == "stability_floor" else _number(row.get(key))
        grouped[str(row.get("family", "UNKNOWN"))].append(value)
    return {family: median(values) for family, values in grouped.items()}


def annotate_candidate_uncertainty(
    candidates: Sequence[Mapping[str, Any]],
    rows_by_pair: Mapping[ParameterPair, Sequence[Mapping[str, Any]]],
    *,
    baseline: ParameterPair = ParameterPair(30, 5),
    stability_tolerance: float = 0.02,
    coverage_tolerance: float = 0.03,
) -> list[dict[str, Any]]:
    """Add preregistered family-disagreement and rank-reversal evidence.

    Family disagreement requires at least one representative family to improve
    and another to worsen relative to 30/5 by a material stability amount.
    An interaction is important when a window or step comparison reverses sign
    across the other factor and both opposing effects exceed the metric's
    practical tolerance.  Every corner of the implicated 2x2 contrast is
    flagged so an axis candidate cannot hide an interaction it participates in.
    """
    annotated = [dict(candidate) for candidate in candidates]
    by_pair: dict[ParameterPair, dict[str, Any]] = {}
    for row in annotated:
        pair = row.get("pair")
        if not isinstance(pair, ParameterPair):
            pair = ParameterPair(int(row["window_size"]), int(row["window_step"]))
            row["pair"] = pair
        row.update({
            "family_direction_disagreement": False,
            "family_compared_count": 0,
            "family_improved_count": 0,
            "family_worsened_count": 0,
            "important_interaction": False,
            "interaction_reversal_count": 0,
        })
        by_pair[pair] = row

    baseline_families = _family_metric_medians(
        rows_by_pair.get(baseline, ()), "stability_floor"
    )
    for pair, row in by_pair.items():
        if pair == baseline:
            continue
        family_values = _family_metric_medians(
            rows_by_pair.get(pair, ()), "stability_floor"
        )
        common = sorted(set(baseline_families) & set(family_values))
        deltas = [family_values[family] - baseline_families[family]
                  for family in common]
        improved = sum(delta >= stability_tolerance for delta in deltas)
        worsened = sum(delta <= -stability_tolerance for delta in deltas)
        row["family_compared_count"] = len(common)
        row["family_improved_count"] = improved
        row["family_worsened_count"] = worsened
        row["family_direction_disagreement"] = improved > 0 and worsened > 0

    windows = sorted({pair.window_size for pair in by_pair})
    steps = sorted({pair.window_step for pair in by_pair})
    metrics = (
        ("stability_floor", stability_tolerance),
        ("stable_protein_coverage", coverage_tolerance),
    )
    for window in windows:
        if window == baseline.window_size:
            continue
        for step in steps:
            if step == baseline.window_step:
                continue
            corners = (
                baseline,
                ParameterPair(window, baseline.window_step),
                ParameterPair(baseline.window_size, step),
                ParameterPair(window, step),
            )
            if any(pair not in by_pair for pair in corners):
                continue
            for metric, tolerance in metrics:
                values = {pair: _number(by_pair[pair].get(metric)) for pair in corners}
                window_effect_at_step = values[corners[3]] - values[corners[2]]
                window_effect_at_baseline = values[corners[1]] - values[corners[0]]
                step_effect_at_window = values[corners[3]] - values[corners[1]]
                step_effect_at_baseline = values[corners[2]] - values[corners[0]]
                reverses = (
                    window_effect_at_step * window_effect_at_baseline < 0
                    and abs(window_effect_at_step) >= tolerance
                    and abs(window_effect_at_baseline) >= tolerance
                ) or (
                    step_effect_at_window * step_effect_at_baseline < 0
                    and abs(step_effect_at_window) >= tolerance
                    and abs(step_effect_at_baseline) >= tolerance
                )
                if reverses:
                    for pair in corners:
                        by_pair[pair]["important_interaction"] = True
                        by_pair[pair]["interaction_reversal_count"] += 1
    return annotated
