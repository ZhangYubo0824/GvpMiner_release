from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Callable, Sequence

import numpy as np
from scipy.stats import wilcoxon


PRIMARY_ENDPOINTS = [
    "contact_density",
    "relative_SASA",
    "long_range_contact_count",
    "local_RMSD_or_structural_similarity",
]


SECONDARY_ENDPOINTS = [
    "motif_mean_pLDDT",
    "motif_min_pLDDT",
    "helix_fraction",
    "sheet_fraction",
    "coil_fraction",
    "radius_of_gyration",
    "distance_to_protein_centroid",
    "surface_residue_fraction",
    "low_confidence_fraction",
]


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    p_values = [float(p) for p in p_values]
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    q_values = [1.0] * n
    running_min = 1.0
    for rank, original_index in reversed(list(enumerate(order, start=1))):
        q = min(running_min, p_values[original_index] * n / rank)
        running_min = q
        q_values[original_index] = min(1.0, q)
    return q_values


def paired_permutation_p_value(differences: Sequence[float], *, num_permutations: int = 2000, seed: int = 42) -> float:
    diffs = np.asarray([float(value) for value in differences], dtype=float)
    if len(diffs) == 0:
        return 1.0
    if num_permutations <= 0:
        return 1.0
    observed = abs(float(np.mean(diffs)))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(num_permutations, len(diffs)))
    permuted = np.abs(np.mean(diffs * signs, axis=1))
    count = int(np.sum(permuted >= observed))
    return float((count + 1) / (num_permutations + 1))


def bootstrap_ci(values: Sequence[float], *, seed: int = 42, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray([float(value) for value in values], dtype=float)
    if len(values) == 0:
        return (0.0, 0.0)
    if n_bootstrap <= 0:
        estimate = float(np.median(values))
        return estimate, estimate
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    estimates = np.median(samples, axis=1)
    lo, hi = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


class ControlFeatureAccumulator:
    def __init__(self, endpoints: Sequence[str]) -> None:
        self.endpoints = list(endpoints)
        self._sums: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.control_type_counts: dict[str, int] = defaultdict(int)

    def add(self, row: dict[str, object]) -> None:
        control_type = str(row.get("control_type", ""))
        motif_instance_id = str(row.get("motif_instance_id", ""))
        if not control_type or not motif_instance_id:
            return
        self.control_type_counts[control_type] += 1
        instance_sums = self._sums[control_type][motif_instance_id]
        for endpoint in self.endpoints:
            value = _to_float(row.get(endpoint))
            if value is None:
                continue
            total_and_count = instance_sums.setdefault(endpoint, [0.0, 0.0])
            total_and_count[0] += value
            total_and_count[1] += 1.0

    def has_control_type(self, control_type: str) -> bool:
        return bool(self._sums.get(control_type))

    def mean(self, control_type: str, motif_instance_id: str, endpoint: str) -> float | None:
        total_and_count = self._sums.get(control_type, {}).get(motif_instance_id, {}).get(endpoint)
        if not total_and_count or total_and_count[1] <= 0:
            return None
        return float(total_and_count[0] / total_and_count[1])


def _summarize_with_control_mean(
    motif_rows: Sequence[dict[str, object]],
    *,
    control_mean_for: Callable[[str, str], float | None],
    endpoints: Sequence[str],
    control_type: str,
    seed: int,
) -> list[dict[str, object]]:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in motif_rows:
        cluster = str(row.get("motif_cluster", row.get("motif_id", "")))
        grouped_rows[cluster].append(row)

    results: list[dict[str, object]] = []
    for motif_cluster, cluster_rows in sorted(grouped_rows.items()):
        for endpoint in endpoints:
            protein_differences: dict[str, list[float]] = defaultdict(list)
            protein_motif_values: dict[str, list[float]] = defaultdict(list)
            protein_control_values: dict[str, list[float]] = defaultdict(list)
            n_occurrences = 0
            for motif_row in cluster_rows:
                instance_id = str(motif_row["motif_instance_id"])
                motif_value = _to_float(motif_row.get(endpoint))
                if motif_value is None:
                    continue
                control_mean = control_mean_for(instance_id, endpoint)
                if control_mean is None:
                    continue
                protein_id = str(motif_row.get("sequence_id") or instance_id)
                protein_differences[protein_id].append(motif_value - control_mean)
                protein_motif_values[protein_id].append(motif_value)
                protein_control_values[protein_id].append(control_mean)
                n_occurrences += 1
            if not protein_differences:
                continue

            differences = [float(median(values)) for values in protein_differences.values()]
            motif_values = [float(median(values)) for values in protein_motif_values.values()]
            control_values = [float(median(values)) for values in protein_control_values.values()]
            if len(differences) >= 2 and any(abs(value) > 1e-12 for value in differences):
                try:
                    wilcoxon_p = float(wilcoxon(differences).pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
            else:
                wilcoxon_p = 1.0
            permutation_p = paired_permutation_p_value(differences, seed=seed)
            ci_low, ci_high = bootstrap_ci(differences, seed=seed)
            results.append(
                {
                    "motif_cluster": motif_cluster,
                    "feature": endpoint,
                    "control_type": control_type,
                    "n_occurrences": n_occurrences,
                    "n_proteins": len(differences),
                    "motif_median": float(median(motif_values)),
                    "control_median": float(median(control_values)),
                    "effect_median_difference": float(median(differences)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_value": wilcoxon_p,
                    "wilcoxon_p_value": wilcoxon_p,
                    "permutation_p_value": permutation_p,
                }
            )

    q_values = benjamini_hochberg([float(row["p_value"]) for row in results])
    for row, q_value in zip(results, q_values):
        row["q_value"] = q_value
    return results


def summarize_structural_enrichment(
    motif_rows: Sequence[dict[str, object]],
    control_rows: Sequence[dict[str, object]],
    *,
    endpoints: Sequence[str] | None = None,
    control_type: str = "within_random",
    seed: int = 42,
) -> list[dict[str, object]]:
    controls_by_instance: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in control_rows:
        if str(row.get("control_type", "")) == control_type:
            controls_by_instance[str(row.get("motif_instance_id", ""))].append(row)

    def control_mean_for(instance_id: str, endpoint: str) -> float | None:
        values = [_to_float(row.get(endpoint)) for row in controls_by_instance.get(instance_id, [])]
        valid_values = [value for value in values if value is not None]
        return float(np.mean(valid_values)) if valid_values else None

    return _summarize_with_control_mean(
        motif_rows,
        control_mean_for=control_mean_for,
        endpoints=list(endpoints or PRIMARY_ENDPOINTS),
        control_type=control_type,
        seed=seed,
    )


def summarize_structural_enrichment_from_control_accumulator(
    motif_rows: Sequence[dict[str, object]],
    control_accumulator: ControlFeatureAccumulator,
    *,
    endpoints: Sequence[str] | None = None,
    control_type: str = "within_random",
    seed: int = 42,
) -> list[dict[str, object]]:
    return _summarize_with_control_mean(
        motif_rows,
        control_mean_for=lambda instance_id, endpoint: control_accumulator.mean(
            control_type, instance_id, endpoint
        ),
        endpoints=list(endpoints or PRIMARY_ENDPOINTS),
        control_type=control_type,
        seed=seed,
    )
