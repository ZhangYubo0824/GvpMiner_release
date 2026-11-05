from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from .clustering import ClusterRunner, RepresentationBuilder, cluster_parameter_grid
from .matching import hungarian_match
from .occurrences import collapse_cluster_occurrences, occurrence_jaccard, protein_jaccard


@dataclass(frozen=True)
class StabilityGridResult:
    subsamples: tuple[frozenset[str], ...]
    reference_labels: Mapping[tuple[int, int], np.ndarray]
    resample_rows: tuple[dict[str, object], ...]
    cluster_rows: tuple[dict[str, object], ...]
    parameter_rows: tuple[dict[str, object], ...]


def make_protein_subsamples(
    windows: Sequence[Mapping[str, object]],
    *,
    fraction: float,
    repeats: int,
    seed: int,
) -> tuple[frozenset[str], ...]:
    """Draw deterministic protein-level samples; every window of a protein stays together."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    proteins = np.asarray(sorted({str(row["parent_id"]) for row in windows}), dtype=object)
    if not len(proteins):
        raise ValueError("windows must contain at least one protein")
    sample_size = max(1, min(len(proteins), int(round(fraction * len(proteins)))))
    generator = np.random.default_rng(int(seed))
    return tuple(
        frozenset(str(value) for value in generator.choice(proteins, size=sample_size, replace=False))
        for _ in range(int(repeats))
    )


def _clusters(
    windows: Sequence[Mapping[str, object]],
    labels: np.ndarray,
) -> dict[int, set[str]]:
    clusters: dict[int, set[str]] = {}
    for row, label in zip(windows, labels):
        label = int(label)
        if label < 0:
            continue
        clusters.setdefault(label, set()).add(str(row["window_id"]))
    return clusters


def _labelled_windows(
    windows: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    wanted_label: int,
) -> list[dict[str, object]]:
    return [
        {**dict(row), "cluster_label": int(label)}
        for row, label in zip(windows, labels)
        if int(label) == int(wanted_label)
    ]


def _safe_median(values: Sequence[float], default: float = 0.0) -> float:
    return float(median(values)) if values else float(default)


def run_stability_grid(
    *,
    windows: Sequence[Mapping[str, object]],
    embeddings: np.ndarray,
    mcs_values: Iterable[int],
    min_samples_values: Iterable[int],
    fraction: float,
    repeats: int,
    seed: int,
    representation_builder: RepresentationBuilder,
    cluster_runner: ClusterRunner,
    min_match_jaccard: float = 0.3,
    split_merge_jaccard: float = 0.2,
    stable_window_jaccard: float = 0.5,
    stable_protein_jaccard: float = 0.7,
    stable_occurrence_jaccard: float = 0.6,
    stable_recovery_fraction: float = 0.8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> StabilityGridResult:
    """Evaluate clustering stability on paired protein-level subsamples."""
    values = np.asarray(embeddings)
    if values.ndim != 2 or len(values) != len(windows):
        raise ValueError("embeddings must be 2D and aligned one-to-one with windows")
    mcs = tuple(int(value) for value in mcs_values)
    min_samples = tuple(int(value) for value in min_samples_values)
    if not mcs or not min_samples:
        raise ValueError("parameter grids cannot be empty")

    reference_labels = cluster_parameter_grid(
        values,
        mcs_values=mcs,
        min_samples_values=min_samples,
        representation_builder=representation_builder,
        cluster_runner=cluster_runner,
    )
    subsamples = make_protein_subsamples(
        windows,
        fraction=fraction,
        repeats=repeats,
        seed=seed,
    )

    resample_rows: list[dict[str, object]] = []
    global_rows: dict[tuple[int, int], list[dict[str, float]]] = {
        key: [] for key in reference_labels
    }
    for repeat_index, retained_proteins in enumerate(subsamples):
        indices = np.asarray(
            [index for index, row in enumerate(windows) if str(row["parent_id"]) in retained_proteins],
            dtype=int,
        )
        sample_windows = tuple(windows[index] for index in indices)
        sample_labels_by_parameter = cluster_parameter_grid(
            values[indices],
            mcs_values=mcs,
            min_samples_values=min_samples,
            representation_builder=representation_builder,
            cluster_runner=cluster_runner,
        )

        for key, full_labels in reference_labels.items():
            sample_labels = sample_labels_by_parameter[key]
            restricted_labels = full_labels[indices]
            reference_clusters = _clusters(sample_windows, restricted_labels)
            sample_clusters = _clusters(sample_windows, sample_labels)
            matching = hungarian_match(
                reference_clusters,
                sample_clusters,
                min_match_jaccard=min_match_jaccard,
                split_merge_jaccard=split_merge_jaccard,
            )
            matches = {match.reference_label: match for match in matching.matches}
            ami = float(adjusted_mutual_info_score(restricted_labels, sample_labels))
            noise_agreement = float(np.mean((restricted_labels == -1) == (sample_labels == -1)))
            global_rows[key].append({"ami_all": ami, "noise_agreement": noise_agreement})

            for reference_label, reference_ids in reference_clusters.items():
                match = matches.get(reference_label)
                sample_label = match.sample_label if match is not None else None
                reference_rows = _labelled_windows(sample_windows, restricted_labels, reference_label)
                sample_rows = (
                    _labelled_windows(sample_windows, sample_labels, sample_label)
                    if sample_label is not None
                    else []
                )
                reference_occurrences = collapse_cluster_occurrences(reference_rows)
                sample_occurrences = collapse_cluster_occurrences(sample_rows)
                window_score = float(match.window_jaccard) if match is not None else 0.0
                protein_score = protein_jaccard(
                    reference_rows,
                    sample_rows,
                    retained_proteins=set(retained_proteins),
                )
                occurrence_score = occurrence_jaccard(
                    reference_occurrences,
                    sample_occurrences,
                    retained_proteins=set(retained_proteins),
                )
                sample_ids = sample_clusters.get(sample_label, set()) if sample_label is not None else set()
                recovery = len(reference_ids & sample_ids) / len(reference_ids) if reference_ids else 0.0
                stable = (
                    window_score >= stable_window_jaccard
                    and protein_score >= stable_protein_jaccard
                    and occurrence_score >= stable_occurrence_jaccard
                    and recovery >= stable_recovery_fraction
                )
                resample_rows.append(
                    {
                        "repeat": repeat_index,
                        "min_cluster_size": key[0],
                        "min_samples": key[1],
                        "reference_label": reference_label,
                        "sample_label": sample_label,
                        "window_jaccard": window_score,
                        "protein_jaccard": protein_score,
                        "occurrence_jaccard": occurrence_score,
                        "recovery_fraction": recovery,
                        "stable": stable,
                        "split": reference_label in matching.split_reference_labels,
                        "merge": sample_label in matching.merge_sample_labels if sample_label is not None else False,
                    }
                )
        if progress_callback is not None:
            progress_callback(repeat_index + 1, len(subsamples), "protein_subsample")

    cluster_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    all_proteins = {str(row["parent_id"]) for row in windows}
    for key, full_labels in reference_labels.items():
        reference_cluster_labels = sorted(set(int(label) for label in full_labels if int(label) >= 0))
        stable_labels: set[int] = set()
        for reference_label in reference_cluster_labels:
            rows = [
                row
                for row in resample_rows
                if row["min_cluster_size"] == key[0]
                and row["min_samples"] == key[1]
                and row["reference_label"] == reference_label
            ]
            stable_fraction = float(np.mean([bool(row["stable"]) for row in rows])) if rows else 0.0
            is_stable = stable_fraction >= stable_recovery_fraction
            if is_stable:
                stable_labels.add(reference_label)
            cluster_rows.append(
                {
                    "min_cluster_size": key[0],
                    "min_samples": key[1],
                    "reference_label": reference_label,
                    "n_windows": int(np.count_nonzero(full_labels == reference_label)),
                    "median_window_jaccard": _safe_median([float(row["window_jaccard"]) for row in rows]),
                    "median_protein_jaccard": _safe_median([float(row["protein_jaccard"]) for row in rows]),
                    "median_occurrence_jaccard": _safe_median([float(row["occurrence_jaccard"]) for row in rows]),
                    "median_recovery_fraction": _safe_median([float(row["recovery_fraction"]) for row in rows]),
                    "stable_repeat_fraction": stable_fraction,
                    "stable": is_stable,
                    "split_repeat_fraction": float(np.mean([bool(row["split"]) for row in rows])) if rows else 0.0,
                    "merge_repeat_fraction": float(np.mean([bool(row["merge"]) for row in rows])) if rows else 0.0,
                }
            )

        setting_rows = [
            row
            for row in resample_rows
            if row["min_cluster_size"] == key[0] and row["min_samples"] == key[1]
        ]
        stable_window_mask = np.asarray([int(label) in stable_labels for label in full_labels], dtype=bool)
        stable_proteins = {
            str(row["parent_id"])
            for row, keep in zip(windows, stable_window_mask)
            if bool(keep)
        }
        global_metrics = global_rows[key]
        parameter_rows.append(
            {
                "min_cluster_size": key[0],
                "min_samples": key[1],
                "raw_cluster_number": len(reference_cluster_labels),
                "stable_cluster_number": len(stable_labels),
                "median_window_jaccard": _safe_median([float(row["window_jaccard"]) for row in setting_rows]),
                "median_protein_jaccard": _safe_median([float(row["protein_jaccard"]) for row in setting_rows]),
                "median_occurrence_jaccard": _safe_median([float(row["occurrence_jaccard"]) for row in setting_rows]),
                "noise_ratio": float(np.mean(full_labels == -1)),
                "stable_window_coverage": float(np.mean(stable_window_mask)),
                "stable_protein_coverage": len(stable_proteins) / len(all_proteins) if all_proteins else 0.0,
                "median_ami_all": _safe_median([row["ami_all"] for row in global_metrics]),
                "median_noise_agreement": _safe_median([row["noise_agreement"] for row in global_metrics]),
            }
        )

    return StabilityGridResult(
        subsamples=subsamples,
        reference_labels=reference_labels,
        resample_rows=tuple(resample_rows),
        cluster_rows=tuple(cluster_rows),
        parameter_rows=tuple(parameter_rows),
    )
