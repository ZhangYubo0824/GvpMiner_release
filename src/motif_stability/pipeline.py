from __future__ import annotations

import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Sequence

import numpy as np
from joblib import Memory

from exp3_domain_discovery import (
    build_sequence_motif_occurrences,
    finalize_motif_catalog,
    summarize_motif_clusters,
)

from .clustering import (
    ClusterRunner,
    RepresentationBuilder,
    cluster_parameter_grid,
    make_cached_hdbscan_clusterer,
    make_pca_builder,
    make_umap_builder,
)
from .io import FamilyInput
from .matching import hungarian_match
from .null_model import (
    LengthMatchedSequenceSampler,
    benjamini_hochberg,
    conservation_score,
    empirical_pvalue,
)
from .selection import count_seed_recoveries, select_parameter_setting
from .stability import run_stability_grid


CATALOG_FIELDS = [
    "motif_uid",
    "family",
    "method",
    "cluster_label",
    "rank_within_method",
    "min_start",
    "max_end",
    "mean_start",
    "mean_end",
    "position_std",
    "n_windows",
    "n_unique_sequences",
    "support_ratio",
    "dominant_family",
    "cross_species_count",
    "median_window_jaccard",
    "median_protein_jaccard",
    "median_occurrence_jaccard",
    "median_recovery_fraction",
    "stable_repeat_fraction",
    "split_repeat_fraction",
    "merge_repeat_fraction",
    "seed_recovery_count",
    "seed_total",
    "pca_window_jaccard",
    "null_empirical_p",
    "null_fdr",
]

OCCURRENCE_FIELDS = [
    "occurrence_id",
    "motif_uid",
    "family",
    "method",
    "cluster_label",
    "parent_id",
    "occ_start",
    "occ_end",
    "occ_length",
    "n_support_windows",
    "support_window_ids",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    temporary.replace(path)


def _clusters(windows, labels, allowed_labels=None) -> dict[int, set[str]]:
    allowed = None if allowed_labels is None else set(int(value) for value in allowed_labels)
    clusters: dict[int, set[str]] = {}
    for row, label in zip(windows, labels):
        label = int(label)
        if label < 0 or (allowed is not None and label not in allowed):
            continue
        clusters.setdefault(label, set()).add(str(row["window_id"]))
    return clusters


def _profile_counts(args) -> tuple[int, int]:
    confirmatory = getattr(args, "profile", "pilot") == "confirmatory"
    repeats = getattr(args, "repeats", None)
    null_repeats = getattr(args, "null_repeats", None)
    return (
        int(repeats if repeats is not None else (50 if confirmatory else 10)),
        int(null_repeats if null_repeats is not None else (1000 if confirmatory else 0)),
    )


def _profile_seeds(args) -> tuple[int, ...]:
    configured = getattr(args, "seeds", None)
    if configured:
        return tuple(int(seed) for seed in configured)
    if getattr(args, "profile", "pilot") == "confirmatory":
        return (0, 1, 2, 3, 4)
    return (0,)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_only_source_root(item: FamilyInput) -> Path:
    return item.family_dir.parent.resolve()


def _run_signature(
    item: FamilyInput,
    *,
    args,
    repeats: int,
    null_repeats: int,
    seeds: Sequence[int],
) -> str:
    metadata_stat = item.metadata_path.stat()
    embedding_stat = item.embedding_path.stat()
    payload = {
        "metadata": [str(item.metadata_path.resolve()), metadata_stat.st_size, metadata_stat.st_mtime_ns],
        "embeddings": [str(item.embedding_path.resolve()), embedding_stat.st_size, embedding_stat.st_mtime_ns],
        "embedding_shape": list(item.embeddings.shape),
        "profile": args.profile,
        "mcs": [int(value) for value in args.mcs],
        "min_samples": [int(value) for value in args.min_samples],
        "n_neighbors": int(args.n_neighbors),
        "n_components": int(args.n_components),
        "subsample_fraction": float(args.subsample_fraction),
        "repeats": int(repeats),
        "seeds": [int(seed) for seed in seeds],
        "null_repeats": int(null_repeats),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _robustness_metrics(
    *,
    item: FamilyInput,
    reference_labels: np.ndarray,
    stable_labels: set[int],
    selected_key: tuple[int, int],
    args,
    cluster_runner: ClusterRunner,
    representation_builder: RepresentationBuilder | None,
) -> dict[int, dict[str, object]]:
    reference_clusters = _clusters(item.windows, reference_labels, stable_labels)
    seeds = _profile_seeds(args)
    seed_runs = []
    if representation_builder is None:
        for seed in seeds[1:]:
            labels = cluster_parameter_grid(
                item.embeddings,
                mcs_values=[selected_key[0]],
                min_samples_values=[selected_key[1]],
                representation_builder=make_umap_builder(
                    n_neighbors=int(args.n_neighbors),
                    n_components=int(args.n_components),
                    seed=seed,
                ),
                cluster_runner=cluster_runner,
            )[selected_key]
            seed_runs.append(_clusters(item.windows, labels))
    recovery = count_seed_recoveries(reference_clusters, seed_runs, min_jaccard=0.5)
    for label in recovery:
        recovery[label] += 1

    pca_scores = {label: None for label in reference_clusters}
    if getattr(args, "profile", "pilot") == "confirmatory":
        pca_labels = cluster_parameter_grid(
            item.embeddings,
            mcs_values=[selected_key[0]],
            min_samples_values=[selected_key[1]],
            representation_builder=make_pca_builder(
                n_components=int(args.n_components),
                seed=seeds[0],
            ),
            cluster_runner=cluster_runner,
        )[selected_key]
        matched = hungarian_match(reference_clusters, _clusters(item.windows, pca_labels))
        for match in matched.matches:
            pca_scores[match.reference_label] = match.window_jaccard

    return {
        label: {
            "seed_recovery_count": recovery[label],
            "seed_total": len(seeds),
            "pca_window_jaccard": pca_scores[label],
        }
        for label in reference_clusters
    }


def _add_null_statistics(
    motifs: list[dict[str, object]],
    windows: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    *,
    repeats: int,
    seed: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    sampler_factory=LengthMatchedSequenceSampler,
) -> None:
    if not motifs:
        return
    if repeats <= 0:
        for motif in motifs:
            motif["null_empirical_p"] = ""
            motif["null_fdr"] = ""
        return
    pvalues = []
    total_draws = len(motifs) * repeats
    milestones = set(progress_milestones(total_draws))
    completed_draws = 0
    for motif_index, motif in enumerate(motifs):
        motif_label = int(motif.get("cluster_label", -1))
        observed_by_protein: dict[str, str] = {}
        for row, label in sorted(
            zip(windows, labels),
            key=lambda item: str(item[0].get("window_id", "")),
        ):
            if int(label) != motif_label:
                continue
            sequence = str(row.get("window_seq", ""))
            if sequence:
                observed_by_protein.setdefault(str(row["parent_id"]), sequence)
        fragments = list(observed_by_protein.values())
        support_proteins = set(observed_by_protein)
        background_by_protein: dict[str, str] = {}
        for row, label in sorted(
            zip(windows, labels),
            key=lambda item: str(item[0].get("window_id", "")),
        ):
            parent_id = str(row["parent_id"])
            if int(label) == motif_label or parent_id in support_proteins:
                continue
            candidate = str(row.get("window_seq", ""))
            if candidate:
                background_by_protein.setdefault(parent_id, candidate)
        pool = list(background_by_protein.values())
        if not fragments or not pool:
            pvalues.append(1.0)
            continue
        sampler = sampler_factory(fragments, pool)
        null_scores = []
        for repeat in range(repeats):
            sampled = sampler.sample(
                seed=int(seed) + motif_index * repeats + repeat,
            )
            null_scores.append(conservation_score(sampled))
            completed_draws += 1
            if progress_callback is not None and completed_draws in milestones:
                progress_callback(completed_draws, total_draws, "matched_null")
        pvalues.append(empirical_pvalue(conservation_score(fragments), null_scores))
    adjusted = benjamini_hochberg(pvalues)
    for motif, pvalue, fdr in zip(motifs, pvalues, adjusted):
        motif["null_empirical_p"] = float(pvalue)
        motif["null_fdr"] = float(fdr)


def progress_milestones(total: int, *, max_updates: int = 101) -> tuple[int, ...]:
    """Return monotonic completion marks, including the final item."""
    if total <= 0:
        return ()
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    if total <= max_updates:
        return tuple(range(1, total + 1))
    return tuple(sorted({max(1, round(index * total / max_updates))
                         for index in range(1, max_updates + 1)} | {total}))


def _run_unit(
    item: FamilyInput,
    *,
    args,
    representation_builder: RepresentationBuilder | None,
    cluster_runner: ClusterRunner,
) -> dict[str, object]:
    output_root = Path(args.output_root)
    diagnostic_dir = output_root / "motif_stability" / item.family / item.method
    completion_path = diagnostic_dir / ".complete.json"
    exp3_family_dir = output_root / "exp3_domain_discovery" / item.family
    protected_root = _read_only_source_root(item)
    for destination in (diagnostic_dir, exp3_family_dir):
        if _is_within(destination, protected_root):
            raise ValueError(
                f"Output path {destination} is inside read-only source root {protected_root}"
            )
    repeats, null_repeats = _profile_counts(args)
    seeds = _profile_seeds(args)
    run_signature = _run_signature(
        item,
        args=args,
        repeats=repeats,
        null_repeats=null_repeats,
        seeds=seeds,
    )
    required_outputs = (
        diagnostic_dir / "selected_setting.json",
        diagnostic_dir / "parameter_screen.csv",
        diagnostic_dir / "cluster_stability.csv",
        diagnostic_dir / "resample_matches.csv",
        diagnostic_dir / "stable_cluster_labels.npy",
        exp3_family_dir / f"motif_catalog_{item.method}.csv",
        exp3_family_dir / f"motif_occurrences_{item.method}.csv",
    )
    prior_signature = None
    if completion_path.is_file():
        try:
            prior_signature = json.loads(completion_path.read_text(encoding="utf-8")).get(
                "run_signature"
            )
        except (OSError, ValueError, AttributeError):
            prior_signature = None
    if (
        completion_path.is_file()
        and all(path.is_file() for path in required_outputs)
        and prior_signature == run_signature
        and not bool(getattr(args, "force", False))
    ):
        return {"family": item.family, "method": item.method, "status": "skipped"}

    builder = representation_builder or make_umap_builder(
        n_neighbors=int(args.n_neighbors),
        n_components=int(args.n_components),
        seed=seeds[0],
    )
    manifest = {
        "status": "running",
        "started_at": _utc_now(),
        "family": item.family,
        "method": item.method,
        "source_metadata": str(item.metadata_path.resolve()),
        "source_embeddings": str(item.embedding_path.resolve()),
        "embedding_shape": list(item.embeddings.shape),
        "profile": args.profile,
        "mcs": list(args.mcs),
        "min_samples": list(args.min_samples),
        "subsample_fraction": float(args.subsample_fraction),
        "repeats": repeats,
        "seeds": list(seeds),
        "null_repeats": null_repeats,
        "run_signature": run_signature,
    }
    _write_json(diagnostic_dir / "run_manifest.json", manifest)

    show_progress = not bool(getattr(args, "no_progress", False))

    def progress(done: int, total: int, phase: str) -> None:
        if show_progress:
            print(
                f"[{phase.upper()}] {item.family}/{item.method} {done}/{total} "
                f"({100.0 * done / total:.1f}%)",
                flush=True,
            )

    grid = run_stability_grid(
        windows=item.windows,
        embeddings=item.embeddings,
        mcs_values=args.mcs,
        min_samples_values=args.min_samples,
        fraction=float(args.subsample_fraction),
        repeats=repeats,
        seed=seeds[0],
        representation_builder=builder,
        cluster_runner=cluster_runner,
        progress_callback=progress,
    )
    selected = select_parameter_setting(grid.parameter_rows)
    selected_key = (int(selected["min_cluster_size"]), int(selected["min_samples"]))
    full_labels = np.asarray(grid.reference_labels[selected_key], dtype=int)
    selected_cluster_rows = [
        dict(row)
        for row in grid.cluster_rows
        if int(row["min_cluster_size"]) == selected_key[0]
        and int(row["min_samples"]) == selected_key[1]
    ]
    stable_by_label = {
        int(row["reference_label"]): row
        for row in selected_cluster_rows
        if bool(row["stable"])
    }
    stable_labels = set(stable_by_label)
    filtered_labels = np.asarray(
        [label if int(label) in stable_labels else -1 for label in full_labels],
        dtype=int,
    )
    robustness = _robustness_metrics(
        item=item,
        reference_labels=full_labels,
        stable_labels=stable_labels,
        selected_key=selected_key,
        args=args,
        cluster_runner=cluster_runner,
        representation_builder=representation_builder,
    )

    motifs = summarize_motif_clusters(item.windows, filtered_labels, item.method)
    total_proteins = len({str(row["parent_id"]) for row in item.windows})
    for motif in motifs:
        label = int(motif["cluster_label"])
        motif["support_ratio"] = (
            float(motif["n_unique_sequences"]) / total_proteins if total_proteins else 0.0
        )
        motif.update(
            {
                key: value
                for key, value in stable_by_label[label].items()
                if key
                in {
                    "median_window_jaccard",
                    "median_protein_jaccard",
                    "median_occurrence_jaccard",
                    "median_recovery_fraction",
                    "stable_repeat_fraction",
                    "split_repeat_fraction",
                    "merge_repeat_fraction",
                }
            }
        )
        motif.update(robustness.get(label, {}))
    _add_null_statistics(
        motifs,
        item.windows,
        filtered_labels,
        repeats=null_repeats,
        seed=seeds[0],
        progress_callback=progress,
    )
    motifs = finalize_motif_catalog(item.family, item.method, motifs)
    occurrences = build_sequence_motif_occurrences(
        item.windows,
        filtered_labels,
        motifs,
        item.method,
    )

    _write_csv(exp3_family_dir / f"motif_catalog_{item.method}.csv", motifs, CATALOG_FIELDS)
    _write_json(exp3_family_dir / f"motif_catalog_{item.method}.json", motifs)
    _write_csv(
        exp3_family_dir / f"motif_occurrences_{item.method}.csv",
        occurrences,
        OCCURRENCE_FIELDS,
    )
    _write_json(exp3_family_dir / f"motif_occurrences_{item.method}.json", occurrences)
    np.save(diagnostic_dir / "stable_cluster_labels.npy", filtered_labels)
    _write_csv(diagnostic_dir / "parameter_screen.csv", grid.parameter_rows)
    _write_csv(diagnostic_dir / "cluster_stability.csv", grid.cluster_rows)
    _write_csv(diagnostic_dir / "resample_matches.csv", grid.resample_rows)
    _write_json(diagnostic_dir / "selected_setting.json", selected)

    completed = {
        **manifest,
        "status": "completed",
        "completed_at": _utc_now(),
        "selected_setting": selected,
        "stable_cluster_number": len(stable_labels),
        "motif_number": len(motifs),
    }
    _write_json(diagnostic_dir / "run_manifest.json", completed)
    _write_json(completion_path, completed)
    return {
        "family": item.family,
        "method": item.method,
        "status": "completed",
        "motif_number": len(motifs),
    }


def run_pipeline(
    *,
    inputs: Sequence[FamilyInput],
    args,
    representation_builder: RepresentationBuilder | None = None,
    cluster_runner: ClusterRunner | None = None,
) -> list[dict[str, object]]:
    """Run every validated family/method unit and return resumable status rows."""
    results = []
    for item in inputs:
        with ExitStack() as cache_lifetime:
            cached_runner = None

            def run_cached_hdbscan(values, *, min_cluster_size, min_samples):
                nonlocal cached_runner
                if cached_runner is None:
                    output_root = Path(args.output_root)
                    output_root.mkdir(parents=True, exist_ok=True)
                    cache_dir = cache_lifetime.enter_context(
                        TemporaryDirectory(prefix=".hdbscan-tree-", dir=output_root)
                    )
                    cached_runner = make_cached_hdbscan_clusterer(
                        Memory(cache_dir, verbose=0)
                    )
                return cached_runner(
                    values,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )

            result = _run_unit(
                item,
                args=args,
                representation_builder=representation_builder,
                cluster_runner=(
                    cluster_runner if cluster_runner is not None else run_cached_hdbscan
                ),
            )
        results.append(result)
        print(f"{result['status'].upper()} {item.family}/{item.method}", flush=True)
    return results
