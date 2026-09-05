from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import multiprocessing
from pathlib import Path
import re
import stat
from typing import Callable

import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .io import read_csv_rows, write_csv_rows
from .parsing import ParsedStructure


_ALL_CLUSTER_FRAGMENTS: tuple[tuple[np.ndarray | None, ...], ...] = ()
_ALL_CLUSTER_FRAGMENT_KEYS: tuple[tuple[object | None, ...], ...] = ()
_ALL_CLUSTER_HAS_DUPLICATES: tuple[bool, ...] = ()
_USE_DISTANCE_SIGNATURE_CACHE = False
_USE_BATCHED_RMSD = False
_RMSD_BATCH_SIZE = 64
_DISTANCE_SIGNATURE_CACHE: dict[tuple[int, int], tuple[bool, np.ndarray]] = {}
_DISTANCE_SIGNATURE_CACHE_CLUSTER_INDEX: int | None = None
_CHECKPOINT_FIELDS = (
    "motif_cluster",
    "motif_instance_id",
    "local_RMSD_or_structural_similarity",
)
_CHECKPOINT_FILE_PATTERN = re.compile(r"[0-9a-f]{64}\.csv")
_CHECKPOINT_TEMP_FILE_PATTERN = re.compile(r"\.[0-9a-f]{64}\.csv\.tmp")
_PARTIAL_CHECKPOINT_FILE_PATTERN = re.compile(r"[0-9a-f]{64}\.partial\.csv")
_PARTIAL_CHECKPOINT_TEMP_FILE_PATTERN = re.compile(
    r"\.[0-9a-f]{64}\.partial\.csv\.tmp"
)


def _wrap_progress(iterable, *, show_progress: bool, **kwargs):
    if not show_progress or tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


@dataclass(frozen=True)
class RmsdBackfillSummary:
    completed_clusters: int
    total_clusters: int
    completed_rows: int
    total_rows: int


def _uncached_kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    n = min(len(coords_a), len(coords_b))
    if n < 2:
        return 0.0
    a = np.asarray(coords_a[:n], dtype=float)
    b = np.asarray(coords_b[:n], dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    indices = np.triu_indices(n, k=1)
    da = np.sqrt(np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=2))[indices]
    db = np.sqrt(np.sum((b[:, None, :] - b[None, :, :]) ** 2, axis=2))[indices]
    if len(da) == 0:
        return 0.0
    return float(np.sqrt(np.mean((da - db) ** 2)))


def _distance_signature(coords: np.ndarray, n: int) -> tuple[bool, np.ndarray]:
    values = np.asarray(coords[:n], dtype=float)
    if not np.isfinite(values).all():
        return False, np.empty(0, dtype=float)
    indices = np.triu_indices(n, k=1)
    signature = np.sqrt(
        np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2)
    )[indices]
    return True, signature


def _cached_distance_signature(
    coords: np.ndarray,
    n: int,
) -> tuple[bool, np.ndarray]:
    key = (id(coords), n)
    cached = _DISTANCE_SIGNATURE_CACHE.get(key)
    if cached is None:
        cached = _distance_signature(coords, n)
        _DISTANCE_SIGNATURE_CACHE[key] = cached
    return cached


def _cached_kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    n = min(len(coords_a), len(coords_b))
    if n < 2:
        return 0.0
    finite_a, da = _cached_distance_signature(coords_a, n)
    finite_b, db = _cached_distance_signature(coords_b, n)
    if not finite_a or not finite_b or len(da) == 0:
        return 0.0
    return float(np.sqrt(np.mean((da - db) ** 2)))


def _kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute the legacy fragment dissimilarity with an exact cache fast path."""
    if _USE_DISTANCE_SIGNATURE_CACHE:
        return _cached_kabsch_rmsd(coords_a, coords_b)
    return _uncached_kabsch_rmsd(coords_a, coords_b)


_ORIGINAL_KABSCH_RMSD = _kabsch_rmsd


def _batched_pair_values(
    coords_a: np.ndarray,
    coords_b_values: list[np.ndarray],
) -> list[float]:
    results = [0.0] * len(coords_b_values)
    by_length: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for index, coords_b in enumerate(coords_b_values):
        n = min(len(coords_a), len(coords_b))
        if n >= 2:
            by_length[n].append((index, coords_b))

    for n, entries in by_length.items():
        finite_a, signature_a = _cached_distance_signature(coords_a, n)
        if not finite_a or len(signature_a) == 0:
            continue

        valid_indices: list[int] = []
        signatures_b: list[np.ndarray] = []
        for index, coords_b in entries:
            finite_b, signature_b = _cached_distance_signature(coords_b, n)
            if not finite_b:
                continue
            valid_indices.append(index)
            signatures_b.append(signature_b)
        if not signatures_b:
            continue

        signature_matrix = np.stack(signatures_b, axis=0)
        values = np.sqrt(
            np.mean((signature_a[None, :] - signature_matrix) ** 2, axis=1)
        )
        for index, value in zip(valid_indices, values):
            results[index] = float(value)
    return results


def _verify_distance_signature_cache_exact(
    all_fragments: tuple[tuple[np.ndarray | None, ...], ...],
) -> bool:
    representatives: dict[int, np.ndarray] = {}
    invalid: np.ndarray | None = None
    for cluster_fragments in all_fragments:
        for coords in cluster_fragments:
            if coords is None:
                continue
            if not np.isfinite(coords).all():
                invalid = coords
            representatives.setdefault(len(coords), coords)
            if len(representatives) >= 32 and invalid is not None:
                break
        if len(representatives) >= 32 and invalid is not None:
            break

    probes = list(representatives.values())[:32]
    if invalid is not None:
        probes.append(invalid)
    if not probes:
        return True

    _DISTANCE_SIGNATURE_CACHE.clear()
    try:
        for coords_a in probes:
            for coords_b in probes:
                if _uncached_kabsch_rmsd(coords_a, coords_b) != _cached_kabsch_rmsd(
                    coords_a,
                    coords_b,
                ):
                    return False
        return True
    finally:
        _DISTANCE_SIGNATURE_CACHE.clear()


def _verify_batched_rmsd_exact(
    all_fragments: tuple[tuple[np.ndarray | None, ...], ...],
) -> bool:
    representatives: dict[int, np.ndarray] = {}
    invalid: np.ndarray | None = None
    for cluster_fragments in all_fragments:
        for coords in cluster_fragments:
            if coords is None:
                continue
            if not np.isfinite(coords).all():
                invalid = coords
            representatives.setdefault(len(coords), coords)
            if len(representatives) >= 32 and invalid is not None:
                break
        if len(representatives) >= 32 and invalid is not None:
            break

    probes = list(representatives.values())[:32]
    if invalid is not None:
        probes.append(invalid)
    if not probes:
        return True

    _DISTANCE_SIGNATURE_CACHE.clear()
    try:
        for coords_a in probes:
            scalar = [_cached_kabsch_rmsd(coords_a, coords_b) for coords_b in probes]
            batched = _batched_pair_values(coords_a, probes)
            if scalar != batched:
                return False
        return True
    finally:
        _DISTANCE_SIGNATURE_CACHE.clear()


def _backfill_local_rmsd(
    feature_rows: list[dict[str, object]],
    structures: dict[str, ParsedStructure],
    *,
    flank: int = 5,
) -> None:
    """Backfill local_RMSD_or_structural_similarity for each motif instance.

    For each motif instance, compute its average pairwise RMSD with all other
    instances in the same motif_cluster (using motif +/- flank residues).
    """
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(feature_rows):
        by_cluster[str(row.get("motif_cluster", ""))].append(idx)

    for cluster, indices in by_cluster.items():
        if len(indices) < 2:
            continue
        # Extract coordinate fragments
        fragments: list[tuple[int, np.ndarray | None]] = []
        for idx in indices:
            row = feature_rows[idx]
            parsed = structures.get(str(row.get("sequence_id", "")))
            if parsed is None:
                fragments.append((idx, None))
                continue
            start = max(0, int(float(row.get("motif_start", 0))) - flank)
            end = min(len(parsed.ca_coords), int(float(row.get("motif_end", 0))) + flank)
            if end <= start:
                fragments.append((idx, None))
                continue
            fragments.append((idx, np.asarray(parsed.ca_coords[start:end], dtype=float)))

        # Compute pairwise RMSD
        for i, (idx_a, coords_a) in enumerate(fragments):
            if coords_a is None:
                continue
            rmsds = []
            for j, (idx_b, coords_b) in enumerate(fragments):
                if i == j or coords_b is None:
                    continue
                rmsds.append(_kabsch_rmsd(coords_a, coords_b))
            if rmsds:
                feature_rows[idx_a]["local_RMSD_or_structural_similarity"] = float(np.mean(rmsds))


def _row_context(
    cluster_index: int,
    row_index: int,
) -> tuple[
    tuple[np.ndarray | None, ...],
    np.ndarray | None,
    tuple[object | None, ...],
    bool,
]:
    cluster_fragments = _ALL_CLUSTER_FRAGMENTS[cluster_index]
    coords_a = cluster_fragments[row_index]
    fragment_keys = (
        _ALL_CLUSTER_FRAGMENT_KEYS[cluster_index]
        if cluster_index < len(_ALL_CLUSTER_FRAGMENT_KEYS)
        else ()
    )
    reuse_duplicates = (
        cluster_index < len(_ALL_CLUSTER_HAS_DUPLICATES)
        and _ALL_CLUSTER_HAS_DUPLICATES[cluster_index]
    )
    return cluster_fragments, coords_a, fragment_keys, reuse_duplicates


def _compute_row_rmsd_scalar(cluster_index: int, row_index: int) -> float | None:
    cluster_fragments, coords_a, fragment_keys, reuse_duplicates = _row_context(
        cluster_index,
        row_index,
    )
    if coords_a is None:
        return None
    duplicate_pair_values: dict[object, float] = {}
    rmsds = []
    for j, coords_b in enumerate(cluster_fragments):
        if row_index == j or coords_b is None:
            continue
        fragment_key = fragment_keys[j] if reuse_duplicates else None
        if reuse_duplicates and fragment_key in duplicate_pair_values:
            value = duplicate_pair_values[fragment_key]
        else:
            value = _kabsch_rmsd(coords_a, coords_b)
            if reuse_duplicates:
                duplicate_pair_values[fragment_key] = value
        rmsds.append(value)
    return float(np.mean(rmsds)) if rmsds else None


def _compute_row_rmsd_batched(cluster_index: int, row_index: int) -> float | None:
    cluster_fragments, coords_a, fragment_keys, reuse_duplicates = _row_context(
        cluster_index,
        row_index,
    )
    if coords_a is None:
        return None

    duplicate_pair_values: dict[object, float] = {}
    rmsds: list[float] = []
    for chunk_start in range(0, len(cluster_fragments), _RMSD_BATCH_SIZE):
        chunk_end = min(len(cluster_fragments), chunk_start + _RMSD_BATCH_SIZE)
        chunk_values: list[float | None] = []
        pending_coords: list[np.ndarray] = []
        pending_keys: list[object | None] = []
        pending_slots: list[list[int]] = []
        pending_by_key: dict[object, int] = {}

        for j in range(chunk_start, chunk_end):
            coords_b = cluster_fragments[j]
            if row_index == j or coords_b is None:
                continue
            slot = len(chunk_values)
            chunk_values.append(None)
            fragment_key = fragment_keys[j] if reuse_duplicates else None
            if reuse_duplicates and fragment_key in duplicate_pair_values:
                chunk_values[slot] = duplicate_pair_values[fragment_key]
                continue
            if reuse_duplicates and fragment_key in pending_by_key:
                pending_slots[pending_by_key[fragment_key]].append(slot)
                continue

            pending_index = len(pending_coords)
            pending_coords.append(coords_b)
            pending_keys.append(fragment_key)
            pending_slots.append([slot])
            if reuse_duplicates:
                pending_by_key[fragment_key] = pending_index

        if pending_coords:
            pending_values = _batched_pair_values(coords_a, pending_coords)
            for pending_index, value in enumerate(pending_values):
                fragment_key = pending_keys[pending_index]
                if reuse_duplicates:
                    duplicate_pair_values[fragment_key] = value
                for slot in pending_slots[pending_index]:
                    chunk_values[slot] = value
        rmsds.extend(float(value) for value in chunk_values if value is not None)
    return float(np.mean(rmsds)) if rmsds else None


def _compute_row_rmsd(task: tuple[int, int]) -> float | None:
    global _DISTANCE_SIGNATURE_CACHE_CLUSTER_INDEX
    cluster_index, row_index = task
    if (
        _USE_DISTANCE_SIGNATURE_CACHE
        and _DISTANCE_SIGNATURE_CACHE_CLUSTER_INDEX != cluster_index
    ):
        _DISTANCE_SIGNATURE_CACHE.clear()
        _DISTANCE_SIGNATURE_CACHE_CLUSTER_INDEX = cluster_index
    if _USE_BATCHED_RMSD and _kabsch_rmsd is _ORIGINAL_KABSCH_RMSD:
        return _compute_row_rmsd_batched(cluster_index, row_index)
    return _compute_row_rmsd_scalar(cluster_index, row_index)


def _checkpoint_path(output_dir: Path, cluster: str) -> Path:
    digest = hashlib.sha256(cluster.encode("utf-8")).hexdigest()
    return output_dir / "rmsd_checkpoints" / f"{digest}.csv"


def _partial_checkpoint_path(output_dir: Path, cluster: str) -> Path:
    digest = hashlib.sha256(cluster.encode("utf-8")).hexdigest()
    return output_dir / "rmsd_checkpoints" / f"{digest}.partial.csv"


def _purge_owned_cluster_checkpoints(output_dir: Path) -> None:
    checkpoint_dir = output_dir / "rmsd_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for path in checkpoint_dir.iterdir():
        if not (
            _CHECKPOINT_FILE_PATTERN.fullmatch(path.name)
            or _CHECKPOINT_TEMP_FILE_PATTERN.fullmatch(path.name)
            or _PARTIAL_CHECKPOINT_FILE_PATTERN.fullmatch(path.name)
            or _PARTIAL_CHECKPOINT_TEMP_FILE_PATTERN.fullmatch(path.name)
        ):
            continue
        try:
            path_stat = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(path_stat.st_mode):
            path.unlink()


def _load_checkpoint_values(
    path: Path,
    *,
    cluster: str,
    instance_ids: list[str],
) -> list[float | None]:
    rows = read_csv_rows(path)
    if len(rows) != len(instance_ids):
        raise ValueError(
            f"RMSD checkpoint {path} row count does not match expected IDs for cluster {cluster!r}"
        )
    for row in rows:
        if None in row or any(field not in row for field in _CHECKPOINT_FIELDS):
            raise ValueError(f"RMSD checkpoint {path} is malformed")
    if any(str(row["motif_cluster"]) != cluster for row in rows):
        raise ValueError(f"RMSD checkpoint {path} motif_cluster does not match {cluster!r}")
    checkpoint_ids = [str(row["motif_instance_id"]) for row in rows]
    if checkpoint_ids != instance_ids:
        raise ValueError(
            f"RMSD checkpoint {path} ordered motif_instance_id values do not match"
        )

    values: list[float | None] = []
    for row in rows:
        raw_value = row["local_RMSD_or_structural_similarity"]
        if raw_value in (None, ""):
            values.append(None)
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RMSD checkpoint {path} has invalid local_RMSD value") from exc
        if not np.isfinite(value):
            raise ValueError(f"RMSD checkpoint {path} has invalid local_RMSD value")
        values.append(value)
    return values


def _load_cluster_checkpoint(
    path: Path,
    *,
    cluster: str,
    instance_ids: list[str],
) -> list[float | None]:
    return _load_checkpoint_values(
        path,
        cluster=cluster,
        instance_ids=instance_ids,
    )


def _load_partial_cluster_checkpoint(
    path: Path,
    *,
    cluster: str,
    instance_ids: list[str],
) -> list[float | None]:
    rows = read_csv_rows(path)
    if not rows or len(rows) >= len(instance_ids):
        raise ValueError(
            f"Partial RMSD checkpoint {path} must contain a non-empty proper row prefix"
        )
    prefix_ids = instance_ids[: len(rows)]
    return _load_checkpoint_values(
        path,
        cluster=cluster,
        instance_ids=prefix_ids,
    )


def _write_cluster_checkpoint(
    path: Path,
    *,
    cluster: str,
    instance_ids: list[str],
    values: list[float | None],
) -> None:
    write_csv_rows(
        path,
        [
            {
                "motif_cluster": cluster,
                "motif_instance_id": instance_id,
                "local_RMSD_or_structural_similarity": "" if value is None else value,
            }
            for instance_id, value in zip(instance_ids, values)
        ],
        _CHECKPOINT_FIELDS,
    )


def backfill_local_rmsd_resumable(
    feature_rows: list[dict[str, object]],
    structures: dict[str, ParsedStructure],
    output_dir: Path,
    *,
    workers: int = 1,
    flank: int = 5,
    show_progress: bool = True,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
    reuse_checkpoints: bool = True,
    checkpoint_rows: int = 500,
) -> RmsdBackfillSummary:
    global _ALL_CLUSTER_FRAGMENTS
    global _ALL_CLUSTER_FRAGMENT_KEYS
    global _ALL_CLUSTER_HAS_DUPLICATES
    global _USE_DISTANCE_SIGNATURE_CACHE
    global _USE_BATCHED_RMSD
    global _DISTANCE_SIGNATURE_CACHE_CLUSTER_INDEX
    _ALL_CLUSTER_FRAGMENTS = ()
    _ALL_CLUSTER_FRAGMENT_KEYS = ()
    _ALL_CLUSTER_HAS_DUPLICATES = ()
    _USE_DISTANCE_SIGNATURE_CACHE = False
    _USE_BATCHED_RMSD = False
    _DISTANCE_SIGNATURE_CACHE.clear()
    _DISTANCE_SIGNATURE_CACHE_CLUSTER_INDEX = None
    output_dir = Path(output_dir)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if checkpoint_rows < 1:
        raise ValueError("checkpoint_rows must be at least 1")
    if workers > 1 and "fork" not in multiprocessing.get_all_start_methods():
        raise ValueError("workers > 1 requires the multiprocessing 'fork' start method")
    if not reuse_checkpoints:
        _purge_owned_cluster_checkpoints(output_dir)

    by_cluster: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(feature_rows):
        by_cluster[str(row.get("motif_cluster", ""))].append(idx)

    all_fragments: list[tuple[np.ndarray | None, ...]] = []
    all_fragment_keys: list[tuple[object | None, ...]] = []
    all_has_duplicates: list[bool] = []
    clusters = list(by_cluster.items())
    cluster_indices = [indices for _, indices in clusters]
    for indices in cluster_indices:
        fragments: list[np.ndarray | None] = []
        fragment_keys: list[object | None] = []
        for idx in indices:
            row = feature_rows[idx]
            sequence_id = str(row.get("sequence_id", ""))
            parsed = structures.get(sequence_id)
            if parsed is None:
                fragments.append(None)
                fragment_keys.append(None)
                continue
            start = max(0, int(float(row.get("motif_start", 0))) - flank)
            end = min(len(parsed.ca_coords), int(float(row.get("motif_end", 0))) + flank)
            if end <= start:
                fragments.append(None)
                fragment_keys.append(None)
                continue
            fragments.append(np.asarray(parsed.ca_coords[start:end], dtype=float))
            fragment_keys.append((sequence_id, start, end))
        all_fragments.append(tuple(fragments))
        all_fragment_keys.append(tuple(fragment_keys))
        valid_keys = [key for key in fragment_keys if key is not None]
        all_has_duplicates.append(len(valid_keys) != len(set(valid_keys)))

    _ALL_CLUSTER_FRAGMENTS = tuple(all_fragments)
    _ALL_CLUSTER_FRAGMENT_KEYS = tuple(all_fragment_keys)
    _ALL_CLUSTER_HAS_DUPLICATES = tuple(all_has_duplicates)
    _USE_DISTANCE_SIGNATURE_CACHE = _verify_distance_signature_cache_exact(
        _ALL_CLUSTER_FRAGMENTS
    )
    _USE_BATCHED_RMSD = (
        _USE_DISTANCE_SIGNATURE_CACHE
        and _verify_batched_rmsd_exact(_ALL_CLUSTER_FRAGMENTS)
    )
    _DISTANCE_SIGNATURE_CACHE.clear()
    _DISTANCE_SIGNATURE_CACHE_CLUSTER_INDEX = None
    if show_progress:
        if _USE_BATCHED_RMSD:
            print(
                f"Using exact cached fragment-distance signatures in {_RMSD_BATCH_SIZE}-pair batches.",
                flush=True,
            )
        elif _USE_DISTANCE_SIGNATURE_CACHE:
            print("Using exact cached fragment-distance signatures.", flush=True)
        else:
            print("Distance-cache exactness check failed; using the legacy RMSD path.", flush=True)

    completed_rows = 0
    total_rows = len(feature_rows)
    total_clusters = len(cluster_indices)
    executor = None
    try:
        if workers > 1:
            context = multiprocessing.get_context("fork")
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=context)
        cluster_iterator = _wrap_progress(
            enumerate(clusters),
            show_progress=show_progress,
            desc="Backfilling local RMSD",
            unit="cluster",
            total=total_clusters,
        )
        for cluster_index, (cluster, indices) in cluster_iterator:
            instance_ids = [str(feature_rows[idx].get("motif_instance_id", "")) for idx in indices]
            checkpoint_path = _checkpoint_path(output_dir, cluster)
            partial_checkpoint_path = _partial_checkpoint_path(output_dir, cluster)
            if reuse_checkpoints and checkpoint_path.exists():
                values = _load_cluster_checkpoint(
                    checkpoint_path,
                    cluster=cluster,
                    instance_ids=instance_ids,
                )
                partial_checkpoint_path.unlink(missing_ok=True)
            else:
                values = []
                if reuse_checkpoints and partial_checkpoint_path.exists():
                    values = _load_partial_cluster_checkpoint(
                        partial_checkpoint_path,
                        cluster=cluster,
                        instance_ids=instance_ids,
                    )
                tasks = [
                    (cluster_index, row_index)
                    for row_index in range(len(values), len(indices))
                ]
                last_saved_rows = len(values)
                try:
                    if executor is None:
                        row_iterator = (_compute_row_rmsd(task) for task in tasks)
                    else:
                        row_iterator = executor.map(_compute_row_rmsd, tasks)
                    row_iterator = _wrap_progress(
                        row_iterator,
                        show_progress=show_progress,
                        desc=f"RMSD rows {cluster}",
                        unit="row",
                        total=len(indices),
                        initial=len(values),
                        leave=False,
                    )
                    for value in row_iterator:
                        values.append(value)
                        if (
                            len(values) < len(indices)
                            and len(values) - last_saved_rows >= checkpoint_rows
                        ):
                            _write_cluster_checkpoint(
                                partial_checkpoint_path,
                                cluster=cluster,
                                instance_ids=instance_ids[: len(values)],
                                values=values,
                            )
                            last_saved_rows = len(values)
                            if progress_callback is not None:
                                progress_callback(
                                    cluster_index,
                                    total_clusters,
                                    completed_rows + len(values),
                                    total_rows,
                                )
                except BaseException:
                    if 0 < len(values) < len(indices) and len(values) > last_saved_rows:
                        _write_cluster_checkpoint(
                            partial_checkpoint_path,
                            cluster=cluster,
                            instance_ids=instance_ids[: len(values)],
                            values=values,
                        )
                    raise
                _write_cluster_checkpoint(
                    checkpoint_path,
                    cluster=cluster,
                    instance_ids=instance_ids,
                    values=values,
                )
                partial_checkpoint_path.unlink(missing_ok=True)
            for idx, value in zip(indices, values):
                if value is not None:
                    feature_rows[idx]["local_RMSD_or_structural_similarity"] = value
            completed_rows += len(indices)
            if progress_callback is not None:
                progress_callback(cluster_index + 1, total_clusters, completed_rows, total_rows)
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        if executor is not None:
            executor.shutdown(wait=True)
    finally:
        _ALL_CLUSTER_FRAGMENTS = ()
        _ALL_CLUSTER_FRAGMENT_KEYS = ()
        _ALL_CLUSTER_HAS_DUPLICATES = ()
        _USE_DISTANCE_SIGNATURE_CACHE = False
        _USE_BATCHED_RMSD = False
        _DISTANCE_SIGNATURE_CACHE.clear()
        _DISTANCE_SIGNATURE_CACHE_CLUSTER_INDEX = None

    return RmsdBackfillSummary(
        completed_clusters=total_clusters,
        total_clusters=total_clusters,
        completed_rows=completed_rows,
        total_rows=total_rows,
    )
