from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPRESENTATIVE_FIELDS = [
    "motif_uid",
    "family",
    "method",
    "cluster_label",
    "representative_window_id",
    "representative_parent_id",
    "representative_start",
    "representative_end",
    "representative_length",
    "representative_sequence",
    "distance_to_cluster_centroid",
    "n_cluster_windows",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_int(value: object) -> int:
    return int(float(str(value)))


def _normalized_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(
        vectors,
        norms,
        out=np.zeros_like(vectors, dtype=np.float64),
        where=norms > 0,
    )


def expand_cached_labels(
    windows: Sequence[Mapping[str, object]],
    cached_labels: np.ndarray,
) -> np.ndarray:
    cached_labels = np.asarray(cached_labels).reshape(-1)
    if len(cached_labels) == len(windows):
        return cached_labels

    seen_by_parent: dict[str, set[str]] = defaultdict(set)
    retained_indices: list[int] = []
    for index, window in enumerate(windows):
        parent_id = str(window.get("parent_id", ""))
        window_sequence = str(window.get("window_seq", "") or "")
        if window_sequence and window_sequence in seen_by_parent[parent_id]:
            continue
        if window_sequence:
            seen_by_parent[parent_id].add(window_sequence)
        retained_indices.append(index)

    if len(cached_labels) != len(retained_indices):
        raise ValueError(
            "cached labels do not match either full windows or Exp3 sequence-deduplicated windows: "
            f"windows={len(windows)}, retained={len(retained_indices)}, labels={len(cached_labels)}"
        )

    expanded = np.full(len(windows), -1, dtype=cached_labels.dtype)
    expanded[np.asarray(retained_indices, dtype=int)] = cached_labels
    return expanded


def select_representative_windows(
    *,
    windows: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    embeddings: np.ndarray,
    catalog_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    labels = np.asarray(labels).reshape(-1)
    embeddings = np.asarray(embeddings)
    if len(windows) != len(labels) or len(labels) != embeddings.shape[0]:
        raise ValueError(
            "windows, labels, and embeddings must contain the same number of rows: "
            f"{len(windows)}, {len(labels)}, {embeddings.shape[0]}"
        )
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be a 2D matrix, got shape {embeddings.shape}")

    catalog_by_label: dict[int, Mapping[str, object]] = {}
    for row in catalog_rows:
        label = _to_int(row.get("cluster_label", -1))
        if label >= 0:
            catalog_by_label[label] = row

    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for index, label_value in enumerate(labels):
        label = int(label_value)
        if label >= 0 and label in catalog_by_label:
            indices_by_label[label].append(index)

    representative_rows: list[dict[str, object]] = []
    for label, indices in sorted(indices_by_label.items()):
        catalog = catalog_by_label[label]
        cluster_vectors = _normalized_rows(embeddings[indices])
        centroid = cluster_vectors.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
            distances = 1.0 - np.clip(cluster_vectors @ centroid, -1.0, 1.0)
        else:
            distances = np.ones(len(indices), dtype=np.float64)

        min_distance = float(np.min(distances))
        tied_positions = np.flatnonzero(
            np.isclose(distances, min_distance, rtol=1e-12, atol=1e-12)
        )
        selected_position = min(
            tied_positions.tolist(),
            key=lambda position: str(windows[indices[position]].get("window_id", "")),
        )
        selected_index = indices[selected_position]
        window = windows[selected_index]
        start = _to_int(window.get("start", 0))
        end = _to_int(window.get("end", 0))
        sequence = str(window.get("window_seq", "") or "")
        if not sequence:
            full_sequence = str(window.get("window_seq_full", "") or "")
            if 0 <= start < end <= len(full_sequence):
                sequence = full_sequence[start:end]

        representative_rows.append(
            {
                "motif_uid": str(catalog.get("motif_uid", "")),
                "family": str(catalog.get("family") or window.get("family") or ""),
                "method": str(catalog.get("method", "")),
                "cluster_label": label,
                "representative_window_id": str(window.get("window_id", "")),
                "representative_parent_id": str(window.get("parent_id", "")),
                "representative_start": start,
                "representative_end": end,
                "representative_length": max(0, end - start),
                "representative_sequence": sequence,
                "distance_to_cluster_centroid": float(distances[selected_position]),
                "n_cluster_windows": len(indices),
            }
        )
    return representative_rows


def write_representative_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPRESENTATIVE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_family_method_representatives(
    *,
    family_dir: Path,
    method: str,
    windows: Sequence[Mapping[str, object]] | None = None,
    labels: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
    catalog_rows: Sequence[Mapping[str, object]] | None = None,
) -> tuple[Path, list[dict[str, object]]]:
    family_dir = Path(family_dir)
    windows = list(windows) if windows is not None else read_csv_rows(family_dir / "windows_metadata.csv")
    labels = labels if labels is not None else np.load(family_dir / f"clustering_{method}_labels.npy")
    labels = expand_cached_labels(windows, labels)
    embeddings = (
        embeddings
        if embeddings is not None
        else np.load(family_dir / f"embeddings_{method}_optimized.npy", mmap_mode="r")
    )
    catalog_rows = (
        list(catalog_rows)
        if catalog_rows is not None
        else read_csv_rows(family_dir / f"motif_catalog_{method}.csv")
    )
    rows = select_representative_windows(
        windows=windows,
        labels=labels,
        embeddings=embeddings,
        catalog_rows=catalog_rows,
    )
    output_path = family_dir / f"motif_representative_windows_{method}.csv"
    write_representative_rows(output_path, rows)
    return output_path, rows




