from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def window_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    return _jaccard(set(left), set(right))


def protein_jaccard(
    reference_windows: Sequence[Mapping[str, object]],
    sample_windows: Sequence[Mapping[str, object]],
    *,
    retained_proteins: set[str] | None = None,
) -> float:
    reference = {
        str(row["parent_id"])
        for row in reference_windows
        if retained_proteins is None or str(row["parent_id"]) in retained_proteins
    }
    sample = {
        str(row["parent_id"])
        for row in sample_windows
        if retained_proteins is None or str(row["parent_id"]) in retained_proteins
    }
    return _jaccard(reference, sample)


def interval_iou(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    left_start, left_end = int(left["start"]), int(left["end"])
    right_start, right_end = int(right["start"]), int(right["end"])
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return overlap / union if union > 0 else 0.0


def _overlap_fraction(start: int, end: int, other_start: int, other_end: int) -> float:
    overlap = max(0, min(end, other_end) - max(start, other_start))
    denominator = min(end - start, other_end - other_start)
    return overlap / denominator if denominator > 0 else 0.0


def collapse_cluster_occurrences(
    windows: Sequence[Mapping[str, object]],
    *,
    min_overlap_fraction: float = 0.5,
    diffuse_length: int = 90,
) -> list[dict[str, object]]:
    if not 0.0 <= min_overlap_fraction <= 1.0:
        raise ValueError("min_overlap_fraction must be between 0 and 1")
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in windows:
        label = int(row.get("cluster_label", -1))
        if label < 0:
            continue
        grouped[(label, str(row["parent_id"]))].append(row)

    occurrences: list[dict[str, object]] = []
    for (label, parent_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: (int(row["start"]), int(row["end"]), str(row["window_id"])))
        current: list[Mapping[str, object]] = []
        region_start = region_end = 0

        def emit() -> None:
            if not current:
                return
            length = region_end - region_start
            occurrences.append(
                {
                    "cluster_label": label,
                    "parent_id": parent_id,
                    "start": region_start,
                    "end": region_end,
                    "length": length,
                    "n_windows": len(current),
                    "window_ids": tuple(str(row["window_id"]) for row in current),
                    "is_diffuse": length > diffuse_length,
                }
            )

        for row in ordered:
            start, end = int(row["start"]), int(row["end"])
            if not current:
                current = [row]
                region_start, region_end = start, end
                continue
            if _overlap_fraction(region_start, region_end, start, end) >= min_overlap_fraction:
                current.append(row)
                region_start = min(region_start, start)
                region_end = max(region_end, end)
                continue
            emit()
            current = [row]
            region_start, region_end = start, end
        emit()

    return sorted(
        occurrences,
        key=lambda row: (
            int(row["cluster_label"]),
            str(row["parent_id"]),
            int(row["start"]),
            int(row["end"]),
        ),
    )


def occurrence_jaccard(
    reference_occurrences: Sequence[Mapping[str, object]],
    sample_occurrences: Sequence[Mapping[str, object]],
    *,
    retained_proteins: set[str] | None = None,
    min_region_iou: float = 0.5,
) -> float:
    reference = [
        row
        for row in reference_occurrences
        if retained_proteins is None or str(row["parent_id"]) in retained_proteins
    ]
    sample = [
        row
        for row in sample_occurrences
        if retained_proteins is None or str(row["parent_id"]) in retained_proteins
    ]
    if not reference and not sample:
        return 1.0

    candidates = []
    for left_index, left in enumerate(reference):
        for right_index, right in enumerate(sample):
            if str(left["parent_id"]) != str(right["parent_id"]):
                continue
            score = interval_iou(left, right)
            if score >= min_region_iou:
                candidates.append((score, left_index, right_index))
    candidates.sort(reverse=True)

    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = 0
    for _, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches += 1

    union_count = len(reference) + len(sample) - matches
    return matches / union_count if union_count else 1.0
