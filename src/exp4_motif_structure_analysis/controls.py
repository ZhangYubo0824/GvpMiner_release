from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .features import amino_acid_composition_features

Window = Tuple[int, int]
CompositionWindowIndex = Dict[Tuple[str, int], List[Tuple[str, Window, Dict[str, float]]]]
COMPOSITION_FEATURE_KEYS = (
    "hydrophobic_fraction",
    "charged_fraction",
    "aromatic_fraction",
    "polar_fraction",
    "motif_entropy",
)


@dataclass(frozen=True)
class PositionSequenceIndex:
    sequence_lengths: dict[str, int]
    family_sequence_ids: dict[str, tuple[str, ...]]
    family_lengths: dict[str, np.ndarray]


@dataclass(frozen=True)
class CompositionFeatureBlock:
    sequence_ids: tuple[str, ...]
    sequence_ranks: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    features: np.ndarray


CompositionFeatureIndex = Dict[Tuple[str, int], CompositionFeatureBlock]


def intervals_overlap(a: Window, b: Window) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def candidate_within_protein_windows(
    *,
    sequence_length: int,
    window_length: int,
    excluded_intervals: Iterable[Window],
) -> list[Window]:
    if sequence_length <= 0 or window_length <= 0 or sequence_length < window_length:
        return []
    blocked = np.zeros(sequence_length, dtype=np.int32)
    for start, end in excluded_intervals:
        start = max(0, int(start))
        end = min(sequence_length, int(end))
        if end > start:
            blocked[start:end] = 1
    prefix = np.concatenate(([0], np.cumsum(blocked)))
    candidates = []
    for start in range(0, sequence_length - window_length + 1):
        end = start + window_length
        if int(prefix[end] - prefix[start]) == 0:
            candidates.append((start, end))
    return candidates


def sample_windows_from_candidates(
    candidates: Sequence[Window],
    *,
    num_controls: int,
    rng_seed: int | None = None,
) -> list[Window]:
    if not candidates or num_controls <= 0:
        return []
    rng = np.random.default_rng(rng_seed)
    replace = len(candidates) < num_controls
    indices = rng.choice(len(candidates), size=num_controls, replace=replace)
    return [candidates[int(index)] for index in indices]


def sample_within_protein_windows(
    *,
    sequence_length: int,
    window_length: int,
    excluded_intervals: Iterable[Window],
    num_controls: int,
    rng_seed: int | None = None,
) -> list[Window]:
    candidates = candidate_within_protein_windows(
        sequence_length=sequence_length,
        window_length=window_length,
        excluded_intervals=excluded_intervals,
    )
    return sample_windows_from_candidates(
        candidates,
        num_controls=num_controls,
        rng_seed=rng_seed,
    )


def _relative_position(start: int, end: int, sequence_length: int) -> float:
    if sequence_length <= 0:
        return 0.0
    return ((start + end) / 2.0) / float(sequence_length)


def sample_position_matched_windows(
    *,
    motif_sequence_id: str,
    motif_family: str,
    motif_start: int,
    motif_end: int,
    sequences: dict[str, dict[str, object]],
    num_controls: int,
    position_tolerance: float = 0.1,
) -> list[tuple[str, Window]]:
    motif_length = motif_end - motif_start
    source_length = int(sequences.get(motif_sequence_id, {}).get("sequence_length", 0) or 0)
    target_position = _relative_position(motif_start, motif_end, source_length)
    candidates = []
    for sequence_id, row in sorted(sequences.items()):
        if sequence_id == motif_sequence_id:
            continue
        if str(row.get("family", "")) != motif_family:
            continue
        seq_len = int(row.get("sequence_length", len(str(row.get("sequence", "")))) or 0)
        if seq_len < motif_length:
            continue
        start = int(round(target_position * seq_len - motif_length / 2.0))
        start = max(0, min(seq_len - motif_length, start))
        window = (start, start + motif_length)
        actual_position = _relative_position(window[0], window[1], seq_len)
        if abs(actual_position - target_position) <= position_tolerance:
            candidates.append((sequence_id, window))
    return candidates[:num_controls]


def build_position_sequence_index(sequences: dict[str, dict[str, object]]) -> PositionSequenceIndex:
    sequence_lengths = {}
    by_family: dict[str, list[tuple[str, int]]] = {}
    for sequence_id, row in sorted(sequences.items()):
        sequence = str(row.get("sequence", ""))
        sequence_length = int(row.get("sequence_length", len(sequence)) or 0)
        sequence_lengths[sequence_id] = sequence_length
        by_family.setdefault(str(row.get("family", "")), []).append((sequence_id, sequence_length))
    return PositionSequenceIndex(
        sequence_lengths=sequence_lengths,
        family_sequence_ids={
            family: tuple(sequence_id for sequence_id, _length in rows)
            for family, rows in by_family.items()
        },
        family_lengths={
            family: np.asarray([length for _sequence_id, length in rows], dtype=np.int64)
            for family, rows in by_family.items()
        },
    )


def sample_position_matched_windows_from_index(
    *,
    motif_sequence_id: str,
    motif_family: str,
    motif_start: int,
    motif_end: int,
    index: PositionSequenceIndex,
    num_controls: int,
    position_tolerance: float = 0.1,
) -> list[tuple[str, Window]]:
    if num_controls <= 0:
        return []
    motif_length = motif_end - motif_start
    source_length = int(index.sequence_lengths.get(motif_sequence_id, 0) or 0)
    target_position = _relative_position(motif_start, motif_end, source_length)
    sequence_ids = index.family_sequence_ids.get(motif_family, ())
    lengths = index.family_lengths.get(motif_family)
    if not sequence_ids or lengths is None or motif_length <= 0:
        return []
    valid = lengths >= motif_length
    if motif_sequence_id in sequence_ids:
        valid = valid & (np.asarray(sequence_ids, dtype=object) != motif_sequence_id)
    if not np.any(valid):
        return []
    starts = np.rint(target_position * lengths - motif_length / 2.0).astype(np.int64)
    starts = np.maximum(0, np.minimum(lengths - motif_length, starts))
    actual_positions = ((starts + starts + motif_length) / 2.0) / lengths
    valid = valid & (np.abs(actual_positions - target_position) <= position_tolerance)
    selected_indices = np.nonzero(valid)[0][:num_controls]
    return [
        (sequence_ids[int(index_value)], (int(starts[int(index_value)]), int(starts[int(index_value)] + motif_length)))
        for index_value in selected_indices
    ]


def sample_composition_matched_windows(
    *,
    motif_sequence: str,
    motif_sequence_id: str,
    motif_family: str,
    window_length: int,
    sequences: dict[str, dict[str, object]],
    excluded_by_sequence: dict[str, list[Window]],
    num_controls: int,
    max_candidates_per_sequence: int = 200,
) -> list[tuple[str, Window]]:
    motif_features = amino_acid_composition_features(motif_sequence)
    scored: list[tuple[float, str, Window]] = []
    for sequence_id, row in sorted(sequences.items()):
        if str(row.get("family", "")) != motif_family:
            continue
        sequence = str(row.get("sequence", ""))
        if len(sequence) < window_length:
            continue
        stride = max(1, math.ceil((len(sequence) - window_length + 1) / max_candidates_per_sequence))
        for start in range(0, len(sequence) - window_length + 1, stride):
            window = (start, start + window_length)
            if any(intervals_overlap(window, interval) for interval in excluded_by_sequence.get(sequence_id, [])):
                continue
            candidate_features = amino_acid_composition_features(sequence[start : start + window_length])
            distance = sum(
                abs(float(candidate_features[key]) - float(motif_features[key]))
                for key in ("hydrophobic_fraction", "charged_fraction", "aromatic_fraction", "polar_fraction", "motif_entropy")
            )
            scored.append((distance, sequence_id, window))
    scored.sort(key=lambda item: (item[0], item[1], item[2][0]))
    return [(sequence_id, window) for _, sequence_id, window in scored[:num_controls]]


def build_composition_window_index(
    *,
    sequences: dict[str, dict[str, object]],
    excluded_by_sequence: dict[str, list[Window]],
    window_lengths: Iterable[int],
    max_candidates_per_sequence: int = 200,
) -> CompositionWindowIndex:
    index: CompositionWindowIndex = {}
    lengths = sorted({int(length) for length in window_lengths if int(length) > 0})
    for window_length in lengths:
        by_family: dict[str, list[tuple[str, Window, dict[str, float]]]] = {}
        for sequence_id, row in sorted(sequences.items()):
            family = str(row.get("family", ""))
            sequence = str(row.get("sequence", ""))
            if len(sequence) < window_length:
                continue
            stride = max(1, math.ceil((len(sequence) - window_length + 1) / max_candidates_per_sequence))
            for start in range(0, len(sequence) - window_length + 1, stride):
                window = (start, start + window_length)
                if any(intervals_overlap(window, interval) for interval in excluded_by_sequence.get(sequence_id, [])):
                    continue
                candidate_features = amino_acid_composition_features(sequence[start : start + window_length])
                by_family.setdefault(family, []).append((sequence_id, window, candidate_features))
        for family, candidates in by_family.items():
            index[(family, window_length)] = candidates
    return index


def build_composition_feature_index(index: CompositionWindowIndex) -> CompositionFeatureIndex:
    result: CompositionFeatureIndex = {}
    for key, candidates in index.items():
        if not candidates:
            continue
        sequence_ids = tuple(sequence_id for sequence_id, _window, _features in candidates)
        sequence_rank_map = {sequence_id: rank for rank, sequence_id in enumerate(sorted(set(sequence_ids)))}
        starts = np.asarray([window[0] for _sequence_id, window, _features in candidates], dtype=np.int64)
        ends = np.asarray([window[1] for _sequence_id, window, _features in candidates], dtype=np.int64)
        features = np.asarray(
            [
                [float(candidate_features[feature_key]) for feature_key in COMPOSITION_FEATURE_KEYS]
                for _sequence_id, _window, candidate_features in candidates
            ],
            dtype=float,
        )
        sequence_ranks = np.asarray([sequence_rank_map[sequence_id] for sequence_id in sequence_ids], dtype=np.int64)
        result[key] = CompositionFeatureBlock(
            sequence_ids=sequence_ids,
            sequence_ranks=sequence_ranks,
            starts=starts,
            ends=ends,
            features=features,
        )
    return result


def sample_composition_matched_windows_from_index(
    *,
    motif_sequence: str,
    motif_sequence_id: str,
    motif_family: str,
    window_length: int,
    index: CompositionWindowIndex,
    num_controls: int,
) -> list[tuple[str, Window]]:
    if num_controls <= 0:
        return []
    motif_features = amino_acid_composition_features(motif_sequence)
    scored: list[tuple[float, str, Window]] = []
    for sequence_id, window, candidate_features in index.get((motif_family, window_length), []):
        distance = sum(
            abs(float(candidate_features[key]) - float(motif_features[key]))
            for key in ("hydrophobic_fraction", "charged_fraction", "aromatic_fraction", "polar_fraction", "motif_entropy")
        )
        scored.append((distance, sequence_id, window))
    scored.sort(key=lambda item: (item[0], item[1], item[2][0]))
    return [(sequence_id, window) for _, sequence_id, window in scored[:num_controls]]


def sample_composition_matched_windows_from_feature_index(
    *,
    motif_sequence: str,
    motif_family: str,
    window_length: int,
    index: CompositionFeatureIndex,
    num_controls: int,
) -> list[tuple[str, Window]]:
    if num_controls <= 0:
        return []
    block = index.get((motif_family, window_length))
    if block is None or len(block.sequence_ids) == 0:
        return []
    motif_features = amino_acid_composition_features(motif_sequence)
    motif_vector = np.asarray([float(motif_features[key]) for key in COMPOSITION_FEATURE_KEYS], dtype=float)
    distances = np.abs(block.features[:, 0] - motif_vector[0])
    for column_index in range(1, len(COMPOSITION_FEATURE_KEYS)):
        distances = distances + np.abs(block.features[:, column_index] - motif_vector[column_index])
    selected_indices = np.lexsort((block.starts, block.sequence_ranks, distances))[:num_controls]
    return [
        (
            block.sequence_ids[int(index_value)],
            (int(block.starts[int(index_value)]), int(block.ends[int(index_value)])),
        )
        for index_value in selected_indices
    ]
