from __future__ import annotations

import argparse
import ast
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp4_motif_structure_analysis.io import read_csv_rows, write_csv_rows


@dataclass(frozen=True)
class MotifRepresentationResult:
    representation_rows: list[dict[str, object]]
    fragment_rows: list[dict[str, object]]
    profile_rows: list[dict[str, object]]


@dataclass(frozen=True)
class _ReferenceAlignment:
    ref_start: int
    ref_end: int
    sequence_start: int
    sequence_end: int
    score: float
    matches: int
    aligned_pairs: int
    identity: float
    residues_by_ref_pos: Mapping[int, str]


_BLOSUM62_AA = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_DATA = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
_BLOSUM62 = {
    aa: {bb: _BLOSUM62_DATA[i][j] for j, bb in enumerate(_BLOSUM62_AA)}
    for i, aa in enumerate(_BLOSUM62_AA)
}


def _substitution_score(left: str, right: str) -> int:
    if left == right:
        return _BLOSUM62.get(left, {}).get(right, 4)
    return _BLOSUM62.get(left, {}).get(right, -4)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _maybe_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _natural_method_key(method: str) -> tuple[int, str]:
    text = str(method)
    if len(text) > 1 and text[0].upper() == "M" and text[1:].isdigit():
        return (int(text[1:]), text)
    return (10_000, text)


def parse_support_window_ids(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if str(item)]
    for separator in ["|", ","]:
        text = text.replace(separator, ";")
    return [part.strip().strip("'\"[] ") for part in text.split(";") if part.strip().strip("'\"[] ")]


def _row_key(row: Mapping[str, object]) -> tuple[str, int, int, str]:
    return (
        str(row.get("parent_id", "")),
        _to_int(row.get("start")),
        _to_int(row.get("end")),
        str(row.get("window_id", "")),
    )


def _connected_segments(windows: Sequence[Mapping[str, object]], merge_gap: int) -> list[list[Mapping[str, object]]]:
    ordered = sorted(windows, key=_row_key)
    if not ordered:
        return []

    segments: list[list[Mapping[str, object]]] = [[ordered[0]]]
    current_end = _to_int(ordered[0].get("end"))
    for window in ordered[1:]:
        start = _to_int(window.get("start"))
        end = _to_int(window.get("end"))
        if start <= current_end + merge_gap:
            segments[-1].append(window)
            current_end = max(current_end, end)
            continue
        segments.append([window])
        current_end = end
    return segments


def _choose_representative_window(segment: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    starts = [_to_int(row.get("start")) for row in segment]
    ends = [_to_int(row.get("end")) for row in segment]
    segment_center = (min(starts) + max(ends)) / 2.0

    def key(row: Mapping[str, object]) -> tuple[float, int, int, str]:
        start = _to_int(row.get("start"))
        end = _to_int(row.get("end"))
        center = (start + end) / 2.0
        return (abs(center - segment_center), -(end - start), start, str(row.get("window_id", "")))

    return min(segment, key=key)


def _window_sequence(row: Mapping[str, object]) -> str:
    seq = str(row.get("window_seq", "") or "")
    if seq:
        return seq
    full = str(row.get("window_seq_full", "") or "")
    start = _to_int(row.get("start"))
    end = _to_int(row.get("end"))
    if full and 0 <= start < end <= len(full):
        return full[start:end]
    return ""


def _ordered_unique_windows(
    occurrence: Mapping[str, object],
    windows_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], list[str]]:
    parent_id = str(occurrence.get("parent_id", ""))
    seen: set[str] = set()
    windows: list[Mapping[str, object]] = []
    missing: list[str] = []
    for window_id in parse_support_window_ids(occurrence.get("support_window_ids", "")):
        if window_id in seen:
            continue
        seen.add(window_id)
        window = windows_by_id.get(window_id)
        if not window:
            missing.append(window_id)
            continue
        if parent_id and str(window.get("parent_id", "")) != parent_id:
            missing.append(window_id)
            continue
        windows.append(window)
    return sorted(windows, key=_row_key), missing


def _occurrence_span(
    occurrence: Mapping[str, object],
    windows: Sequence[Mapping[str, object]],
) -> tuple[int, int]:
    start = _maybe_int(occurrence.get("occ_start"))
    end = _maybe_int(occurrence.get("occ_end"))
    if start is not None and end is not None and start < end:
        return start, end

    starts = [_to_int(row.get("start")) for row in windows]
    ends = [_to_int(row.get("end")) for row in windows]
    if starts and ends:
        return min(starts), max(ends)
    return 0, 0


def _full_sequence_from_windows(
    windows: Sequence[Mapping[str, object]],
    span_start: int,
    span_end: int,
) -> str:
    for window in windows:
        full = str(window.get("window_seq_full", "") or "")
        if full and 0 <= span_start < span_end <= len(full):
            return full[span_start:span_end]
    return ""


def _stitched_window_sequence(
    windows: Sequence[Mapping[str, object]],
    span_start: int,
    span_end: int,
) -> str:
    if span_start >= span_end:
        return ""
    chars: list[str | None] = [None] * (span_end - span_start)
    for window in sorted(windows, key=_row_key):
        sequence = str(window.get("window_seq", "") or "")
        if not sequence:
            continue
        window_start = _to_int(window.get("start"))
        window_end = _to_int(window.get("end"))
        if window_start >= window_end:
            continue
        left = max(span_start, window_start)
        right = min(span_end, window_end)
        if left >= right:
            continue
        for position in range(left, right):
            sequence_index = position - window_start
            if sequence_index < 0 or sequence_index >= len(sequence):
                continue
            out_index = position - span_start
            if chars[out_index] is None:
                chars[out_index] = sequence[sequence_index]
    if any(char is None for char in chars):
        return ""
    return "".join(str(char) for char in chars)


def _fragment_sequence_for_occurrence(
    windows: Sequence[Mapping[str, object]],
    span_start: int,
    span_end: int,
) -> tuple[str, int, int, str, Mapping[str, object] | None]:
    if not windows:
        return "", span_start, span_end, "missing_support_windows", None

    full_sequence = _full_sequence_from_windows(windows, span_start, span_end)
    representative = _choose_representative_window(windows)
    if full_sequence:
        return full_sequence, span_start, span_end, "occurrence_span_full_sequence", representative

    stitched = _stitched_window_sequence(windows, span_start, span_end)
    if stitched:
        return stitched, span_start, span_end, "occurrence_span_stitched_windows", representative

    representative_sequence = _window_sequence(representative)
    representative_start = _to_int(representative.get("start"))
    representative_end = _to_int(representative.get("end"))
    return (
        representative_sequence,
        representative_start,
        representative_end,
        "representative_window_fallback",
        representative,
    )


def build_occurrence_fragments(
    occurrence_rows: Sequence[Mapping[str, object]],
    windows_by_id: Mapping[str, Mapping[str, object]],
    *,
    method: str,
    merge_gap: int = 5,
) -> list[dict[str, object]]:
    del merge_gap  # Occurrence boundaries are defined by the upstream occurrence file.

    fragments: list[dict[str, object]] = []
    motif_counts: Counter[str] = Counter()
    for occurrence in occurrence_rows:
        motif_uid = str(occurrence.get("motif_uid", ""))
        parent_id = str(occurrence.get("parent_id", ""))
        if not motif_uid or not parent_id:
            continue
        support_windows, missing_window_ids = _ordered_unique_windows(occurrence, windows_by_id)
        if not support_windows:
            continue

        motif_counts[motif_uid] += 1
        span_start, span_end = _occurrence_span(occurrence, support_windows)
        fragment, fragment_start, fragment_end, sequence_source, representative = _fragment_sequence_for_occurrence(
            support_windows,
            span_start,
            span_end,
        )
        source_id = str(occurrence.get("occurrence_id", "") or f"{motif_uid}_occ_{motif_counts[motif_uid]:03d}")
        fragment_id = f"{motif_uid}_fragment_{motif_counts[motif_uid]:05d}"
        family = str(occurrence.get("family") or (representative.get("family", "") if representative else ""))
        fragments.append({
            "fragment_id": fragment_id,
            "occurrence_id": source_id,
            "motif_uid": motif_uid,
            "family": family,
            "method": str(occurrence.get("method") or method),
            "parent_id": parent_id,
            "fragment_start": fragment_start,
            "fragment_end": fragment_end,
            "fragment_length": len(fragment),
            "fragment_sequence": fragment,
            "fragment_sequence_source": sequence_source,
            "occurrence_span_start": span_start,
            "occurrence_span_end": span_end,
            "occurrence_span_length": (span_end - span_start) if span_start < span_end else "",
            "representative_window_id": str(representative.get("window_id", "")) if representative else "",
            "n_support_windows": len(support_windows),
            "n_missing_support_windows": len(missing_window_ids),
            "support_window_ids": ";".join(str(row.get("window_id", "")) for row in support_windows),
            "missing_support_window_ids": ";".join(missing_window_ids),
            "source_occurrence_ids": source_id,
        })
    return fragments


def _modal_length(fragments: Sequence[Mapping[str, object]]) -> int:
    counts = Counter(_to_int(row.get("fragment_length")) for row in fragments if _to_int(row.get("fragment_length")) > 0)
    if not counts:
        return 0
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _entropy(counts: Mapping[str, int], total: int) -> float:
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        value -= p * math.log2(p)
    return value


def _column_counts(sequences: Sequence[str]) -> list[Counter[str]]:
    if not sequences:
        return []
    width = min(len(seq) for seq in sequences)
    return [Counter(seq[pos] for seq in sequences if len(seq) > pos) for pos in range(width)]


def _smith_waterman_reference_alignment(
    reference: str,
    sequence: str,
    *,
    gap_penalty: int,
) -> _ReferenceAlignment | None:
    if not reference or not sequence:
        return None

    rows = len(reference) + 1
    cols = len(sequence) + 1
    scores = [[0] * cols for _ in range(rows)]
    trace = [[""] * cols for _ in range(rows)]
    best_score = 0
    best_cell = (0, 0)

    for i in range(1, rows):
        for j in range(1, cols):
            diag = scores[i - 1][j - 1] + _substitution_score(reference[i - 1], sequence[j - 1])
            up = scores[i - 1][j] + gap_penalty
            left = scores[i][j - 1] + gap_penalty
            score = max(0, diag, up, left)
            scores[i][j] = score
            if score == 0:
                trace[i][j] = ""
            elif score == diag:
                trace[i][j] = "diag"
            elif score == up:
                trace[i][j] = "up"
            else:
                trace[i][j] = "left"
            if score > best_score:
                best_score = score
                best_cell = (i, j)

    if best_score <= 0:
        return None

    i, j = best_cell
    aligned_reference: list[str] = []
    aligned_sequence: list[str] = []
    while i > 0 and j > 0 and scores[i][j] > 0:
        direction = trace[i][j]
        if direction == "diag":
            aligned_reference.append(reference[i - 1])
            aligned_sequence.append(sequence[j - 1])
            i -= 1
            j -= 1
        elif direction == "up":
            aligned_reference.append(reference[i - 1])
            aligned_sequence.append("-")
            i -= 1
        elif direction == "left":
            aligned_reference.append("-")
            aligned_sequence.append(sequence[j - 1])
            j -= 1
        else:
            break

    aligned_reference.reverse()
    aligned_sequence.reverse()

    ref_pos = i
    seq_pos = j
    ref_start = None
    ref_end = None
    sequence_start = None
    sequence_end = None
    residues_by_ref_pos: dict[int, str] = {}
    matches = 0
    aligned_pairs = 0
    for ref_aa, seq_aa in zip(aligned_reference, aligned_sequence):
        if ref_aa != "-":
            if ref_start is None:
                ref_start = ref_pos
            if seq_aa != "-":
                residues_by_ref_pos[ref_pos] = seq_aa
                aligned_pairs += 1
                if ref_aa == seq_aa:
                    matches += 1
            ref_pos += 1
            ref_end = ref_pos
        if seq_aa != "-":
            if sequence_start is None:
                sequence_start = seq_pos
            seq_pos += 1
            sequence_end = seq_pos

    if ref_start is None or ref_end is None or sequence_start is None or sequence_end is None or aligned_pairs <= 0:
        return None

    while ref_start > 0 and sequence_start > 0:
        ref_start -= 1
        sequence_start -= 1
        residues_by_ref_pos[ref_start] = sequence[sequence_start]
        aligned_pairs += 1
        if reference[ref_start] == sequence[sequence_start]:
            matches += 1

    while ref_end < len(reference) and sequence_end < len(sequence):
        residues_by_ref_pos[ref_end] = sequence[sequence_end]
        aligned_pairs += 1
        if reference[ref_end] == sequence[sequence_end]:
            matches += 1
        ref_end += 1
        sequence_end += 1

    return _ReferenceAlignment(
        ref_start=ref_start,
        ref_end=ref_end,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        score=float(best_score),
        matches=matches,
        aligned_pairs=aligned_pairs,
        identity=matches / aligned_pairs,
        residues_by_ref_pos=residues_by_ref_pos,
    )


def _contiguous_position_segments(positions: Sequence[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(positions))
    if not ordered:
        return []
    segments: list[tuple[int, int]] = []
    start = ordered[0]
    previous = ordered[0]
    for position in ordered[1:]:
        if position == previous + 1:
            previous = position
            continue
        segments.append((start, previous + 1))
        start = position
        previous = position
    segments.append((start, previous + 1))
    return segments


def _choose_profile_reference(fragments: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    usable = [row for row in fragments if str(row.get("fragment_sequence", "") or "")]
    if not usable:
        return None
    modal_length = _modal_length(usable)
    candidates = [
        row for row in usable
        if modal_length <= 0 or len(str(row.get("fragment_sequence", "") or "")) == modal_length
    ] or usable
    sequence_counts = Counter(str(row.get("fragment_sequence", "") or "") for row in candidates)
    return max(
        candidates,
        key=lambda row: (
            sequence_counts[str(row.get("fragment_sequence", "") or "")],
            _to_int(row.get("n_support_windows")),
            len(str(row.get("fragment_sequence", "") or "")),
            str(row.get("fragment_id", "")),
        ),
    )


def _aligned_profile_sequences(
    fragments: Sequence[dict[str, object]],
    *,
    min_overlap_fraction: float,
    min_identity: float,
    min_column_coverage_fraction: float,
    gap_penalty: int,
) -> tuple[list[dict[str, object]], list[str], dict[str, object]]:
    updated = [dict(row) for row in fragments]
    reference_row = _choose_profile_reference(updated)
    strategy = "smith_waterman_blosum62_reference"
    if reference_row is None:
        for row in updated:
            row["used_for_profile"] = False
            row["profile_exclusion_reason"] = "empty_fragment"
        return updated, [], {
            "profile_alignment_strategy": strategy,
            "profile_reference_fragment_id": "",
            "profile_reference_sequence": "",
            "profile_reference_start": "",
            "profile_reference_end": "",
            "profile_alignment_min_overlap_fraction": min_overlap_fraction,
            "profile_alignment_min_identity": min_identity,
            "profile_alignment_min_column_coverage_fraction": min_column_coverage_fraction,
            "profile_alignment_gap_penalty": gap_penalty,
        }

    reference = str(reference_row.get("fragment_sequence", "") or "")
    aligned: list[tuple[int, _ReferenceAlignment]] = []
    for idx, row in enumerate(updated):
        sequence = str(row.get("fragment_sequence", "") or "")
        row["profile_alignment_ref_start"] = ""
        row["profile_alignment_ref_end"] = ""
        row["profile_alignment_sequence_start"] = ""
        row["profile_alignment_sequence_end"] = ""
        row["profile_alignment_overlap"] = ""
        row["profile_alignment_identity"] = ""
        row["profile_alignment_score"] = ""
        row["profile_sequence"] = ""
        if not sequence:
            row["used_for_profile"] = False
            row["profile_exclusion_reason"] = "empty_fragment"
            continue

        min_overlap = math.ceil(min(len(reference), len(sequence)) * min_overlap_fraction)
        alignment = _smith_waterman_reference_alignment(reference, sequence, gap_penalty=gap_penalty)
        if alignment is None:
            row["used_for_profile"] = False
            row["profile_exclusion_reason"] = "no_sufficient_overlap"
            continue

        row["profile_alignment_ref_start"] = alignment.ref_start
        row["profile_alignment_ref_end"] = alignment.ref_end
        row["profile_alignment_sequence_start"] = alignment.sequence_start
        row["profile_alignment_sequence_end"] = alignment.sequence_end
        row["profile_alignment_overlap"] = alignment.aligned_pairs
        row["profile_alignment_identity"] = alignment.identity
        row["profile_alignment_score"] = alignment.score
        if alignment.aligned_pairs < min_overlap:
            row["used_for_profile"] = False
            row["profile_exclusion_reason"] = "no_sufficient_overlap"
            continue
        if alignment.identity < min_identity:
            row["used_for_profile"] = False
            row["profile_exclusion_reason"] = "low_alignment_identity"
            continue
        aligned.append((idx, alignment))

    if not aligned:
        return updated, [], {
            "profile_alignment_strategy": strategy,
            "profile_reference_fragment_id": str(reference_row.get("fragment_id", "")),
            "profile_reference_sequence": reference,
            "profile_reference_start": "",
            "profile_reference_end": "",
            "profile_alignment_min_overlap_fraction": min_overlap_fraction,
            "profile_alignment_min_identity": min_identity,
            "profile_alignment_min_column_coverage_fraction": min_column_coverage_fraction,
            "profile_alignment_gap_penalty": gap_penalty,
        }

    column_coverage: Counter[int] = Counter()
    for _, alignment in aligned:
        for ref_pos in alignment.residues_by_ref_pos:
            column_coverage[ref_pos] += 1
    required_coverage = max(1, math.ceil(len(aligned) * min_column_coverage_fraction))
    supported_positions = sorted(pos for pos, count in column_coverage.items() if count >= required_coverage)
    segments = _contiguous_position_segments(supported_positions)
    if not segments:
        for idx, _ in aligned:
            updated[idx]["used_for_profile"] = False
            updated[idx]["profile_exclusion_reason"] = "no_shared_aligned_interval"
        return updated, [], {
            "profile_alignment_strategy": strategy,
            "profile_reference_fragment_id": str(reference_row.get("fragment_id", "")),
            "profile_reference_sequence": reference,
            "profile_reference_start": "",
            "profile_reference_end": "",
            "profile_alignment_min_overlap_fraction": min_overlap_fraction,
            "profile_alignment_min_identity": min_identity,
            "profile_alignment_min_column_coverage_fraction": min_column_coverage_fraction,
            "profile_alignment_gap_penalty": gap_penalty,
        }

    profile_start, profile_end = max(
        segments,
        key=lambda segment: (
            sum(column_coverage[pos] for pos in range(segment[0], segment[1])),
            segment[1] - segment[0],
            -segment[0],
        ),
    )

    used_sequences: list[str] = []
    for idx, alignment in aligned:
        row = updated[idx]
        profile_chars: list[str] = []
        missing_core_position = False
        for ref_pos in range(profile_start, profile_end):
            residue = alignment.residues_by_ref_pos.get(ref_pos)
            if not residue:
                missing_core_position = True
                break
            profile_chars.append(residue)
        if missing_core_position:
            row["used_for_profile"] = False
            row["profile_exclusion_reason"] = "core_interval_gap"
            continue
        profile_sequence = "".join(profile_chars)
        row["used_for_profile"] = True
        row["profile_exclusion_reason"] = ""
        row["profile_sequence"] = profile_sequence
        row["profile_reference_start"] = profile_start
        row["profile_reference_end"] = profile_end
        used_sequences.append(profile_sequence)

    return updated, used_sequences, {
        "profile_alignment_strategy": strategy,
        "profile_reference_fragment_id": str(reference_row.get("fragment_id", "")),
        "profile_reference_sequence": reference,
        "profile_reference_start": profile_start,
        "profile_reference_end": profile_end,
        "profile_alignment_min_overlap_fraction": min_overlap_fraction,
        "profile_alignment_min_identity": min_identity,
        "profile_alignment_min_column_coverage_fraction": min_column_coverage_fraction,
        "profile_alignment_gap_penalty": gap_penalty,
    }


def _degenerate_token(
    counts: Counter[str],
    total: int,
    *,
    conserved_threshold: float,
    degenerate_cumulative: float,
    max_degenerate_size: int,
) -> str:
    if not counts or total <= 0:
        return "X"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_aa, top_count = ordered[0]
    top_freq = top_count / total
    if top_freq >= conserved_threshold:
        return top_aa

    selected: list[str] = []
    cumulative = 0
    for aa, count in ordered:
        selected.append(aa)
        cumulative += count
        if cumulative / total >= degenerate_cumulative:
            break
    if 1 < len(selected) <= max_degenerate_size:
        return "[" + "".join(sorted(selected)) + "]"
    return "X"


def _profile_score(sequence: str, profile: Sequence[Mapping[str, float]]) -> float:
    if not sequence or not profile:
        return 0.0
    width = min(len(sequence), len(profile))
    return sum(float(profile[pos].get(sequence[pos], 0.0)) for pos in range(width)) / width


def _hamming_to_consensus(sequence: str, consensus: str) -> float:
    if not sequence or not consensus:
        return 1.0
    width = min(len(sequence), len(consensus))
    if width <= 0:
        return 1.0
    return sum(1 for idx in range(width) if sequence[idx] != consensus[idx]) / width


def _confidence_level(
    *,
    n_used: int,
    n_total: int,
    min_occurrences: int,
    mean_conservation: float,
) -> str:
    length_consistency = (n_used / n_total) if n_total else 0.0
    if n_used >= min_occurrences and mean_conservation >= 0.75 and length_consistency >= 0.8:
        return "high"
    if n_used >= min_occurrences and mean_conservation >= 0.55 and length_consistency >= 0.6:
        return "medium"
    return "low"


def _alignment_quality(n_used: int, n_total: int) -> str:
    if not n_total:
        return "no_fragments"
    ratio = n_used / n_total
    if ratio >= 0.9:
        return "length_consistent"
    if ratio >= 0.6:
        return "partly_length_variable"
    return "length_variable"


def _build_profile_for_motif(
    fragments: Sequence[Mapping[str, object]],
    *,
    min_occurrences: int,
    conserved_threshold: float,
    degenerate_cumulative: float,
    max_degenerate_size: int,
    alignment_min_overlap_fraction: float,
    alignment_min_identity: float,
    alignment_min_column_coverage_fraction: float,
    alignment_gap_penalty: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    motif_uid = str(fragments[0].get("motif_uid", "")) if fragments else ""
    family = str(fragments[0].get("family", "")) if fragments else ""
    method = str(fragments[0].get("method", "")) if fragments else ""
    updated_fragments, used_sequences, alignment_meta = _aligned_profile_sequences(
        [dict(fragment) for fragment in fragments],
        min_overlap_fraction=alignment_min_overlap_fraction,
        min_identity=alignment_min_identity,
        min_column_coverage_fraction=alignment_min_column_coverage_fraction,
        gap_penalty=alignment_gap_penalty,
    )

    counts_by_pos = _column_counts(used_sequences)
    profile: list[dict[str, float]] = []
    profile_rows: list[dict[str, object]] = []
    consensus_tokens: list[str] = []
    degenerate_tokens: list[str] = []
    conservation_values: list[float] = []
    entropy_values: list[float] = []

    for pos, counts in enumerate(counts_by_pos, start=1):
        position_total = sum(counts.values())
        frequencies = {aa: count / position_total for aa, count in counts.items()} if position_total else {}
        profile.append(frequencies)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        consensus_aa = ordered[0][0] if ordered else "X"
        consensus_tokens.append(consensus_aa)
        degenerate_tokens.append(
            _degenerate_token(
                counts,
                position_total,
                conserved_threshold=conserved_threshold,
                degenerate_cumulative=degenerate_cumulative,
                max_degenerate_size=max_degenerate_size,
            )
        )
        conservation = (ordered[0][1] / position_total) if ordered and position_total else 0.0
        entropy = _entropy(counts, position_total)
        conservation_values.append(conservation)
        entropy_values.append(entropy)
        for aa, count in sorted(counts.items()):
            profile_rows.append({
                "motif_uid": motif_uid,
                "family": family,
                "method": method,
                "position": pos,
                "aa": aa,
                "count": count,
                "frequency": count / position_total if position_total else 0.0,
                "consensus_aa": consensus_aa,
                "position_conservation": conservation,
                "position_entropy": entropy,
            })

    consensus = "".join(consensus_tokens)
    degenerate = "".join(degenerate_tokens)
    used_rows = [row for row in updated_fragments if row.get("used_for_profile")]
    representative = None
    if used_rows:
        representative = max(
            used_rows,
            key=lambda row: (
                _profile_score(str(row.get("profile_sequence", "")), profile),
                -_hamming_to_consensus(str(row.get("profile_sequence", "")), consensus),
                _to_int(row.get("n_support_windows")),
                str(row.get("parent_id", "")),
            ),
        )

    mean_conservation = sum(conservation_values) / len(conservation_values) if conservation_values else 0.0
    mean_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
    n_total = len(fragments)
    n_used = len(used_sequences)
    profile_width = len(used_sequences[0]) if used_sequences else 0
    length_counts = Counter(_to_int(row.get("fragment_length")) for row in fragments if _to_int(row.get("fragment_length")) > 0)
    representation = {
        "motif_uid": motif_uid,
        "family": family,
        "method": method,
        "consensus_sequence": consensus,
        "degenerate_template": degenerate,
        "representative_sequence": str(representative.get("profile_sequence", "") or representative.get("fragment_sequence", "")) if representative else "",
        "representative_fragment_sequence": str(representative.get("fragment_sequence", "")) if representative else "",
        "representative_parent_id": str(representative.get("parent_id", "")) if representative else "",
        "representative_fragment_id": str(representative.get("fragment_id", "")) if representative else "",
        "representative_start": representative.get("fragment_start", "") if representative else "",
        "representative_end": representative.get("fragment_end", "") if representative else "",
        "n_occurrences": n_total,
        "n_profile_occurrences": n_used,
        "n_unique_proteins": len({str(row.get("parent_id", "")) for row in fragments if str(row.get("parent_id", ""))}),
        "motif_length": profile_width,
        "fragment_length_distribution": ";".join(f"{length}:{count}" for length, count in sorted(length_counts.items())),
        "mean_conservation": mean_conservation,
        "mean_entropy": mean_entropy,
        "alignment_quality": _alignment_quality(n_used, n_total),
        "confidence_level": _confidence_level(
            n_used=n_used,
            n_total=n_total,
            min_occurrences=min_occurrences,
            mean_conservation=mean_conservation,
        ),
    }
    representation.update(alignment_meta)
    return representation, updated_fragments, profile_rows


def build_motif_representations(
    occurrence_rows: Sequence[Mapping[str, object]],
    windows_by_id: Mapping[str, Mapping[str, object]],
    *,
    method: str,
    merge_gap: int = 5,
    min_occurrences: int = 3,
    conserved_threshold: float = 0.75,
    degenerate_cumulative: float = 0.80,
    max_degenerate_size: int = 4,
    alignment_min_overlap_fraction: float = 0.60,
    alignment_min_identity: float = 0.45,
    alignment_min_column_coverage_fraction: float = 0.80,
    alignment_gap_penalty: int = -6,
) -> MotifRepresentationResult:
    fragments = build_occurrence_fragments(
        occurrence_rows,
        windows_by_id,
        method=method,
        merge_gap=merge_gap,
    )
    by_motif: dict[str, list[dict[str, object]]] = defaultdict(list)
    for fragment in fragments:
        by_motif[str(fragment.get("motif_uid", ""))].append(fragment)

    representation_rows: list[dict[str, object]] = []
    fragment_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    for motif_uid in sorted(by_motif):
        representation, motif_fragments, motif_profile = _build_profile_for_motif(
            by_motif[motif_uid],
            min_occurrences=min_occurrences,
            conserved_threshold=conserved_threshold,
            degenerate_cumulative=degenerate_cumulative,
            max_degenerate_size=max_degenerate_size,
            alignment_min_overlap_fraction=alignment_min_overlap_fraction,
            alignment_min_identity=alignment_min_identity,
            alignment_min_column_coverage_fraction=alignment_min_column_coverage_fraction,
            alignment_gap_penalty=alignment_gap_penalty,
        )
        representation_rows.append(representation)
        fragment_rows.extend(motif_fragments)
        profile_rows.extend(motif_profile)

    return MotifRepresentationResult(
        representation_rows=representation_rows,
        fragment_rows=fragment_rows,
        profile_rows=profile_rows,
    )


def _method_from_occurrence_file(path: Path) -> str:
    return path.stem.replace("motif_occurrences_", "", 1)


def discover_units(exp3_dir: Path, families: Sequence[str] | None, methods: Sequence[str] | None) -> list[tuple[str, str]]:
    supported_methods = {"M2", "M3"}
    method_filter = set(methods or supported_methods)
    if families:
        family_dirs = [exp3_dir / family for family in families]
    else:
        family_dirs = [path for path in exp3_dir.iterdir() if path.is_dir()] if exp3_dir.exists() else []

    units: list[tuple[str, str]] = []
    for family_dir in sorted(family_dirs, key=lambda p: p.name):
        if not family_dir.exists():
            continue
        discovered = sorted(
            {_method_from_occurrence_file(path) for path in family_dir.glob("motif_occurrences_*.csv")},
            key=_natural_method_key,
        )
        for method in discovered:
            if method not in supported_methods:
                continue
            if method_filter and method not in method_filter:
                continue
            units.append((family_dir.name, method))
    return units


def _load_windows(path: Path) -> dict[str, dict[str, object]]:
    rows = read_csv_rows(path)
    return {str(row.get("window_id", "")): row for row in rows if str(row.get("window_id", ""))}


def _write_fasta(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8", newline="") as handle:
        for row in rows:
            seq = str(row.get("representative_sequence", "") or row.get("consensus_sequence", "") or "")
            if not seq:
                continue
            header = (
                f">{row.get('motif_uid')}|method={row.get('method')}|"
                f"n_occurrences={row.get('n_occurrences')}|confidence={row.get('confidence_level')}"
            )
            handle.write(header + "\n")
            handle.write(seq + "\n")
    temp_path.replace(path)


def run_motif_representation(
    *,
    exp3_dir: Path,
    output_dir: Path | None = None,
    families: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    merge_gap: int = 5,
    min_occurrences: int = 3,
    conserved_threshold: float = 0.75,
    degenerate_cumulative: float = 0.80,
    max_degenerate_size: int = 4,
    alignment_min_overlap_fraction: float = 0.60,
    alignment_min_identity: float = 0.45,
    alignment_min_column_coverage_fraction: float = 0.80,
    alignment_gap_penalty: int = -6,
) -> dict[str, list[Path]]:
    exp3_dir = Path(exp3_dir)
    units = discover_units(exp3_dir, families=families, methods=methods)
    if not units:
        raise FileNotFoundError(f"No Exp3 motif occurrence files found in {exp3_dir}")

    written: dict[str, list[Path]] = {}
    for family, method in units:
        family_dir = exp3_dir / family
        windows_path = family_dir / "windows_metadata.csv"
        occurrence_path = family_dir / f"motif_occurrences_{method}.csv"
        if not windows_path.exists():
            raise FileNotFoundError(f"Missing windows metadata: {windows_path}")
        if not occurrence_path.exists():
            raise FileNotFoundError(f"Missing motif occurrences: {occurrence_path}")

        windows_by_id = _load_windows(windows_path)
        occurrence_rows = read_csv_rows(occurrence_path)
        result = build_motif_representations(
            occurrence_rows,
            windows_by_id,
            method=method,
            merge_gap=merge_gap,
            min_occurrences=min_occurrences,
            conserved_threshold=conserved_threshold,
            degenerate_cumulative=degenerate_cumulative,
            max_degenerate_size=max_degenerate_size,
            alignment_min_overlap_fraction=alignment_min_overlap_fraction,
            alignment_min_identity=alignment_min_identity,
            alignment_min_column_coverage_fraction=alignment_min_column_coverage_fraction,
            alignment_gap_penalty=alignment_gap_penalty,
        )

        family_output_dir = Path(output_dir) if output_dir else family_dir / "motif_sequence_representations"
        if output_dir and len({unit_family for unit_family, _ in units}) > 1:
            family_output_dir = Path(output_dir) / family
        method_output_dir = family_output_dir / method

        representation_path = method_output_dir / "motif_representations.csv"
        fragment_path = method_output_dir / "motif_occurrence_fragments.csv"
        profile_path = method_output_dir / "motif_position_profile.csv"
        fasta_path = method_output_dir / "motif_representations.fasta"

        write_csv_rows(representation_path, result.representation_rows)
        write_csv_rows(fragment_path, result.fragment_rows)
        write_csv_rows(profile_path, result.profile_rows)
        _write_fasta(fasta_path, result.representation_rows)
        written[f"{family}:{method}"] = [representation_path, fragment_path, profile_path, fasta_path]
    return written


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


try:
    from config import OUTPUT_DIR as _default_results_dir
except ImportError:
    _default_results_dir = _project_root() / "work" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build occurrence-level motif sequence representations from Exp3 outputs.",
    )
    parser.add_argument("--family", action="append", dest="families", help="Family to process; can be provided more than once.")
    parser.add_argument("--families", nargs="+", dest="families_list", help="Families to process.")
    parser.add_argument("--methods", nargs="+", choices=["M2", "M3"], help="Methods to process, for example M2 M3.")
    parser.add_argument("--results-dir", type=Path, default=_default_results_dir, help="Project results directory.")
    parser.add_argument("--exp3-dir", type=Path, help="Exp3 output directory. Defaults to <results-dir>/exp3_domain_discovery.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory override.")
    parser.add_argument("--merge-gap", type=int, default=5, help="Deprecated; occurrence boundaries are read from motif_occurrences files.")
    parser.add_argument("--min-occurrences", type=int, default=3, help="Minimum profile occurrences for medium/high confidence.")
    parser.add_argument("--conserved-threshold", type=float, default=0.75, help="Top-residue frequency threshold for fixed consensus positions.")
    parser.add_argument("--degenerate-cumulative", type=float, default=0.80, help="Cumulative frequency threshold for degenerate tokens.")
    parser.add_argument("--max-degenerate-size", type=int, default=4, help="Maximum residues inside a degenerate token before using X.")
    parser.add_argument("--alignment-min-overlap-fraction", type=float, default=0.60, help="Minimum local-alignment overlap fraction for profile inclusion.")
    parser.add_argument("--alignment-min-identity", type=float, default=0.45, help="Minimum exact-match identity after local alignment for profile inclusion.")
    parser.add_argument("--alignment-min-column-coverage-fraction", type=float, default=0.80, help="Minimum aligned-sequence coverage for retaining a reference profile column.")
    parser.add_argument("--alignment-gap-penalty", type=int, default=-6, help="Linear gap penalty used by Smith-Waterman BLOSUM62 profile alignment.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    families = list(args.families or [])
    if args.families_list:
        families.extend(args.families_list)
    families = families or None
    exp3_dir = args.exp3_dir or (args.results_dir / "exp3_domain_discovery")

    written = run_motif_representation(
        exp3_dir=exp3_dir,
        output_dir=args.output_dir,
        families=families,
        methods=args.methods,
        merge_gap=args.merge_gap,
        min_occurrences=args.min_occurrences,
        conserved_threshold=args.conserved_threshold,
        degenerate_cumulative=args.degenerate_cumulative,
        max_degenerate_size=args.max_degenerate_size,
        alignment_min_overlap_fraction=args.alignment_min_overlap_fraction,
        alignment_min_identity=args.alignment_min_identity,
        alignment_min_column_coverage_fraction=args.alignment_min_column_coverage_fraction,
        alignment_gap_penalty=args.alignment_gap_penalty,
    )
    for unit, paths in written.items():
        print(f"{unit}:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
