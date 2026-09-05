from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SequenceAlignment:
    aligned_query: str
    aligned_reference: str
    query_to_reference: dict[int, int | None]
    identity: float
    query_coverage: float


@dataclass(frozen=True)
class IntervalMapping:
    reference_start: int | None
    reference_end: int | None
    mapped_residue_count: int
    query_residue_count: int
    mapped_fraction: float
    identity: float


@dataclass(frozen=True)
class QueryIntervalProjection:
    query_start: int | None
    query_end: int | None
    mapped_residue_count: int
    reference_residue_count: int
    mapped_fraction: float


@dataclass(frozen=True)
class RegionOverlap:
    overlap_residue_count: int
    motif_overlap_fraction: float
    region_overlap_fraction: float


@dataclass(frozen=True)
class ReferenceSelection:
    reference_id: str
    alignment: SequenceAlignment


def align_query_to_reference(
    query: str,
    reference: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -2,
) -> SequenceAlignment:
    query = query.upper()
    reference = reference.upper()
    n = len(query)
    m = len(reference)
    scores = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        scores[i][0] = i * gap_score
        trace[i][0] = "up"
    for j in range(1, m + 1):
        scores[0][j] = j * gap_score
        trace[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = scores[i - 1][j - 1] + (
                match_score if query[i - 1] == reference[j - 1] else mismatch_score
            )
            up = scores[i - 1][j] + gap_score
            left = scores[i][j - 1] + gap_score
            best = max(diagonal, up, left)
            scores[i][j] = best
            if diagonal == best:
                trace[i][j] = "diagonal"
            elif up == best:
                trace[i][j] = "up"
            else:
                trace[i][j] = "left"

    aligned_query: list[str] = []
    aligned_reference: list[str] = []
    i = n
    j = m
    while i or j:
        direction = trace[i][j]
        if direction == "diagonal":
            aligned_query.append(query[i - 1])
            aligned_reference.append(reference[j - 1])
            i -= 1
            j -= 1
        elif direction == "up":
            aligned_query.append(query[i - 1])
            aligned_reference.append("-")
            i -= 1
        else:
            aligned_query.append("-")
            aligned_reference.append(reference[j - 1])
            j -= 1

    aligned_query.reverse()
    aligned_reference.reverse()
    query_to_reference: dict[int, int | None] = {}
    query_position = 0
    reference_position = 0
    matches = 0
    mapped = 0
    for query_residue, reference_residue in zip(
        aligned_query, aligned_reference, strict=True
    ):
        if query_residue != "-":
            query_position += 1
        if reference_residue != "-":
            reference_position += 1
        if query_residue != "-":
            mapped_position = (
                reference_position if reference_residue != "-" else None
            )
            query_to_reference[query_position] = mapped_position
            if mapped_position is not None:
                mapped += 1
                matches += int(query_residue == reference_residue)

    return SequenceAlignment(
        aligned_query="".join(aligned_query),
        aligned_reference="".join(aligned_reference),
        query_to_reference=query_to_reference,
        identity=matches / mapped if mapped else 0.0,
        query_coverage=mapped / n if n else 0.0,
    )


def map_query_interval(
    alignment: SequenceAlignment,
    start: int,
    end: int,
) -> IntervalMapping:
    if start < 0 or end <= start:
        raise ValueError("Expected a non-empty zero-based half-open interval")
    query_positions = range(start + 1, end + 1)
    mapped_positions = [
        alignment.query_to_reference.get(position)
        for position in query_positions
    ]
    mapped_positions = [
        position for position in mapped_positions if position is not None
    ]
    identity_count = 0
    alignment_pairs: dict[int, tuple[str, str]] = {}
    query_position = 0
    for query_residue, reference_residue in zip(
        alignment.aligned_query,
        alignment.aligned_reference,
        strict=True,
    ):
        if query_residue == "-":
            continue
        query_position += 1
        alignment_pairs[query_position] = (query_residue, reference_residue)
    for position in query_positions:
        query_residue, reference_residue = alignment_pairs.get(
            position, ("", "-")
        )
        identity_count += int(
            reference_residue != "-" and query_residue == reference_residue
        )

    query_count = end - start
    mapped_count = len(mapped_positions)
    return IntervalMapping(
        reference_start=min(mapped_positions) if mapped_positions else None,
        reference_end=max(mapped_positions) if mapped_positions else None,
        mapped_residue_count=mapped_count,
        query_residue_count=query_count,
        mapped_fraction=mapped_count / query_count,
        identity=identity_count / mapped_count if mapped_count else 0.0,
    )


def project_reference_interval_to_query(
    alignment: SequenceAlignment,
    start: int,
    end: int,
) -> QueryIntervalProjection:
    if start < 1 or end < start:
        raise ValueError("Expected a non-empty one-based inclusive interval")
    query_positions: list[int] = []
    query_position = 0
    reference_position = 0
    for query_residue, reference_residue in zip(
        alignment.aligned_query,
        alignment.aligned_reference,
        strict=True,
    ):
        if query_residue != "-":
            query_position += 1
        if reference_residue != "-":
            reference_position += 1
        if (
            reference_residue != "-"
            and start <= reference_position <= end
            and query_residue != "-"
        ):
            query_positions.append(query_position)
    reference_count = end - start + 1
    return QueryIntervalProjection(
        query_start=min(query_positions) - 1 if query_positions else None,
        query_end=max(query_positions) if query_positions else None,
        mapped_residue_count=len(query_positions),
        reference_residue_count=reference_count,
        mapped_fraction=(
            len(query_positions) / reference_count if reference_count else 0.0
        ),
    )


def region_overlap(
    motif_start: int | None,
    motif_end: int | None,
    region_start: int,
    region_end: int,
) -> RegionOverlap:
    if motif_start is None or motif_end is None:
        return RegionOverlap(0, 0.0, 0.0)
    overlap_start = max(motif_start, region_start)
    overlap_end = min(motif_end, region_end)
    overlap_count = max(0, overlap_end - overlap_start + 1)
    motif_length = motif_end - motif_start + 1
    region_length = region_end - region_start + 1
    return RegionOverlap(
        overlap_residue_count=overlap_count,
        motif_overlap_fraction=overlap_count / motif_length,
        region_overlap_fraction=overlap_count / region_length,
    )


def reference_region_exact_match_fraction(
    alignment: SequenceAlignment,
    region_start: int,
    region_end: int,
) -> float:
    reference_position = 0
    compared = 0
    matches = 0
    for query_residue, reference_residue in zip(
        alignment.aligned_query,
        alignment.aligned_reference,
        strict=True,
    ):
        if reference_residue == "-":
            continue
        reference_position += 1
        if not region_start <= reference_position <= region_end:
            continue
        compared += 1
        matches += int(
            query_residue != "-" and query_residue == reference_residue
        )
    return matches / compared if compared else 0.0


def select_best_reference(
    query: str,
    references: Mapping[str, str],
) -> ReferenceSelection:
    if not references:
        raise ValueError("At least one reference sequence is required")
    selections = [
        ReferenceSelection(
            reference_id=reference_id,
            alignment=align_query_to_reference(query, reference),
        )
        for reference_id, reference in references.items()
    ]
    return max(
        selections,
        key=lambda selection: (
            selection.alignment.identity,
            selection.alignment.query_coverage,
        ),
    )


def find_region_hits(
    family: str,
    query_sequence: str,
    motif_start: int,
    motif_end: int,
    reference_sequences: Mapping[str, str],
    evidence_regions: Iterable[Mapping[str, object]],
    minimum_global_identity: float = 0.25,
    minimum_mapped_fraction: float = 0.8,
) -> list[dict[str, object]]:
    alignments: dict[str, SequenceAlignment] = {}
    hits: list[dict[str, object]] = []
    for region in evidence_regions:
        if str(region.get("family", "")) != family:
            continue
        reference_id = str(region["reference_id"])
        reference_sequence = reference_sequences.get(reference_id)
        if not reference_sequence:
            continue
        if reference_id not in alignments:
            alignments[reference_id] = align_query_to_reference(
                query_sequence, reference_sequence
            )
        alignment = alignments[reference_id]
        if alignment.identity < minimum_global_identity:
            continue
        interval = map_query_interval(
            alignment,
            start=motif_start,
            end=motif_end,
        )
        if interval.mapped_fraction < minimum_mapped_fraction:
            continue
        overlap = region_overlap(
            motif_start=interval.reference_start,
            motif_end=interval.reference_end,
            region_start=int(region["region_start"]),
            region_end=int(region["region_end"]),
        )
        if overlap.overlap_residue_count == 0:
            continue
        exact_match_fraction = reference_region_exact_match_fraction(
            alignment,
            region_start=int(region["region_start"]),
            region_end=int(region["region_end"]),
        )
        if (
            bool(region.get("requires_exact_residue_match", False))
            and exact_match_fraction < 1.0
        ):
            continue
        hits.append(
            {
                **dict(region),
                "reference_id": reference_id,
                "global_alignment_identity": alignment.identity,
                "global_query_coverage": alignment.query_coverage,
                "mapped_reference_start": interval.reference_start,
                "mapped_reference_end": interval.reference_end,
                "motif_mapping_fraction": interval.mapped_fraction,
                "motif_alignment_identity": interval.identity,
                "overlap_residue_count": overlap.overlap_residue_count,
                "motif_overlap_fraction": overlap.motif_overlap_fraction,
                "region_overlap_fraction": overlap.region_overlap_fraction,
                "reference_region_exact_match_fraction": exact_match_fraction,
            }
        )
    return sorted(
        hits,
        key=lambda hit: (
            -float(hit["motif_overlap_fraction"]),
            -float(hit["motif_alignment_identity"]),
            str(hit["evidence_id"]),
        ),
    )
