"""Exact residue selections; nulls preserve missing alignment columns."""
from __future__ import annotations

import json
from collections.abc import Sequence

RESIDUE_SEMANTICS_VERSION = "exact-core-residues-v1"


def parse_residue_positions(value: object) -> tuple[int | None, ...] | None:
    if value is None or value == "":
        return None
    values = json.loads(value) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        raise ValueError("core residue positions must be a JSON array")
    result = []
    for item in values:
        if item is None:
            result.append(None)
        elif isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"Invalid core residue position: {item!r}")
        else:
            result.append(item)
    observed = [p for p in result if p is not None]
    if observed != sorted(set(observed)):
        raise ValueError("Core residue positions must be unique and increasing")
    return tuple(result)


def row_residue_positions(row: dict[str, object]) -> tuple[int | None, ...] | None:
    return parse_residue_positions(row.get("core_residue_positions_0based"))


def observed_positions(start: int, end: int, positions: Sequence[int | None] | None) -> tuple[int, ...]:
    result = tuple(range(start, end)) if positions is None else tuple(p for p in positions if p is not None)
    if not result or any(p < start or p >= end for p in result):
        raise ValueError(f"Core residue positions are empty or outside interval {start}-{end}")
    if result != tuple(sorted(set(result))):
        raise ValueError("Core residue positions must be unique and increasing")
    return result


def residue_offsets(start: int, end: int, positions: Sequence[int | None] | None) -> tuple[int | None, ...]:
    observed_positions(start, end, positions)
    return tuple(range(end - start)) if positions is None else tuple(None if p is None else p - start for p in positions)


def selection_sequence(sequence: str, start: int, end: int, positions: Sequence[int | None] | None) -> str:
    if start < 0 or end > len(sequence):
        raise ValueError(f"Core interval {start}-{end} is outside sequence")
    return "".join(sequence[p] for p in observed_positions(start, end, positions))


def positions_json(positions: Sequence[int | None]) -> str:
    return json.dumps(list(positions), separators=(",", ":"))
