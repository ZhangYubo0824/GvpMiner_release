from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from .contracts import (
    ALL_FAMILIES,
    REPRESENTATIVE_FAMILIES,
    ParameterPair,
    StagePlan,
    WorkUnit,
)


BASELINE_PAIR = ParameterPair(30, 5)


def stage_a_pairs() -> tuple[ParameterPair, ...]:
    return tuple(
        ParameterPair(window, step)
        for window in (20, 25, 30, 35, 40)
        for step in (3, 5, 10, 15)
    )


def _units(
    stage: str,
    pairs: Iterable[ParameterPair],
    families: Iterable[str],
    methods: Iterable[str],
    profile: str,
    output_root: Path,
) -> tuple[WorkUnit, ...]:
    return tuple(
        WorkUnit(stage, pair, family, method, profile, Path(output_root))
        for pair in pairs
        for family in families
        for method in methods
    )


def build_stage_a_plan(output_root: Path) -> StagePlan:
    pairs = stage_a_pairs()
    root = Path(output_root) / "stage_a_main_effects"
    return StagePlan("A", pairs, _units("A", pairs, REPRESENTATIVE_FAMILIES,
                                        ("M2",), "pilot", root), root)


def build_stage_b_plan(output_root: Path, pairs: Sequence[ParameterPair]) -> StagePlan:
    pairs = tuple(dict.fromkeys(pairs))
    if BASELINE_PAIR not in pairs or len(pairs) not in (2, 3):
        raise ValueError("stage B requires baseline 30/5 and one or two challengers")
    root = Path(output_root) / "stage_b_m2_all_families"
    return StagePlan("B", pairs, _units("B", pairs, ALL_FAMILIES,
                                        ("M2",), "pilot", root), root)


def build_stage_c_plan(output_root: Path, pairs: Sequence[ParameterPair]) -> StagePlan:
    pairs = tuple(dict.fromkeys(pairs))
    if len(pairs) != 2:
        raise ValueError("stage C requires exactly two final candidates")
    root = Path(output_root) / "stage_c_confirmatory"
    return StagePlan("C", pairs, _units("C", pairs, ALL_FAMILIES,
                                        ("M2", "M3"), "confirmatory", root), root)
