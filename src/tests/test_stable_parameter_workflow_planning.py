import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_parameter_workflow.contracts import (
    ALL_FAMILIES,
    DEFAULT_MCS,
    DEFAULT_MIN_SAMPLES,
    REPRESENTATIVE_FAMILIES,
    ParameterPair,
)
from stable_parameter_workflow.planning import (
    build_stage_a_plan,
    build_stage_b_plan,
    build_stage_c_plan,
)


def test_approved_design_constants_are_frozen():
    assert DEFAULT_MCS == (10, 20, 30, 40, 60, 80)
    assert DEFAULT_MIN_SAMPLES == (10,)
    assert REPRESENTATIVE_FAMILIES == (
        "GvpK", "GvpN", "GvpF", "GvpM", "GvpL", "GvpH",
    )
    assert len(ALL_FAMILIES) == 16


def test_parameter_pair_has_stable_slug_and_validates_values():
    assert ParameterPair(30, 5).slug == "w30_s5"
    try:
        ParameterPair(0, 5)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero window size must be rejected")


def test_stage_a_has_complete_twenty_pair_grid_and_120_units(tmp_path):
    plan = build_stage_a_plan(tmp_path)
    assert set(plan.pairs) == {
        ParameterPair(window, step)
        for window in (20, 25, 30, 35, 40)
        for step in (3, 5, 10, 15)
    }
    assert len(plan.pairs) == 20
    assert ParameterPair(30, 5) in plan.pairs
    assert len(plan.units) == 120
    assert {unit.method for unit in plan.units} == {"M2"}
    assert {unit.family for unit in plan.units} == set(REPRESENTATIVE_FAMILIES)


def test_stage_b_and_c_unit_counts_follow_adaptive_design(tmp_path):
    pairs = (ParameterPair(30, 5), ParameterPair(30, 3))
    stage_b = build_stage_b_plan(tmp_path, pairs)
    stage_c = build_stage_c_plan(tmp_path, pairs)
    assert stage_b.stage == "B"
    assert stage_b.output_root.name == "stage_b_m2_all_families"
    assert stage_c.stage == "C"
    assert stage_c.output_root.name == "stage_c_confirmatory"
    assert len(stage_b.units) == 32
    assert {unit.method for unit in stage_b.units} == {"M2"}
    assert len(stage_c.units) == 64
    assert {unit.method for unit in stage_c.units} == {"M2", "M3"}
    assert {unit.profile for unit in stage_c.units} == {"confirmatory"}
