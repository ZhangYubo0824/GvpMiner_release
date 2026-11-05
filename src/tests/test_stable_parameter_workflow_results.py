import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_parameter_workflow.contracts import ParameterPair
from stable_parameter_workflow.results import (
    annotate_candidate_uncertainty,
    summarize_candidate,
)


def _row(family, *, qualified=True, floor=.7, coverage=.6, noise=.2, mcs=20):
    return {
        "family": family,
        "method": "M2",
        "selection_mode": "qualified" if qualified else "fallback",
        "median_window_jaccard": floor,
        "median_protein_jaccard": floor + .1,
        "median_occurrence_jaccard": floor + .05,
        "stable_protein_coverage": coverage,
        "stable_window_coverage": coverage - .1,
        "noise_ratio": noise,
        "min_cluster_size": mcs,
        "stable_cluster_number": 3,
    }


def test_candidate_summary_equal_weights_families_and_flags_mcs_boundaries():
    rows = [_row("GvpA", floor=.8, coverage=.8, mcs=10)]
    rows += [_row("GvpA", floor=.2, coverage=.2, mcs=20)] * 9
    rows += [_row("GvpC", floor=.6, coverage=.6, mcs=80)]
    summary = summarize_candidate(ParameterPair(30, 5), rows)
    # GvpA is reduced first, so its ten rows cannot outweigh GvpC.
    assert summary["family_count"] == 2
    assert summary["qualified_family_count"] == 2
    assert summary["mcs_lower_boundary"] is True
    assert summary["mcs_upper_boundary"] is True
    assert 0.39 < summary["stability_floor"] < 0.41


def test_uncertainty_marks_material_family_direction_disagreement():
    baseline = ParameterPair(30, 5)
    challenger = ParameterPair(25, 5)
    rows_by_pair = {
        baseline: [_row("GvpK", floor=.70), _row("GvpN", floor=.70)],
        challenger: [_row("GvpK", floor=.73), _row("GvpN", floor=.67)],
    }
    summaries = [
        summarize_candidate(pair, rows) for pair, rows in rows_by_pair.items()
    ]

    annotated = annotate_candidate_uncertainty(summaries, rows_by_pair)
    by_pair = {row["pair"]: row for row in annotated}

    assert by_pair[baseline]["family_direction_disagreement"] is False
    assert by_pair[challenger]["family_direction_disagreement"] is True
    assert by_pair[challenger]["family_improved_count"] == 1
    assert by_pair[challenger]["family_worsened_count"] == 1


def test_uncertainty_marks_all_pairs_in_material_interaction_reversal():
    pairs = {
        ParameterPair(30, 5): .70,
        ParameterPair(20, 5): .74,
        ParameterPair(30, 3): .75,
        ParameterPair(20, 3): .70,
    }
    rows_by_pair = {
        pair: [_row("GvpK", floor=floor)] for pair, floor in pairs.items()
    }
    summaries = [
        summarize_candidate(pair, rows) for pair, rows in rows_by_pair.items()
    ]

    annotated = annotate_candidate_uncertainty(summaries, rows_by_pair)

    assert all(row["important_interaction"] for row in annotated)
    assert all(row["interaction_reversal_count"] >= 1 for row in annotated)
