import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motif_stability.null_model import (
    LengthMatchedSequenceSampler,
    benjamini_hochberg,
    conservation_score,
    empirical_pvalue,
)
from motif_stability.selection import count_seed_recoveries, select_parameter_setting


def _row(
    mcs,
    window,
    protein,
    occurrence,
    coverage,
    noise,
    stable=2,
    window_coverage=0.0,
):
    return {
        "min_cluster_size": mcs,
        "min_samples": 1,
        "median_window_jaccard": window,
        "median_protein_jaccard": protein,
        "median_occurrence_jaccard": occurrence,
        "stable_protein_coverage": coverage,
        "stable_window_coverage": window_coverage,
        "noise_ratio": noise,
        "stable_cluster_number": stable,
    }


def test_selector_prefers_best_threshold_qualified_setting():
    rows = [
        _row(20, 0.70, 0.82, 0.71, 0.50, 0.20),
        _row(30, 0.80, 0.85, 0.75, 0.65, 0.30),
        _row(40, 0.90, 0.90, 0.79, 0.80, 0.10, stable=0),
    ]

    selected = select_parameter_setting(rows)

    assert selected["min_cluster_size"] == 30
    assert selected["selection_mode"] == "qualified"


def test_selector_fallback_is_deterministic():
    rows = [
        _row(40, 0.49, 0.90, 0.90, 0.70, 0.10),
        _row(20, 0.49, 0.90, 0.90, 0.70, 0.10),
    ]

    left = select_parameter_setting(rows)
    right = select_parameter_setting(list(reversed(rows)))

    assert left == right
    assert left["min_cluster_size"] == 20
    assert left["selection_mode"] == "fallback"
    assert left["selection_strategy"] == "exploratory_fallback"


def test_selector_uses_coverage_within_stability_plateau():
    rows = [
        _row(70, 0.953, 0.97, 0.97, 0.736, 0.05, stable=21, window_coverage=0.440),
        _row(50, 0.941, 0.96, 0.96, 0.822, 0.07, stable=26, window_coverage=0.494),
        _row(80, 0.939, 0.96, 0.96, 0.831, 0.06, stable=17, window_coverage=0.416),
    ]

    selected = select_parameter_setting(rows)

    assert selected["min_cluster_size"] == 50
    assert selected["selection_strategy"] == "stability_plateau"


def test_selector_excludes_coverage_gain_outside_stability_tolerance():
    rows = [
        _row(60, 0.90, 0.95, 0.95, 0.75, 0.10, window_coverage=0.40),
        _row(40, 0.87, 0.95, 0.95, 0.99, 0.05, window_coverage=0.70),
    ]

    selected = select_parameter_setting(rows)

    assert selected["min_cluster_size"] == 60


def test_selector_avoids_large_fragmentation_for_small_protein_gain():
    rows = [
        _row(20, 0.855, 0.88, 0.88, 0.955, 0.10, stable=200, window_coverage=0.550),
        _row(40, 0.871, 0.91, 0.91, 0.943, 0.11, stable=116, window_coverage=0.459),
    ]

    selected = select_parameter_setting(rows)

    assert selected["min_cluster_size"] == 40
    assert selected["selection_strategy"] == "stability_plateau"


def test_selector_rescues_low_coverage_with_high_stability_alternative():
    rows = [
        _row(40, 0.971, 1.0, 1.0, 0.514, 0.01, stable=3, window_coverage=0.476),
        _row(20, 0.914, 0.95, 0.95, 1.0, 0.07, stable=10, window_coverage=0.692),
    ]

    selected = select_parameter_setting(rows)

    assert selected["min_cluster_size"] == 20
    assert selected["selection_strategy"] == "coverage_rescue"


def test_selector_fallback_normalizes_blank_and_nan_metrics_deterministically():
    rows = [
        _row(20, "", 0.90, 0.90, 0.70, float("nan")),
        _row(10, float("nan"), 0.90, 0.90, 0.70, 0.10),
    ]

    left = select_parameter_setting(rows)
    right = select_parameter_setting(list(reversed(rows)))

    assert left == right
    assert left["min_cluster_size"] == 10
    assert left["stability_floor"] == 0.0


def test_fragmentation_guard_keeps_window_noise_stability_order():
    rows = [
        _row(20, 0.86, 0.90, 0.90, 0.95, 0.08, stable=200, window_coverage=0.90),
        _row(40, 0.87, 0.90, 0.90, 0.94, 0.10, stable=100, window_coverage=0.50),
        _row(60, 0.85, 0.90, 0.90, 0.94, 0.05, stable=90, window_coverage=0.70),
    ]

    selected = select_parameter_setting(rows)

    assert selected["min_cluster_size"] == 60


def test_selector_includes_exact_quality_and_rescue_boundaries():
    qualified = _row(40, 0.50, 0.70, 0.60, 0.70, 0.20, window_coverage=0.20)
    assert select_parameter_setting([qualified])["selection_mode"] == "qualified"

    rescued = select_parameter_setting(
        [
            _row(40, 0.95, 0.95, 0.95, 0.69, 0.01, window_coverage=0.40),
            _row(20, 0.90, 0.95, 0.95, 0.89, 0.05, window_coverage=0.60),
        ]
    )
    assert rescued["min_cluster_size"] == 20
    assert rescued["selection_strategy"] == "coverage_rescue"


def test_seed_recovery_count_uses_reference_label_sets():
    reference = {0: {"a", "b"}, 1: {"c", "d"}}
    runs = [
        {10: {"a", "b"}, 11: {"c"}},
        {20: {"a"}, 21: {"c", "d"}},
        {30: {"x"}},
    ]

    counts = count_seed_recoveries(reference, runs, min_jaccard=0.5)

    assert counts == {0: 2, 1: 2}


def test_empirical_pvalue_uses_plus_one_correction():
    assert empirical_pvalue(10.0, [1.0, 2.0, 3.0]) == 0.25
    assert empirical_pvalue(2.0, [1.0, 2.0, 3.0]) == 0.75


def test_benjamini_hochberg_is_monotone_in_rank():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(adjusted, [0.04, 0.05333333333333334, 0.05333333333333334, 0.20])


def _legacy_length_matched_sample(target_sequences, sequence_pool, *, seed):
    pool = [str(sequence) for sequence in sequence_pool]
    generator = np.random.default_rng(int(seed))
    remaining = set(range(len(pool)))
    selected = []
    for target in target_sequences:
        target_length = len(str(target))
        all_exact = [
            index
            for index, value in enumerate(pool)
            if len(value) == target_length
        ]
        if not all_exact:
            raise ValueError(
                f"sequence_pool has no exact length-{target_length} match"
            )
        available = [index for index in all_exact if index in remaining]
        candidates = available or all_exact
        chosen = int(generator.choice(np.asarray(candidates, dtype=int)))
        remaining.discard(chosen)
        selected.append(pool[chosen])
    return tuple(selected)


@pytest.mark.parametrize("seed", [0, 1, 9, 37, 999])
def test_preindexed_length_sampler_matches_legacy_draws_exactly(seed):
    targets = ["M" * 4, "K" * 2, "L" * 4, "Q" * 4, "R" * 4]
    pool = ["AAAA", "CC", "GGGG", "TT", "AAAA", "VVV"]
    expected = _legacy_length_matched_sample(targets, pool, seed=seed)

    sampler = LengthMatchedSequenceSampler(targets, pool)

    assert sampler.sample(seed=seed) == expected


def test_preindexed_length_sampler_preserves_missing_length_error():
    sampler = LengthMatchedSequenceSampler(["MMMM"], ["AAA"])

    with pytest.raises(
        ValueError,
        match=r"sequence_pool has no exact length-4 match",
    ):
        sampler.sample(seed=1)


def test_conservation_and_length_matched_sampling_are_seeded():
    assert conservation_score(["AAAA", "AAAT"]) == 0.875
    pool = ["AAAA", "CC", "GGGG", "TTT", "ACAC"]
    left = LengthMatchedSequenceSampler(["MMMM", "KK"], pool).sample(seed=9)
    right = LengthMatchedSequenceSampler(["MMMM", "KK"], pool).sample(seed=9)
    assert left == right
    assert [len(value) for value in left] == [4, 2]


def test_length_matched_sampling_rejects_missing_lengths_and_can_replace():
    with np.testing.assert_raises(ValueError):
        LengthMatchedSequenceSampler(["MMMM"], ["AAA"]).sample(seed=1)

    sampled = LengthMatchedSequenceSampler(["MMMM", "KKKK"], ["AAAA"]).sample(seed=1)
    assert sampled == ("AAAA", "AAAA")
