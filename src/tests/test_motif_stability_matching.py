import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motif_stability.matching import hungarian_match


def test_hungarian_matching_is_one_to_one_when_clusters_compete():
    reference = {0: {"a", "b"}, 1: {"c", "d"}}
    sample = {10: {"a", "b", "c", "d"}}

    result = hungarian_match(reference, sample, min_match_jaccard=0.3)

    assert len(result.matches) == 1
    assert len(result.unmatched_reference) == 1
    assert result.matched_sample_labels == {10}


def test_split_cluster_is_reported():
    reference = {0: {"a", "b", "c", "d"}}
    sample = {10: {"a", "b"}, 11: {"c", "d"}}

    result = hungarian_match(
        reference,
        sample,
        min_match_jaccard=0.3,
        split_merge_jaccard=0.2,
    )

    assert result.split_reference_labels == {0}


def test_merge_cluster_is_reported():
    reference = {0: {"a", "b"}, 1: {"c", "d"}}
    sample = {10: {"a", "b", "c", "d"}}

    result = hungarian_match(
        reference,
        sample,
        min_match_jaccard=0.3,
        split_merge_jaccard=0.2,
    )

    assert result.merge_sample_labels == {10}


def test_below_threshold_assignment_remains_unmatched():
    reference = {0: {"a", "b", "c"}}
    sample = {10: {"c", "d", "e"}}

    result = hungarian_match(reference, sample, min_match_jaccard=0.5)

    assert result.matches == ()
    assert result.unmatched_reference == {0}
    assert result.unmatched_sample == {10}
