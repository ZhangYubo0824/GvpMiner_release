import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motif_stability.occurrences import (
    collapse_cluster_occurrences,
    occurrence_jaccard,
    protein_jaccard,
    window_jaccard,
)


def window(window_id, parent_id, start, end, cluster=0):
    return {
        "window_id": window_id,
        "parent_id": parent_id,
        "family": "GvpA",
        "start": start,
        "end": end,
        "cluster_label": cluster,
    }


def test_overlapping_windows_collapse_but_disjoint_regions_remain_separate():
    windows = [
        window("w1", "p1", 0, 10),
        window("w2", "p1", 5, 15),
        window("w3", "p1", 30, 40),
    ]

    occurrences = collapse_cluster_occurrences(windows, min_overlap_fraction=0.5)

    assert [(row["start"], row["end"]) for row in occurrences] == [(0, 15), (30, 40)]
    assert occurrences[0]["window_ids"] == ("w1", "w2")


def test_chain_overlap_collapses_and_flags_diffuse_region():
    windows = [
        window("w1", "p1", 0, 10),
        window("w2", "p1", 5, 15),
        window("w3", "p1", 10, 20),
    ]

    occurrences = collapse_cluster_occurrences(
        windows,
        min_overlap_fraction=0.5,
        diffuse_length=15,
    )

    assert len(occurrences) == 1
    assert occurrences[0]["start"] == 0
    assert occurrences[0]["end"] == 20
    assert occurrences[0]["is_diffuse"] is True


def test_window_jaccard_has_known_value():
    assert window_jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)
    assert window_jaccard(set(), set()) == 1.0


def test_protein_jaccard_excludes_proteins_omitted_by_subsample():
    reference = [
        window("a", "p1", 0, 10),
        window("b", "p2", 0, 10),
        window("c", "p3", 0, 10),
    ]
    sample = [window("d", "p1", 0, 10), window("e", "p2", 0, 10)]

    score = protein_jaccard(reference, sample, retained_proteins={"p1", "p2"})

    assert score == 1.0


def test_occurrence_jaccard_matches_regions_only_within_same_protein():
    reference = [
        {"parent_id": "p1", "start": 0, "end": 20},
        {"parent_id": "p2", "start": 50, "end": 70},
    ]
    sample = [
        {"parent_id": "p1", "start": 5, "end": 20},
        {"parent_id": "p3", "start": 50, "end": 70},
    ]

    score = occurrence_jaccard(
        reference,
        sample,
        retained_proteins={"p1", "p2", "p3"},
        min_region_iou=0.5,
    )

    assert score == pytest.approx(1 / 3)


def test_occurrence_jaccard_excludes_omitted_reference_proteins():
    reference = [
        {"parent_id": "p1", "start": 0, "end": 20},
        {"parent_id": "p2", "start": 50, "end": 70},
    ]
    sample = [{"parent_id": "p1", "start": 2, "end": 20}]

    score = occurrence_jaccard(
        reference,
        sample,
        retained_proteins={"p1"},
        min_region_iou=0.5,
    )

    assert score == 1.0
