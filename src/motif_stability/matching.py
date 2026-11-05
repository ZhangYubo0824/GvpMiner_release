from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from .occurrences import window_jaccard


@dataclass(frozen=True)
class ClusterMatch:
    reference_label: int
    sample_label: int
    window_jaccard: float


@dataclass(frozen=True)
class MatchingResult:
    matches: tuple[ClusterMatch, ...]
    unmatched_reference: Set[int]
    unmatched_sample: Set[int]
    split_reference_labels: Set[int]
    merge_sample_labels: Set[int]
    reference_labels: tuple[int, ...]
    sample_labels: tuple[int, ...]
    score_matrix: np.ndarray

    @property
    def matched_sample_labels(self) -> set[int]:
        return {match.sample_label for match in self.matches}


def hungarian_match(
    reference_clusters: Mapping[int, set[str]],
    sample_clusters: Mapping[int, set[str]],
    *,
    min_match_jaccard: float = 0.3,
    split_merge_jaccard: float = 0.2,
) -> MatchingResult:
    reference_labels = tuple(sorted(int(label) for label in reference_clusters))
    sample_labels = tuple(sorted(int(label) for label in sample_clusters))
    matrix = np.zeros((len(reference_labels), len(sample_labels)), dtype=float)
    for row_index, reference_label in enumerate(reference_labels):
        for column_index, sample_label in enumerate(sample_labels):
            matrix[row_index, column_index] = window_jaccard(
                reference_clusters[reference_label],
                sample_clusters[sample_label],
            )

    matches = []
    if matrix.size:
        row_indices, column_indices = linear_sum_assignment(matrix, maximize=True)
        for row_index, column_index in zip(row_indices, column_indices):
            score = float(matrix[row_index, column_index])
            if score < min_match_jaccard:
                continue
            matches.append(
                ClusterMatch(
                    reference_label=reference_labels[row_index],
                    sample_label=sample_labels[column_index],
                    window_jaccard=score,
                )
            )

    matched_reference = {match.reference_label for match in matches}
    matched_sample = {match.sample_label for match in matches}
    split_reference = {
        reference_labels[row_index]
        for row_index in range(len(reference_labels))
        if int(np.count_nonzero(matrix[row_index, :] >= split_merge_jaccard)) >= 2
    }
    merge_sample = {
        sample_labels[column_index]
        for column_index in range(len(sample_labels))
        if int(np.count_nonzero(matrix[:, column_index] >= split_merge_jaccard)) >= 2
    }

    return MatchingResult(
        matches=tuple(sorted(matches, key=lambda match: match.reference_label)),
        unmatched_reference=set(reference_labels) - matched_reference,
        unmatched_sample=set(sample_labels) - matched_sample,
        split_reference_labels=split_reference,
        merge_sample_labels=merge_sample,
        reference_labels=reference_labels,
        sample_labels=sample_labels,
        score_matrix=matrix,
    )
