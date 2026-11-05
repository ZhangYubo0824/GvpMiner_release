"""Preprint-v1 peptide segmentation used by current official inference."""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable


def _paper_peptide_points(
    sparse: Iterable[tuple[int, int, float]],
) -> list[tuple[int, int, int, float]]:
    """Implement the explicitly described 2025 preprint-v1 peptide protocol.

    A gap of up to two inactive residues is bridged. The peptide mean includes
    those inactive residues. The paper does not publish extraction source code,
    so this is an explicit operational interpretation of its prose method.
    The 2026 published Methods do not restate these numerical activation/gap
    thresholds. Do not describe this helper as verified published-version code.
    """
    by_feature: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for position, feature_id, value in sparse:
        if value > 0:
            by_feature[int(feature_id)].append((int(position), float(value)))

    peptides: list[tuple[int, int, int, float]] = []
    for feature_id, activations in sorted(by_feature.items()):
        if sum(value for _, value in activations) <= 1.0:
            continue
        activations.sort()
        start = last = activations[0][0]
        value_sum = activations[0][1]

        def retain(end_position: int, total: float, begin: int) -> None:
            length = end_position - begin + 1
            mean = total / length
            if 5 <= length <= 30 and mean > 0.2:
                peptides.append((feature_id, begin, end_position + 1, mean))

        for position, value in activations[1:]:
            if position - last <= 3:
                last = position
                value_sum += value
            else:
                retain(last, value_sum, start)
                start = last = position
                value_sum = value
        retain(last, value_sum, start)
    return peptides
