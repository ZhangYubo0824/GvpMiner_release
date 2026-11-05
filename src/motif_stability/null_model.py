from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


def empirical_pvalue(observed: float, null_values: Iterable[float]) -> float:
    """One-sided empirical P value with the finite-sample plus-one correction."""
    values = np.asarray(tuple(float(value) for value in null_values), dtype=float)
    if not len(values):
        raise ValueError("null_values cannot be empty")
    return float((1 + np.count_nonzero(values >= float(observed))) / (len(values) + 1))


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted P values in original order."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pvalues must be finite and between 0 and 1")
    if not len(values):
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def conservation_score(sequences: Sequence[str]) -> float:
    """Mean per-position majority fraction for an aligned motif-sequence collection."""
    cleaned = [str(sequence).upper() for sequence in sequences if str(sequence)]
    if not cleaned:
        return 0.0
    width = min(len(sequence) for sequence in cleaned)
    position_scores = []
    for position in range(width):
        residues = [sequence[position] for sequence in cleaned]
        counts: dict[str, int] = defaultdict(int)
        for residue in residues:
            counts[residue] += 1
        position_scores.append(max(counts.values()) / len(residues))
    return float(np.mean(position_scores)) if position_scores else 0.0


class LengthMatchedSequenceSampler:
    """Reuse ordered length indexes while preserving direct seeded draws."""

    def __init__(
        self,
        target_sequences: Sequence[str],
        sequence_pool: Sequence[str],
    ) -> None:
        self._targets = tuple(str(sequence) for sequence in target_sequences)
        self._pool = tuple(str(sequence) for sequence in sequence_pool)
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, sequence in enumerate(self._pool):
            grouped[len(sequence)].append(index)
        self._indices_by_length = {
            length: np.asarray(indices, dtype=int)
            for length, indices in grouped.items()
        }

    def sample(self, *, seed: int) -> tuple[str, ...]:
        generator = np.random.default_rng(int(seed))
        available_by_length = {
            length: np.ones(len(indices), dtype=bool)
            for length, indices in self._indices_by_length.items()
        }
        selected: list[str] = []
        for target in self._targets:
            target_length = len(target)
            all_exact = self._indices_by_length.get(target_length)
            if all_exact is None:
                raise ValueError(
                    f"sequence_pool has no exact length-{target_length} match"
                )
            availability = available_by_length[target_length]
            available = all_exact[availability]
            candidates = available if len(available) else all_exact
            chosen = int(generator.choice(candidates))
            availability[int(np.searchsorted(all_exact, chosen))] = False
            selected.append(self._pool[chosen])
        return tuple(selected)


