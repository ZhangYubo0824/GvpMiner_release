"""Pure profile calculations; no alignment fitting or candidate selection."""
from collections import Counter
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


def project(aligned, start, end, expected):
    """Project a zero-based half-open actual interval, preserving internal gaps."""
    positions = [i for i, a in enumerate(aligned) if a != "-"]
    if start < 0 or end <= start or end > len(positions):
        raise ValueError("Invalid actual occurrence coordinates")
    lo, hi = positions[start], positions[end - 1] + 1
    fragment = aligned[lo:hi].upper()
    if fragment.replace("-", "") != expected.upper():
        raise ValueError("Occurrence sequence differs from mapped alignment")
    return lo, hi, fragment


def profile(rows):
    """rows: (protein_id, first_alignment_column_0based, aligned_fragment).

    Each protein contributes one unit, split evenly over its actual occurrences.
    Probabilities are conditional on observed canonical residues at each column.
    """
    if not rows:
        raise ValueError("Empty candidate")
    per_protein = Counter(r[0] for r in rows)
    n = len(per_protein)
    lo = min(r[1] for r in rows)
    hi = max(r[1] + len(r[2]) for r in rows)
    width = hi - lo
    counts = np.zeros((width, 20))
    raw = np.zeros((width, 20), dtype=np.int64)
    gap, ambiguous, span = (np.zeros(width) for _ in range(3))
    support = np.zeros(width, dtype=int)
    protein_columns = {p: set() for p in per_protein}
    for protein, start, sequence in rows:
        weight = 1 / per_protein[protein]
        sl = slice(start-lo, start-lo+len(sequence))
        chars = np.frombuffer(sequence.encode("ascii"), dtype="S1")
        span[sl] += weight
        gaps = chars == b"-"
        gap[sl] += gaps * weight
        canonical = np.zeros(len(sequence), dtype=bool)
        for j, aa in enumerate(AA):
            mask = chars == aa.encode()
            counts[sl, j] += weight * mask
            raw[sl, j] += mask
            canonical |= mask
        ambiguous[sl] += (~canonical & ~gaps) * weight
        protein_columns[protein].update(np.flatnonzero(canonical) + start-lo)
    for columns in protein_columns.values():
        support[list(columns)] += 1
    total, raw_total = counts.sum(axis=1), raw.sum(axis=1)
    probabilities = np.divide(counts, total[:, None],
                              out=np.full_like(counts, np.nan), where=total[:, None] > 0)
    occurrence_probabilities = np.divide(raw, raw_total[:, None],
                                        out=np.full_like(counts, np.nan), where=raw_total[:, None] > 0)
    return dict(columns=np.arange(lo, hi), weighted_counts=counts, raw_counts=raw,
                probabilities=probabilities, occurrence_probabilities=occurrence_probabilities,
                coverage=total/n, gap=gap/n, ambiguous=ambiguous/n,
                outside=np.clip(1-span/n, 0, 1), n_proteins_observed=support,
                tv=np.abs(probabilities-occurrence_probabilities).sum(axis=1)/2,
                n_proteins=n, n_occurrences=len(rows))


def consensus(p, coverage_threshold=.5, majority_threshold=.5):
    result = []
    for cov, freq in zip(p["coverage"], p["probabilities"]):
        if cov + 1e-12 < coverage_threshold or not np.isfinite(freq).all():
            result.append(".")
        else:
            j = int(freq.argmax())
            result.append(AA[j] if freq[j] > majority_threshold + 1e-12 else "X")
    return "".join(result)
