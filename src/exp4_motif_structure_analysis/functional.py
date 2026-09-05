from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from scipy.stats import fisher_exact

from .io import split_terms
from .stats import benjamini_hochberg


ANNOTATION_COLUMNS = {
    "GO": "GO_terms",
    "Pfam": "Pfam",
    "InterPro": "InterPro",
    "UniProt_keyword": "UniProt_keywords",
}


def _annotation_terms(row: dict[str, object]) -> dict[str, set[str]]:
    terms = {}
    for namespace, column in ANNOTATION_COLUMNS.items():
        terms[namespace] = set(split_terms(row.get(column, "")))
    return terms


def _row_sequence_id(row: dict[str, object]) -> str:
    return str(row.get("sequence_id") or row.get("protein_id") or row.get("seq_id") or "").strip()


def _rows_by_sequence_id(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        sequence_id = _row_sequence_id(row)
        if sequence_id:
            result[sequence_id] = row
    return result


def run_functional_enrichment(
    *,
    sequence_rows: Sequence[dict[str, object]],
    motif_rows: Sequence[dict[str, object]],
    annotation_rows: Sequence[dict[str, object]],
    min_term_count: int = 1,
) -> list[dict[str, object]]:
    sequences_by_id = _rows_by_sequence_id(sequence_rows)
    annotations_by_id = _rows_by_sequence_id(annotation_rows)
    motif_positive_by_cluster: dict[str, set[str]] = defaultdict(set)
    motif_family_by_cluster: dict[str, set[str]] = defaultdict(set)
    for row in motif_rows:
        cluster = str(row.get("motif_cluster", row.get("motif_id", "")))
        sequence_id = str(row.get("sequence_id", ""))
        motif_positive_by_cluster[cluster].add(sequence_id)
        family = str(row.get("family", sequences_by_id.get(sequence_id, {}).get("family", "")))
        if family:
            motif_family_by_cluster[cluster].add(family)

    results: list[dict[str, object]] = []
    for motif_cluster, positive_ids in sorted(motif_positive_by_cluster.items()):
        families = motif_family_by_cluster.get(motif_cluster) or {str(sequences_by_id.get(seq_id, {}).get("family", "")) for seq_id in positive_ids}
        background_ids = {
            sequence_id
            for sequence_id, row in sequences_by_id.items()
            if str(row.get("family", "")) in families and sequence_id not in positive_ids
        }
        if not background_ids:
            background_ids = set(sequences_by_id.keys()) - positive_ids
        foreground_total = len(positive_ids)
        background_total = len(background_ids)
        if foreground_total == 0 or background_total == 0:
            continue

        term_index: dict[tuple[str, str], set[str]] = defaultdict(set)
        annotated_ids = positive_ids | background_ids
        for sequence_id in annotated_ids:
            annotation = annotations_by_id.get(sequence_id, {})
            for namespace, terms in _annotation_terms(annotation).items():
                for term in terms:
                    term_index[(namespace, term)].add(sequence_id)

        cluster_rows: list[dict[str, object]] = []
        for (namespace, term), ids_with_term in sorted(term_index.items()):
            total_count = len(ids_with_term & annotated_ids)
            if total_count < min_term_count:
                continue
            fg_hits = len(ids_with_term & positive_ids)
            bg_hits = len(ids_with_term & background_ids)
            table = [
                [fg_hits, foreground_total - fg_hits],
                [bg_hits, background_total - bg_hits],
            ]
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
            coverage = len([seq_id for seq_id in annotated_ids if annotations_by_id.get(seq_id)]) / max(1, len(annotated_ids))
            cluster_rows.append(
                {
                    "motif_cluster": motif_cluster,
                    "namespace": namespace,
                    "function_term": term,
                    "n_with_motif": fg_hits,
                    "n_without_motif": bg_hits,
                    "foreground_total": foreground_total,
                    "background_total": background_total,
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_value),
                    "annotation_coverage": float(coverage),
                }
            )
        q_values = benjamini_hochberg([float(row["p_value"]) for row in cluster_rows])
        for row, q_value in zip(cluster_rows, q_values):
            row["q_value"] = q_value
        results.extend(cluster_rows)
    return results
