from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

from .io import read_csv_rows


def _sequence_map(sequence_rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    result = {}
    for row in sequence_rows:
        sequence_id = str(row.get("sequence_id") or row.get("protein_id") or row.get("seq_id") or "")
        if not sequence_id:
            continue
        normalized = dict(row)
        normalized["sequence_id"] = sequence_id
        result[sequence_id] = normalized
    return result


def _load_catalog_by_uid(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {str(row.get("motif_uid", "")): row for row in read_csv_rows(path)}


def _resolve_motif_paths(family_dir: Path, method: str) -> tuple[Path, Path]:
    flat_occurrence = family_dir / f"motif_occurrences_{method}.csv"
    nested_occurrence = family_dir / method / f"motif_occurrences_{method}.csv"
    for occurrence_path in [flat_occurrence, nested_occurrence]:
        if occurrence_path.exists():
            return occurrence_path, occurrence_path.parent / f"motif_catalog_{method}.csv"
    return flat_occurrence, flat_occurrence.parent / f"motif_catalog_{method}.csv"


def _int_value(row: dict[str, object], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return int(float(value))


def _motif_sequence(sequence: str, start: int, end: int) -> str:
    if sequence and 0 <= start < end <= len(sequence):
        return sequence[start:end]
    return ""


def build_exp3_motif_rows(
    *,
    exp3_dir: Path,
    sequence_rows: Sequence[dict[str, object]],
    families: Sequence[str],
    methods: Sequence[str],
) -> list[dict[str, object]]:
    exp3_dir = Path(exp3_dir)
    sequences = _sequence_map(sequence_rows)
    motif_rows: list[dict[str, object]] = []
    missing_occurrence_files = []

    for family in families:
        family_dir = exp3_dir / family
        for method in methods:
            occurrence_path, catalog_path = _resolve_motif_paths(family_dir, method)
            if not occurrence_path.exists():
                missing_occurrence_files.append(str(occurrence_path))
                continue
            catalog_by_uid = _load_catalog_by_uid(catalog_path)
            for occurrence in read_csv_rows(occurrence_path):
                sequence_id = str(occurrence.get("parent_id") or occurrence.get("sequence_id") or "")
                if not sequence_id:
                    continue
                sequence_row = sequences.get(sequence_id, {})
                sequence = str(sequence_row.get("sequence", ""))
                motif_uid = str(occurrence.get("motif_uid") or occurrence.get("motif_id") or occurrence.get("cluster_label") or "")
                catalog = catalog_by_uid.get(motif_uid, {})
                start = _int_value(occurrence, "occ_start")
                end = _int_value(occurrence, "occ_end")
                motif_sequence = _motif_sequence(sequence, start, end)
                if not motif_sequence:
                    continue
                motif_rows.append(
                    {
                        "motif_instance_id": f"{motif_uid}:{sequence_id}:{start}-{end}",
                        "motif_uid": motif_uid,
                        "motif_id": motif_uid,
                        "motif_cluster": motif_uid,
                        "sequence_id": sequence_id,
                        "family": str(occurrence.get("family") or family),
                        "method": str(occurrence.get("method") or method),
                        "cluster_label": str(occurrence.get("cluster_label") or catalog.get("cluster_label", "")),
                        "motif_start": start,
                        "motif_end": end,
                        "motif_sequence": motif_sequence,
                        "window_length": str(occurrence.get("occ_length") or max(0, end - start)),
                        "cluster_score": str(
                            catalog.get("stable_repeat_fraction")
                            or catalog.get("median_protein_jaccard")
                            or ""
                        ),
                        "cluster_frequency": str(catalog.get("n_windows", occurrence.get("n_support_windows", ""))),
                    }
                )

    instance_ids = [str(row["motif_instance_id"]) for row in motif_rows]
    duplicate_ids = sorted(
        instance_id for instance_id, count in Counter(instance_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "Duplicate composite motif occurrence IDs detected: "
            + ", ".join(duplicate_ids[:10])
        )

    if not motif_rows and missing_occurrence_files:
        raise FileNotFoundError(f"No Exp3 motif occurrence files found. Checked: {missing_occurrence_files[:10]}")
    return motif_rows
