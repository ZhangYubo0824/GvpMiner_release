"""Per-motif control availability, overlap, and reuse diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence


CONTROL_TYPES = ("within_random", "position_matched", "composition_matched")


def _integer(row: dict[str, object], *keys: str) -> int:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    return 0


class ControlAudit:
    def __init__(self, motif_rows: Sequence[dict[str, object]]) -> None:
        self._intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for row in motif_rows:
            sequence_id = str(row.get("sequence_id") or row.get("parent_id") or "")
            start = _integer(row, "motif_start", "occ_start")
            end = _integer(row, "motif_end", "occ_end")
            if sequence_id and end > start:
                self._intervals[sequence_id].append(
                    (start, end, str(row.get("motif_cluster") or row.get("motif_uid") or row.get("motif_id") or ""))
                )
        self._counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self._instances: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._control_proteins: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    def add(self, control_row: dict[str, object]) -> None:
        cluster = str(control_row.get("motif_cluster", ""))
        control_type = str(control_row.get("control_type", ""))
        if not cluster or control_type not in CONTROL_TYPES:
            return
        key = cluster, control_type
        counts = self._counts[key]
        counts["n_controls"] += 1
        instance_id = str(control_row.get("motif_instance_id", ""))
        if instance_id:
            self._instances[key].add(instance_id)
        protein_id = str(control_row.get("control_sequence_id", ""))
        if protein_id:
            self._control_proteins[key][protein_id] += 1
        start = _integer(control_row, "control_start")
        end = _integer(control_row, "control_end")
        overlaps = [
            motif_cluster
            for motif_start, motif_end, motif_cluster in self._intervals.get(protein_id, [])
            if start < motif_end and motif_start < end
        ]
        if overlaps:
            counts["overlap_any_motif"] += 1
        if cluster in overlaps:
            counts["overlap_same_cluster"] += 1

    def summaries(self, feature_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        instances_by_cluster: dict[str, set[str]] = defaultdict(set)
        proteins_by_cluster: dict[str, set[str]] = defaultdict(set)
        for row in feature_rows:
            cluster = str(row.get("motif_cluster") or row.get("motif_id") or "")
            if not cluster:
                continue
            instances_by_cluster[cluster].add(str(row.get("motif_instance_id", "")))
            proteins_by_cluster[cluster].add(str(row.get("sequence_id", "")))
        rows = []
        for cluster in sorted(instances_by_cluster):
            for control_type in CONTROL_TYPES:
                key = cluster, control_type
                counts = self._counts[key]
                total = counts["n_controls"]
                proteins = self._control_proteins[key]
                requested = len(instances_by_cluster[cluster])
                rows.append(
                    {
                        "motif_cluster": cluster,
                        "control_type": control_type,
                        "n_motif_instances": requested,
                        "n_motif_proteins": len(proteins_by_cluster[cluster]),
                        "n_instances_with_control": len(self._instances[key]),
                        "instance_coverage_fraction": len(self._instances[key]) / requested if requested else 0.0,
                        "n_controls": total,
                        "n_unique_control_proteins": len(proteins),
                        "max_control_protein_reuse": max(proteins.values(), default=0),
                        "fraction_overlap_any_motif": counts["overlap_any_motif"] / total if total else 0.0,
                        "fraction_overlap_same_cluster": counts["overlap_same_cluster"] / total if total else 0.0,
                    }
                )
        return rows
