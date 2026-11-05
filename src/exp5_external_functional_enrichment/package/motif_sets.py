from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .prepare import SequenceRecord


SUPPORTED_MOTIF_METHODS = (
    "M2",
    "M3",
    "MEME",
    "GLAM2",
    "STREME",
    "MotifAE",
    "InterPLM",
)


@dataclass(frozen=True)
class MotifSet:
    motif_cluster: str
    family: str
    method: str
    foreground_ids: tuple[str, ...]
    background_ids: tuple[str, ...]
    universe_ids: tuple[str, ...]


def motif_cluster_id(row: Mapping[str, object]) -> str:
    motif_uid = str(row.get("motif_uid") or row.get("motif_cluster") or row.get("motif_id") or "").strip()
    if motif_uid:
        return motif_uid
    family = str(row.get("family") or "").strip()
    method = str(row.get("method") or "").strip()
    label = str(row.get("cluster_label") or "").strip()
    return "_".join(part for part in [family, method, label] if part)


def occurrence_protein_id(row: Mapping[str, object]) -> str:
    return str(row.get("parent_id") or row.get("sequence_id") or row.get("protein_id") or "").strip()


def accession_id(protein_id: str) -> str:
    token = protein_id.strip().split()[0] if protein_id.strip() else ""
    fields = token.split("|")
    if len(fields) >= 3 and fields[0] in {"sp", "tr"}:
        return fields[1]
    return fields[0]


def build_sequence_lookup(
    protein_rows: Sequence[SequenceRecord],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], str]]:
    proteins_by_family: dict[str, set[str]] = defaultdict(set)
    ids_by_accession: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in protein_rows:
        if record.protein_id and record.family:
            proteins_by_family[record.family].add(record.protein_id)
            ids_by_accession[(record.family, accession_id(record.protein_id))].add(record.protein_id)
    unique_accessions = {
        key: next(iter(ids)) for key, ids in ids_by_accession.items() if len(ids) == 1
    }
    return proteins_by_family, unique_accessions


def resolve_index_protein_id(
    family: str,
    protein_id: str,
    proteins_by_family: Mapping[str, set[str]],
    unique_accessions: Mapping[tuple[str, str], str],
) -> str:
    if protein_id in proteins_by_family.get(family, set()):
        return protein_id
    return unique_accessions.get((family, accession_id(protein_id)), "")


def read_occurrence_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _discover_family_occurrence_paths(family_dir: Path) -> list[Path]:
    return sorted(
        list(family_dir.glob("motif_occurrences_*.csv"))
        + list(family_dir.glob("*/motif_occurrences_*.csv"))
    )


def discover_occurrence_files(
    exp3_dir: Path,
    *,
    families: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> list[Path]:
    exp3_dir = Path(exp3_dir)
    family_filter = set(families or [])
    method_filter = set(methods or SUPPORTED_MOTIF_METHODS)
    family_dirs = [exp3_dir / family for family in families] if families else [
        path for path in exp3_dir.iterdir() if path.is_dir()
    ]
    files: list[Path] = []
    for family_dir in sorted(family_dirs, key=lambda p: p.name):
        if family_filter and family_dir.name not in family_filter:
            continue
        for path in _discover_family_occurrence_paths(family_dir):
            method = path.stem.replace("motif_occurrences_", "", 1)
            if method not in method_filter:
                continue
            files.append(path)
    return files


def build_motif_sets(
    occurrence_rows: Sequence[Mapping[str, object]],
    protein_rows: Sequence[SequenceRecord],
) -> list[MotifSet]:
    proteins_by_family, unique_accessions = build_sequence_lookup(protein_rows)

    foreground: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in occurrence_rows:
        family = str(row.get("family") or "").strip()
        method = str(row.get("method") or "").strip()
        cluster = motif_cluster_id(row)
        protein_id = resolve_index_protein_id(
            family, occurrence_protein_id(row), proteins_by_family, unique_accessions
        )
        if not family or not method or not cluster or not protein_id:
            continue
        foreground[(family, method, cluster)].add(protein_id)

    motif_sets: list[MotifSet] = []
    for (family, method, cluster), foreground_ids in sorted(foreground.items()):
        # Enrichment foregrounds must be subsets of the indexed family universe.
        universe = set(proteins_by_family.get(family, set()))
        if not universe:
            continue
        foreground_ids = foreground_ids & universe
        if not foreground_ids:
            continue
        background = universe - foreground_ids
        motif_sets.append(
            MotifSet(
                motif_cluster=cluster,
                family=family,
                method=method,
                foreground_ids=tuple(sorted(foreground_ids)),
                background_ids=tuple(sorted(background)),
                universe_ids=tuple(sorted(universe)),
            )
        )
    return motif_sets


def write_motif_sets_long(motif_set_rows: Sequence[MotifSet], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family", "method", "motif_cluster", "set_type", "protein_id"],
        )
        writer.writeheader()
        for motif_set in motif_set_rows:
            for set_type, protein_ids in [
                ("foreground", motif_set.foreground_ids),
                ("background", motif_set.background_ids),
                ("universe", motif_set.universe_ids),
            ]:
                for protein_id in protein_ids:
                    writer.writerow(
                        {
                            "family": motif_set.family,
                            "method": motif_set.method,
                            "motif_cluster": motif_set.motif_cluster,
                            "set_type": set_type,
                            "protein_id": protein_id,
                        }
                    )
