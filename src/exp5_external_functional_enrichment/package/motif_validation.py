from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .motif_sets import motif_cluster_id, occurrence_protein_id
from .prepare import SequenceRecord


@dataclass(frozen=True)
class MotifWindow:
    motif_cluster: str
    family: str
    method: str
    protein_id: str
    start: int
    end: int
    sequence: str


@dataclass(frozen=True)
class HmmerCommandPlan:
    hmmscan: str
    mafft: str
    hmmbuild: str
    hmmsearch: str


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def extract_motif_windows(
    occurrence_rows: Sequence[Mapping[str, object]],
    sequences_by_id: Mapping[str, SequenceRecord],
) -> dict[str, list[MotifWindow]]:
    windows: dict[str, list[MotifWindow]] = defaultdict(list)
    for row in occurrence_rows:
        protein_id = occurrence_protein_id(row)
        record = sequences_by_id.get(protein_id)
        if record is None:
            continue
        start = _to_int(row.get("occ_start"))
        end = _to_int(row.get("occ_end"))
        if start < 0 or end <= start or end > len(record.sequence):
            continue
        cluster = motif_cluster_id(row)
        window = MotifWindow(
            motif_cluster=cluster,
            family=str(row.get("family") or record.family),
            method=str(row.get("method") or ""),
            protein_id=protein_id,
            start=start,
            end=end,
            sequence=record.sequence[start:end],
        )
        windows[cluster].append(window)
    return dict(sorted(windows.items()))


def write_motif_window_fastas(
    windows_by_cluster: Mapping[str, Sequence[MotifWindow]],
    output_dir: Path,
    *,
    min_windows: int = 1,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for cluster, windows in sorted(windows_by_cluster.items()):
        if len(windows) < min_windows:
            continue
        path = output_dir / f"{cluster}.fasta"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for idx, window in enumerate(windows, start=1):
                handle.write(
                    f">{window.motif_cluster}|{idx}|{window.protein_id}:{window.start}-{window.end}\n"
                )
                handle.write(f"{window.sequence}\n")
        paths[cluster] = path
    return paths


def build_hmmer_command_plan(
    *,
    motif_fasta: Path,
    all_protein_fasta: Path,
    pfam_hmm: Path,
    output_dir: Path,
) -> HmmerCommandPlan:
    output_dir = Path(output_dir)
    stem = Path(motif_fasta).stem
    alignment = output_dir / f"{stem}.mafft.fasta"
    motif_hmm = output_dir / f"{stem}.hmm"
    return HmmerCommandPlan(
        hmmscan=f'hmmscan --tblout "{output_dir / (stem + ".pfam.tbl")}" "{pfam_hmm}" "{motif_fasta}"',
        mafft=f'mafft --auto "{motif_fasta}" > "{alignment}"',
        hmmbuild=f'hmmbuild "{motif_hmm}" "{alignment}"',
        hmmsearch=f'hmmsearch --tblout "{output_dir / (stem + ".all_gvp.tbl")}" "{motif_hmm}" "{all_protein_fasta}"',
    )
