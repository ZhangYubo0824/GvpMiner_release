from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SequenceRecord:
    protein_id: str
    family: str
    sequence: str
    fasta_path: str
    description: str = ""


def _wrap_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i:i + width] for i in range(0, len(sequence), width))


def parse_fasta(path: Path, *, family: str) -> list[SequenceRecord]:
    path = Path(path)
    records: list[SequenceRecord] = []
    current_id = ""
    current_desc = ""
    chunks: list[str] = []

    def flush() -> None:
        nonlocal chunks
        if not current_id:
            return
        records.append(
            SequenceRecord(
                protein_id=current_id,
                family=family,
                sequence="".join(chunks).replace(" ", "").replace("\t", "").upper(),
                fasta_path=str(path),
                description=current_desc,
            )
        )
        chunks = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                current_id = header.split()[0]
                current_desc = header
            else:
                chunks.append(line)
    flush()
    return records


def discover_family_fastas(
    ncbi_root: Path,
    *,
    families: Sequence[str] | None = None,
) -> list[tuple[str, Path]]:
    ncbi_root = Path(ncbi_root)
    family_filter = set(families or [])
    family_dirs = [ncbi_root / family for family in families] if families else [
        path for path in ncbi_root.iterdir() if path.is_dir()
    ]
    discovered: list[tuple[str, Path]] = []
    for family_dir in sorted(family_dirs, key=lambda p: p.name):
        if family_filter and family_dir.name not in family_filter:
            continue
        fasta = family_dir / f"{family_dir.name}_proteins.fasta"
        if fasta.exists():
            discovered.append((family_dir.name, fasta))
    return discovered


def write_sequence_index(records: Sequence[SequenceRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_id", "family", "fasta_path", "description", "sequence"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "protein_id": record.protein_id,
                    "family": record.family,
                    "fasta_path": record.fasta_path,
                    "description": record.description,
                    "sequence": record.sequence,
                }
            )


def read_sequence_index(path: Path) -> list[SequenceRecord]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            SequenceRecord(
                protein_id=str(row.get("protein_id", "")).strip(),
                family=str(row.get("family", "")).strip(),
                sequence=str(row.get("sequence", "")).strip().upper(),
                fasta_path=str(row.get("fasta_path", "")).strip(),
                description=str(row.get("description", "")).strip(),
            )
            for row in reader
            if str(row.get("protein_id", "")).strip()
        ]


def write_merged_fasta(records: Iterable[SequenceRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.protein_id} family={record.family} {record.description}\n")
            handle.write(_wrap_sequence(record.sequence))
            handle.write("\n")


def build_sequence_index(
    *,
    ncbi_root: Path,
    output_dir: Path,
    families: Sequence[str] | None = None,
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for family, fasta in discover_family_fastas(ncbi_root, families=families):
        records.extend(parse_fasta(fasta, family=family))

    input_dir = Path(output_dir) / "input"
    write_merged_fasta(records, input_dir / "all_gvp_proteins.fasta")
    write_sequence_index(records, input_dir / "sequence_index.csv")
    return records
