from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


OCCURRENCE_FIELDS = [
    "family",
    "method",
    "motif_uid",
    "sequence_id",
    "parent_id",
    "occ_start",
    "occ_end",
    "occ_core_start",
    "occ_core_end",
    "score",
    "significance",
    "motif_sequence",
    "consensus_or_label",
    "source_model",
]

CATALOG_FIELDS = [
    "family",
    "method",
    "motif_uid",
    "consensus_or_label",
    "n_occurrences",
    "score",
    "significance",
    "source_model",
]

REQUIRED_OCCURRENCE_FIELDS = [
    "family",
    "method",
    "motif_uid",
    "sequence_id",
    "occ_start",
    "occ_end",
    "motif_sequence",
]


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def canonical_sequence_id(value: object) -> str:
    text = clean(value)
    if not text:
        return ""
    token = text.split()[0]
    return token.split("|", 1)[0]


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = ""
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if header:
                    records.append((header, "".join(chunks)))
                header = text[1:].split()[0]
                chunks = []
            else:
                chunks.append(text)
    if header:
        records.append((header, "".join(chunks)))
    return records


def read_delimited(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        lines = [line for line in handle if line.strip() and not line.startswith("#")]
    if not lines:
        return []
    return [dict(row) for row in csv.DictReader(lines, delimiter=delimiter)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row.get(field)) for field in fields})


def find_first(input_dir: Path, names: Sequence[str]) -> Path | None:
    input_dir = Path(input_dir)
    for name in names:
        direct = input_dir / name
        if direct.exists():
            return direct
    for path in input_dir.rglob("*"):
        if path.is_file() and path.name in names:
            return path
    return None


def normalize_occurrence(row: Mapping[str, object]) -> dict[str, str]:
    result = {field: clean(row.get(field)) for field in OCCURRENCE_FIELDS}
    if result["sequence_id"]:
        result["sequence_id"] = canonical_sequence_id(result["sequence_id"])
    if result["parent_id"]:
        result["parent_id"] = canonical_sequence_id(result["parent_id"])
    if not result["parent_id"]:
        result["parent_id"] = result["sequence_id"]
    if not result["occ_core_start"]:
        result["occ_core_start"] = result["occ_start"]
    if not result["occ_core_end"]:
        result["occ_core_end"] = result["occ_end"]
    return result


