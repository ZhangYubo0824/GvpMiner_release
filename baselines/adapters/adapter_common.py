from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
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


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--family", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


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


def resolve_family_fasta(family: str) -> Path:
    candidates: list[Path] = []
    preprocessed = os.environ.get("PREPROCESSED")
    project = os.environ.get("PROJECT")
    if preprocessed:
        candidates.append(Path(preprocessed) / "by_family" / f"{family}_cleaned.fasta")
    if project:
        project_path = Path(project)
        candidates.extend(
            [
                project_path / "data" / "reference" / "preprocessed" / "by_family" / f"{family}_cleaned.fasta",
            ]
        )
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "data" / "reference" / "preprocessed" / "by_family" / f"{family}_cleaned.fasta",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Family FASTA not found for {family}. Set PREPROCESSED to the directory containing by_family/{family}_cleaned.fasta."
    )


def resolve_sequence_id(display_id: object, sequence_ids: Sequence[str]) -> str:
    display = clean(display_id)
    if not display:
        return ""
    canonical_display = canonical_sequence_id(display)
    canonical_by_full = {canonical_sequence_id(sequence_id): sequence_id for sequence_id in sequence_ids}
    if canonical_display in canonical_by_full:
        return canonical_display

    full_matches = [sequence_id for sequence_id in sequence_ids if sequence_id.startswith(display)]
    if len(full_matches) == 1:
        return canonical_sequence_id(full_matches[0])

    canonical_matches = [
        canonical_sequence_id(sequence_id)
        for sequence_id in sequence_ids
        if canonical_sequence_id(sequence_id).startswith(canonical_display)
    ]
    unique_matches = sorted(set(match for match in canonical_matches if match))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return canonical_display


def first_value(row: Mapping[str, object], names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = clean(row.get(name))
        if value:
            return value
    return default


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


def make_motif_uid(family: str, method: str, motif_label: str) -> str:
    label = clean(motif_label).replace(" ", "_").replace("/", "_")
    return f"{family}_{method}_{label or 'motif'}"


def one_based_to_zero_based(start_text: str, end_text: str, motif_sequence: str) -> tuple[int, int]:
    start = int(float(start_text))
    if end_text:
        end = int(float(end_text))
    else:
        end = start + len(motif_sequence) - 1
    return max(0, start - 1), max(0, end)


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


def validate_occurrence_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("No motif occurrence rows were produced.")
    for index, row in enumerate(rows, start=1):
        for field in REQUIRED_OCCURRENCE_FIELDS:
            if not clean(row.get(field)):
                raise ValueError(f"Occurrence row {index} is missing required field: {field}")
        start = int(float(clean(row.get("occ_start"))))
        end = int(float(clean(row.get("occ_end"))))
        if start < 0 or end <= start:
            raise ValueError(f"Occurrence row {index} has invalid coordinates: {start}-{end}")


def catalog_from_occurrences(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[clean(row.get("motif_uid"))].append(row)

    catalog = []
    for motif_uid, group in sorted(grouped.items()):
        first = group[0]
        catalog.append(
            {
                "family": clean(first.get("family")),
                "method": clean(first.get("method")),
                "motif_uid": motif_uid,
                "consensus_or_label": clean(first.get("consensus_or_label")) or clean(first.get("motif_sequence")),
                "n_occurrences": str(len(group)),
                "score": clean(first.get("score")),
                "significance": clean(first.get("significance")),
                "source_model": clean(first.get("source_model")),
            }
        )
    return catalog


def write_standard_outputs(
    *,
    output_dir: Path,
    method: str,
    occurrences: Sequence[Mapping[str, object]],
    catalog: Sequence[Mapping[str, object]] | None = None,
) -> None:
    normalized = [normalize_occurrence(row) for row in occurrences]
    validate_occurrence_rows(normalized)
    catalog_rows = list(catalog) if catalog is not None else catalog_from_occurrences(normalized)
    write_rows(Path(output_dir) / f"motif_occurrences_{method}.csv", normalized, OCCURRENCE_FIELDS)
    write_rows(Path(output_dir) / f"motif_catalog_{method}.csv", catalog_rows, CATALOG_FIELDS)


def copy_standard_csvs(*, family: str, method: str, input_dir: Path, output_dir: Path) -> None:
    occurrence = find_first(input_dir, [f"motif_occurrences_{method}.csv", "motif_occurrences.csv"])
    catalog = find_first(input_dir, [f"motif_catalog_{method}.csv", "motif_catalog.csv"])
    if occurrence is None or catalog is None:
        raise FileNotFoundError(
            "Standard CSV adapter requires motif_occurrences_<method>.csv and "
            "motif_catalog_<method>.csv, or generic motif_occurrences.csv and motif_catalog.csv."
        )

    occurrence_rows = read_delimited(occurrence)
    for row in occurrence_rows:
        row.setdefault("family", family)
        row.setdefault("method", method)
        row.setdefault("source_model", method)
    normalized = [normalize_occurrence(row) for row in occurrence_rows]
    validate_occurrence_rows(normalized)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / f"motif_occurrences_{method}.csv", normalized, OCCURRENCE_FIELDS)
    shutil.copyfile(catalog, output_dir / f"motif_catalog_{method}.csv")
