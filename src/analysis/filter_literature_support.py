#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import build_region_literature_validation as regional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "result6"
    / "motif_validation"
    / "cluster_commonality_validation"
    / "region_literature_validation"
    / "external_nonsignificant_motif_localized_function_supported.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name(
    "external_nonsignificant_motif_high_confidence_function_candidates.csv"
)

POINT_MUTATION_EVIDENCE_IDS = {
    "FU_J_ALPHA1_LOOP",
    "FU_J_RAAIA_EYGL",
    "FU_M_ALPHA1",
    "FU_M_GAV",
    "FU_M_RAAIA",
}
MIN_MOTIF_ALIGNMENT_IDENTITY = 0.70
MIN_UNIQUE_SEQUENCES = 10
STRICT_STATUS = "high_confidence_homologous_point_mutation_candidate"
SELECTION_RULE = (
    "n_unique_sequences>=10; motif_alignment_identity>=0.70; "
    "point-mutation evidence only; region_overlap_fraction=1.0; "
    "reference_region_exact_match_fraction=1.0"
)


def select_strict_hits(
    hits: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        hit
        for hit in hits
        if str(hit["evidence_id"]) in POINT_MUTATION_EVIDENCE_IDS
        and float(hit["motif_alignment_identity"])
        >= MIN_MOTIF_ALIGNMENT_IDENTITY
        and float(hit["region_overlap_fraction"]) == 1.0
        and float(hit["reference_region_exact_match_fraction"]) == 1.0
    ]


def filter_candidates(
    rows: Iterable[dict[str, str]],
    full_sequences: dict[str, str],
    references: dict[str, str],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    alignment_cache: dict[tuple[str, str], object] = {}

    for row in rows:
        if int(float(row["n_unique_sequences"])) < MIN_UNIQUE_SEQUENCES:
            continue

        accession = row["representative_accession"]
        full_sequence = full_sequences.get(accession, "")
        if not full_sequence:
            continue

        hits = regional.collect_hits(
            family=row["family"],
            accession=accession,
            full_sequence=full_sequence,
            motif_start=int(float(row["representative_start"])),
            motif_end=int(float(row["representative_end"])),
            regions=regional.FUNCTION_REGIONS,
            references=references,
            alignment_cache=alignment_cache,
        )
        strict_hits = select_strict_hits(hits)
        if not strict_hits:
            continue

        strict_summary = regional.summarize_hit_rows(strict_hits)
        output = {
            **row,
            "original_localized_function_support_status": row[
                "localized_function_support_status"
            ],
            "localized_function_support_status": STRICT_STATUS,
            **strict_summary,
            "strict_selection_rule": SELECTION_RULE,
            "direct_motif_validation_status": "not_directly_validated",
            "interpretation": (
                "High-confidence literature-guided homologous candidate; "
                "the exact motif window was not tested as a unit."
            ),
        }
        selected.append(output)

    return sorted(
        selected,
        key=lambda row: (
            str(row["family"]),
            str(row["method"]),
            str(row["motif"]),
        ),
    )


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(
    path: Path,
    original_fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    added_fields = [
        "original_localized_function_support_status",
        "strict_selection_rule",
        "direct_motif_validation_status",
        "interpretation",
    ]
    fieldnames = original_fieldnames + [
        name for name in added_fields if name not in original_fieldnames
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retain only strict literature-guided motif candidates."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_rows(args.input)
    full_sequences = regional.read_fasta(regional.FULL_FASTA)
    references = regional.load_reference_sequences()
    selected = filter_candidates(rows, full_sequences, references)
    write_rows(args.output, fieldnames, selected)
    print(f"input_candidates={len(rows)}")
    print(f"high_confidence_candidates={len(selected)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
