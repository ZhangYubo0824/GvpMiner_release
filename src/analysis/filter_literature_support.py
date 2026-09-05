#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gvp_literature_enrichment import (
    load_experimental_regions,
    load_reference_sequences,
    map_motifs_to_regions,
    read_fasta,
)

RESULTS = ROOT / "work" / "results"
DEFAULT_FULL_FASTA = (
    RESULTS
    / "exp5_external_functional_enrichment"
    / "input"
    / "all_gvp_proteins.fasta"
)
DEFAULT_REGIONS = ROOT / "data" / "literature" / "experimental_regions.csv"
DEFAULT_REFERENCES = ROOT / "data" / "literature" / "reference_sequences"
DEFAULT_INPUT = (
    RESULTS
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


def collect_hits(
    family: str,
    accession: str,
    full_sequence: str,
    motif_start: int,
    motif_end: int,
    regions: list[dict[str, object]],
    references: dict[str, str],
) -> list[dict[str, object]]:
    motif = {
        "family": family,
        "method": "",
        "motif_cluster": f"{accession}:{motif_start}-{motif_end}",
        "representative_parent_id": accession,
        "representative_start": motif_start,
        "representative_end": motif_end,
        "n_unique_sequences": 1,
    }
    _, hits, _ = map_motifs_to_regions(
        [motif],
        full_sequences={accession: full_sequence},
        reference_sequences=references,
        regions=regions,
    )
    for hit in hits:
        hit.setdefault("source_ids", hit.get("source_id", ""))
    return hits


def _join_unique(values: list[object]) -> str:
    output: list[str] = []
    for value in values:
        for item in str(value).split(";"):
            item = item.strip()
            if item and item not in output:
                output.append(item)
    return ";".join(output)


def summarize_hit_rows(hits: list[dict[str, object]]) -> dict[str, object]:
    top = hits[0]
    return {
        "reference_id": top["reference_id"],
        "mapped_reference_start": top["mapped_reference_start"],
        "mapped_reference_end": top["mapped_reference_end"],
        "global_alignment_identity": top["global_alignment_identity"],
        "motif_alignment_identity": top["motif_alignment_identity"],
        "motif_mapping_fraction": top["motif_mapping_fraction"],
        "matched_region_ids": _join_unique([hit["evidence_id"] for hit in hits]),
        "matched_region_labels": _join_unique([hit.get("region_label", "") for hit in hits]),
        "experimental_features": _join_unique([hit.get("experimental_feature", "") for hit in hits]),
        "candidate_function": _join_unique(
            [hit.get("supported_function") or hit.get("phenotype", "") for hit in hits]
        ),
        "experiment_types": _join_unique([hit.get("experiment_type", "") for hit in hits]),
        "source_ids": _join_unique([hit.get("source_id", "") for hit in hits]),
        "evidence_scope": _join_unique([hit.get("evidence_scope", "") for hit in hits]),
        "limitations": _join_unique([hit.get("limitations", "") for hit in hits]),
        "maximum_motif_overlap_fraction": max(float(hit["motif_overlap_fraction"]) for hit in hits),
        "minimum_reference_region_exact_match_fraction": min(
            float(hit["reference_region_exact_match_fraction"]) for hit in hits
        ),
        "matched_region_count": len(hits),
    }


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
    function_regions: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []

    for row in rows:
        if int(float(row["n_unique_sequences"])) < MIN_UNIQUE_SEQUENCES:
            continue

        accession = row["representative_accession"]
        full_sequence = full_sequences.get(accession, "")
        if not full_sequence:
            continue

        hits = collect_hits(
            family=row["family"],
            accession=accession,
            full_sequence=full_sequence,
            motif_start=int(float(row["representative_start"])),
            motif_end=int(float(row["representative_end"])),
            regions=function_regions,
            references=references,
        )
        strict_hits = select_strict_hits(hits)
        if not strict_hits:
            continue

        strict_summary = summarize_hit_rows(strict_hits)
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
    parser.add_argument("--full-fasta", type=Path, default=DEFAULT_FULL_FASTA)
    parser.add_argument("--experimental-regions", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_rows(args.input)
    full_sequences = read_fasta(args.full_fasta)
    references = load_reference_sequences(args.reference_dir)
    function_regions = [
        row
        for row in load_experimental_regions(args.experimental_regions)
        if str(row.get("analysis_type")) != "structure"
    ]
    selected = filter_candidates(rows, full_sequences, references, function_regions)
    write_rows(args.output, fieldnames, selected)
    print(f"input_candidates={len(rows)}")
    print(f"high_confidence_candidates={len(selected)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
