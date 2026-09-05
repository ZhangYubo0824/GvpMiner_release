#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gvp_literature_enrichment import (
    load_experimental_regions,
    load_literature_sources,
    load_reference_sequences as _load_reference_sequences,
    map_motifs_to_regions,
    read_fasta,
    run_literature_enrichment,
)


DATA_DIR = ROOT / "data" / "literature"
SOURCE_CSV = DATA_DIR / "literature_sources.csv"
REGION_CSV = DATA_DIR / "experimental_regions.csv"
REFERENCE_FASTA_DIR = DATA_DIR / "reference_sequences"
FULL_FASTA = (
    ROOT
    / "result6"
    / "exp5_external_functional_enrichment"
    / "input"
    / "all_gvp_proteins.fasta"
)
INPUT = ROOT / "result6" / "motif_validation" / "motif_evidence_integration.csv"
INTERPRO = (
    ROOT
    / "result6"
    / "exp5_external_functional_enrichment"
    / "interproscan"
    / "all_gvp_proteins.fasta.tsv"
)
SEARCH_LOG = DATA_DIR / "literature_search_log.csv"
OUTDIR = ROOT / "result6" / "motif_validation" / "literature_enrichment"
OCCURRENCE_ROOT = ROOT / "result6" / "exp3_domain_discovery"


ALL_REGIONS = load_experimental_regions(REGION_CSV)
STRUCTURAL_REGIONS = [
    row for row in ALL_REGIONS if str(row.get("analysis_type")) == "structure"
]
FUNCTION_REGIONS = [
    row for row in ALL_REGIONS if str(row.get("analysis_type")) != "structure"
]


def load_reference_sequences() -> dict[str, str]:
    return _load_reference_sequences(REFERENCE_FASTA_DIR)


def collect_hits(
    family: str,
    accession: str,
    full_sequence: str,
    motif_start: int,
    motif_end: int,
    regions: list[dict[str, object]],
    references: Mapping[str, str],
    alignment_cache: dict[tuple[str, str], object] | None = None,
) -> list[dict[str, object]]:
    del alignment_cache
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
        "matched_region_ids": _join_unique(
            [hit["evidence_id"] for hit in hits]
        ),
        "matched_region_labels": _join_unique(
            [hit.get("region_label", "") for hit in hits]
        ),
        "experimental_features": _join_unique(
            [hit.get("experimental_feature", "") for hit in hits]
        ),
        "candidate_function": _join_unique(
            [
                hit.get("supported_function")
                or hit.get("phenotype", "")
                for hit in hits
            ]
        ),
        "experiment_types": _join_unique(
            [hit.get("experiment_type", "") for hit in hits]
        ),
        "source_ids": _join_unique(
            [hit.get("source_id", "") for hit in hits]
        ),
        "evidence_scope": _join_unique(
            [hit.get("evidence_scope", "") for hit in hits]
        ),
        "limitations": _join_unique(
            [hit.get("limitations", "") for hit in hits]
        ),
        "maximum_motif_overlap_fraction": max(
            float(hit["motif_overlap_fraction"]) for hit in hits
        ),
        "minimum_reference_region_exact_match_fraction": min(
            float(hit["reference_region_exact_match_fraction"])
            for hit in hits
        ),
        "matched_region_count": len(hits),
    }


def source_table() -> pd.DataFrame:
    sources, _ = load_literature_sources(SOURCE_CSV)
    return pd.DataFrame(sources)


def main() -> None:
    summary = run_literature_enrichment(
        motif_table=INPUT,
        full_fasta=FULL_FASTA,
        literature_sources=SOURCE_CSV,
        experimental_regions=REGION_CSV,
        reference_dir=REFERENCE_FASTA_DIR,
        interpro_path=INTERPRO,
        search_log=SEARCH_LOG,
        output_dir=OUTDIR,
        occurrence_root=OCCURRENCE_ROOT,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
