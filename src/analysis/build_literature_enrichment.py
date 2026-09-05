#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gvp_literature_enrichment import run_literature_enrichment


DEFAULT_MOTIFS = (
    ROOT / "result6" / "motif_validation" / "motif_evidence_integration.csv"
)
DEFAULT_FULL_FASTA = (
    ROOT
    / "result6"
    / "exp5_external_functional_enrichment"
    / "input"
    / "all_gvp_proteins.fasta"
)
DEFAULT_SOURCES = ROOT / "data" / "literature" / "literature_sources.csv"
DEFAULT_REGIONS = ROOT / "data" / "literature" / "experimental_regions.csv"
DEFAULT_REFERENCES = ROOT / "data" / "literature" / "reference_sequences"
DEFAULT_INTERPRO = (
    ROOT
    / "result6"
    / "exp5_external_functional_enrichment"
    / "interproscan"
    / "all_gvp_proteins.fasta.tsv"
)
DEFAULT_SEARCH_LOG = (
    ROOT / "data" / "literature" / "literature_search_log.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "result6" / "motif_validation" / "literature_enrichment"
)
DEFAULT_OCCURRENCES = ROOT / "result6" / "exp3_domain_discovery"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map every GVP motif to curated experimental literature regions "
            "and cross-check local database annotations."
        )
    )
    parser.add_argument("--motif-table", type=Path, default=DEFAULT_MOTIFS)
    parser.add_argument("--full-fasta", type=Path, default=DEFAULT_FULL_FASTA)
    parser.add_argument(
        "--literature-sources", type=Path, default=DEFAULT_SOURCES
    )
    parser.add_argument(
        "--experimental-regions", type=Path, default=DEFAULT_REGIONS
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=DEFAULT_REFERENCES
    )
    parser.add_argument("--interpro", type=Path, default=DEFAULT_INTERPRO)
    parser.add_argument("--search-log", type=Path, default=DEFAULT_SEARCH_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--occurrence-root", type=Path, default=DEFAULT_OCCURRENCES
    )
    parser.add_argument("--max-cluster-instances", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_literature_enrichment(
        motif_table=args.motif_table,
        full_fasta=args.full_fasta,
        literature_sources=args.literature_sources,
        experimental_regions=args.experimental_regions,
        reference_dir=args.reference_dir,
        interpro_path=args.interpro,
        search_log=args.search_log,
        output_dir=args.output_dir,
        occurrence_root=args.occurrence_root,
        max_cluster_instances=args.max_cluster_instances,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
