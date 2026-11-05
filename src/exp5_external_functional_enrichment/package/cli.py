from __future__ import annotations

import argparse
from itertools import chain
import os
from pathlib import Path
from typing import Sequence

from . import annotations
from . import clusterprofiler
from . import external_tools
from . import motif_sets
from . import motif_validation
from . import prepare
from . import reproducibility




def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_results_dir() -> Path:
    return Path(
        os.environ.get("GVP_OUTPUT_DIR", _default_project_root() / "work" / "results")
    ).expanduser().resolve()


def _default_ncbi_root() -> Path:
    return _default_project_root() / "data" / "ncbi"


def _default_exp3_dir() -> Path:
    return _default_results_dir() / "exp3_domain_discovery"


def _default_output_dir() -> Path:
    return _default_results_dir() / "exp5_external_functional_enrichment"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone external functional enrichment workflow for Exp3 motif clusters"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Merge family FASTA files and build sequence_index.csv")
    p_prepare.add_argument("--ncbi-root", type=Path, default=_default_ncbi_root())
    p_prepare.add_argument("--output-dir", type=Path, default=_default_output_dir())
    p_prepare.add_argument("--families", nargs="*", default=None)

    p_parse = sub.add_parser("parse-annotations", help="Parse external annotation outputs into TERM2GENE tables")
    p_parse.add_argument("--interproscan-tsv", type=Path, default=None)
    p_parse.add_argument("--eggnog-annotations", type=Path, default=None)
    p_parse.add_argument("--kofamscan-tsv", type=Path, default=None)
    p_parse.add_argument("--output-dir", type=Path, default=_default_output_dir() / "annotations")
    p_parse.add_argument(
        "--include-interpro-pathways",
        action="store_true",
        help=(
            "Opt in to InterProScan Pathway rows. They are excluded by default "
            "because their broad imported labels dominated the earlier annotation table."
        ),
    )

    p_sets = sub.add_parser("make-motif-sets", help="Build foreground/background/universe rows from Exp3 occurrences")
    p_sets.add_argument("--exp3-dir", type=Path, default=_default_exp3_dir())
    p_sets.add_argument("--sequence-index", type=Path, default=_default_output_dir() / "input" / "sequence_index.csv")
    p_sets.add_argument("--output-dir", type=Path, default=_default_output_dir() / "clusterprofiler")
    p_sets.add_argument("--families", nargs="*", default=None)
    p_sets.add_argument("--methods", nargs="*", choices=motif_sets.SUPPORTED_MOTIF_METHODS, default=None)

    p_r = sub.add_parser("write-r-script", help="Write the clusterProfiler enricher R script")
    p_r.add_argument("--output", type=Path, default=_default_output_dir() / "clusterprofiler" / "run_clusterprofiler_enricher.R")

    p_tools = sub.add_parser("tool-commands", help="Print command templates for external annotation tools")
    p_tools.add_argument("--input-fasta", type=Path, default=_default_output_dir() / "input" / "all_gvp_proteins.fasta")
    p_tools.add_argument("--output-dir", type=Path, default=_default_output_dir())
    p_tools.add_argument("--cpu", type=int, default=32)
    p_tools.add_argument("--kofam-profile-dir", type=Path, default=Path("profiles"))
    p_tools.add_argument("--kofam-ko-list", type=Path, default=Path("ko_list"))

    p_windows = sub.add_parser("motif-windows", help="Extract motif windows and write per-cluster FASTA files")
    p_windows.add_argument("--exp3-dir", type=Path, default=_default_exp3_dir())
    p_windows.add_argument("--sequence-index", type=Path, default=_default_output_dir() / "input" / "sequence_index.csv")
    p_windows.add_argument("--output-dir", type=Path, default=_default_output_dir() / "motif_windows")
    p_windows.add_argument("--families", nargs="*", default=None)
    p_windows.add_argument("--methods", nargs="*", choices=motif_sets.SUPPORTED_MOTIF_METHODS, default=None)
    p_windows.add_argument("--min-windows", type=int, default=2)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.command == "prepare":
        rows = prepare.build_sequence_index(
            ncbi_root=args.ncbi_root,
            output_dir=args.output_dir,
            families=args.families,
        )
        print(f"Wrote {len(rows)} protein records to {args.output_dir / 'input'}")
        return

    if args.command == "parse-annotations":
        term_streams = []
        filter_audit = annotations.AnnotationFilterAudit()
        if args.interproscan_tsv:
            term_streams.append(
                annotations.iter_interproscan_tsv(
                    args.interproscan_tsv,
                    include_pathways=args.include_interpro_pathways,
                    audit=filter_audit,
                )
            )
        if args.eggnog_annotations:
            term_streams.append(
                annotations.iter_eggnog_annotations(args.eggnog_annotations)
            )
        if args.kofamscan_tsv:
            term_streams.append(
                annotations.iter_kofamscan_tsv(args.kofamscan_tsv)
            )
        paths = annotations.write_term_tables(chain.from_iterable(term_streams), args.output_dir)
        paths["filter_audit"] = annotations.write_filter_audit(
            filter_audit,
            args.output_dir / "annotation_filter_counts.csv",
        )
        manifest_path = reproducibility.write_manifest(
            args.output_dir / "annotation_run_manifest.json",
            operation="parse-annotations",
            parameters={
                "include_interpro_pathways": args.include_interpro_pathways,
            },
            inputs=[
                args.interproscan_tsv,
                args.eggnog_annotations,
                args.kofamscan_tsv,
            ],
            outputs=paths.values(),
            code_files=[
                Path(__file__),
                Path(annotations.__file__),
                Path(reproducibility.__file__),
            ],
        )
        print("Wrote deduplicated annotation tables")
        for label, path in paths.items():
            print(f"{label}: {path}")
        print(f"manifest: {manifest_path}")
        return

    if args.command == "make-motif-sets":
        sequence_rows = prepare.read_sequence_index(args.sequence_index)
        occurrence_rows: list[dict[str, str]] = []
        for path in motif_sets.discover_occurrence_files(
            args.exp3_dir,
            families=args.families,
            methods=args.methods,
        ):
            occurrence_rows.extend(motif_sets.read_occurrence_csv(path))
        proteins_by_family, unique_accessions = motif_sets.build_sequence_lookup(sequence_rows)
        excluded_proteins = {
            (
                str(row.get("family") or "").strip(),
                str(row.get("method") or "").strip(),
                motif_sets.occurrence_protein_id(row),
            )
            for row in occurrence_rows
            if motif_sets.occurrence_protein_id(row)
            if not motif_sets.resolve_index_protein_id(
                str(row.get("family") or "").strip(),
                motif_sets.occurrence_protein_id(row),
                proteins_by_family,
                unique_accessions,
            )
        }
        rows = motif_sets.build_motif_sets(occurrence_rows, sequence_rows)
        output_path = args.output_dir / "motif_sets_long.csv"
        motif_sets.write_motif_sets_long(rows, output_path)
        print(f"Wrote {len(rows)} motif cluster set definitions to {output_path}")
        if excluded_proteins:
            print(
                f"Excluded {len(excluded_proteins)} family/method/protein IDs absent from "
                f"the sequence index: {args.sequence_index}"
            )
        return

    if args.command == "write-r-script":
        path = clusterprofiler.write_clusterprofiler_r_script(args.output)
        print(f"Wrote clusterProfiler R script: {path}")
        return

    if args.command == "tool-commands":
        interpro_dir = args.output_dir / "interproscan"
        eggnog_dir = args.output_dir / "eggnog"
        kofam_out = args.output_dir / "kofamscan" / "gvp_kofam.tsv"
        print(external_tools.build_interproscan_command(input_fasta=args.input_fasta, output_dir=interpro_dir, cpu=args.cpu))
        print(external_tools.build_eggnog_mapper_command(input_fasta=args.input_fasta, output_dir=eggnog_dir, cpu=args.cpu))
        print(
            external_tools.build_kofamscan_command(
                input_fasta=args.input_fasta,
                output_path=kofam_out,
                profile_dir=args.kofam_profile_dir,
                ko_list=args.kofam_ko_list,
                cpu=args.cpu,
            )
        )
        return

    if args.command == "motif-windows":
        sequence_rows = prepare.read_sequence_index(args.sequence_index)
        sequences_by_id = {row.protein_id: row for row in sequence_rows}
        occurrence_rows: list[dict[str, str]] = []
        for path in motif_sets.discover_occurrence_files(
            args.exp3_dir,
            families=args.families,
            methods=args.methods,
        ):
            occurrence_rows.extend(motif_sets.read_occurrence_csv(path))
        windows = motif_validation.extract_motif_windows(occurrence_rows, sequences_by_id)
        paths = motif_validation.write_motif_window_fastas(
            windows,
            args.output_dir,
            min_windows=args.min_windows,
        )
        print(f"Wrote {len(paths)} motif-window FASTA files to {args.output_dir}")
        return


if __name__ == "__main__":
    main()
