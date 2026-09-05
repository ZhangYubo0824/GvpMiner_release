#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence


EXP5_ROOT = Path(__file__).resolve().parents[1]
if str(EXP5_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP5_ROOT))

from package import reproducibility, support_analysis


MOTIF_FIELDS = [
    "family",
    "method",
    "motif_cluster",
    "motif_rank",
    "rank_source",
    "protein_count",
    "eligible_term_count",
    "family_eligible_annotated_protein_count",
    "eligible_annotated_foreground_count",
    "annotation_coverage",
    "protein_size_bin",
    "annotation_coverage_bin",
    "is_testable",
    "is_supported",
    "status",
    "minimum_p_adjust",
    "alpha",
    "significance_rule",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build corrected motif testability, deterministic ranking, Top-K, "
            "and matched random-background tables without changing multiplicity handling."
        )
    )
    parser.add_argument("--motif-sets", action="append", required=True, type=Path)
    parser.add_argument("--enrichment", action="append", required=True, type=Path)
    parser.add_argument("--family-term2gene-dir", required=True, type=Path)
    parser.add_argument("--catalog", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-annotated-foreground", type=int, default=3)
    parser.add_argument("--min-gene-set-size", type=int, default=10)
    parser.add_argument("--max-gene-set-size", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-iterations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260714)
    return parser


def _expand_csv_inputs(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.csv"))
        else:
            raise FileNotFoundError(path)
    return sorted({path.resolve() for path in files}, key=str)


def _iter_rows(paths: Iterable[Path]) -> Iterable[dict[str, str]]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            yield from (dict(row) for row in csv.DictReader(handle))


def _validate_schema(
    paths: Iterable[Path],
    *,
    required: set[str],
    kind: str,
    any_identifier: set[str] | None = None,
) -> None:
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            fields = set(csv.DictReader(handle).fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(
                f"{kind} file {path} missing columns: {sorted(missing)}"
            )
        if any_identifier and not (fields & any_identifier):
            raise ValueError(
                f"{kind} file {path} requires one of: {sorted(any_identifier)}"
            )


def _read_family_terms(
    directory: Path,
) -> tuple[dict[str, Iterable[dict[str, str]]], list[Path]]:
    files = sorted(Path(directory).glob("term2gene_*.csv"), key=str)
    family_rows: dict[str, Iterable[dict[str, str]]] = {}
    for path in files:
        family = path.stem.removeprefix("term2gene_")
        family_rows[family] = _iter_rows([path])
    return family_rows, files


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    motif_files = _expand_csv_inputs(args.motif_sets)
    enrichment_files = _expand_csv_inputs(args.enrichment)
    catalog_files = _expand_csv_inputs(args.catalog) if args.catalog else []
    family_terms, family_files = _read_family_terms(args.family_term2gene_dir)
    if not motif_files:
        raise ValueError("no motif-set CSV files were found")
    if not enrichment_files:
        raise ValueError("no enrichment CSV files were found")
    if not family_files:
        raise ValueError("no family TERM2GENE CSV files were found")
    _validate_schema(
        motif_files,
        required={"family", "method", "motif_cluster", "set_type", "protein_id"},
        kind="motif-set",
    )
    _validate_schema(
        enrichment_files,
        required={"family", "method", "motif_cluster", "p.adjust"},
        kind="enrichment",
    )
    _validate_schema(
        family_files,
        required={"term_id", "protein_id"},
        kind="family TERM2GENE",
    )
    if catalog_files:
        _validate_schema(
            catalog_files,
            required={"family", "method", "rank_within_method"},
            kind="motif catalog",
            any_identifier={"motif_cluster", "motif_uid", "motif_id"},
        )

    # The motif-set and enrichment inputs can be hundreds of MB. Stream them
    # into the compact per-motif aggregations instead of materialising every
    # CSV row as a Python dictionary.
    motif_rows = _iter_rows(motif_files)
    enrichment_rows = _iter_rows(enrichment_files)
    catalog_rows = _iter_rows(catalog_files)
    analysed = support_analysis.analyse_motifs(
        motif_rows,
        enrichment_rows,
        family_terms,
        catalog_rows=catalog_rows,
        alpha=args.alpha,
        min_annotated_foreground=args.min_annotated_foreground,
        min_gene_set_size=args.min_gene_set_size,
        max_gene_set_size=args.max_gene_set_size,
    )
    if not analysed:
        raise ValueError(
            "no foreground motif rows were found; check the --motif-sets inputs"
        )
    topk = support_analysis.summarise_top_k(analysed, args.top_k)
    method_summary = support_analysis.summarise_methods(topk)
    random_summary = support_analysis.random_background_summaries(
        analysed,
        top_k=args.top_k,
        iterations=args.random_iterations,
        global_seed=args.random_seed,
    )

    output_dir = Path(args.output_dir)
    outputs = [
        _write_csv(output_dir / "motif_level_fairness.csv", analysed, MOTIF_FIELDS),
        _write_csv(
            output_dir / "topk_family_method_summary.csv",
            topk,
            [
                "family",
                "method",
                "top_k_requested",
                "selected_motif_count",
                "testable_motif_count",
                "supported_motif_count",
                "supported_fraction_of_selected",
                "supported_fraction_of_testable",
            ],
        ),
        _write_csv(
            output_dir / "topk_method_summary.csv",
            method_summary,
            [
                "method",
                "family_count",
                "selected_motif_count",
                "testable_motif_count",
                "supported_motif_count",
                "supported_fraction_of_selected",
                "supported_fraction_of_testable",
            ],
        ),
        _write_csv(
            output_dir / "random_background_summary.csv",
            random_summary,
            [
                "family",
                "method",
                "top_k_requested",
                "observed_selected_count",
                "observed_testable_count",
                "observed_supported_count",
                "random_iterations",
                "global_seed",
                "random_testable_mean",
                "random_supported_mean",
                "random_supported_q025",
                "random_supported_q975",
                "empirical_p_ge_observed",
                "sampling_contract",
            ],
        ),
    ]
    manifest = reproducibility.write_manifest(
        output_dir / "support_analysis_manifest.json",
        operation="build-fair-support-tables",
        parameters={
            "alpha": args.alpha,
            "min_annotated_foreground": args.min_annotated_foreground,
            "min_gene_set_size": args.min_gene_set_size,
            "max_gene_set_size": args.max_gene_set_size,
            "top_k": args.top_k,
            "random_iterations": args.random_iterations,
            "random_seed": args.random_seed,
            "multiplicity_change": "none",
        },
        inputs=[*motif_files, *enrichment_files, *family_files, *catalog_files],
        outputs=outputs,
        code_files=[
            Path(__file__),
            Path(support_analysis.__file__),
            Path(reproducibility.__file__),
        ],
    )
    for path in outputs:
        print(path)
    print(manifest)


if __name__ == "__main__":
    main()
