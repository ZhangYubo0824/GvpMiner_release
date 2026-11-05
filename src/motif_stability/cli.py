from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .io import discover_family_inputs


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_source_exp3_dir() -> Path:
    return project_root() / "work" / "results" / "exp3_domain_discovery"


def default_output_root() -> Path:
    return project_root() / "work" / "results"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Protein-level subsampling stability analysis for cached Exp3 motifs"
    )
    parser.add_argument("--source-exp3-dir", type=Path, default=default_source_exp3_dir())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=["M2", "M3"])
    parser.add_argument("--profile", choices=["pilot", "confirmatory"], default="pilot")
    parser.add_argument(
        "--mcs",
        nargs="+",
        type=int,
        default=list(range(10, 81, 10)),
    )
    parser.add_argument("--min-samples", nargs="+", type=int, default=[10])
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--subsample-fraction", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="UMAP seeds (default: pilot=0; confirmatory=0 1 2 3 4)",
    )
    parser.add_argument("--null-repeats", type=int, default=None)
    parser.add_argument("--validate-inputs", action="store_true")
    parser.add_argument(
        "--reselect-only",
        action="store_true",
        help=(
            "select final parameters from existing parameter_screen.csv files "
            "without loading embeddings or running clustering"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable repeat/null progress messages",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.reselect_only:
        from .reselection import collect_parameter_selections, write_selection_summary

        selections = collect_parameter_selections(
            args.output_root,
            families=args.families,
            methods=args.methods,
        )
        csv_path, json_path = write_selection_summary(args.output_root, selections)
        for selected in selections:
            print(
                f"SELECTED {selected['family']}/{selected['method']}: "
                f"mcs={selected['min_cluster_size']} "
                f"min_samples={selected['min_samples']} "
                f"strategy={selected['selection_strategy']}"
            )
        print(f"WROTE {csv_path}")
        print(f"WROTE {json_path}")
        return 0

    inputs = discover_family_inputs(
        args.source_exp3_dir,
        families=args.families,
        methods=args.methods,
    )
    if args.validate_inputs:
        for item in inputs:
            print(
                f"VALID {item.family}/{item.method}: "
                f"{len(item.windows)} windows x {item.embeddings.shape[1]} dimensions"
            )
        return 0

    from .pipeline import run_pipeline

    run_pipeline(inputs=inputs, args=args)
    return 0
