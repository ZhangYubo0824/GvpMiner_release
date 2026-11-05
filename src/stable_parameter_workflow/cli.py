from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .commands import CommandContext
from .advancement import recommend_decision_file
from .contracts import DEFAULT_MCS, ParameterPair
from .progress import TerminalProgress
from .workflow import WorkflowContext, run_workflow


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the staged stable-motif window/stride selection workflow. "
            "Run A, B, then C; C is the terminal motif-discovery stage."
        )
    )
    parser.add_argument("--stage", required=True, choices=("A", "B", "C"))
    parser.add_argument(
        "--auto-select", action="store_true",
        help="Choose next-stage parameters from the completed prior stage using fixed auditable rules",
    )
    parser.add_argument("--dataset-csv", required=True, type=Path)
    parser.add_argument(
        "--baseline-exp3-dir", required=True, type=Path,
        help="Read-only existing Exp3 root; used for output-overlap protection",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--residue-cache-dir", action="append", type=Path, default=[])
    parser.add_argument("--residue-cache-fallback-dir", type=Path, default=None)
    parser.add_argument(
        "--mcs", nargs="+", type=int, default=list(DEFAULT_MCS),
        help="Frozen selection grid; must remain: 10 20 30 40 60 80",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--minimum-free-gb", type=float, default=5.0)
    parser.add_argument("--exp3-workers", type=int, default=1)
    parser.add_argument("--stability-workers", type=int, default=1)
    parser.add_argument("--confirmatory-workers", type=int, default=1)
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.auto_select and args.stage == "A":
        raise ValueError("--auto-select is supported only for Stage B or C")
    if args.stage in {"B", "C"} and not args.auto_select:
        raise ValueError(f"Stage {args.stage} requires --auto-select")
    values = tuple(int(value) for value in args.mcs)
    if any(value > 80 for value in values):
        raise ValueError("MCS values above 80 are not allowed in this workflow")
    if values != DEFAULT_MCS:
        raise ValueError(f"the fixed MCS grid is {' '.join(map(str, DEFAULT_MCS))}")
    if args.minimum_free_gb < 0:
        raise ValueError("minimum free space cannot be negative")
    if min(
        args.exp3_workers,
        args.stability_workers,
        args.confirmatory_workers,
    ) < 1:
        raise ValueError("worker counts must be positive integers")
    if args.stage in {"A", "B"} and args.confirmatory_workers != 1:
        raise ValueError("confirmatory workers above one are only supported for Stage C")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    root = project_root()
    command_context = CommandContext(
        project_root=root,
        python_executable=Path(sys.executable),
        dataset_csv=args.dataset_csv,
        residue_cache_dirs=tuple(args.residue_cache_dir),
        residue_cache_fallback_dir=args.residue_cache_fallback_dir,
    )
    reporter = TerminalProgress(enabled=not args.no_progress)
    context = WorkflowContext(
        commands=command_context,
        output_root=args.output_root,
        baseline_roots=(args.baseline_exp3_dir,),
        resume=args.resume,
        force=args.force,
        reuse_stage_a=args.stage == "B" and not args.force,
        dry_run=args.dry_run,
        minimum_free_gb=args.minimum_free_gb,
        reporter=reporter,
        exp3_workers=args.exp3_workers,
        stability_workers=args.stability_workers,
        confirmatory_workers=args.confirmatory_workers,
    )
    reviewed_pairs = None
    stage_c_pairs = None
    selection_report = None
    if args.auto_select:
        source_stage = {"B": "A", "C": "B"}[args.stage]
        decision_file = (args.output_root / "comparisons" /
                         f"stage_{source_stage.lower()}_decision.json")
        selection_report = recommend_decision_file(decision_file)
        pairs = tuple(ParameterPair(**pair) for pair in selection_report["selected_pairs"])
        if args.stage == "B":
            reviewed_pairs = pairs
        else:
            stage_c_pairs = pairs
    decision = run_workflow(
        args.stage, context,
        reviewed_pairs=reviewed_pairs,
        stage_c_pairs=stage_c_pairs,
        selection_report=selection_report,
    )
    if decision is not None:
        print(f"DECISION {decision}", flush=True)
    return 0
