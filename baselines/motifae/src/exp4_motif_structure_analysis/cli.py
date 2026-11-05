from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Sequence
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .exp3_adapter import build_exp3_motif_rows
from .io import read_csv_rows, write_csv_rows, write_json
from .pipeline import run_exp4_motif_structure_analysis, MOTIF_FEATURE_FIELDS, CONTROL_FEATURE_FIELDS
from .residues import RESIDUE_SEMANTICS_VERSION

_root = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(os.environ.get("GVP_OUTPUT_DIR", _root / "work" / "results")).expanduser().resolve()
SOURCE_RESULTS_DIR = Path(os.environ.get("GVP_SOURCE_RESULTS_DIR", _root / "data" / "reference")).expanduser().resolve()
EXP4_SEQUENCE_TABLE = OUTPUT_DIR / "preprocessed" / "cleaned_gvp_dataset.csv"
if not EXP4_SEQUENCE_TABLE.is_file():
    EXP4_SEQUENCE_TABLE = SOURCE_RESULTS_DIR / "preprocessed" / "cleaned_gvp_dataset.csv"
EXP4_EXP3_INPUT_DIR = OUTPUT_DIR / "exp3_domain_discovery"
EXP4_STRUCTURE_INPUT_DIR = OUTPUT_DIR / "structures"

MOTIF_STRUCTURE_DIR = OUTPUT_DIR / "exp4_motif_structure_analysis"
SUPPORTED_MOTIF_METHODS = (
    "M2",
    "M3",
    "MEME",
    "GLAM2",
    "STREME",
    "MotifAE",
    "InterPLM",
)


def _default_sequence_table() -> Path:
    return EXP4_SEQUENCE_TABLE


def _default_exp3_dir() -> Path:
    return EXP4_EXP3_INPUT_DIR


def _default_output_root() -> Path:
    return MOTIF_STRUCTURE_DIR


def _default_structure_dir(family: str, method: str, backend: str) -> Path:
    return EXP4_STRUCTURE_INPUT_DIR / backend / family


def _method_sort_key(method: str) -> tuple[int, str]:
    text = str(method)
    if len(text) > 1 and text[0].upper() == "M" and text[1:].isdigit():
        return int(text[1:]), text
    return 10_000, text


def _method_from_occurrence_file(path: Path) -> str:
    return path.stem.replace("motif_occurrences_", "", 1)


def _discover_occurrence_paths(family_dir: Path) -> list[Path]:
    return sorted(
        list(family_dir.glob("motif_occurrences_*.csv"))
        + list(family_dir.glob("*/motif_occurrences_*.csv"))
    )


def _wrap_progress(iterable, *, show_progress: bool, desc: str, unit: str):
    if not show_progress or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, dynamic_ncols=True)


def discover_exp3_units(
    exp3_dir: Path,
    *,
    families: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    exp3_dir = Path(exp3_dir)
    method_filter = set(methods or SUPPORTED_MOTIF_METHODS)
    family_dirs = [exp3_dir / family for family in families] if families else [
        path for path in exp3_dir.iterdir() if path.is_dir()
    ]

    units = []
    for family_dir in sorted(family_dirs, key=lambda path: path.name):
        if not family_dir.exists():
            continue
        discovered_methods = {_method_from_occurrence_file(path) for path in _discover_occurrence_paths(family_dir)}
        for method in sorted(discovered_methods, key=_method_sort_key):
            if method not in method_filter:
                continue
            units.append((family_dir.name, method))
    return units


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gvp motif structure prediction and functional analysis")
    parser.add_argument("--sequence-table", type=Path, default=None)
    parser.add_argument("--motif-table", type=Path)
    parser.add_argument("--from-exp3", action="store_true", help="Read motif instances from Exp3 motif_occurrences_<method>.csv files")
    parser.add_argument("--exp3-dir", type=Path, default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_MOTIF_METHODS, default=None)
    structure_group = parser.add_mutually_exclusive_group()
    structure_group.add_argument("--structure-dir", type=Path, default=None)
    structure_group.add_argument(
        "--structure-root",
        type=Path,
        default=None,
        help=(
            "Root containing <backend>/<family> full-length structure caches; "
            "used with --from-exp3."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--predict-missing-structures",
        dest="predict_missing_structures",
        action="store_true",
        default=False,
        help="Predict missing full-length structures during motif analysis.",
    )
    parser.add_argument(
        "--no-predict-missing-structures",
        dest="predict_missing_structures",
        action="store_false",
        help="Only use the existing family structure cache (default).",
    )
    parser.add_argument("--prediction-backend", default="esmfold")
    parser.add_argument("--num-random-controls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-threshold", type=float, default=8.0)
    parser.add_argument("--long-range-min-separation", type=int, default=12)
    parser.add_argument("--low-plddt-cutoff", type=float, default=70.0)
    parser.add_argument(
        "--generate-figures",
        dest="generate_figures",
        action="store_true",
        default=True,
        help="Generate publication-ready static figures (default).",
    )
    parser.add_argument(
        "--no-generate-figures",
        dest="generate_figures",
        action="store_false",
        help="Skip publication figure generation while still writing tables and report.",
    )
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure output formats, e.g. png pdf.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of family-method units to run in parallel. Default: 1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from validated Exp4 outputs without rebuilding motif/control features.",
    )
    parser.add_argument(
        "--rmsd-workers",
        type=int,
        default=1,
        help="Independent CPU workers for exact local RMSD rows. Default: 1.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and stage messages")
    return parser


def resolve_structure_dir(
    args: argparse.Namespace,
    *,
    family: str,
    method: str,
) -> Path:
    if args.structure_dir is not None:
        return Path(args.structure_dir)
    if args.structure_root is not None:
        return Path(args.structure_root) / args.prediction_backend / family
    return _default_structure_dir(family, method, args.prediction_backend)


def _run_one_exp3_unit(
    args: argparse.Namespace,
    *,
    sequence_table: Path,
    sequence_rows: Sequence[dict[str, str]],
    exp3_dir: Path,
    output_root: Path,
    family: str,
    method: str,
) -> dict[str, object]:
    structure_dir = resolve_structure_dir(
        args,
        family=family,
        method=method,
    )
    motif_rows = build_exp3_motif_rows(
        exp3_dir=exp3_dir,
        sequence_rows=sequence_rows,
        families=[family],
        methods=[method],
    )
    if not motif_rows:
        output_dir = output_root / family / method
        payload = dict(family=family, method=method, sequences=[r for r in sequence_rows if r.get('family') == family], residue_semantics_version=RESIDUE_SEMANTICS_VERSION, motifs=[])
        signature = hashlib.sha256(json.dumps(payload,sort_keys=True).encode('utf-8')).hexdigest()
        signature_path = output_dir / 'analysis_input_signature.json'
        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.resume or not signature_path.is_file() or json.loads(signature_path.read_text(encoding='utf-8')).get('signature') != signature:
                raise ValueError('Empty-family Exp4 output exists with a different input signature; use a new output directory')
        summary = dict(family=family, method=method, status='no_motif_instances', motif_instances_analyzed=0, skipped_motif_instances=0, structural_tests=0, biological_function_validated=False)
        write_csv_rows(output_dir/'motif_structure_features.csv',[],MOTIF_FEATURE_FIELDS)
        write_csv_rows(output_dir/'matched_control_features.csv',[],CONTROL_FEATURE_FIELDS)
        write_json(signature_path,dict(signature=signature,residue_semantics_version=RESIDUE_SEMANTICS_VERSION))
        write_json(output_dir/'analysis_summary.json',summary)
        write_json(output_dir/'analysis_progress.json',dict(stage='complete',complete=True,counts=dict(motif_features=0,control_features=0)))
        return summary
    if not structure_dir.exists() and not args.predict_missing_structures:
        raise SystemExit(
            "Structure directory not found for "
            f"family={family}, method={method}: {structure_dir}\n"
            "Run the family structure prediction step first:\n"
            f"  python predict_family_structures.py --families {family} --backend {args.prediction_backend}\n"
            "Or pass --structure-dir to an existing full-length structure cache."
        )
    structure_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = exp3_dir.parent / "motif_stability" / family / method / "run_manifest.json"
    exp3_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    exp3_profile = str(exp3_manifest.get("profile") or "unknown")
    exp3_run_signature = str(exp3_manifest.get("run_signature") or "")
    output_dir = output_root / family / method
    progress_path = output_dir / "analysis_progress.json"
    summary_path = output_dir / "analysis_summary.json"
    resume_run = args.resume and progress_path.is_file()
    if args.resume and progress_path.is_file() and summary_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete") is True and progress.get("stage") == "complete":
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "motif_instances_analyzed" not in summary:
                raise ValueError(
                    f"Completed Exp4 summary is invalid: {summary_path}"
                )
            # Completion is not an input identity check. In particular the
            # MotifAE handoff has no legacy motif_stability run_signature.
            # Let the pipeline verify its exact residue/sequence/settings
            # signature before loading any saved feature or RMSD result.
            resume_run = True
    print(f"Analyzing {family} {method}: {len(motif_rows)} motif instances")
    return run_exp4_motif_structure_analysis(
        sequence_table=sequence_table,
        motif_table=None,
        structure_dir=structure_dir,
        output_dir=output_dir,
        motif_rows_override=motif_rows,
        predict_missing_structures=args.predict_missing_structures,
        prediction_backend=args.prediction_backend,
        num_random_controls=args.num_random_controls,
        seed=args.seed,
        contact_threshold=args.contact_threshold,
        long_range_min_separation=args.long_range_min_separation,
        low_plddt_cutoff=args.low_plddt_cutoff,
        resume=resume_run,
        exp3_profile=exp3_profile,
        exp3_run_signature=exp3_run_signature,
        rmsd_workers=args.rmsd_workers,
        show_progress=not args.no_progress,
        generate_figures=args.generate_figures,
        figure_formats=args.figure_formats,
    )


def _run_one_exp3_unit_payload(payload: tuple[argparse.Namespace, Path, list[dict[str, str]], Path, Path, str, str]) -> dict[str, object]:
    args, sequence_table, sequence_rows, exp3_dir, output_root, family, method = payload
    return _run_one_exp3_unit(
        args,
        sequence_table=sequence_table,
        sequence_rows=sequence_rows,
        exp3_dir=exp3_dir,
        output_root=output_root,
        family=family,
        method=method,
    )


def _run_exp3_defaults(args: argparse.Namespace, *, sequence_table: Path) -> list[dict[str, object]]:
    exp3_dir = args.exp3_dir or _default_exp3_dir()
    output_root = args.output_dir or _default_output_root()
    if not exp3_dir.exists():
        raise SystemExit(f"Exp3 output directory not found: {exp3_dir}")

    units = discover_exp3_units(
        exp3_dir,
        families=args.families,
        methods=args.methods,
    )
    if not units:
        raise SystemExit(f"No Exp3 motif occurrence files found in: {exp3_dir}")

    sequence_rows = read_csv_rows(sequence_table)
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    if workers == 1 or len(units) <= 1:
        summaries = []
        for family, method in _wrap_progress(
            units,
            show_progress=not args.no_progress,
            desc="Analyzing Exp3 units",
            unit="unit",
        ):
            summaries.append(
                _run_one_exp3_unit(
                    args,
                    sequence_table=sequence_table,
                    sequence_rows=sequence_rows,
                    exp3_dir=exp3_dir,
                    output_root=output_root,
                    family=family,
                    method=method,
                )
            )
        return summaries

    worker_args = argparse.Namespace(**vars(args))
    worker_args.no_progress = True
    payloads = [
        (worker_args, sequence_table, sequence_rows, exp3_dir, output_root, family, method)
        for family, method in units
    ]
    max_workers = min(workers, len(payloads))
    print(f"Running {len(payloads)} Exp3 units with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(
            _wrap_progress(
                executor.map(_run_one_exp3_unit_payload, payloads),
                show_progress=not args.no_progress,
                desc="Analyzing Exp3 units",
                unit="unit",
            )
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    sequence_table = args.sequence_table or _default_sequence_table()
    use_exp3 = args.from_exp3 or args.motif_table is None

    print(f"Sequence table (read-only input): {sequence_table}")
    print(f"Exp3 root (read-only input): {args.exp3_dir or _default_exp3_dir()}")
    print(f"Structure root (read-only input): {EXP4_STRUCTURE_INPUT_DIR}")
    print(f"Output root: {args.output_dir or _default_output_root()}")

    if use_exp3:
        summaries = _run_exp3_defaults(args, sequence_table=sequence_table)
        total = sum(int(summary["motif_instances_analyzed"]) for summary in summaries)
        print(f"Analyzed {total} motif instances across {len(summaries)} family/method units.")
        print(f"Wrote outputs to {args.output_dir or _default_output_root()}")
        return

    if args.structure_dir is None:
        raise SystemExit("--structure-dir is required when using --motif-table")

    output_dir = args.output_dir or (_default_output_root() / "custom")
    summary = run_exp4_motif_structure_analysis(
        sequence_table=sequence_table,
        motif_table=args.motif_table,
        structure_dir=args.structure_dir,
        output_dir=output_dir,
        motif_rows_override=None,
        predict_missing_structures=args.predict_missing_structures,
        prediction_backend=args.prediction_backend,
        num_random_controls=args.num_random_controls,
        seed=args.seed,
        contact_threshold=args.contact_threshold,
        long_range_min_separation=args.long_range_min_separation,
        low_plddt_cutoff=args.low_plddt_cutoff,
        resume=args.resume,
        rmsd_workers=args.rmsd_workers,
        show_progress=not args.no_progress,
        generate_figures=args.generate_figures,
        figure_formats=args.figure_formats,
    )
    print(f"Analyzed {summary['motif_instances_analyzed']} motif instances.")
    print(f"Wrote outputs to {summary['output_dir']}")


if __name__ == "__main__":
    main()
