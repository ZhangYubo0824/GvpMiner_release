from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .io import read_csv_rows, write_csv_rows, write_json
from .parsing import parse_structure_file, resolve_structure_path
from .structure_backends import get_structure_backend


Predictor = Callable[[str, str, Path], Path]


MANIFEST_FIELDS = [
    "sequence_id",
    "family",
    "sequence_length",
    "backend",
    "structure_path",
    "status",
    "global_mean_pLDDT",
    "error",
]


def _sequence_id(row: dict[str, object], index: int) -> str:
    return str(row.get("sequence_id") or row.get("protein_id") or row.get("seq_id") or f"seq_{index}")


def _selected_families(rows: Sequence[dict[str, object]], families: Sequence[str] | None) -> list[str]:
    if families:
        seen = set()
        selected = []
        for family in families:
            family = str(family)
            if family not in seen:
                selected.append(family)
                seen.add(family)
        return selected
    return sorted({str(row.get("family", "")) for row in rows if row.get("family")})


def _global_mean_plddt(path: Path) -> str:
    try:
        parsed = parse_structure_file(path)
    except Exception:
        return ""
    if len(parsed.plddt) == 0:
        return ""
    return str(float(np.mean(parsed.plddt)))


def _summarize_errors(rows: Sequence[dict[str, object]], *, limit: int = 5) -> list[dict[str, object]]:
    errors = [str(row.get("error", "")) for row in rows if row.get("error")]
    counts = Counter(errors)
    return [
        {"error": error, "count": count}
        for error, count in counts.most_common(limit)
    ]


def _wrap_progress(iterable, *, show_progress: bool, desc: str, unit: str):
    if not show_progress or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, dynamic_ncols=True)


def predict_family_structures(
    *,
    sequence_table: Path,
    structure_dir: Path,
    families: Sequence[str] | None = None,
    backend_name: str = "esmfold",
    predictor: Predictor | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    rows = read_csv_rows(Path(sequence_table))
    selected_families = _selected_families(rows, families)
    selected_set = set(selected_families)
    structure_dir = Path(structure_dir)
    structure_dir.mkdir(parents=True, exist_ok=True)
    backend = get_structure_backend(backend_name, predictor=predictor)

    manifest_rows: list[dict[str, object]] = []
    predicted = 0
    reused = 0
    failed = 0

    selected_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if not selected_set or str(row.get("family", "")) in selected_set
    ]

    for index, row in _wrap_progress(
        selected_rows,
        show_progress=show_progress,
        desc=f"Predicting {','.join(selected_families) or 'families'} structures",
        unit="seq",
    ):
        family = str(row.get("family", ""))
        sequence_id = _sequence_id(row, index)
        sequence = str(row.get("sequence", ""))
        sequence_length = str(row.get("sequence_length") or len(sequence))
        existing_path = resolve_structure_path(structure_dir, sequence_id)

        status = "reused"
        error = ""
        structure_path = existing_path
        if structure_path is None:
            try:
                structure_path = Path(backend.predict(sequence_id, sequence, structure_dir))
                status = "predicted"
            except Exception as exc:
                failed += 1
                status = "failed"
                error = str(exc)

        if status == "predicted":
            predicted += 1
        elif status == "reused":
            reused += 1

        manifest_rows.append(
            {
                "sequence_id": sequence_id,
                "family": family,
                "sequence_length": sequence_length,
                "backend": backend.name,
                "structure_path": str(structure_path or ""),
                "status": status,
                "global_mean_pLDDT": _global_mean_plddt(structure_path) if structure_path else "",
                "error": error,
            }
        )

    write_csv_rows(structure_dir / "structure_prediction_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    common_errors = _summarize_errors(manifest_rows)
    summary = {
        "sequence_table": str(sequence_table),
        "structure_dir": str(structure_dir),
        "backend": backend.name,
        "families": selected_families,
        "requested_sequences": len(manifest_rows),
        "predicted": predicted,
        "reused": reused,
        "failed": failed,
        "manifest": str(structure_dir / "structure_prediction_manifest.csv"),
        "first_error": common_errors[0]["error"] if common_errors else "",
        "common_errors": common_errors,
    }
    write_json(structure_dir / "structure_prediction_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict full-length Gvp family structures")
    parser.add_argument("--sequence-table", type=Path, default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--backend", default="esmfold")
    parser.add_argument("--structure-dir", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    return parser


try:
    from config import PREPROCESSED_DIR, STRUCTURES_DIR
except ImportError:
    _root = Path(__file__).resolve().parents[2]
    PREPROCESSED_DIR = _root / "work" / "results" / "preprocessed"
    STRUCTURES_DIR = _root / "work" / "results" / "structures"


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sequence_table() -> Path:
    return PREPROCESSED_DIR / "cleaned_gvp_dataset.csv"


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    sequence_table = args.sequence_table or _default_sequence_table()
    if args.structure_dir is not None:
        _run_single(args, sequence_table, args.structure_dir)
        return

    all_families = _selected_families(read_csv_rows(sequence_table), args.families)
    if not all_families:
        raise SystemExit("No families found in sequence table.")

    total = {"predicted": 0, "reused": 0, "failed": 0}
    for family in all_families:
        family_dir = STRUCTURES_DIR / args.backend / family
        summary = predict_family_structures(
            sequence_table=sequence_table,
            structure_dir=family_dir,
            families=[family],
            backend_name=args.backend,
            show_progress=not args.no_progress,
        )
        total["predicted"] += int(summary["predicted"])
        total["reused"] += int(summary["reused"])
        total["failed"] += int(summary["failed"])
        print(
            f"  {family}: predicted {summary['predicted']}, reused {summary['reused']}, "
            f"failed {summary['failed']} → {summary['structure_dir']}"
        )
    print(
        f"\nTotal across {len(all_families)} families: "
        f"predicted {total['predicted']}, reused {total['reused']}, failed {total['failed']}"
    )


def _run_single(args: argparse.Namespace, sequence_table: Path, structure_dir: Path) -> None:
    summary = predict_family_structures(
        sequence_table=sequence_table,
        structure_dir=structure_dir,
        families=args.families,
        backend_name=args.backend,
        show_progress=not args.no_progress,
    )
    print(
        "Predicted {predicted}, reused {reused}, failed {failed} full-length structures. "
        "Manifest: {manifest}".format(**summary)
    )
    if summary.get("first_error"):
        print(f"First/common error: {summary['first_error']}")
