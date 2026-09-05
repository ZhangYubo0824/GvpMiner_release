"""Generate paper-ready figures from an existing GvpMiner result snapshot."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from publication_figures.data import load_exp2_summary, resolve_results_dir
from publication_figures.exp1 import build_exp1_figure
from publication_figures.exp2 import build_exp2_figure
from publication_figures.exp3 import build_exp3_figure
from publication_figures.exp4 import build_exp4_figure
from publication_figures.exp5 import build_exp5_figure
from publication_figures.export import export_figure_bundle, write_manifest


STEMS = {
    "exp1": "figure_1_classification",
    "exp2": "figure_2_generation_validation",
    "exp3": "figure_3_motif_discovery",
    "exp4": "figure_4_structural_validation",
    "exp5": "figure_5_external_functional_enrichment",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready figures from existing GvpMiner results."
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <results-dir>/publication_figures",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(STEMS),
        default=list(STEMS),
    )
    return parser


def resolve_output_dir(results_dir: Path, explicit: Path | None) -> Path:
    return Path(explicit) if explicit is not None else results_dir / "publication_figures"


def _supported_exp1_families(results_dir: Path) -> set[str]:
    report_path = (
        results_dir / "exp1_classification" / "exp1_detailed_reports.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return set(payload["M2"]["per_class"]) & set(payload["M3"]["per_class"])


def _find_exp2_summary(results_dir: Path) -> Path:
    preferred = (
        results_dir
        / "exp2_generation_validation"
        / "summaries"
        / "exp3_generation_validation_summary.json"
    )
    if preferred.is_file():
        return preferred
    candidates = sorted(
        (results_dir / "exp2_generation_validation").rglob("*summary.json")
    )
    aggregate = [path for path in candidates if path.parent.name == "summaries"]
    if aggregate:
        return aggregate[0]
    raise FileNotFoundError("No aggregate Exp2 summary JSON found")


def _build_experiment(experiment: str, results_dir: Path):
    if experiment == "exp1":
        return build_exp1_figure(results_dir / "exp1_classification")
    if experiment == "exp2":
        frame = load_exp2_summary(
            _find_exp2_summary(results_dir),
            _supported_exp1_families(results_dir),
        )
        return build_exp2_figure(frame)
    if experiment == "exp3":
        path = results_dir / "exp3_domain_discovery" / "clustering_comparison.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        return build_exp3_figure(pd.read_csv(path))
    if experiment == "exp4":
        return build_exp4_figure(results_dir / "exp4_motif_structure_analysis")
    if experiment == "exp5":
        return build_exp5_figure(results_dir / "exp5_external_functional_enrichment")
    raise ValueError(f"Unsupported experiment: {experiment}")


def generate_selected(
    results_dir: Path,
    output_dir: Path,
    experiments: list[str],
) -> dict[str, object]:
    generated: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for experiment in experiments:
        try:
            fig, plotted, warnings = _build_experiment(experiment, results_dir)
            paths = export_figure_bundle(
                fig,
                plotted,
                output_dir,
                STEMS[experiment],
            )
            plt.close(fig)
            generated[experiment] = {
                "status": "generated",
                "files": [path.name for path in paths.values()],
                "warnings": warnings,
            }
        except (FileNotFoundError, ValueError) as error:
            skipped[experiment] = str(error)
    return {"generated": generated, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    results_dir = resolve_results_dir(args.results_dir, project_root=project_root)
    output_dir = resolve_output_dir(results_dir, args.output_dir).resolve()
    outcome = generate_selected(results_dir, output_dir, args.experiments)
    manifest = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        **outcome,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }
    manifest_path = write_manifest(output_dir, manifest)
    print(f"Wrote publication figure manifest: {manifest_path}")
    for experiment in args.experiments:
        if experiment in outcome["generated"]:
            print(f"{experiment}: generated")
        else:
            print(f"{experiment}: skipped - {outcome['skipped'][experiment]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
