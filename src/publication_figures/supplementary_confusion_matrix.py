"""Supplementary confusion-matrix figure for Exp1 classification results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .data import resolve_results_dir
    from .export import export_figure_bundle
    from .style import FAMILY_ORDER, apply_publication_style
except ImportError:  # pragma: no cover - used when running this file directly
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from publication_figures.data import resolve_results_dir
    from publication_figures.export import export_figure_bundle
    from publication_figures.style import FAMILY_ORDER, apply_publication_style


DEFAULT_STEM = "supplementary_figure_s1_confusion_matrix"


def _read_labeled_matrix(path: Path) -> pd.DataFrame:
    """Read a labeled confusion-matrix CSV with true labels in the first column."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if "true_label" not in table.columns:
        raise ValueError(f"{path} must include a true_label column")
    matrix = table.set_index("true_label")
    return matrix.apply(pd.to_numeric, errors="raise")


def _ordered_labels(*tables: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for table in tables:
        labels.extend(map(str, table.index))
        labels.extend(map(str, table.columns))
    unique = list(dict.fromkeys(labels))
    ordered = [label for label in FAMILY_ORDER if label in unique]
    ordered.extend(sorted(set(unique) - set(ordered)))
    return ordered


def _row_normalize(raw: pd.DataFrame) -> pd.DataFrame:
    values = raw.to_numpy(dtype=float)
    denominator = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        denominator,
        out=np.zeros_like(values, dtype=float),
        where=denominator != 0,
    )
    return pd.DataFrame(normalized, index=raw.index, columns=raw.columns)


def load_confusion_tables(
    exp1_dir: Path,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw and row-normalized confusion matrices for one method."""
    exp1_dir = Path(exp1_dir)
    raw = _read_labeled_matrix(exp1_dir / f"confusion_matrix_{method}.csv")
    normalized_path = exp1_dir / f"confusion_matrix_{method}_row_normalized.csv"
    normalized = (
        _read_labeled_matrix(normalized_path)
        if normalized_path.is_file()
        else _row_normalize(raw)
    )

    labels = _ordered_labels(raw, normalized)
    raw = raw.reindex(index=labels, columns=labels, fill_value=0)
    normalized = normalized.reindex(index=labels, columns=labels, fill_value=0.0)
    return raw, normalized


def build_cell_annotations(
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    threshold_fraction: float = 0.01,
    threshold_count: int = 5,
) -> pd.DataFrame:
    """Create sparse labels that show diagonal recall and non-trivial errors."""
    annotations = pd.DataFrame("", index=normalized.index, columns=normalized.columns)
    for true_label in normalized.index:
        for predicted_label in normalized.columns:
            fraction = float(normalized.loc[true_label, predicted_label])
            count = float(raw.loc[true_label, predicted_label])
            if true_label == predicted_label:
                annotations.loc[true_label, predicted_label] = f"{fraction:.0%}"
            elif fraction >= threshold_fraction or count >= threshold_count:
                annotations.loc[true_label, predicted_label] = (
                    f"{fraction:.1%}\n({int(round(count))})"
                )
    return annotations


def _source_data(
    raw_tables: dict[str, pd.DataFrame],
    normalized_tables: dict[str, pd.DataFrame],
    annotations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, raw in raw_tables.items():
        normalized = normalized_tables[method]
        method_annotations = annotations[method]
        for true_label in raw.index:
            for predicted_label in raw.columns:
                rows.append(
                    {
                        "method": method,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": raw.loc[true_label, predicted_label],
                        "row_fraction": normalized.loc[true_label, predicted_label],
                        "annotation": method_annotations.loc[
                            true_label,
                            predicted_label,
                        ],
                    }
                )
    return pd.DataFrame(rows)


def build_supplementary_confusion_figure(
    exp1_dir: Path,
    *,
    threshold_fraction: float = 0.01,
    threshold_count: int = 5,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Build a two-panel supplementary heatmap for M2 and M3."""
    apply_publication_style()
    raw_tables: dict[str, pd.DataFrame] = {}
    normalized_tables: dict[str, pd.DataFrame] = {}
    annotations: dict[str, pd.DataFrame] = {}

    for method in ("M2", "M3"):
        raw, normalized = load_confusion_tables(exp1_dir, method)
        raw_tables[method] = raw
        normalized_tables[method] = normalized
        annotations[method] = build_cell_annotations(
            raw,
            normalized,
            threshold_fraction=threshold_fraction,
            threshold_count=threshold_count,
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.85),
        gridspec_kw={"width_ratios": [1, 1, 0.045], "wspace": 0.08},
        constrained_layout=True,
    )
    heatmap_axes = axes[:2]
    colorbar_ax = axes[2]

    for index, method in enumerate(("M2", "M3")):
        ax = heatmap_axes[index]
        sns.heatmap(
            normalized_tables[method],
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.25,
            linecolor="white",
            annot=annotations[method],
            fmt="",
            annot_kws={"fontsize": 4.8},
            cbar=index == 1,
            cbar_ax=colorbar_ax if index == 1 else None,
            cbar_kws={"label": "Row-normalized fraction"},
        )
        ax.set_title(method)
        ax.set_xlabel("Predicted family")
        ax.set_ylabel("True family" if index == 0 else "")
        ax.tick_params(axis="x", rotation=90, pad=1.5)
        ax.tick_params(axis="y", rotation=0, pad=1.5)
        if index == 1:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

    colorbar_ax.tick_params(labelsize=6)
    colorbar_ax.set_ylabel("Row-normalized fraction", fontsize=6.5)

    plotted = _source_data(raw_tables, normalized_tables, annotations)
    return fig, plotted


def export_supplementary_confusion_figure(
    exp1_dir: Path,
    output_dir: Path,
    stem: str = DEFAULT_STEM,
    *,
    threshold_fraction: float = 0.01,
    threshold_count: int = 5,
) -> dict[str, Path]:
    """Export the supplementary confusion matrix and its source data."""
    fig, plotted = build_supplementary_confusion_figure(
        exp1_dir,
        threshold_fraction=threshold_fraction,
        threshold_count=threshold_count,
    )
    try:
        return export_figure_bundle(fig, plotted, output_dir, stem)
    finally:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redraw the Exp1 M2/M3 confusion matrices from saved CSV data."
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--exp1-dir",
        type=Path,
        help="Default: <results-dir>/exp1_classification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <results-dir>/publication_figures_optimized",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--threshold-fraction", type=float, default=0.01)
    parser.add_argument("--threshold-count", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    results_dir = resolve_results_dir(args.results_dir, project_root=project_root)
    exp1_dir = (
        Path(args.exp1_dir).resolve()
        if args.exp1_dir is not None
        else results_dir / "exp1_classification"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else results_dir / "publication_figures_optimized"
    )
    paths = export_supplementary_confusion_figure(
        exp1_dir,
        output_dir,
        args.stem,
        threshold_fraction=args.threshold_fraction,
        threshold_count=args.threshold_count,
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
