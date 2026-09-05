"""Publication figure for Exp1 classification performance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import read_csv
from .style import (
    FAMILY_ORDER,
    GRID,
    METHOD_COLORS,
    METHOD_MARKERS,
    add_panel_label,
    apply_publication_style,
    clean_axis,
    method_handles,
)


def load_exp1_data(exp1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load overall and per-family classification metrics."""
    exp1_dir = Path(exp1_dir)
    summary = read_csv(
        exp1_dir / "exp1_results_summary.csv",
        ["method", "mean_acc", "std_acc", "mean_f1", "std_f1"],
    )
    overall = pd.concat(
        [
            summary[["method", "mean_acc", "std_acc"]]
            .rename(columns={"mean_acc": "value", "std_acc": "std"})
            .assign(metric="Accuracy"),
            summary[["method", "mean_f1", "std_f1"]]
            .rename(columns={"mean_f1": "value", "std_f1": "std"})
            .assign(metric="Macro F1"),
        ],
        ignore_index=True,
    )[["metric", "method", "value", "std"]]

    report_path = exp1_dir / "exp1_detailed_reports.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [
        {"family": family, "method": method, "f1": float(metrics["f1-score"])}
        for method in ("M2", "M3")
        for family, metrics in payload[method]["per_class"].items()
    ]
    return overall, pd.DataFrame(rows)


def load_confusion_matrices(
    exp1_dir: Path,
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Load numeric confusion matrices without reverse engineering image pixels."""
    exp1_dir = Path(exp1_dir)
    payload = json.loads(
        (exp1_dir / "exp1_detailed_reports.json").read_text(encoding="utf-8")
    )
    class_names = list(payload["M2"]["per_class"])
    matrices: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    for method in ("M2", "M3"):
        candidates = (
            exp1_dir / f"confusion_matrix_{method}.npy",
            exp1_dir / f"exp1_confusion_matrix_{method}.npy",
            exp1_dir / f"confusion_matrix_{method}.csv",
            exp1_dir / f"exp1_confusion_matrix_{method}.csv",
            exp1_dir / f"confusion_matrix_{method}.json",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            warnings.append(f"No numeric confusion matrix found for {method}")
            continue
        if source.suffix == ".npy":
            matrix = np.load(source)
        elif source.suffix == ".csv":
            table = pd.read_csv(source)
            if "true_label" in table.columns:
                matrix = (
                    table.set_index("true_label")
                    .reindex(index=class_names, columns=class_names)
                    .to_numpy(dtype=float)
                )
            else:
                matrix = pd.read_csv(source, header=None).to_numpy(dtype=float)
        else:
            matrix = np.asarray(
                json.loads(source.read_text(encoding="utf-8")),
                dtype=float,
            )
        expected = (len(class_names), len(class_names))
        if matrix.shape != expected:
            raise ValueError(f"{source} has shape {matrix.shape}; expected {expected}")
        denominator = matrix.sum(axis=1, keepdims=True)
        matrices[method] = np.divide(
            matrix,
            denominator,
            out=np.zeros_like(matrix, dtype=float),
            where=denominator != 0,
        )
    return matrices, class_names, warnings


def _plot_overall(ax, overall: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Macro F1"]
    offsets = {"M2": -0.10, "M3": 0.10}
    for method in ("M2", "M3"):
        subset = overall[overall["method"].eq(method)].set_index("metric")
        y = np.arange(len(metrics), dtype=float) + offsets[method]
        ax.errorbar(
            subset.loc[metrics, "value"],
            y,
            xerr=subset.loc[metrics, "std"],
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markersize=4.8,
            capsize=2.5,
            linewidth=1.0,
            label=method,
        )
    ax.set_yticks(range(len(metrics)), metrics)
    ax.set_xlim(0.944, 0.964)
    ax.set_xlabel("Cross-validated score")
    ax.set_title("Cross-validation")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    clean_axis(ax)


def _plot_family_f1(ax, family: pd.DataFrame) -> None:
    pivot = family.pivot(index="family", columns="method", values="f1")
    order = [family for family in FAMILY_ORDER if family in pivot.index]
    order += sorted(set(pivot.index) - set(order))
    pivot = pivot.loc[order]
    y = np.arange(len(pivot))
    for index, (_, row) in enumerate(pivot.iterrows()):
        if row.notna().sum() == 2:
            ax.plot(
                [row["M2"], row["M3"]],
                [index, index],
                color="#C7C7C7",
                linewidth=0.9,
                zorder=1,
            )
    for method in ("M2", "M3"):
        ax.scatter(
            pivot[method],
            y,
            s=21,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
            label=method,
        )
    ax.axvline(0.90, color=GRID, linestyle="--", linewidth=0.8)
    ax.set_yticks(y, pivot.index)
    ax.set_xlim(max(0, float(np.nanmin(pivot.to_numpy())) - 0.04), 1.01)
    ax.set_xlabel("Per-family F1")
    ax.set_title("Family-level F1")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    clean_axis(ax)


def build_exp1_figure(
    exp1_dir: Path,
) -> tuple[plt.Figure, pd.DataFrame, list[str]]:
    """Build the two-panel classification figure."""
    apply_publication_style()
    overall, family = load_exp1_data(exp1_dir)
    _, _, warnings = load_confusion_matrices(exp1_dir)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.7, 3.6),
        gridspec_kw={"width_ratios": [0.72, 1.55]},
    )
    _plot_overall(axes[0], overall)
    _plot_family_f1(axes[1], family)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.legend(
        handles=method_handles(),
        labels=["M2", "M3"],
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.985, 1.01),
    )
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])

    plotted = pd.concat(
        [
            overall.assign(panel="overall"),
            family.assign(panel="family_f1"),
        ],
        ignore_index=True,
        sort=False,
    )
    return fig, plotted, warnings
