"""Publication figure for Exp3 motif-domain discovery."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import require_columns
from .style import (
    GRID,
    METHOD_COLORS,
    METHOD_MARKERS,
    add_panel_label,
    apply_publication_style,
    clean_axis,
    method_handles,
)


def prepare_exp3_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate clustering metrics and derive the retained-sample fraction."""
    required = [
        "family",
        "method",
        "n_motifs",
        "n_core_samples",
        "noise_ratio",
        "silhouette_final",
    ]
    require_columns(frame, required, "<dataframe>")
    result = frame.copy()
    numeric = [
        "n_motifs",
        "n_core_samples",
        "noise_ratio",
        "silhouette_final",
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["retained_fraction"] = (1 - result["noise_ratio"]).clip(0, 1)
    return result


def _paired_family_plot(
    ax,
    frame: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    log_scale: bool = False,
) -> None:
    pivot = frame.pivot(index="family", columns="method", values=column).sort_index()
    y = np.arange(len(pivot))
    for position, (_, row) in enumerate(pivot.iterrows()):
        if row.notna().sum() == 2:
            ax.plot(
                [row.get("M2"), row.get("M3")],
                [position, position],
                color="#C7C7C7",
                linewidth=0.8,
                zorder=1,
            )
    for method in ("M2", "M3"):
        if method not in pivot:
            continue
        ax.scatter(
            pivot[method],
            y,
            s=19,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
            label=method,
        )
    if log_scale:
        ax.set_xscale("log")
    ax.set_yticks(y, pivot.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    clean_axis(ax)


def _plot_quality_yield(ax, frame: pd.DataFrame) -> None:
    for method in ("M2", "M3"):
        subset = frame[frame["method"].eq(method)]
        ax.scatter(
            subset["n_motifs"],
            subset["silhouette_final"],
            s=25,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            label=method,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Discovered motifs (log scale)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Yield versus separation")
    ax.grid(color=GRID, linewidth=0.6)
    clean_axis(ax)


def build_exp3_figure(
    frame: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame, list[str]]:
    """Build four complementary views of motif discovery quality."""
    apply_publication_style()
    plotted = prepare_exp3_data(frame)
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.2))
    _paired_family_plot(
        axes[0, 0],
        plotted,
        "silhouette_final",
        "Cluster separation",
        "Silhouette score",
    )
    _paired_family_plot(
        axes[0, 1],
        plotted,
        "retained_fraction",
        "Retained sequence fraction",
        "Retained fraction",
    )
    _paired_family_plot(
        axes[1, 0],
        plotted,
        "n_motifs",
        "Recovered motif diversity",
        "Number of motifs",
        log_scale=True,
    )
    _plot_quality_yield(axes[1, 1], plotted)
    for ax, label in zip(axes.flat, "ABCD"):
        add_panel_label(ax, label)
    fig.legend(
        handles=method_handles(),
        labels=["M2", "M3"],
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.995),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, plotted, []
