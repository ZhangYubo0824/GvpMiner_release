"""Publication figure for Exp2 generated-sequence validation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    GRID,
    METHOD_COLORS,
    METHOD_MARKERS,
    NEUTRAL,
    add_panel_label,
    apply_publication_style,
    clean_axis,
    method_handles,
)


def prepare_exp2_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Add display labels while retaining unsupported targets as missing evidence."""
    result = frame.copy()
    result["display_status"] = np.where(
        result["evaluation_status"].eq("supported"),
        "Evaluable",
        "Not evaluable",
    )
    return result


def _family_order(frame: pd.DataFrame) -> list[str]:
    return sorted(
        frame["family"].dropna().unique(),
        key=lambda value: (len(str(value)), str(value)),
    )


def _plot_metric(ax, frame: pd.DataFrame, column: str, title: str) -> None:
    order = _family_order(frame)
    supported = frame[frame["display_status"].eq("Evaluable")]
    pivot = supported.pivot(index="family", columns="method", values=column).reindex(
        order
    )
    y = np.arange(len(order))
    for index, row in pivot.iterrows():
        position = order.index(index)
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
    ax.set_yticks(y, order)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Proportion")
    ax.set_title(title)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    clean_axis(ax)


def _plot_status(ax, frame: pd.DataFrame) -> None:
    order = _family_order(frame)
    status = (
        frame.groupby("family")["display_status"]
        .first()
        .reindex(order)
        .fillna("Not evaluable")
    )
    colors = ["#4D4D4D" if value == "Evaluable" else "#C7C7C7" for value in status]
    y = np.arange(len(order))
    ax.barh(y, np.ones(len(order)), color=colors, height=0.52)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Label-space coverage")
    ax.spines["bottom"].set_visible(False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4D4D4D", label="Evaluable"),
            plt.Rectangle((0, 0), 1, 1, color="#C7C7C7", label="Outside label space"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        fontsize=6.2,
    )
    clean_axis(ax)


def build_exp2_figure(
    frame: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame, list[str]]:
    """Build validation panels without treating unsupported labels as failures."""
    apply_publication_style()
    plotted = prepare_exp2_plot_data(frame)
    height = max(4.8, 0.18 * plotted["family"].nunique() + 1.6)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.9, height),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.15, 0.62]},
    )
    _plot_metric(
        axes[0],
        plotted,
        "predicted_target_ratio",
        "Target assignment",
    )
    _plot_metric(
        axes[1],
        plotted,
        "mean_target_probability",
        "Target probability",
    )
    _plot_status(axes[2], plotted)
    for ax, label in zip(axes, "ABC"):
        add_panel_label(ax, label)
    fig.legend(
        handles=method_handles(),
        labels=["M2", "M3"],
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.93, 1.00),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig, plotted, []
