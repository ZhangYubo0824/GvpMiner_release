"""Shared visual constants and Matplotlib defaults for publication figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


METHOD_COLORS = {"M2": "#2F6C99", "M3": "#D27A2C"}
METHOD_MARKERS = {"M2": "o", "M3": "s"}

NEUTRAL = "#9E9E9E"
GRID = "#D9D9D9"
POSITIVE = "#B2182B"
NEGATIVE = "#2166AC"

FAMILY_ORDER = (
    "GvpA",
    "GvpC",
    "GvpD",
    "GvpF",
    "GvpG",
    "GvpH",
    "GvpJ",
    "GvpK",
    "GvpL",
    "GvpM",
    "GvpN",
    "GvpO",
    "GvpQ",
    "GvpT",
    "GvpU",
    "GvpV",
)


def apply_publication_style() -> None:
    """Apply the shared publication rendering defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "figure.titlesize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": GRID,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(ax, label: str):
    """Add a bold panel label in axes-relative coordinates."""
    return ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axis(ax) -> None:
    """Apply common tick and grid polish after plotting."""
    ax.tick_params(axis="both", which="major", pad=2)


def method_handles():
    """Return reusable M2/M3 legend handles."""
    return [
        plt.Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=5,
            label=method,
        )
        for method in ("M2", "M3")
    ]
