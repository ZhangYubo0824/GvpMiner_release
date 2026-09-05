"""Publication figure for Exp5 external functional enrichment."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import collapse_function_terms, read_csv
from .style import (
    GRID,
    METHOD_COLORS,
    METHOD_MARKERS,
    add_panel_label,
    apply_publication_style,
    clean_axis,
)


EXTERNAL_STATUS_ORDER = [
    "external_enrichment_supported",
    "external_enrichment_not_significant",
    "external_filter_failed",
    "external_enrichment_no_terms",
    "external_enrichment_unavailable",
    "not_tested_external",
]
EXTERNAL_STATUS_LABELS = {
    "external_enrichment_supported": "supported",
    "external_enrichment_not_significant": "tested, not significant",
    "external_filter_failed": "filter failed",
    "external_enrichment_no_terms": "no external terms",
    "external_enrichment_unavailable": "unavailable",
    "not_tested_external": "not tested",
}
EXTERNAL_STATUS_COLORS = {
    "external_enrichment_supported": "#2166AC",
    "external_enrichment_not_significant": "#D9D9D9",
    "external_filter_failed": "#B2182B",
    "external_enrichment_no_terms": "#E0E0E0",
    "external_enrichment_unavailable": "#F0A35E",
    "not_tested_external": "#F2F2F2",
}
GENERIC_TERM_PATTERNS = (
    r"^COG:[A-Z]$",
    r"^uncharacterized protein$",
)


def load_external_enrichment(root: Path) -> pd.DataFrame:
    """Collect usable clusterProfiler CSV rows and standardize their schema."""
    required = [
        "ID",
        "Description",
        "p.adjust",
        "family",
        "method",
        "motif_cluster",
    ]
    root = Path(root)
    if root.is_file():
        candidates = [root]
    else:
        preferred = (
            root
            / "clusterprofiler"
            / "clusterprofiler_enrichment_significant.csv"
        )
        candidates = [preferred] if preferred.is_file() else sorted(root.rglob("*.csv"))
    parts: list[pd.DataFrame] = []
    for path in candidates:
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in required)
        except (pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
            continue
        if not set(required).issubset(frame.columns):
            continue
        parts.append(frame[required])
    if not parts:
        raise FileNotFoundError(f"No usable clusterProfiler CSVs found below {root}")
    result = (
        pd.concat(parts, ignore_index=True)
        .rename(
            columns={
                "ID": "term_id",
                "Description": "term",
                "p.adjust": "adjusted_p",
            }
        )
        .dropna(subset=["family", "method", "motif_cluster", "adjusted_p"])
    )
    result["adjusted_p"] = pd.to_numeric(result["adjusted_p"], errors="coerce")
    result = result.dropna(subset=["adjusted_p"])
    columns = [
        "term_id",
        "term",
        "adjusted_p",
        "family",
        "method",
        "motif_cluster",
    ]
    return result[columns].drop_duplicates()


def load_external_motif_universe(root: Path) -> pd.DataFrame:
    """Load motif clusters tested by Exp5 external enrichment."""
    path = Path(root) / "clusterprofiler" / "motif_sets_long.csv"
    required = ["family", "method", "motif_cluster"]
    if not path.is_file():
        return pd.DataFrame(columns=required)
    frame = pd.read_csv(path, usecols=lambda column: column in required)
    if not set(required).issubset(frame.columns):
        return pd.DataFrame(columns=required)
    return frame[required].dropna().drop_duplicates()


def load_external_enrichment_summary(root: Path) -> pd.DataFrame:
    """Load motif-level Exp5 enrichment summary when available."""
    path = Path(root) / "clusterprofiler" / "clusterprofiler_enrichment_summary.csv"
    required = [
        "family",
        "method",
        "motif_cluster",
        "significant_terms",
        "minimum_p_adjust",
    ]
    if not path.is_file():
        return pd.DataFrame(columns=required)
    frame = read_csv(path, required)
    return frame[required].drop_duplicates()


def _corrected_status_path(external_root: Path) -> Path:
    return (
        Path(external_root).parent
        / "motif_validation"
        / "cluster_commonality_validation"
        / "external_enrichment_status_recomputed.csv"
    )


def load_corrected_external_status(external_root: Path) -> pd.DataFrame:
    """Load the motif-level corrected external-enrichment status table."""
    path = _corrected_status_path(external_root)
    required = [
        "family",
        "method",
        "motif_cluster",
        "external_enrichment_status",
    ]
    if not path.is_file():
        return pd.DataFrame(columns=required)
    frame = read_csv(path, required)
    optional = [column for column in ("top_terms",) if column in frame.columns]
    return frame[required + optional].dropna(subset=required).drop_duplicates(
        required[:3]
    )


def load_gvpn_batch_status(
    external_root: Path,
    significance_threshold: float = 0.05,
) -> pd.DataFrame:
    """Derive corrected GvpN motif status from available batch CSV outputs."""
    batch_dir = Path(external_root) / "clusterprofiler" / "batch_results"
    required = ["family", "method", "motif_cluster", "p.adjust"]
    parts: list[pd.DataFrame] = []
    for method in ("M2", "M3"):
        path = batch_dir / f"GvpN_{method}.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path, usecols=lambda column: column in required)
        if not set(required).issubset(frame.columns):
            continue
        frame = frame[required].dropna(subset=["family", "method", "motif_cluster"])
        frame["p.adjust"] = pd.to_numeric(frame["p.adjust"], errors="coerce")
        grouped = (
            frame.groupby(["family", "method", "motif_cluster"], as_index=False)
            .agg(minimum_p_adjust=("p.adjust", "min"))
            .dropna(subset=["minimum_p_adjust"])
        )
        grouped["external_enrichment_status"] = np.where(
            grouped["minimum_p_adjust"].le(significance_threshold),
            "external_enrichment_supported",
            "external_enrichment_not_significant",
        )
        parts.append(
            grouped[
                [
                    "family",
                    "method",
                    "motif_cluster",
                    "external_enrichment_status",
                ]
            ]
        )
    if not parts:
        return pd.DataFrame(
            columns=[
                "family",
                "method",
                "motif_cluster",
                "external_enrichment_status",
            ]
        )
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["family", "method", "motif_cluster"]
    )


def apply_gvpn_batch_status(
    corrected: pd.DataFrame,
    gvpn_status: pd.DataFrame,
) -> pd.DataFrame:
    """Overlay actual GvpN batch results on the corrected status table."""
    if corrected.empty or gvpn_status.empty:
        return corrected
    keys = ["family", "method", "motif_cluster"]
    overlay = gvpn_status.rename(
        columns={"external_enrichment_status": "_gvpn_external_enrichment_status"}
    )
    result = corrected.merge(overlay, on=keys, how="outer")
    if "external_enrichment_status" not in result.columns:
        result["external_enrichment_status"] = np.nan
    result["external_enrichment_status"] = result[
        "_gvpn_external_enrichment_status"
    ].combine_first(result["external_enrichment_status"])
    return result.drop(columns=["_gvpn_external_enrichment_status"])


def derive_external_evidence(
    external_root: Path,
    significance_threshold: float = 0.05,
) -> pd.DataFrame:
    """Derive motif-level external enrichment status from Exp5 outputs."""
    keys = ["family", "method", "motif_cluster"]
    corrected = load_corrected_external_status(external_root)
    if not corrected.empty:
        corrected = apply_gvpn_batch_status(
            corrected,
            load_gvpn_batch_status(external_root, significance_threshold),
        )
        return corrected.sort_values(keys)

    universe = load_external_motif_universe(external_root)
    summary = load_external_enrichment_summary(external_root)
    if summary.empty:
        terms = load_external_enrichment(external_root)
        summary = (
            terms.groupby(keys, as_index=False)
            .agg(
                significant_terms=("term_id", "nunique"),
                minimum_p_adjust=("adjusted_p", "min"),
            )
        )
    if universe.empty:
        universe = summary[keys].drop_duplicates()
    evidence = universe.merge(summary, on=keys, how="left")
    evidence["significant_terms"] = pd.to_numeric(
        evidence["significant_terms"],
        errors="coerce",
    ).fillna(0)
    evidence["minimum_p_adjust"] = pd.to_numeric(
        evidence["minimum_p_adjust"],
        errors="coerce",
    )
    evidence["external_enrichment_status"] = np.where(
        evidence["significant_terms"].gt(0)
        & evidence["minimum_p_adjust"].le(significance_threshold),
        "external_enrichment_supported",
        "external_enrichment_not_significant",
    )
    return evidence[
        [
            "family",
            "method",
            "motif_cluster",
            "external_enrichment_status",
        ]
    ].sort_values(keys)


def _clean_term(term_id: str, description: str) -> str:
    if not description or description == term_id:
        return term_id
    return re.sub(
        r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s+",
        "",
        description,
    ).strip()


def format_term_label(value: str, width: int = 50) -> str:
    """Wrap long functional descriptions for a compact two-column figure."""
    return textwrap.fill(
        str(value),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def parse_evidence_top_terms(evidence: pd.DataFrame) -> pd.DataFrame:
    """Expand the saved top-term summaries without rereading gigabyte tables."""
    rows: list[dict[str, object]] = []
    for item in evidence.itertuples(index=False):
        raw = getattr(item, "top_terms", "")
        if pd.isna(raw) or not str(raw).strip():
            continue
        for term_spec in str(raw).split(";"):
            pieces = term_spec.split("|", 2)
            if len(pieces) != 3:
                continue
            term_id, description, adjusted = pieces
            adjusted_p = pd.to_numeric(adjusted, errors="coerce")
            if pd.isna(adjusted_p):
                continue
            rows.append(
                {
                    "term_id": term_id,
                    "term": _clean_term(term_id, description),
                    "adjusted_p": float(adjusted_p),
                    "family": getattr(item, "family"),
                    "method": getattr(item, "method"),
                    "motif_cluster": getattr(item, "motif_cluster"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "term_id",
            "term",
            "adjusted_p",
            "family",
            "method",
            "motif_cluster",
        ],
    )


def summarize_external_status(evidence: pd.DataFrame) -> pd.DataFrame:
    """Calculate external-enrichment status proportions."""
    summary = (
        evidence.groupby(
            ["family", "method", "external_enrichment_status"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_motifs"})
    )
    summary["proportion"] = summary["n_motifs"] / summary.groupby(
        ["family", "method"]
    )["n_motifs"].transform("sum")
    return summary


def summarize_method_external_status(coverage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external-enrichment status proportions by method."""
    summary = (
        coverage.groupby(["method", "external_enrichment_status"], as_index=False)[
            "n_motifs"
        ]
        .sum()
    )
    summary["proportion"] = summary["n_motifs"] / summary.groupby("method")[
        "n_motifs"
    ].transform("sum")
    return summary


def select_main_terms(
    collapsed: pd.DataFrame,
    limit: int = 16,
) -> pd.DataFrame:
    """Select nonredundant significant terms with broad motif support."""
    significant = collapsed[collapsed["min_adjusted_p"].lt(0.05)].copy()
    if significant.empty:
        return significant
    generic = significant["term"].astype(str).apply(
        lambda term: any(
            re.search(pattern, term, flags=re.IGNORECASE)
            for pattern in GENERIC_TERM_PATTERNS
        )
    )
    significant = significant[~generic].copy()
    if significant.empty:
        return significant
    ranking = (
        significant.groupby(["normalized_term", "term"], as_index=False)
        .agg(
            total_motif_support=("n_motif_clusters", "sum"),
            best_minus_log10_q=("minus_log10_q", "max"),
        )
        .sort_values(
            ["total_motif_support", "best_minus_log10_q", "term"],
            ascending=[False, False, True],
        )
        .head(limit)
    )
    selected = significant[
        significant["normalized_term"].isin(ranking["normalized_term"])
    ].copy()
    rank_lookup = {
        term: index
        for index, term in enumerate(ranking["normalized_term"].tolist())
    }
    selected["_rank"] = selected["normalized_term"].map(rank_lookup)
    return selected.sort_values(
        ["_rank", "family", "method"]
    ).drop(columns="_rank")


def _stacked_external_status_summary(ax, summary: pd.DataFrame) -> None:
    methods = ["M2", "M3"]
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for status in EXTERNAL_STATUS_ORDER:
        lookup = summary[summary["external_enrichment_status"].eq(status)].set_index(
            "method"
        )["proportion"]
        values = np.array([lookup.get(method, 0.0) for method in methods])
        if not np.any(values):
            continue
        ax.barh(
            y,
            values,
            left=left,
            height=0.52,
            color=EXTERNAL_STATUS_COLORS[status],
            label=EXTERNAL_STATUS_LABELS[status],
        )
        for position, value in enumerate(values):
            if value >= 0.08:
                ax.text(
                    left[position] + value / 2,
                    position,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=(
                        "white"
                        if status
                        in {"external_enrichment_supported", "external_filter_failed"}
                        else "#333333"
                    ),
                )
        left += values
    ax.set_yticks(y, methods)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Motif fraction")
    ax.set_title("Supported motif fraction by method")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
    )
    clean_axis(ax)


def _external_support_by_family(ax, coverage: pd.DataFrame) -> None:
    supported = coverage[
        coverage["external_enrichment_status"].eq("external_enrichment_supported")
    ]
    pivot = supported.pivot(
        index="family",
        columns="method",
        values="proportion",
    ).fillna(0.0)
    pivot["_mean"] = pivot[[column for column in ("M2", "M3") if column in pivot]].mean(
        axis=1
    )
    pivot = pivot.sort_values("_mean").drop(columns="_mean")
    y = np.arange(len(pivot))
    for position, (_, row) in enumerate(pivot.iterrows()):
        if row.notna().sum() == 2:
            ax.plot(
                [row["M2"], row["M3"]],
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
            s=20,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
            label=method,
        )
    ax.set_yticks(y, pivot.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Functionally supported motif fraction")
    ax.set_title("Family-level supported motif fraction")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)


def _term_support_summary(ax, terms: pd.DataFrame) -> None:
    if terms.empty:
        ax.text(0.5, 0.5, "No nonredundant significant external terms", ha="center", va="center")
        ax.set_axis_off()
        return None
    work = terms.copy()
    work["family_method"] = work["family"] + "-" + work["method"]
    summary = (
        work.groupby(["normalized_term", "term"], as_index=False)
        .agg(
            total_motif_support=("n_motif_clusters", "sum"),
            best_minus_log10_q=("minus_log10_q", "max"),
            n_family_methods=("family_method", "nunique"),
        )
        .sort_values(
            ["total_motif_support", "best_minus_log10_q", "term"],
            ascending=[True, True, False],
        )
    )
    y = np.arange(len(summary))
    ax.hlines(
        y,
        0,
        summary["total_motif_support"],
        color="#D0D0D0",
        linewidth=0.8,
        zorder=1,
    )
    scatter = ax.scatter(
        summary["total_motif_support"],
        y,
        s=22 + 18 * summary["n_family_methods"].clip(upper=10),
        c=summary["best_minus_log10_q"].clip(upper=20),
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    ax.set_yticks(
        y,
        [format_term_label(term, width=42) for term in summary["term"]],
    )
    ax.set_xlabel("Supporting candidate motif clusters")
    ax.set_title("Representative enriched functional terms")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    clean_axis(ax)
    size_handles = [
        ax.scatter(
            [],
            [],
            s=22 + 18 * count,
            color="#CFCFCF",
            edgecolor="white",
            linewidth=0.4,
            label=str(count),
        )
        for count in (2, 5, 10)
    ]
    ax.legend(
        handles=size_handles,
        title="family-methods",
        frameon=False,
        loc="lower right",
    )
    return scatter


def _stacked_status_plot(ax, summary: pd.DataFrame) -> None:
    labels = sorted(
        summary[["family", "method"]]
        .drop_duplicates()
        .assign(label=lambda x: x["family"] + "-" + x["method"])["label"]
    )
    statuses = list(summary["external_enrichment_status"].dropna().unique())
    palette = {
        "external_enrichment_supported": "#2166AC",
        "external_enrichment_not_significant": "#BDBDBD",
        "external_enrichment_unavailable": "#F0A35E",
        "external_filter_failed": "#B2182B",
        "external_no_terms": "#E0E0E0",
        "external_enrichment_no_terms": "#E0E0E0",
        "not_tested_external": "#F2F2F2",
    }
    labels_by_status = {
        "external_enrichment_supported": "supported",
        "external_enrichment_not_significant": "not significant",
        "external_enrichment_unavailable": "unavailable",
        "external_filter_failed": "filter failed",
        "external_no_terms": "no terms",
        "external_enrichment_no_terms": "no terms",
        "not_tested_external": "not tested",
    }
    left = np.zeros(len(labels))
    for status in statuses:
        lookup = (
            summary[summary["external_enrichment_status"].eq(status)]
            .assign(label=lambda x: x["family"] + "-" + x["method"])
            .set_index("label")["proportion"]
        )
        values = np.array([lookup.get(label, 0.0) for label in labels])
        ax.barh(
            labels,
            values,
            left=left,
            height=0.72,
            color=palette.get(status, "#D9D9D9"),
            label=labels_by_status.get(status, status.replace("_", " ")),
        )
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of motifs")
    ax.set_title("External functional-enrichment coverage")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
    )


def _term_bubble_plot(ax, terms: pd.DataFrame) -> None:
    if terms.empty:
        ax.text(0.5, 0.5, "No significant external terms", ha="center", va="center")
        ax.set_axis_off()
        return
    columns = sorted(
        (terms["family"] + "-" + terms["method"]).unique(),
        key=lambda value: (value.split("-")[0], value.split("-")[1]),
    )
    term_order = list(dict.fromkeys(terms["term"]))
    x_lookup = {value: index for index, value in enumerate(columns)}
    y_lookup = {value: index for index, value in enumerate(term_order)}
    labels = terms["family"] + "-" + terms["method"]
    scatter = ax.scatter(
        labels.map(x_lookup),
        terms["term"].map(y_lookup),
        s=18 + 22 * terms["n_motif_clusters"].clip(upper=12),
        c=terms["minus_log10_q"].clip(upper=20),
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(range(len(columns)), columns, rotation=90)
    ax.set_yticks(
        range(len(term_order)),
        [format_term_label(term) for term in term_order],
    )
    ax.set_title("Nonredundant supported functional terms")
    ax.grid(color=GRID, linewidth=0.4)
    plt.colorbar(scatter, ax=ax, label="-log10 adjusted p")


def build_exp5_figure(
    external_root: Path,
) -> tuple[plt.Figure, pd.DataFrame, list[str]]:
    """Build functional coverage and term panels from Exp5 outputs."""
    apply_publication_style()
    evidence = derive_external_evidence(external_root)
    coverage = summarize_external_status(evidence)
    method_status = summarize_method_external_status(coverage)
    parsed = parse_evidence_top_terms(evidence)
    if parsed.empty:
        parsed = load_external_enrichment(external_root)
    terms = (
        select_main_terms(collapse_function_terms(parsed), limit=9)
        if not parsed.empty
        else pd.DataFrame()
    )

    fig = plt.figure(figsize=(7.1, 5.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.45], width_ratios=[1.0, 1.12])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    _stacked_external_status_summary(axes[0], method_status)
    _external_support_by_family(axes[1], coverage)
    scatter = _term_support_summary(axes[2], terms)
    if scatter is not None:
        colorbar_ax = fig.add_axes([0.88, 0.145, 0.012, 0.23])
        fig.colorbar(scatter, cax=colorbar_ax, label="-log10 adjusted p")
    for ax, label in zip(axes, "ABC"):
        add_panel_label(ax, label)
    fig.subplots_adjust(
        left=0.22,
        right=0.84,
        top=0.965,
        bottom=0.12,
        wspace=0.43,
        hspace=0.78,
    )

    plotted = pd.concat(
        [
            method_status.assign(panel="external_status"),
            coverage.assign(panel="coverage"),
            terms.assign(panel="terms"),
        ],
        ignore_index=True,
        sort=False,
    )
    return fig, plotted, []
