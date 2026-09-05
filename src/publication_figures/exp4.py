"""Publication figure for Exp4 structural validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import read_csv, standardize_structural_effects
from .style import (
    GRID,
    METHOD_COLORS,
    METHOD_MARKERS,
    NEUTRAL,
    add_panel_label,
    apply_publication_style,
    clean_axis,
)


EVIDENCE_ORDER = [
    "strong",
    "moderate",
    "weak",
    "uncertain_low_plddt",
    "not_significant",
]
EVIDENCE_COLORS = {
    "strong": "#2166AC",
    "moderate": "#67A9CF",
    "weak": "#D1E5F0",
    "uncertain_low_plddt": "#FDAE61",
    "not_significant": "#D9D9D9",
}
EVIDENCE_LABELS = {
    "strong": "strong",
    "moderate": "moderate",
    "weak": "weak",
    "uncertain_low_plddt": "uncertain pLDDT",
    "not_significant": "not significant",
}
REPRESENTATIVE_FEATURES = [
    "contact_density",
    "relative_SASA",
    "motif_mean_pLDDT",
    "distance_to_protein_centroid",
]
DISPLAY_ENDPOINTS = [
    "contact_density",
    "relative_SASA",
    "long_range_contact_count",
    "motif_mean_pLDDT",
    "radius_of_gyration",
    "distance_to_protein_centroid",
    "surface_residue_fraction",
    "low_confidence_fraction",
]
ENDPOINT_LABELS = {
    "contact_density": "contact density",
    "relative_SASA": "relative SASA",
    "long_range_contact_count": "long-range contacts",
    "motif_mean_pLDDT": "mean pLDDT",
    "radius_of_gyration": "radius of gyration",
    "distance_to_protein_centroid": "centroid distance",
    "surface_residue_fraction": "surface residues",
    "low_confidence_fraction": "low-confidence fraction",
}
SIGNIFICANCE_THRESHOLD = 0.05
LOW_CONFIDENCE_PLDDT = 70.0


def load_exp4_statistics(exp4_dir: Path) -> pd.DataFrame:
    """Load structural statistics from all family-method directories."""
    parts: list[pd.DataFrame] = []
    for path in sorted(Path(exp4_dir).glob("*/*/structural_enrichment_results.csv")):
        method = path.parent.name
        if method not in {"M2", "M3"}:
            continue
        part = read_csv(
            path,
            [
                "motif_cluster",
                "feature",
                "effect_median_difference",
                "q_value",
            ],
        )
        parts.append(
            part.assign(family=path.parent.parent.name, method=method)
        )
    if not parts:
        raise FileNotFoundError(
            f"No structural_enrichment_results.csv files found below {exp4_dir}"
        )
    return pd.concat(parts, ignore_index=True)


def load_exp4_cluster_summaries(exp4_dir: Path) -> pd.DataFrame:
    """Load motif-level structure summaries produced by Exp4."""
    parts: list[pd.DataFrame] = []
    for path in sorted(Path(exp4_dir).glob("*/*/motif_cluster_summary.csv")):
        method = path.parent.name
        if method not in {"M2", "M3"}:
            continue
        try:
            part = read_csv(path, ["motif_cluster"])
        except ValueError:
            continue
        parts.append(part.assign(family=path.parent.parent.name, method=method))
    if not parts:
        return pd.DataFrame(
            columns=[
                "family",
                "method",
                "motif_cluster",
                "n_instances",
                "n_sequences",
                "mean_motif_pLDDT",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def _structural_evidence_strength(
    structural_significant: bool,
    min_q: float,
    n_proteins: float,
    mean_plddt: float,
) -> str:
    if not structural_significant:
        return "not_significant"
    if np.isfinite(mean_plddt) and mean_plddt < LOW_CONFIDENCE_PLDDT:
        return "uncertain_low_plddt"
    if np.isfinite(min_q) and min_q <= 0.01 and n_proteins >= 20:
        return "strong"
    if n_proteins >= 10:
        return "moderate"
    return "weak"


def derive_structural_evidence(
    statistics: pd.DataFrame,
    cluster_summaries: pd.DataFrame | None = None,
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
) -> pd.DataFrame:
    """Derive motif-level structural evidence directly from Exp4 outputs."""
    keys = ["family", "method", "motif_cluster"]
    work = statistics.copy()
    work["q_value"] = pd.to_numeric(work["q_value"], errors="coerce")
    work["n_proteins"] = pd.to_numeric(
        work.get("n_proteins", work.get("n_instances", pd.Series(index=work.index, dtype=float))),
        errors="coerce",
    )
    clusters = work[keys].drop_duplicates()
    significant = work[work["q_value"].le(significance_threshold)]
    significant_summary = (
        significant.groupby(keys, as_index=False)
        .agg(
            min_structural_q_value=("q_value", "min"),
            significant_endpoint_count=("feature", "nunique"),
            n_proteins_from_enrichment=("n_proteins", "max"),
        )
    )
    summaries = (
        cluster_summaries.copy()
        if cluster_summaries is not None
        else pd.DataFrame(columns=keys)
    )
    if not summaries.empty:
        clusters = pd.concat([clusters, summaries[keys]], ignore_index=True).drop_duplicates()
        summaries["n_instances"] = pd.to_numeric(
            summaries.get("n_instances", pd.Series(index=summaries.index, dtype=float)),
            errors="coerce",
        )
        summaries["n_sequences"] = pd.to_numeric(
            summaries.get("n_sequences", pd.Series(index=summaries.index, dtype=float)),
            errors="coerce",
        )
        plddt_source = (
            summaries["mean_motif_pLDDT"]
            if "mean_motif_pLDDT" in summaries
            else summaries.get("motif_mean_pLDDT", pd.Series(index=summaries.index, dtype=float))
        )
        summaries["mean_motif_pLDDT"] = pd.to_numeric(plddt_source, errors="coerce")
        motif_summary = (
            summaries.groupby(keys, as_index=False)
            .agg(
                n_summary_instances=("n_instances", "max"),
                n_sequences=("n_sequences", "max"),
                mean_motif_pLDDT=("mean_motif_pLDDT", "median"),
            )
        )
    else:
        motif_summary = pd.DataFrame(
            columns=[*keys, "n_summary_instances", "n_sequences", "mean_motif_pLDDT"]
        )

    result = (
        clusters.merge(significant_summary, on=keys, how="left")
        .merge(motif_summary, on=keys, how="left")
    )
    result["significant_endpoint_count"] = result["significant_endpoint_count"].fillna(0)
    result["structural_significant"] = result["significant_endpoint_count"].gt(0)
    result["n_structural_instances"] = result["n_proteins_from_enrichment"].combine_first(
        result.get("n_sequences")
    ).combine_first(result["n_summary_instances"])
    result["n_structural_instances"] = pd.to_numeric(
        result["n_structural_instances"],
        errors="coerce",
    ).fillna(0)
    result["min_structural_q_value"] = pd.to_numeric(
        result["min_structural_q_value"],
        errors="coerce",
    )
    result["mean_motif_pLDDT"] = pd.to_numeric(
        result["mean_motif_pLDDT"],
        errors="coerce",
    )
    result["support_ratio"] = np.where(
        pd.to_numeric(result["n_sequences"], errors="coerce").gt(0),
        result["n_structural_instances"]
        / pd.to_numeric(result["n_sequences"], errors="coerce"),
        np.nan,
    )
    result["structural_evidence_strength"] = result.apply(
        lambda row: _structural_evidence_strength(
            bool(row["structural_significant"]),
            float(row["min_structural_q_value"])
            if pd.notna(row["min_structural_q_value"])
            else np.nan,
            float(row["n_structural_instances"]),
            float(row["mean_motif_pLDDT"])
            if pd.notna(row["mean_motif_pLDDT"])
            else np.nan,
        ),
        axis=1,
    )
    return result[
        [
            "family",
            "method",
            "motif_cluster",
            "structural_evidence_strength",
            "structural_significant",
            "n_structural_instances",
            "support_ratio",
        ]
    ].sort_values(keys)


def aggregate_structural_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate motif-level effects into family-method endpoint summaries."""
    work = frame.copy()
    work["q_value"] = pd.to_numeric(work["q_value"], errors="coerce")
    work["effect_median_difference"] = pd.to_numeric(
        work["effect_median_difference"],
        errors="coerce",
    )
    grouped = (
        work.groupby(["family", "method", "feature"], as_index=False)
        .agg(
            median_effect=("effect_median_difference", "median"),
            significant_rate=("q_value", lambda values: (values < 0.05).mean()),
            n_motifs=("motif_cluster", "nunique"),
        )
    )
    standardized = standardize_structural_effects(
        grouped.rename(columns={"median_effect": "effect_median_difference"})
    )
    return standardized.rename(
        columns={"effect_median_difference": "median_effect"}
    )


def summarize_structural_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    """Calculate evidence-strength proportions per family and method."""
    work = evidence.copy()
    work["evidence_category"] = work["structural_evidence_strength"].fillna(
        "not_significant"
    )
    work.loc[
        ~work["evidence_category"].isin(EVIDENCE_ORDER),
        "evidence_category",
    ] = "not_significant"
    summary = (
        work.groupby(
            ["family", "method", "evidence_category"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_motifs"})
    )
    summary["proportion"] = summary["n_motifs"] / summary.groupby(
        ["family", "method"]
    )["n_motifs"].transform("sum")
    return summary


def summarize_method_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    """Aggregate structural evidence categories across all families."""
    work = evidence.copy()
    work["evidence_category"] = work["structural_evidence_strength"].fillna(
        "not_significant"
    )
    work.loc[
        ~work["evidence_category"].isin(EVIDENCE_ORDER),
        "evidence_category",
    ] = "not_significant"
    summary = (
        work.groupby(["method", "evidence_category"], as_index=False)
        .size()
        .rename(columns={"size": "n_motifs"})
    )
    summary["proportion"] = summary["n_motifs"] / summary.groupby("method")[
        "n_motifs"
    ].transform("sum")
    return summary


def summarize_family_support(evidence: pd.DataFrame) -> pd.DataFrame:
    """Summarize strong/moderate structural support by family and method."""
    work = evidence.copy()
    work["evidence_category"] = work["structural_evidence_strength"].fillna(
        "not_significant"
    )
    work["supported"] = work["evidence_category"].isin(["strong", "moderate"])
    return (
        work.groupby(["family", "method"], as_index=False)
        .agg(
            support_rate=("supported", "mean"),
            n_motifs=("motif_cluster", "nunique"),
        )
        .sort_values(["family", "method"])
    )


def summarize_endpoint_effects(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Compress endpoint-level effects into a method-level visual summary."""
    subset = endpoints[endpoints["feature"].isin(DISPLAY_ENDPOINTS)].copy()
    return (
        subset.groupby(["method", "feature"], as_index=False)
        .agg(
            median_standardized_effect=("standardized_effect", "median"),
            median_significant_rate=("significant_rate", "median"),
            n_family_methods=("family", "nunique"),
        )
        .sort_values(["method", "feature"])
    )


def select_representative_motifs(
    evidence: pd.DataFrame,
    limit: int = 4,
) -> pd.DataFrame:
    """Select reproducible, well-supported examples for distribution panels."""
    work = evidence.copy()
    significant = work["structural_significant"].astype(str).str.lower().isin(
        {"true", "1"}
    )
    instances = pd.to_numeric(work["n_structural_instances"], errors="coerce")
    eligible = work[
        work["structural_evidence_strength"].eq("strong")
        & significant
        & instances.ge(20)
    ].copy()
    eligible["support_ratio"] = pd.to_numeric(
        eligible["support_ratio"],
        errors="coerce",
    )
    eligible["n_structural_instances"] = instances.loc[eligible.index]
    eligible = eligible.sort_values(
        ["support_ratio", "n_structural_instances", "motif_cluster"],
        ascending=[False, False, True],
    )
    return (
        eligible.groupby(["family", "method"], as_index=False)
        .head(1)
        .head(limit)
        .reset_index(drop=True)
    )


def standardize_distribution_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Create comparable display values while preserving raw endpoint values."""
    result = frame.copy()

    def scale(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        center = numeric.median()
        deviations = (numeric - center).abs()
        mad = deviations.median()
        denominator = 1.4826 * mad
        if not np.isfinite(denominator) or denominator == 0:
            denominator = deviations.max()
        if not np.isfinite(denominator) or denominator == 0:
            denominator = 1.0
        return ((numeric - center) / denominator).clip(-4, 4)

    result["display_value"] = result.groupby(
        "feature",
        group_keys=False,
    )["value"].transform(scale)
    return result


def load_representative_distributions(
    exp4_dir: Path,
    selected: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load motif and position-matched control values for selected examples."""
    features = features or REPRESENTATIVE_FEATURES
    parts: list[pd.DataFrame] = []
    warnings: list[str] = []
    for (family, method), group in selected.groupby(["family", "method"]):
        method_dir = Path(exp4_dir) / family / method
        motif_path = method_dir / "motif_structure_features.csv"
        control_path = method_dir / "matched_control_features.csv"
        if not motif_path.is_file() or not control_path.is_file():
            warnings.append(
                f"No motif-control distribution tables for {family} {method}"
            )
            continue
        clusters = set(group["motif_cluster"])
        usecols_motif = ["motif_cluster", *features]
        usecols_control = ["motif_cluster", "control_type", *features]
        motif = pd.read_csv(motif_path, usecols=usecols_motif)
        control = pd.read_csv(control_path, usecols=usecols_control)
        motif = motif[motif["motif_cluster"].isin(clusters)]
        control = control[
            control["motif_cluster"].isin(clusters)
            & control["control_type"].eq("position_matched")
        ]
        for feature in features:
            motif_long = motif[["motif_cluster", feature]].rename(
                columns={feature: "value"}
            )
            motif_long["group"] = "Motif"
            control_long = control[["motif_cluster", feature]].rename(
                columns={feature: "value"}
            )
            control_long["group"] = "Matched control"
            combined = pd.concat([motif_long, control_long], ignore_index=True)
            combined["feature"] = feature
            combined["family"] = family
            combined["method"] = method
            parts.append(combined.dropna(subset=["value"]))
    if not parts:
        warnings.append("No representative motif-control distribution data available")
        return pd.DataFrame(), warnings
    return standardize_distribution_values(
        pd.concat(parts, ignore_index=True)
    ), warnings


def _plot_endpoint_bubbles(ax, endpoints: pd.DataFrame, method: str) -> None:
    subset = endpoints[
        endpoints["method"].eq(method)
        & endpoints["feature"].isin(DISPLAY_ENDPOINTS)
    ]
    if subset.empty:
        ax.text(0.5, 0.5, f"No {method} data", ha="center", va="center")
        ax.set_axis_off()
        return
    features = sorted(subset["feature"].unique())
    families = sorted(subset["family"].unique())
    x_lookup = {value: index for index, value in enumerate(features)}
    y_lookup = {value: index for index, value in enumerate(families)}
    scatter = ax.scatter(
        subset["feature"].map(x_lookup),
        subset["family"].map(y_lookup),
        s=18 + 150 * subset["significant_rate"],
        c=subset["standardized_effect"],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        edgecolors=np.where(subset["n_motifs"].lt(10), NEUTRAL, "white"),
        linewidths=0.6,
    )
    ax.set_xticks(range(len(features)), features, rotation=55, ha="right")
    ax.set_yticks(range(len(families)), families)
    ax.set_title(f"{method}: standardized structural effects")
    ax.grid(color=GRID, linewidth=0.4)
    return scatter


def _plot_evidence_composition(ax, summary: pd.DataFrame) -> None:
    labels = sorted(
        summary[["family", "method"]]
        .drop_duplicates()
        .assign(label=lambda x: x["family"] + "-" + x["method"])["label"]
    )
    left = np.zeros(len(labels))
    for category in EVIDENCE_ORDER:
        lookup = (
            summary[summary["evidence_category"].eq(category)]
            .assign(label=lambda x: x["family"] + "-" + x["method"])
            .set_index("label")["proportion"]
        )
        values = np.array([lookup.get(label, 0.0) for label in labels])
        ax.barh(
            labels,
            values,
            left=left,
            color=EVIDENCE_COLORS[category],
            height=0.72,
            label=category.replace("_", " "),
        )
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of motifs")
    ax.set_title("Structural evidence strength")
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
    )
    ax.grid(axis="x", color=GRID, linewidth=0.5)


def _plot_representative_distributions(ax, distributions: pd.DataFrame) -> None:
    if distributions.empty:
        ax.text(
            0.5,
            0.5,
            "Representative distributions unavailable",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return
    selected_features = (
        distributions.groupby("feature")["value"]
        .count()
        .sort_values(ascending=False)
        .head(2)
        .index
    )
    work = distributions[distributions["feature"].isin(selected_features)].copy()
    work["label"] = (
        work["family"]
        + "-"
        + work["method"]
        + "\n"
        + work["feature"].str.replace("_", " ")
    )
    labels = list(dict.fromkeys(work["label"]))
    positions: list[float] = []
    values: list[np.ndarray] = []
    colors: list[str] = []
    ticks: list[float] = []
    for index, label in enumerate(labels):
        subset = work[work["label"].eq(label)]
        for offset, group in [(-0.18, "Motif"), (0.18, "Matched control")]:
            positions.append(index + offset)
            values.append(
                pd.to_numeric(
                    subset.loc[subset["group"].eq(group), "display_value"],
                    errors="coerce",
                )
                .dropna()
                .to_numpy()
            )
            colors.append("#2166AC" if group == "Motif" else "#BDBDBD")
        ticks.append(index)
    valid = [
        (position, value, color)
        for position, value, color in zip(positions, values, colors)
        if len(value)
    ]
    boxes = ax.boxplot(
        [item[1] for item in valid],
        positions=[item[0] for item in valid],
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    for patch, item in zip(boxes["boxes"], valid):
        patch.set_facecolor(item[2])
        patch.set_alpha(0.75)
    ax.set_xticks(ticks, labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set_ylabel("Within-endpoint robust z-score")
    ax.set_title("Representative motif versus matched control")
    ax.grid(axis="y", color=GRID, linewidth=0.5)


def _plot_method_evidence(ax, summary: pd.DataFrame) -> None:
    methods = ["M2", "M3"]
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    for category in EVIDENCE_ORDER:
        lookup = summary[summary["evidence_category"].eq(category)].set_index(
            "method"
        )["proportion"]
        values = np.array([lookup.get(method, 0.0) for method in methods])
        ax.barh(
            y,
            values,
            left=left,
            color=EVIDENCE_COLORS[category],
            height=0.52,
            label=EVIDENCE_LABELS[category],
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
                    color="white" if category in {"strong", "moderate"} else "#333333",
                )
        left += values
    ax.set_yticks(y, methods)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Motif fraction")
    ax.set_title("Structural evidence composition")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
    )
    clean_axis(ax)


def _plot_family_support(ax, family_support: pd.DataFrame) -> None:
    pivot = family_support.pivot(
        index="family",
        columns="method",
        values="support_rate",
    )
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
    ax.set_xlabel("Strong/moderate motif fraction")
    ax.set_title("Support by Gvp family")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)


def _plot_endpoint_summary(ax, endpoint_summary: pd.DataFrame) -> None:
    methods = ["M2", "M3"]
    features = [feature for feature in DISPLAY_ENDPOINTS if feature in set(endpoint_summary["feature"])]
    x_lookup = {feature: index for index, feature in enumerate(features)}
    y_lookup = {method: index for index, method in enumerate(methods)}
    subset = endpoint_summary[
        endpoint_summary["method"].isin(methods)
        & endpoint_summary["feature"].isin(features)
    ].copy()
    scatter = ax.scatter(
        subset["feature"].map(x_lookup),
        subset["method"].map(y_lookup),
        s=72,
        c=subset["median_standardized_effect"],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(
        range(len(features)),
        [ENDPOINT_LABELS.get(feature, feature.replace("_", " ")) for feature in features],
        rotation=38,
        ha="right",
    )
    ax.set_yticks(range(len(methods)), methods)
    ax.set_ylim(-0.45, len(methods) - 0.55)
    ax.set_title("Structural endpoint effects")
    ax.grid(color=GRID, linewidth=0.4)
    clean_axis(ax)
    return scatter


def build_exp4_figure(
    exp4_dir: Path,
) -> tuple[plt.Figure, pd.DataFrame, list[str]]:
    """Build structural endpoint and evidence panels from Exp4 outputs."""
    apply_publication_style()
    statistics = load_exp4_statistics(exp4_dir)
    cluster_summaries = load_exp4_cluster_summaries(exp4_dir)
    evidence = derive_structural_evidence(statistics, cluster_summaries)
    endpoints = aggregate_structural_endpoints(statistics)
    composition = summarize_method_evidence(evidence)
    family_support = summarize_family_support(evidence)
    endpoint_summary = summarize_endpoint_effects(endpoints)
    warnings: list[str] = []

    fig = plt.figure(figsize=(7.1, 4.75))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.82], width_ratios=[1.0, 1.16])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    _plot_method_evidence(axes[0], composition)
    _plot_family_support(axes[1], family_support)
    scatter = _plot_endpoint_summary(axes[2], endpoint_summary)
    if scatter is not None:
        colorbar_ax = fig.add_axes([0.89, 0.17, 0.012, 0.22])
        fig.colorbar(
            scatter,
            cax=colorbar_ax,
            label="median standardized effect",
        )
    for ax, label in zip(axes, "ABC"):
        add_panel_label(ax, label)
    fig.subplots_adjust(
        left=0.10,
        right=0.86,
        top=0.965,
        bottom=0.20,
        wspace=0.40,
        hspace=0.64,
    )

    plotted = pd.concat(
        [
            endpoints.assign(panel="endpoints"),
            composition.assign(panel="evidence"),
            family_support.assign(panel="family_support"),
            endpoint_summary.assign(panel="endpoint_summary"),
        ],
        ignore_index=True,
        sort=False,
    )
    return fig, plotted, warnings
