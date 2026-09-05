"""Generate the v40 GvpMiner manuscript figures as true-vector SVG files."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "work" / "results" / "publication_figures"
QA_COMPANION_DIR = ROOT / "work" / "figure_qa"
PUBLICATION_DIR = ROOT / "results" / "publication_figures"
EXP1_DIR = ROOT / "results" / "exp1_classification"
PRECISION_YIELD_DIR = ROOT / "results" / "precision_yield"
ASSET_DIR = Path(__file__).resolve().parent / "assets"

EXPECTED_STEMS = (
    "figure_1_gvpminer_architecture",
    "figure_2_family_classification",
    "figure_3_generation_validation",
    "figure_4_candidate_clustering",
    "figure_5_structural_characterization",
    "figure_6_precision_yield",
    "supplementary_figure_s1_confusion_matrices",
    "supplementary_figure_s2_structural_profiles",
    "supplementary_figure_s3_topk_sensitivity",
)

DISPLAY_METHODS = {"M2": "ESM-2", "M3": "ProtBERT-BFD"}
METHOD_COLORS = {"M2": "#2F6C99", "M3": "#D27A2C"}
METHOD_MARKERS = {"M2": "o", "M3": "s"}
FAMILY_ORDER = (
    "GvpA", "GvpC", "GvpD", "GvpF", "GvpG", "GvpH", "GvpJ", "GvpK",
    "GvpL", "GvpM", "GvpN", "GvpO", "GvpQ", "GvpT", "GvpU", "GvpV",
)
DISPLAY_ENDPOINTS = (
    "contact_density",
    "relative_SASA",
    "long_range_contact_count",
    "motif_mean_pLDDT",
    "radius_of_gyration",
    "distance_to_protein_centroid",
    "surface_residue_fraction",
    "low_confidence_fraction",
)
ENDPOINT_LABELS = {
    "contact_density": "Contact\ndensity",
    "relative_SASA": "Relative\nSASA",
    "long_range_contact_count": "Long-range\ncontacts",
    "motif_mean_pLDDT": "Mean\npLDDT",
    "radius_of_gyration": "Radius of\ngyration",
    "distance_to_protein_centroid": "Centroid\ndistance",
    "surface_residue_fraction": "Surface\nresidues",
    "low_confidence_fraction": "Low-confidence\nfraction",
}
ALL_METHODS = ("GLAM2", "MEME", "M2", "MotifAE", "M3", "STREME")
ALL_METHOD_COLORS = {
    "GLAM2": "#7B8490",
    "MEME": "#C68A3A",
    "M2": "#2F6F9F",
    "MotifAE": "#8B6FB1",
    "M3": "#69A6C8",
    "STREME": "#5E9A78",
}
GRID = "#D9D9D9"
EXPORT_QA_COMPANIONS = False


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "gvpminer-v40",
            "pdf.fonttype": 42,
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _method_handles(methods: tuple[str, ...] = ("M2", "M3")) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="",
            marker=METHOD_MARKERS.get(method, "o"),
            markersize=4.5,
            color=METHOD_COLORS.get(method, ALL_METHOD_COLORS.get(method, "#777777")),
            label=DISPLAY_METHODS.get(method, method),
        )
        for method in methods
    ]


def validate_svg(path: Path) -> None:
    root = ET.parse(path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    view_box = root.attrib.get("viewBox", "").split()
    if len(view_box) != 4 or float(view_box[2]) <= 0 or float(view_box[3]) <= 0:
        raise ValueError(f"SVG lacks a positive viewBox: {path}")
    if not root.findall(".//svg:text", namespace):
        raise ValueError(f"SVG lacks editable text: {path}")
    if root.findall(".//svg:image", namespace):
        raise ValueError(f"Raster image embedded in SVG: {path}")


def _save_svg(figure: plt.Figure, stem: str) -> Path:
    path = OUTPUT_DIR / f"{stem}.svg"
    figure.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Date": None},
    )
    if EXPORT_QA_COMPANIONS:
        QA_COMPANION_DIR.mkdir(parents=True, exist_ok=True)
        figure.savefig(QA_COMPANION_DIR / f"{stem}.pdf", bbox_inches="tight")
        figure.savefig(
            QA_COMPANION_DIR / f"{stem}.tiff",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(figure)
    validate_svg(path)
    return path


def _normalize_architecture_svg() -> None:
    source = ASSET_DIR / "gvpminer_architecture.svg"
    destination = OUTPUT_DIR / "figure_1_gvpminer_architecture.svg"
    text = source.read_text(encoding="utf-8")
    text = re.sub(r'width="[^"]+"', 'width="7.5in"', text, count=1)
    text = re.sub(r'height="[^"]+"', 'height="5.35in"', text, count=1)
    text = re.sub(
        r'viewBox="[^"]+"',
        'viewBox="18 18 558 398"',
        text,
        count=1,
    )
    destination.write_text(text, encoding="utf-8", newline="\n")
    validate_svg(destination)


def _paired_family_plot(
    axis: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    *,
    log_scale: bool = False,
) -> None:
    pivot = frame.pivot(index="family", columns="method", values=column)
    order = [family for family in FAMILY_ORDER if family in pivot.index]
    pivot = pivot.reindex(order)
    y = np.arange(len(pivot))
    for position, (_, row) in enumerate(pivot.iterrows()):
        if row.notna().sum() == 2:
            axis.plot(
                [row["M2"], row["M3"]],
                [position, position],
                color="#C7C7C7",
                linewidth=0.8,
                zorder=1,
            )
    for method in ("M2", "M3"):
        axis.scatter(
            pivot[method],
            y,
            s=20,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
    if log_scale:
        axis.set_xscale("log")
    axis.set_yticks(y, order)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", color=GRID, linewidth=0.5)


def _figure2() -> None:
    frame = pd.read_csv(PUBLICATION_DIR / "figure_1_classification_data.csv")
    overall = frame[frame["panel"].eq("overall")].copy()
    family = frame[frame["panel"].eq("family_f1")].copy()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.45),
        gridspec_kw={"width_ratios": [0.75, 1.55]},
    )
    offsets = {"M2": -0.10, "M3": 0.10}
    metrics = ["Accuracy", "Macro F1"]
    for method in ("M2", "M3"):
        subset = overall[overall["method"].eq(method)].set_index("metric")
        y = np.arange(2) + offsets[method]
        axes[0].errorbar(
            subset.loc[metrics, "value"],
            y,
            xerr=subset.loc[metrics, "std"],
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markersize=4.5,
            capsize=2.5,
            linewidth=1,
        )
    axes[0].set_yticks(range(2), metrics)
    axes[0].set_xlim(0.944, 0.964)
    axes[0].set_xlabel("Cross-validated score")
    axes[0].set_title("Cross-validation")
    axes[0].grid(axis="x", color=GRID, linewidth=0.5)
    _paired_family_plot(
        axes[1],
        family,
        "f1",
        "Family-level F1",
        "Per-family F1",
    )
    axes[1].axvline(0.90, color=GRID, linestyle="--", linewidth=0.8)
    axes[1].set_xlim(max(0, family["f1"].min() - 0.04), 1.01)
    for axis, label in zip(axes, "AB"):
        _panel_label(axis, label)
    figure.legend(
        handles=_method_handles(),
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.985, 1.0),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_svg(figure, "figure_2_family_classification")


def _figure3() -> None:
    frame = pd.read_csv(PUBLICATION_DIR / "figure_2_generation_validation_data.csv")
    order = sorted(frame["family"].unique(), key=lambda value: (len(value), value))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 5.15),
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1.2, 0.62]},
    )
    evaluable = frame[frame["display_status"].eq("Evaluable")]
    for axis, column, title in zip(
        axes[:2],
        ("predicted_target_ratio", "mean_target_probability"),
        ("Target assignment", "Target probability"),
    ):
        pivot = evaluable.pivot(index="family", columns="method", values=column).reindex(order)
        y = np.arange(len(order))
        for position, (_, row) in enumerate(pivot.iterrows()):
            if row.notna().sum() == 2:
                axis.plot(
                    [row["M2"], row["M3"]],
                    [position, position],
                    color="#C7C7C7",
                    linewidth=0.8,
                )
        for method in ("M2", "M3"):
            axis.scatter(
                pivot[method],
                y,
                s=18,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        axis.set_xlim(-0.02, 1.02)
        axis.set_xlabel("Proportion")
        axis.set_title(title)
        axis.grid(axis="x", color=GRID, linewidth=0.5)
    axes[0].set_yticks(np.arange(len(order)), order)
    status = frame.groupby("family")["display_status"].first().reindex(order)
    axes[2].barh(
        np.arange(len(order)),
        np.ones(len(order)),
        color=["#4D4D4D" if item == "Evaluable" else "#C7C7C7" for item in status],
        height=0.52,
    )
    axes[2].set_xlim(0, 1)
    axes[2].set_xticks([])
    axes[2].set_title("Label-space coverage")
    for axis, label in zip(axes, "ABC"):
        _panel_label(axis, label)
    figure.legend(
        handles=_method_handles(),
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.89, 0.995),
    )
    figure.text(
        0.795,
        0.02,
        "Dark: evaluable\nLight: outside label space",
        fontsize=6.2,
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save_svg(figure, "figure_3_generation_validation")


def _figure4() -> None:
    frame = pd.read_csv(PUBLICATION_DIR / "figure_3_motif_discovery_data.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    for axis, column, title, xlabel, log_scale in zip(
        axes,
        ("silhouette_final", "retained_fraction", "n_motifs"),
        ("Cluster separation", "Retained window fraction", "Candidate motif-unit count"),
        ("Silhouette score", "Retained fraction", "Candidate motif units"),
        (False, False, True),
    ):
        _paired_family_plot(axis, frame, column, title, xlabel, log_scale=log_scale)
        axis.margins(x=0.05)
    for axis, label in zip(axes, "ABC"):
        _panel_label(axis, label)
    figure.legend(
        handles=_method_handles(),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.78, bottom=0.22, wspace=0.55)
    _save_svg(figure, "figure_4_candidate_clustering")


def _figure5_a(axis: plt.Axes, evidence: pd.DataFrame) -> None:
    order = ("strong", "moderate", "weak", "uncertain_low_plddt", "not_significant")
    labels = {
        "strong": "strong",
        "moderate": "moderate",
        "weak": "weak",
        "uncertain_low_plddt": "uncertain pLDDT",
        "not_significant": "not significant",
    }
    colors = {
        "strong": "#2166AC",
        "moderate": "#67A9CF",
        "weak": "#D1E5F0",
        "uncertain_low_plddt": "#FDAE61",
        "not_significant": "#D9D9D9",
    }
    summary = evidence.groupby(["method", "evidence_category"], as_index=False)["n_motifs"].sum()
    summary["proportion"] = summary["n_motifs"] / summary.groupby("method")["n_motifs"].transform("sum")
    left = np.zeros(2)
    for category in order:
        lookup = summary[summary["evidence_category"].eq(category)].set_index("method")["proportion"]
        values = np.array([lookup.get(method, 0.0) for method in ("M2", "M3")])
        axis.barh(np.arange(2), values, left=left, color=colors[category], height=0.52, label=labels[category])
        for position, value in enumerate(values):
            if value >= 0.08:
                axis.text(
                    left[position] + value / 2,
                    position,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if category in {"strong", "moderate"} else "#333333",
                )
        left += values
    axis.set_yticks(range(2), [DISPLAY_METHODS["M2"], DISPLAY_METHODS["M3"]])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Candidate motif-unit fraction")
    axis.set_title("Structural evidence composition")
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.invert_yaxis()
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))


def _figure5_b(axis: plt.Axes, evidence: pd.DataFrame) -> None:
    work = evidence.copy()
    work["supported"] = work["evidence_category"].isin(("strong", "moderate"))
    family = (
        work.groupby(["family", "method"], as_index=False)
        .apply(lambda group: pd.Series({"support_rate": np.average(group["supported"], weights=group["n_motifs"])}), include_groups=False)
        .reset_index(drop=True)
    )
    pivot = family.pivot(index="family", columns="method", values="support_rate")
    pivot["_mean"] = pivot[["M2", "M3"]].mean(axis=1)
    pivot = pivot.sort_values("_mean").drop(columns="_mean")
    y = np.arange(len(pivot))
    for position, (_, row) in enumerate(pivot.iterrows()):
        axis.plot([row["M2"], row["M3"]], [position, position], color="#C7C7C7", linewidth=0.8)
    for method in ("M2", "M3"):
        axis.scatter(
            pivot[method], y, s=20, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
            edgecolor="white", linewidth=0.4, label=DISPLAY_METHODS[method], zorder=3,
        )
    axis.set_yticks(y, pivot.index)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Strong/moderate motif-unit fraction")
    axis.set_title("Support by Gvp family")
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.legend(frameon=False, loc="lower right")


def _figure5_c(axis: plt.Axes, endpoints: pd.DataFrame) -> None:
    work = endpoints[endpoints["feature"].isin(DISPLAY_ENDPOINTS)].copy()
    rows: list[dict[str, object]] = []
    for feature in DISPLAY_ENDPOINTS:
        numeric = pd.to_numeric(
            work.loc[work["feature"].eq(feature), "median_effect"],
            errors="coerce",
        )
        before_count = len(numeric)
        values = numeric.dropna()
        after_count = len(values)
        if after_count != before_count:
            raise ValueError(
                f"Figure 5C contains {before_count - after_count} missing values for {feature}"
            )
        center = float(values.median())
        scale = 1.4826 * float((values - center).abs().median())
        if not np.isfinite(scale) or scale <= 0:
            if values.nunique() == 1:
                scale = 1.0
            else:
                raise ValueError(f"Figure 5C robust scale is invalid for {feature}")
        for method in ("M2", "M3"):
            subset = work[work["feature"].eq(feature) & work["method"].eq(method)]
            raw = float(subset["median_effect"].median())
            display = float(np.clip(raw / scale, -3, 3) / 3)
            rows.append(
                {
                    "method": method,
                    "display_method": DISPLAY_METHODS[method],
                    "feature": feature,
                    "raw_median_effect": raw,
                    "robust_scale": scale,
                    "display_effect": display,
                    "significant_fraction": float(subset["significant_rate"].median()),
                    "n_families": int(subset["family"].nunique()),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT_DIR / "figure_5c_audit.csv", index=False, lineterminator="\n")
    x_lookup = {feature: index for index, feature in enumerate(DISPLAY_ENDPOINTS)}
    y_lookup = {method: index for index, method in enumerate(("M2", "M3"))}
    scatter = axis.scatter(
        audit["feature"].map(x_lookup),
        audit["method"].map(y_lookup),
        s=28 + 180 * audit["significant_fraction"],
        c=audit["display_effect"],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        edgecolor="#555555",
        linewidth=0.65,
        zorder=3,
    )
    axis.set_xticks(range(8), [ENDPOINT_LABELS[item] for item in DISPLAY_ENDPOINTS])
    axis.set_yticks(range(2), [DISPLAY_METHODS["M2"], DISPLAY_METHODS["M3"]])
    axis.set_ylim(1.48, -0.48)
    axis.set_title("Structural endpoint effects")
    axis.grid(color=GRID, linewidth=0.45)
    rates = (0.25, 0.50, 0.75, 1.00)
    handles = [
        axis.scatter([], [], s=28 + 180 * rate, facecolor="#B7B7B7", edgecolor="#555555", linewidth=0.65, label=f"{rate:.0%}")
        for rate in rates
    ]
    axis.legend(
        handles=handles,
        title="Significant-test fraction",
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.43, -0.31),
        handletextpad=0.15,
        columnspacing=0.65,
    )
    return scatter


def _figure5() -> None:
    frame = pd.read_csv(PUBLICATION_DIR / "figure_4_structural_validation_data.csv")
    evidence = frame[frame["panel"].eq("evidence")].copy()
    endpoints = frame[frame["panel"].eq("endpoints")].copy()
    figure = plt.figure(figsize=(7.2, 5.55))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.02, 0.82), hspace=0.68, wspace=0.42)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, :])]
    _figure5_a(axes[0], evidence)
    _figure5_b(axes[1], evidence)
    scatter = _figure5_c(axes[2], endpoints)
    colorbar_axis = figure.add_axes([0.88, 0.115, 0.015, 0.20])
    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Sign-preserving robust standardized effect", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=6)
    for axis, label in zip(axes, "ABC"):
        _panel_label(axis, label)
    figure.subplots_adjust(left=0.10, right=0.85, top=0.96, bottom=0.20)
    _save_svg(figure, "figure_5_structural_characterization")


def _load_precision_yield_summary() -> pd.DataFrame:
    summary = pd.read_csv(PRECISION_YIELD_DIR / "method_yield_summary.csv")
    topk = pd.read_csv(PRECISION_YIELD_DIR / "topk_sensitivity.csv")
    top10 = topk[topk["top_k_requested"].eq(10)].rename(
        columns={
            "selected_motif_count": "top10_selected_motif_count",
            "supported_motif_count": "top10_supported_motif_count",
            "supported_fraction_selected": "top10_supported_fraction_selected",
            "supported_fraction_selected_ci_low": "top10_ci_low",
            "supported_fraction_selected_ci_high": "top10_ci_high",
        }
    )
    return summary.merge(
        top10[
            [
                "method",
                "top10_selected_motif_count",
                "top10_supported_motif_count",
                "top10_supported_fraction_selected",
                "top10_ci_low",
                "top10_ci_high",
            ]
        ],
        on="method",
        validate="one_to_one",
    ).set_index("method").loc[list(ALL_METHODS)].reset_index()


def _figure6() -> None:
    summary = _load_precision_yield_summary()
    summary.to_csv(OUTPUT_DIR / "figure_6_source_data.csv", index=False, lineterminator="\n")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.65), gridspec_kw={"width_ratios": (1.02, 1.15)})
    y_positions = np.arange(len(summary))[::-1]
    stages = (
        ("candidate_motifs", "All candidates", "#C9CED4", "o"),
        ("testable_motifs", "Annotation-testable", "#7D9FB7", "s"),
        ("supported_motifs", "Enrichment-supported", "#244F6B", "D"),
    )
    for y, row in zip(y_positions, summary.itertuples()):
        values = [float(getattr(row, item[0])) for item in stages]
        axes[0].plot(values, [y] * 3, color="#D9DDE1", linewidth=1.1)
        for index, (value, (_, _, color, marker)) in enumerate(zip(values, stages)):
            axes[0].scatter(value, y, s=24, color=color, marker=marker, zorder=3)
            axes[0].annotate(
                f"{int(value):,}",
                (value, y),
                xytext=(4, (-9, 0, 9)[index]),
                textcoords="offset points",
                fontsize=6.0,
                va="center",
                color="#374151",
            )
    axes[0].set_xscale("log")
    axes[0].set_yticks(y_positions, [DISPLAY_METHODS.get(item, item) for item in summary["method"]])
    axes[0].set_xlabel("Candidate motif units (log scale)")
    axes[0].set_title("Candidate conversion", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axes[0].legend(
        handles=[Line2D([], [], linestyle="", marker=marker, color=color, label=label) for _, label, color, marker in stages],
        loc="lower right",
        frameon=False,
        fontsize=6.0,
    )
    for row in summary.itertuples():
        x = float(row.top10_supported_fraction_selected)
        y = float(row.supported_motifs)
        lower = float(row.top10_ci_low)
        upper = float(row.top10_ci_high)
        method = str(row.method)
        axes[1].errorbar(
            x, y, xerr=[[x - lower], [upper - x]], fmt="none", ecolor=ALL_METHOD_COLORS[method],
            linewidth=1.0, capsize=2.2, alpha=0.9,
        )
        axes[1].scatter(
            x, y, s=75 + 300 * float(row.testable_fraction), color=ALL_METHOD_COLORS[method],
            edgecolor="white", linewidth=0.7, zorder=3,
        )
        axes[1].annotate(
            DISPLAY_METHODS.get(method, method),
            (x, y),
            xytext=(6, {"M2": 8, "M3": -13}.get(method, 6)),
            textcoords="offset points",
            fontsize=6.2,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlim(0, 0.88)
    axes[1].set_xlabel("Top-10 enrichment-supported / selected")
    axes[1].set_ylabel("Total enrichment-supported motif units (log scale)")
    axes[1].set_title("Precision-yield landscape", loc="left", fontweight="bold")
    axes[1].grid(color="#E5E7EB", linewidth=0.6)
    axes[1].text(
        0.02, 0.02, "Horizontal bars: family-cluster bootstrap 95% CI\nPoint area: annotation-testable fraction",
        transform=axes[1].transAxes, fontsize=6.0, color="#4B5563", va="bottom",
    )
    for axis, label in zip(axes, "AB"):
        _panel_label(axis, label)
    figure.subplots_adjust(left=0.105, right=0.985, top=0.88, bottom=0.19, wspace=0.35)
    _save_svg(figure, "figure_6_precision_yield")


def _read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.set_index("true_label").apply(pd.to_numeric, errors="raise")


def _s1() -> None:
    raw_tables: dict[str, pd.DataFrame] = {}
    normalized_tables: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    for method in ("M2", "M3"):
        raw = _read_matrix(EXP1_DIR / f"confusion_matrix_{method}.csv").reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER)
        normalized = _read_matrix(EXP1_DIR / f"confusion_matrix_{method}_row_normalized.csv").reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER)
        raw_tables[method] = raw
        normalized_tables[method] = normalized
        for true_label in FAMILY_ORDER:
            for predicted_label in FAMILY_ORDER:
                fraction = float(normalized.loc[true_label, predicted_label])
                count = float(raw.loc[true_label, predicted_label])
                annotation = ""
                if true_label == predicted_label:
                    annotation = f"{fraction:.0%}"
                elif fraction >= 0.01 or count >= 5:
                    annotation = f"{fraction:.1%}\n({int(round(count))})"
                audit_rows.append(
                    {
                        "method": method,
                        "display_method": DISPLAY_METHODS[method],
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                        "row_fraction": fraction,
                        "annotation": annotation,
                    }
                )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUTPUT_DIR / "supplementary_figure_s1_audit.csv", index=False, lineterminator="\n")
    figure = plt.figure(figsize=(7.2, 7.4))
    grid = figure.add_gridspec(2, 2, width_ratios=(1, 0.045), hspace=0.28, wspace=0.10)
    cmap = mpl.colormaps["Blues"]
    for row_index, method in enumerate(("M2", "M3")):
        axis = figure.add_subplot(grid[row_index, 0])
        matrix = normalized_tables[method]
        method_audit = audit[audit["method"].eq(method)].set_index(["true_label", "predicted_label"])
        for y, true_label in enumerate(FAMILY_ORDER):
            for x, predicted_label in enumerate(FAMILY_ORDER):
                value = float(matrix.loc[true_label, predicted_label])
                axis.add_patch(Rectangle((x, y), 1, 1, facecolor=cmap(value), edgecolor="white", linewidth=0.25))
                annotation = str(method_audit.loc[(true_label, predicted_label), "annotation"])
                if annotation:
                    axis.text(
                        x + 0.5,
                        y + 0.5,
                        annotation,
                        ha="center",
                        va="center",
                        fontsize=6.0,
                        color="white" if value >= 0.58 else "#222222",
                        linespacing=0.82,
                    )
        axis.set_xlim(0, 16)
        axis.set_ylim(16, 0)
        axis.set_aspect("equal")
        axis.set_xticks(np.arange(16) + 0.5, FAMILY_ORDER, rotation=90)
        axis.set_yticks(np.arange(16) + 0.5, FAMILY_ORDER)
        axis.set_xlabel("Predicted family")
        axis.set_ylabel("True family")
        axis.set_title(DISPLAY_METHODS[method], fontweight="bold")
        for spine in axis.spines.values():
            spine.set_visible(False)
        _panel_label(axis, "A" if row_index == 0 else "B")
    colorbar_axis = figure.add_subplot(grid[:, 1])
    steps = 40
    for index in range(steps):
        value = index / (steps - 1)
        colorbar_axis.add_patch(Rectangle((0, value), 1, 1 / steps + 0.002, facecolor=cmap(value), edgecolor="none"))
    colorbar_axis.set_xlim(0, 1)
    colorbar_axis.set_ylim(0, 1)
    colorbar_axis.set_xticks([])
    colorbar_axis.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    colorbar_axis.yaxis.tick_right()
    colorbar_axis.set_ylabel("Row-normalized fraction", rotation=270, labelpad=13)
    colorbar_axis.yaxis.set_label_position("right")
    figure.subplots_adjust(left=0.10, right=0.91, top=0.965, bottom=0.08)
    _save_svg(figure, "supplementary_figure_s1_confusion_matrices")


def _s2() -> None:
    source = PUBLICATION_DIR / "supplementary_figure_s2_structural_method_comparison_data.csv"
    frame = pd.read_csv(source)
    x_lookup = {feature: index for index, feature in enumerate(DISPLAY_ENDPOINTS)}
    y_order = ("M2", "M3", "GLAM2", "MEME", "MotifAE", "STREME")
    y_lookup = {method: index for index, method in enumerate(y_order)}
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    scatter = axis.scatter(
        frame["feature"].map(x_lookup),
        frame["method"].map(y_lookup),
        s=24 + 190 * frame["significant_rate"].clip(0, 1),
        c=frame["scaled_effect"],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        edgecolor="#555555",
        linewidth=0.55,
        zorder=3,
    )
    axis.set_xticks(range(8), [ENDPOINT_LABELS[item] for item in DISPLAY_ENDPOINTS])
    axis.set_yticks(range(6), [DISPLAY_METHODS.get(item, item) for item in y_order])
    axis.set_ylim(5.5, -0.5)
    axis.axhline(1.5, color="#AFAFAF", linewidth=0.9)
    axis.set_title("Structural endpoint profiles across candidate-region methods")
    axis.grid(color=GRID, linewidth=0.45)
    for label, method in zip(axis.get_yticklabels(), y_order):
        label.set_color("#2F6C99" if method in {"M2", "M3"} else "#D27A2C")
        label.set_fontweight("bold")
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.025, fraction=0.035)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Endpoint-scaled median effect")
    rates = (0.25, 0.50, 0.75, 1.00)
    handles = [
        axis.scatter([], [], s=24 + 190 * rate, facecolor="#AFAFAF", edgecolor="#555555", linewidth=0.55, label=f"{rate:.0%}")
        for rate in rates
    ]
    axis.legend(
        handles=handles,
        title="Significant tests",
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.29),
        handletextpad=0.2,
        columnspacing=0.8,
    )
    figure.subplots_adjust(left=0.17, right=0.91, top=0.88, bottom=0.30)
    frame.to_csv(OUTPUT_DIR / "supplementary_figure_s2_source_data.csv", index=False, lineterminator="\n")
    _save_svg(figure, "supplementary_figure_s2_structural_profiles")


def _s3() -> None:
    frame = pd.read_csv(PRECISION_YIELD_DIR / "topk_sensitivity.csv")
    frame.to_csv(OUTPUT_DIR / "supplementary_figure_s3_source_data.csv", index=False, lineterminator="\n")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharex=True)
    panels = (
        (
            "supported_fraction_selected",
            "supported_fraction_selected_ci_low",
            "supported_fraction_selected_ci_high",
            "Supported / selected",
        ),
        (
            "supported_fraction_testable",
            "supported_fraction_testable_ci_low",
            "supported_fraction_testable_ci_high",
            "Supported / testable",
        ),
        (
            "testable_fraction_selected",
            "testable_fraction_selected_ci_low",
            "testable_fraction_selected_ci_high",
            "Testable / selected",
        ),
    )
    for axis, (value_column, low_column, high_column, title), label in zip(axes, panels, "ABC"):
        for method in ALL_METHODS:
            subset = frame[frame["method"].eq(method)].sort_values("top_k_requested")
            x = subset["top_k_requested"].to_numpy(dtype=float)
            y = subset[value_column].to_numpy(dtype=float)
            low = subset[low_column].to_numpy(dtype=float)
            high = subset[high_column].to_numpy(dtype=float)
            display = DISPLAY_METHODS.get(method, method)
            axis.plot(x, y, marker="o", markersize=3.4, linewidth=1.0, color=ALL_METHOD_COLORS[method], label=display)
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                fmt="none",
                ecolor=ALL_METHOD_COLORS[method],
                elinewidth=1.0,
                capsize=1.8,
                alpha=0.82,
            )
        axis.set_xscale("log")
        axis.set_xticks((1, 3, 5, 10, 20, 50), ("1", "3", "5", "10", "20", "50"))
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Top K per family")
        axis.set_title(title)
        axis.grid(color=GRID, linewidth=0.5)
        _panel_label(axis, label)
    axes[0].set_ylabel("Fraction")
    figure.suptitle("Top-K sensitivity", fontsize=9.2, fontweight="bold")
    figure.legend(
        handles=[Line2D([], [], marker="o", color=ALL_METHOD_COLORS[method], markersize=3.5, linewidth=1, label=DISPLAY_METHODS.get(method, method)) for method in ALL_METHODS],
        frameon=False,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.text(0.5, 0.075, "Ranks are defined within each method-specific candidate space.", ha="center", fontsize=6.1)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.25, wspace=0.34)
    _save_svg(figure, "supplementary_figure_s3_topk_sensitivity")


def generate_all() -> tuple[Path, ...]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()
    _normalize_architecture_svg()
    _figure2()
    _figure3()
    _figure4()
    _figure5()
    _figure6()
    _s1()
    _s2()
    _s3()
    paths = tuple(OUTPUT_DIR / f"{stem}.svg" for stem in EXPECTED_STEMS)
    for path in paths:
        validate_svg(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-companions",
        action="store_true",
        help="Also export PDF and 600-dpi TIFF companions below tmp for QA.",
    )
    args = parser.parse_args(argv)
    global EXPORT_QA_COMPANIONS
    EXPORT_QA_COMPANIONS = args.qa_companions
    for path in generate_all():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
