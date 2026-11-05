from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 统一配色 & 标签
# ---------------------------------------------------------------------------
PALETTE = {
    "primary":   "#4C78A8",
    "secondary": "#72B7B2",
    "accent":    "#F58518",
    "danger":    "#E45756",
    "success":   "#54A24B",
    "motif":     "#26A69A",
    "control":   "#EF5350",
    "neutral":   "#888888",
    "grid":      "#E5E5E5",
    "enriched":  "#C44E52",
    "depleted":  "#4C72B0",
    "ns":        "#BBBBBB",
}

CMAP_DIVERGING = "RdBu_r"

ENDPOINT_LABEL = {
    "contact_density":                     "Contact density",
    "relative_SASA":                       "Relative SASA",
    "long_range_contact_count":            "Long-range contacts",
    "local_RMSD_or_structural_similarity": "Local structural similarity",
    "motif_mean_pLDDT":                    "Mean pLDDT",
    "motif_min_pLDDT":                     "Min pLDDT",
    "helix_fraction":                      "Helix fraction",
    "sheet_fraction":                      "Sheet fraction",
    "coil_fraction":                       "Coil fraction",
    "radius_of_gyration":                  "Radius of gyration",
    "distance_to_protein_centroid":        "Dist. to centroid",
    "surface_residue_fraction":            "Surface residue fraction",
    "low_confidence_fraction":             "Low-confidence fraction",
    "hydrophobic_fraction":                "Hydrophobic fraction",
    "charged_fraction":                    "Charged fraction",
    "aromatic_fraction":                   "Aromatic fraction",
    "polar_fraction":                      "Polar fraction",
    "motif_entropy":                       "Sequence entropy",
}

CONTROL_LABEL = {
    "within_random":       "Within-protein\nrandom",
    "position_matched":    "Position-matched",
    "composition_matched": "Composition-matched",
}

PRIMARY_FEATURES = {
    "contact_density", "relative_SASA", "long_range_contact_count",
}


def _label(feature: str) -> str:
    return ENDPOINT_LABEL.get(feature, feature)


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "†"
    return ""


def _save(fig, stem: str, output_dir: Path, formats: Sequence[str]) -> list[Path]:
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    return paths


def _empty_figure(plt, title: str, message: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title, fontweight="bold")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=11,
            color=PALETTE["neutral"])
    return fig


def _build_stat_index(stat_rows: Sequence[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    for row in stat_rows:
        cluster = str(row.get("motif_cluster", ""))
        feature = str(row.get("feature", ""))
        if cluster and feature:
            key = (cluster, feature)
            existing = index.get(key)
            if (existing is None or
                (_to_float(row.get("q_value")) or 1.0) < (_to_float(existing.get("q_value")) or 1.0)):
                index[key] = row
    return index


def _headline_stat_rows(stat_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Use only the prespecified structural comparison for new Exp4 figures."""
    if not any("primary_q_value" in row for row in stat_rows):
        return list(stat_rows)
    selected = []
    for row in stat_rows:
        primary_q = _to_float(row.get("primary_q_value"))
        if (
            str(row.get("control_type", "")) == "composition_matched"
            and str(row.get("feature", "")) in PRIMARY_FEATURES
            and primary_q is not None
            and math.isfinite(primary_q)
        ):
            selected.append({**row, "q_value": primary_q})
    return selected


def _adaptive_fontsize(n_cells: int, base: int = 9) -> float:
    if n_cells <= 30:
        return float(base)
    if n_cells <= 80:
        return float(base - 1)
    if n_cells <= 200:
        return float(base - 2)
    if n_cells <= 500:
        return float(base - 3)
    return max(4.5, base - 4)


def _text_color_for_effect(effect: float, vmax: float) -> str:
    return "white" if abs(effect) > vmax * 0.55 else "black"


# =========================================================================
# 主入口
# =========================================================================


def generate_publication_figures(
    *,
    output_dir: Path,
    feature_rows: Sequence[dict[str, object]],
    control_rows: Sequence[dict[str, object]],
    stat_rows: Sequence[dict[str, object]],
    formats: Sequence[str] = ("png", "pdf"),
    isolate_process: bool = True,
    timeout_seconds: int = 600,
) -> tuple[list[Path], list[str]]:
    if not isolate_process:
        return _generate_publication_figures_direct(
            output_dir=output_dir,
            feature_rows=feature_rows,
            control_rows=control_rows,
            stat_rows=stat_rows,
            formats=formats,
        )

    output_dir = Path(output_dir)
    payload = {
        "output_dir": str(output_dir),
        "feature_rows": list(feature_rows),
        "control_rows": list(control_rows),
        "stat_rows": list(stat_rows),
        "formats": list(formats or ("png",)),
    }
    return _generate_figures_in_subprocess(output_dir, payload, timeout_seconds)


def generate_publication_figures_from_summaries(
    *,
    output_dir: Path,
    feature_rows: Sequence[dict[str, object]],
    control_type_counts: Mapping[str, int],
    control_contact_density: Sequence[object],
    stat_rows: Sequence[dict[str, object]],
    formats: Sequence[str] = ("png", "pdf"),
    isolate_process: bool = True,
    timeout_seconds: int = 600,
) -> tuple[list[Path], list[str]]:
    if not isolate_process:
        return _generate_publication_figures_direct(
            output_dir=output_dir,
            feature_rows=feature_rows,
            control_rows=[],
            control_type_counts=control_type_counts,
            control_contact_density=control_contact_density,
            stat_rows=stat_rows,
            formats=formats,
        )

    output_dir = Path(output_dir)
    payload = {
        "output_dir": str(output_dir),
        "feature_rows": list(feature_rows),
        "control_type_counts": dict(control_type_counts),
        "control_contact_density": list(control_contact_density),
        "stat_rows": list(stat_rows),
        "formats": list(formats or ("png",)),
    }
    return _generate_figures_in_subprocess(output_dir, payload, timeout_seconds)


def _generate_figures_in_subprocess(
    output_dir: Path,
    payload: Mapping[str, object],
    timeout_seconds: int,
) -> tuple[list[Path], list[str]]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    try:
        with tempfile.TemporaryDirectory(prefix=".figure_generation_", dir=str(output_dir.parent)) as tmp:
            tmp_dir = Path(tmp)
            payload_path = tmp_dir / "payload.json"
            result_path = tmp_dir / "result.json"
            payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable, "-m", "exp4_motif_structure_analysis.figure_worker",
                    str(payload_path), str(result_path),
                ],
                cwd=str(package_root.parent), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=timeout_seconds,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:]
                suffix = f" {' '.join(detail)}" if detail else ""
                return [], [f"Figure generation skipped: plotting subprocess exited "
                            f"with code {completed.returncode}.{suffix}"]
            if not result_path.exists():
                return [], ["Figure generation skipped: plotting subprocess did not "
                            "write a result file."]
            result = json.loads(result_path.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        return [], [f"Figure generation skipped: plotting subprocess timed out "
                    f"after {timeout_seconds} seconds."]
    except Exception as exc:
        return [], [f"Figure generation skipped: {type(exc).__name__}: {exc}"]

    return ([Path(p) for p in result.get("paths", [])],
            list(result.get("warnings", [])))


def _generate_publication_figures_direct(
    *,
    output_dir: Path,
    feature_rows: Sequence[dict[str, object]],
    control_rows: Sequence[dict[str, object]],
    stat_rows: Sequence[dict[str, object]],
    control_type_counts: Mapping[str, int] | None = None,
    control_contact_density: Sequence[object] | None = None,
    formats: Sequence[str] = ("png", "pdf"),
) -> tuple[list[Path], list[str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [], [f"Figure generation skipped: {type(exc).__name__}: {exc}"]

    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(formats or ("png",))
    paths: list[Path] = []
    warnings: list[str] = []

    def save_current(fig, stem):
        paths.extend(_save(fig, stem, output_dir, formats))
        plt.close(fig)

    primary_mode = any("primary_q_value" in row for row in stat_rows)
    headline_rows = _headline_stat_rows(stat_rows)
    stat_index = _build_stat_index(headline_rows)

    signif_rows = [r for r in headline_rows if _to_float(r.get("q_value")) is not None]
    features_raw = sorted({str(r.get("feature", "")) for r in signif_rows})
    if not features_raw:
        features_raw = sorted(PRIMARY_FEATURES)
    feature_labels = [_label(f) for f in features_raw]

    # ---- shared data ----
    plddt = [_to_float(r.get("motif_mean_pLDDT")) for r in feature_rows]
    plddt = [v for v in plddt if v is not None]
    categories = [str(r.get("confidence_category", "") or "unlabelled") for r in feature_rows]

    if control_type_counts is None:
        control_types = [str(r.get("control_type", "")) for r in control_rows if r.get("control_type")]
        ct_map = {k: CONTROL_LABEL.get(k, k) for k in sorted(set(control_types))}
        ct_counts = {ct: control_types.count(ct) for ct in ct_map}
    else:
        ct_counts_raw = {str(key): int(value) for key, value in control_type_counts.items()}
        ct_map = {k: CONTROL_LABEL.get(k, k) for k in sorted(ct_counts_raw)}
        ct_counts = {ct: ct_counts_raw[ct] for ct in ct_map}

    # best-q per cluster
    cluster_best_q: dict[str, float] = {}
    for r in signif_rows:
        c = str(r.get("motif_cluster", ""))
        qv = _to_float(r.get("q_value")) or 1.0
        if c:
            cluster_best_q[c] = min(cluster_best_q.get(c, 1.0), qv)
    top15 = [c for c, _ in sorted(cluster_best_q.items(), key=lambda kv: kv[1])[:15]]
    top6  = top15[:6]

    # =====================================================================
    # 图 1: QC — pLDDT 直方图 + KDE + 置信度分类
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if plddt:
        n_bins = min(30, max(10, int(np.sqrt(len(plddt)))))
        axes[0].hist(plddt, bins=n_bins, color=PALETTE["primary"], edgecolor="white",
                     alpha=0.7, density=True, label="Histogram")
        # KDE 叠加
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(plddt)
            x_kde = np.linspace(min(plddt), max(plddt), 200)
            axes[0].plot(x_kde, kde(x_kde), color=PALETTE["danger"], linewidth=1.8, label="KDE")
        except Exception:
            pass
        axes[0].axvline(np.mean(plddt), color=PALETTE["danger"], linestyle="--", linewidth=1.0,
                        label=f"Mean = {np.mean(plddt):.1f}")
        axes[0].legend(fontsize=7)
    else:
        axes[0].text(0.5, 0.5, "No pLDDT data", ha="center", va="center",
                     transform=axes[0].transAxes, color=PALETTE["neutral"])
    axes[0].set_title("Motif pLDDT Distribution", fontweight="bold")
    axes[0].set_xlabel("Mean motif pLDDT")
    axes[0].set_ylabel("Density")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    cat_labels = sorted(set(categories))
    cat_counts = [categories.count(lb) for lb in cat_labels]
    axes[1].bar(cat_labels, cat_counts, color=PALETTE["secondary"], edgecolor="white")
    for x, cnt in enumerate(cat_counts):
        if cnt:
            axes[1].text(x, cnt + max(cat_counts) * 0.02, str(cnt), ha="center",
                         fontsize=9, fontweight="bold")
    axes[1].set_title("Confidence Categories", fontweight="bold")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Count")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    save_current(fig, "qc_plddt_distribution")

    # =====================================================================
    # 图 S1: QC — 对照覆盖度 (Supplementary)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [ct_map[ct] for ct in ct_map]
    counts = list(ct_counts.values())
    ax.bar(names, counts, color=PALETTE["accent"], edgecolor="white", alpha=0.85)
    for x, cnt in enumerate(counts):
        ax.text(x, cnt + max(counts) * 0.02, str(cnt), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Matched Control Coverage  (Supplementary)", fontweight="bold")
    ax.set_ylabel("Windows")
    ax.tick_params(axis="x", rotation=30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_current(fig, "qc_control_coverage")

    # =====================================================================
    # 图 2: Feature × Top-15 Clusters 效应量热图
    # =====================================================================
    if signif_rows and top15 and features_raw:
        matrix = np.zeros((len(features_raw), len(top15)))
        annot_cells = np.empty((len(features_raw), len(top15)), dtype=object)
        for fi, feat in enumerate(features_raw):
            for ci, clst in enumerate(top15):
                entry = stat_index.get((clst, feat), {})
                effect = _to_float(entry.get("effect_median_difference")) or 0.0
                qv = _to_float(entry.get("q_value"))
                matrix[fi, ci] = effect
                star = _sig_stars(qv) if qv is not None else ""
                annot_cells[fi, ci] = _sig_stars(qv) if star else ""

        n_cells = len(features_raw) * len(top15)
        fs = _adaptive_fontsize(n_cells, base=8)
        fig_h = max(3.5, len(features_raw) * 0.45)
        fig_w = max(6, len(top15) * 0.75)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vmax = max(abs(matrix.min()), abs(matrix.max())) or 1.0
        im = ax.imshow(matrix, cmap=CMAP_DIVERGING, aspect="auto", vmin=-vmax, vmax=vmax)
        for fi in range(len(features_raw)):
            for ci in range(len(top15)):
                star = annot_cells[fi, ci]
                if star:
                    ax.text(ci, fi, star, ha="center", va="center", fontsize=fs + 1,
                            fontweight="bold",
                            color=_text_color_for_effect(matrix[fi, ci], vmax))
        ax.set_xticks(range(len(top15)))
        ax.set_xticklabels(top15, rotation=45, ha="right", fontsize=fs)
        ax.set_yticks(range(len(features_raw)))
        ax.set_yticklabels(feature_labels, fontsize=fs)
        title = "Primary Structural Associations" if primary_mode else "Exploratory Structural Enrichment"
        ax.set_title(f"{title}  (Feature × Top 15 clusters)", fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, label="Median effect  (motif − control)", shrink=0.82)
        cbar.ax.yaxis.label.set_size(9)
        fig.tight_layout()
    else:
        fig = _empty_figure(plt, "Structural Enrichment (Feature × Cluster)",
                            "No structural enrichment rows available.")
    save_current(fig, "fig2_structural_enrichment_heatmap")

    # =====================================================================
    # 图 3: Endpoint 显著命中数  (q < 0.05)
    # =====================================================================
    sig_hits = [r for r in signif_rows if (_to_float(r.get("q_value")) or 1.0) < 0.05]
    if sig_hits:
        ep_hits: dict[str, int] = {}
        for r in sig_hits:
            ep_hits[str(r.get("feature", ""))] = ep_hits.get(str(r.get("feature", "")), 0) + 1
        ep_names = sorted(ep_hits, key=ep_hits.get, reverse=True)
        ep_counts = [ep_hits[n] for n in ep_names]
        n_clust = len({str(r.get("motif_cluster", "")) for r in sig_hits})
        fig, ax = plt.subplots(figsize=(max(6, len(ep_names) * 0.75), 4.5))
        colors_bar = [PALETTE["primary"] if n in PRIMARY_FEATURES else PALETTE["secondary"]
                      for n in ep_names]
        ax.bar(range(len(ep_names)), ep_counts, color=colors_bar, edgecolor="white", alpha=0.85)
        for x, cnt in enumerate(ep_counts):
            ax.text(x, cnt + max(ep_counts) * 0.02, str(cnt), ha="center", fontsize=9,
                    fontweight="bold")
        ax.set_xticks(range(len(ep_names)))
        ax.set_xticklabels([_label(n) for n in ep_names], rotation=45, ha="right", fontsize=9)
        prefix = "Primary Endpoint Hits" if primary_mode else "Exploratory Endpoint Hits"
        ax.set_title(f"{prefix}  (q < 0.05, {n_clust} clusters)", fontweight="bold")
        ax.set_ylabel("Number of significant clusters")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    else:
        fig = _empty_figure(plt, "Significant Endpoint Hits", "No significant hits at q < 0.05.")
    save_current(fig, "fig3_endpoint_significant_hits")

    # =====================================================================
    # 图 4: Motif vs Control 箱线图 (contact density)
    # =====================================================================
    motif_cd = [_to_float(r.get("contact_density")) for r in feature_rows]
    motif_cd = [v for v in motif_cd if v is not None]
    if control_contact_density is None:
        box_controls = (
            [r for r in control_rows if r.get("control_type") == "composition_matched"]
            if any("primary_q_value" in r for r in stat_rows) else control_rows
        )
        ctrl_cd = [_to_float(r.get("contact_density")) for r in box_controls]
    else:
        ctrl_cd = [_to_float(value) for value in control_contact_density]
    ctrl_cd  = [v for v in ctrl_cd if v is not None]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bp = ax.boxplot(
        [motif_cd or [0.0], ctrl_cd or [0.0]],
        patch_artist=True, widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    bp["boxes"][0].set_facecolor(PALETTE["motif"])
    bp["boxes"][1].set_facecolor(PALETTE["control"])
    for b in bp["boxes"]:
        b.set_alpha(0.55)

    rng = np.random.default_rng(42)
    for i, vals in enumerate([motif_cd, ctrl_cd]):
        if vals:
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(i + 1 + jitter, vals, s=7, alpha=0.3, color=PALETTE["neutral"],
                       zorder=3, edgecolors="none")

    ax.set_xticks([1, 2])
    control_label = "Composition background" if any("primary_q_value" in r for r in stat_rows) else "Controls"
    ax.set_xticklabels([f"Motif\n(n={len(motif_cd)})", f"{control_label}\n(n={len(ctrl_cd)})"],
                       fontsize=11)
    ax.set_title("Contact Density Distributions (Descriptive)", fontweight="bold")
    ax.set_ylabel("Contact density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ann_lines = ["Pooled windows; see paired primary tests for inference"]
    if ann_lines:
        ax.text(0.98, 0.95, "\n".join(ann_lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "alpha": 0.85, "edgecolor": PALETTE["grid"]})
    fig.tight_layout()
    save_current(fig, "fig4_motif_vs_control_boxplot")

    # =====================================================================
    # 图 5: 全景热图 — 所有 motif × 所有 endpoint
    # =====================================================================
    if signif_rows and len(features_raw) >= 2:
        all_clusters = sorted({str(r.get("motif_cluster", "")) for r in signif_rows})
        # 按平均 |effect| 排序，自适应 top-N
        ce_mean: dict[str, float] = {}
        for c in all_clusters:
            effs = [_to_float(r.get("effect_median_difference")) or 0.0
                    for r in signif_rows if str(r.get("motif_cluster", "")) == c]
            ce_mean[c] = float(np.mean([abs(e) for e in effs])) if effs else 0.0
        sorted_c = sorted(ce_mean, key=ce_mean.get, reverse=True)
        max_clusters = min(35, len(sorted_c))
        sorted_c = sorted_c[:max_clusters]

        panorama = np.zeros((len(sorted_c), len(features_raw)))
        pano_stars = np.empty((len(sorted_c), len(features_raw)), dtype=object)
        for ri, c in enumerate(sorted_c):
            for ci, f in enumerate(features_raw):
                entry = stat_index.get((c, f), {})
                eff = _to_float(entry.get("effect_median_difference")) or 0.0
                qv  = _to_float(entry.get("q_value"))
                panorama[ri, ci] = eff
                pano_stars[ri, ci] = _sig_stars(qv) if qv is not None else ""

        n_cells = len(sorted_c) * len(features_raw)
        fs = _adaptive_fontsize(n_cells, base=7)
        fig_h = max(5, len(sorted_c) * 0.33)
        fig_w = max(8, len(features_raw) * 0.9)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vmax = max(abs(panorama.min()), abs(panorama.max())) or 1.0
        im = ax.imshow(panorama, cmap=CMAP_DIVERGING, aspect="auto", vmin=-vmax, vmax=vmax)
        for ri in range(len(sorted_c)):
            for ci in range(len(features_raw)):
                star = pano_stars[ri, ci]
                if star:
                    ax.text(ci, ri, star, ha="center", va="center", fontsize=fs + 1,
                            fontweight="bold",
                            color=_text_color_for_effect(panorama[ri, ci], vmax))
        ax.set_xticks(range(len(features_raw)))
        ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=fs + 1)
        ax.set_yticks(range(len(sorted_c)))
        ax.set_yticklabels(sorted_c, fontsize=fs)
        ax.set_title(f"Panoramic Effect Heatmap  ({len(sorted_c)} motifs × {len(features_raw)} endpoints)",
                     fontweight="bold")
        cbar = fig.colorbar(im, ax=ax, label="Median effect  (motif − control)", shrink=0.82)
        cbar.ax.yaxis.label.set_size(9)
        fig.tight_layout()
    else:
        fig = _empty_figure(plt, "Panoramic Effect Heatmap",
                            "Insufficient data (need ≥ 2 features).")
    save_current(fig, "fig5_panoramic_effect_heatmap")

    # =====================================================================
    # 图 6: 火山图 — effect size vs −log10(q)
    # =====================================================================
    if signif_rows and len(features_raw) >= 2:
        effects = [_to_float(r.get("effect_median_difference")) or 0.0 for r in signif_rows]
        q_vals  = [max(_to_float(r.get("q_value")) or 1e-300, 1e-300) for r in signif_rows]
        nlog_q  = [-math.log10(q) for q in q_vals]
        ep_lbls = [str(r.get("feature", "")) for r in signif_rows]
        cl_lbls = [str(r.get("motif_cluster", "")) for r in signif_rows]

        feat_set = sorted(set(ep_lbls))
        n_feat = len(feat_set)
        base_colors = ["#4C78A8", "#F58518", "#E45756", "#54A24B", "#72B7B2",
                       "#B279A2", "#FF9DA6", "#9D7660", "#D4C96A", "#8CD17D"]
        feat_color = {f: base_colors[i % len(base_colors)] for i, f in enumerate(feat_set)}

        fig, ax = plt.subplots(figsize=(7.5, 6))
        for feat in feat_set:
            mask = [e == feat for e in ep_lbls]
            x = [effects[i] for i, m in enumerate(mask) if m]
            y = [nlog_q[i] for i, m in enumerate(mask) if m]
            ax.scatter(x, y, c=[feat_color[feat]], label=_label(feat), alpha=0.55,
                       s=22, edgecolors="none")

        sig_line = -math.log10(0.05)
        ax.axhline(sig_line, color="#999999", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.95, sig_line, "q = 0.05", ha="right", va="bottom",
                fontsize=8, color="#666666")
        ax.axvline(0, color="#999999", linestyle=":", linewidth=0.7, alpha=0.4)

        # 标注: 取 -log10(q) 和 |effect| 各自 top 5 的交/并
        top_q_idx = set(sorted(range(len(nlog_q)), key=lambda i: nlog_q[i], reverse=True)[:10])
        top_e_idx = set(sorted(range(len(effects)), key=lambda i: abs(effects[i]), reverse=True)[:10])
        label_idx = sorted(top_q_idx | top_e_idx)[:12]
        for idx in label_idx:
            txt = f"{cl_lbls[idx]} | {_label(ep_lbls[idx])}"
            ax.annotate(txt, (effects[idx], nlog_q[idx]),
                        fontsize=5.5, ha="center", va="bottom",
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                              "alpha": 0.7, "edgecolor": "#DDDDDD"},
                        xytext=(0, 6), textcoords="offset points")

        ax.set_xlabel("Effect size  (median difference, motif − control)")
        ax.set_ylabel("−log10(q-value)")
        ax.set_title(
            "Volcano Plot: Primary Structural Associations" if primary_mode
            else "Volcano Plot: Exploratory Structural Enrichment",
            fontweight="bold",
        )
        ax.legend(fontsize=7, loc="upper left", ncol=max(1, n_feat // 7),
                  framealpha=0.8, edgecolor="#DDDDDD")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    else:
        fig = _empty_figure(plt, "Volcano Plot", "Insufficient data.")
    save_current(fig, "fig6_volcano_plot")

    # =====================================================================
    # 图 7: Parallel Coordinates — Top-6 motif 多维效应量剖面
    # =====================================================================
    if top6 and len(features_raw) >= 3:
        # 只用主要结构 endpoint (排除 pLDDT/低置信度等辅助指标)
        struct_feat = [f for f in features_raw if f in PRIMARY_FEATURES
                       or f in ("helix_fraction", "sheet_fraction", "coil_fraction",
                                "surface_residue_fraction", "distance_to_protein_centroid")]
        if len(struct_feat) < 3:
            struct_feat = features_raw

        pc_matrix = np.zeros((len(top6), len(struct_feat)))
        for ri, c in enumerate(top6):
            for ci, f in enumerate(struct_feat):
                entry = stat_index.get((c, f), {})
                pc_matrix[ri, ci] = _to_float(entry.get("effect_median_difference")) or 0.0

        fig, ax = plt.subplots(figsize=(max(7, len(struct_feat) * 0.9), 5))
        colors_pc = plt.get_cmap("tab10")(np.linspace(0, 1, len(top6)))
        x = range(len(struct_feat))
        for ri, c in enumerate(top6):
            ax.plot(x, pc_matrix[ri], "o-", linewidth=1.8, markersize=5,
                    color=colors_pc[ri], label=c, alpha=0.85)
        ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(f) for f in struct_feat], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Effect size  (median difference)")
        ax.set_title("Structural Fingerprint: Top-6 Motif Effect Profiles", fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
    else:
        fig = _empty_figure(plt, "Structural Fingerprint",
                            "Insufficient data (need ≥ 3 features and ≥ 1 cluster).")
    save_current(fig, "fig7_structural_fingerprint")

    return paths, warnings
