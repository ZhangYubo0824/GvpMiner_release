#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ("GLAM2", "M2", "M3", "MEME", "MotifAE", "STREME")
METHOD_PALETTE = {
    "GLAM2": "#6B7280",
    "M2": "#2F6F9F",
    "M3": "#6AA6C8",
    "MEME": "#D18F3B",
    "MotifAE": "#8B6FB1",
    "STREME": "#5E9A78",
}
FIGURE_NAMES = (
    "provisional_figure6_precision_yield",
    "provisional_figureS3_topk_sensitivity",
    "provisional_figureS4_rank_trend",
)
KNOWN_OUTPUT_NAMES = frozenset(
    [
        f"{stem}.{extension}"
        for stem in FIGURE_NAMES
        for extension in ("png", "pdf", "svg")
    ]
    + [f"{stem}_source_data.csv" for stem in FIGURE_NAMES]
    + ["figure_manifest.json"]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["method"] = pd.Categorical(
        ordered["method"], categories=METHOD_ORDER, ordered=True
    )
    return ordered.sort_values("method").reset_index(drop=True)


def _require_positive(values: pd.Series, label: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or (~np.isfinite(numeric)).any() or (numeric <= 0).any():
        raise ValueError(f"{label} values must be finite and strictly positive for a log axis")


def _stable_ci_mask(rows: pd.DataFrame) -> pd.Series:
    return rows["supported_fraction_testable_ci_status"].astype(str).eq("stable")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _precision_yield_label_offset(method: str) -> tuple[int, int]:
    return {
        "M2": (8, 8),
        "M3": (8, -14),
        "MEME": (8, 7),
        "MotifAE": (8, 8),
        "STREME": (8, 8),
        "GLAM2": (8, 8),
    }.get(method, (8, 8))


def _background_label_layout(value: float) -> tuple[tuple[int, int], str]:
    if value <= -2.5:
        return (7, 0), "left"
    if value < 0:
        return (-7, 0), "right"
    return (7, 0), "left"


def _main_source_data(
    method_yield: pd.DataFrame,
    topk: pd.DataFrame,
    global_test: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in _order(method_yield).itertuples():
        for metric in ["candidate_motifs", "testable_motifs", "supported_motifs"]:
            records.append(
                {
                    "panel": "A",
                    "method": str(row.method),
                    "metric": metric,
                    "value": getattr(row, metric),
                    "ci_low": math.nan,
                    "ci_high": math.nan,
                    "numerator": getattr(row, metric),
                    "denominator": getattr(row, "candidate_motifs"),
                    "p_value": math.nan,
                    "adjusted_q_value": math.nan,
                    "null_mean": math.nan,
                    "null_ci_low": math.nan,
                    "null_ci_high": math.nan,
                }
            )
    top10 = topk[topk["top_k_requested"] == 10]
    for row in _order(top10).itertuples():
        records.append(
            {
                "panel": "B",
                "method": str(row.method),
                "metric": "top10_supported_fraction_selected",
                "value": row.supported_fraction_selected,
                "ci_low": row.supported_fraction_selected_ci_low,
                "ci_high": row.supported_fraction_selected_ci_high,
                "numerator": row.supported_motif_count,
                "denominator": row.selected_motif_count,
                "p_value": math.nan,
                "adjusted_q_value": math.nan,
                "null_mean": math.nan,
                "null_ci_low": math.nan,
                "null_ci_high": math.nan,
            }
        )
    for row in _order(global_test).itertuples():
        records.append(
            {
                "panel": "C",
                "method": str(row.method),
                "metric": "observed_minus_null_supported_percentage_points",
                "value": row.observed_minus_null_percentage_points,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "numerator": row.observed_supported_count,
                "denominator": row.observed_selected_count,
                "p_value": row.empirical_p_ge_observed,
                "adjusted_q_value": row.adjusted_q_value,
                "null_mean": row.null_supported_mean,
                "null_ci_low": row.null_supported_q025,
                "null_ci_high": row.null_supported_q975,
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "panel",
            "method",
            "metric",
            "value",
            "ci_low",
            "ci_high",
            "numerator",
            "denominator",
            "p_value",
            "adjusted_q_value",
            "null_mean",
            "null_ci_low",
            "null_ci_high",
        ],
    )


def _plot_main(
    method_yield: pd.DataFrame,
    topk: pd.DataFrame,
    global_test: pd.DataFrame,
    output_dir: Path,
) -> None:
    method_yield = _order(method_yield)
    top10 = _order(topk[topk["top_k_requested"] == 10])
    global_test = _order(global_test)
    _require_positive(method_yield["candidate_motifs"], "candidate_motifs")
    _require_positive(method_yield["testable_motifs"], "testable_motifs")
    _require_positive(method_yield["supported_motifs"], "supported_motifs")
    fig = plt.figure(figsize=(11.2, 4.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.2, 1.05])
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]

    axis = axes[0]
    y_positions = np.arange(len(method_yield))[::-1]
    stages = [
        ("candidate_motifs", "All candidates", "#C8CDD3", "o"),
        ("testable_motifs", "Annotation-testable", "#7C9DB5", "s"),
        ("supported_motifs", "Supported", "#244F6B", "D"),
    ]
    for y, row in zip(y_positions, method_yield.itertuples()):
        values = [getattr(row, stage[0]) for stage in stages]
        axis.plot(values, [y] * 3, color="#D7DCE0", linewidth=1.1, zorder=1)
        for stage_index, (value, (_, _, color, marker)) in enumerate(zip(values, stages)):
            axis.scatter(value, y, color=color, marker=marker, s=24, zorder=2)
            axis.annotate(
                f"{int(value):,}",
                (value, y),
                xytext=(4, (-9, 1, 8)[stage_index]),
                textcoords="offset points",
                fontsize=5.8,
                color="#374151",
                va="center",
            )
    axis.set_xscale("log")
    axis.set_yticks(y_positions, method_yield["method"].astype(str))
    axis.set_xlabel("Candidate motif units (log scale)")
    axis.set_title("Candidate conversion")
    handles = [
        mpl.lines.Line2D([], [], color=color, marker=marker, linestyle="", label=label)
        for _, label, color, marker in stages
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=6)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    _panel_label(axis, "A")

    axis = axes[1]
    yield_lookup = method_yield.set_index("method")
    for row in top10.itertuples():
        method = str(row.method)
        yield_row = yield_lookup.loc[method]
        x = float(row.supported_fraction_selected)
        y = float(yield_row["supported_motifs"])
        size = 80 + 320 * float(yield_row["testable_fraction"])
        if str(row.supported_fraction_selected_ci_status) == "stable":
            axis.errorbar(
                x,
                y,
                xerr=[
                    [x - float(row.supported_fraction_selected_ci_low)],
                    [float(row.supported_fraction_selected_ci_high) - x],
                ],
                fmt="none",
                ecolor=METHOD_PALETTE[method],
                linewidth=1,
                capsize=2,
                alpha=0.8,
            )
        axis.scatter(
            x,
            y,
            s=size,
            color=METHOD_PALETTE[method],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
        )
        offset = _precision_yield_label_offset(method)
        axis.annotate(method, (x, y), xytext=offset, textcoords="offset points", fontsize=6.8)
    axis.set_yscale("log")
    axis.set_xlim(0, min(1.0, max(0.7, float(top10["supported_fraction_selected"].max()) + 0.12)))
    axis.set_xlabel("Top-10 supported / selected")
    axis.set_ylabel("Total supported motif units (log scale)")
    axis.set_title("Precision–yield landscape")
    axis.grid(color="#E5E7EB", linewidth=0.6)
    axis.text(
        0.02,
        0.02,
        "Point area scales with annotation-testable fraction\nHorizontal bars: family-cluster bootstrap 95% CI",
        transform=axis.transAxes,
        fontsize=5.8,
        color="#4B5563",
        va="bottom",
    )
    _panel_label(axis, "B")

    axis = axes[2]
    y_positions = np.arange(len(global_test))[::-1]
    for y, row in zip(y_positions, global_test.itertuples()):
        value = float(row.observed_minus_null_percentage_points)
        method = str(row.method)
        axis.plot([0, value], [y, y], color="#D1D5DB", linewidth=1.3)
        axis.scatter(value, y, s=34, color=METHOD_PALETTE[method], zorder=3)
        offset, alignment = _background_label_layout(value)
        axis.annotate(
            f"{value:+.2f} pp\np={row.empirical_p_ge_observed:.3g}; q={row.adjusted_q_value:.3g}",
            (value, y),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=5.7,
        )
    axis.axvline(0, color="#6B7280", linewidth=0.8, linestyle="--")
    axis.set_yticks(y_positions, global_test["method"].astype(str))
    axis.set_xlabel("Observed − matched-null support (percentage points)")
    axis.set_title("Method-level matched background")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axis.text(
        0.5,
        -0.21,
        "10,000 joint draws; BH across six methods.\nOriginal family–method audit: 0/96 at empirical p < 0.05.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
        color="#4B5563",
    )
    _panel_label(axis, "C")

    _save_figure(fig, output_dir, FIGURE_NAMES[0])
    _main_source_data(method_yield, topk, global_test).to_csv(
        output_dir / f"{FIGURE_NAMES[0]}_source_data.csv", index=False
    )


def _plot_topk(topk: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("supported_fraction_selected", "Supported / selected"),
        ("supported_fraction_testable", "Supported / testable"),
        ("testable_fraction_selected", "Testable / selected"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharex=True, sharey=True, constrained_layout=True)
    for panel, (axis, (metric, label)) in enumerate(zip(axes, metrics)):
        for method in METHOD_ORDER:
            group = topk[topk["method"] == method].sort_values("top_k_requested")
            if group.empty:
                continue
            x = group["top_k_requested"].to_numpy(dtype=float)
            y = group[metric].to_numpy(dtype=float)
            low = group[f"{metric}_ci_low"].to_numpy(dtype=float)
            high = group[f"{metric}_ci_high"].to_numpy(dtype=float)
            stable = group[f"{metric}_ci_status"].astype(str).eq("stable").to_numpy()
            axis.plot(x, y, marker="o", markersize=3, linewidth=1.3, color=METHOD_PALETTE[method], label=method)
            stable_indices = np.flatnonzero(stable)
            if len(stable_indices):
                axis.errorbar(
                    x[stable_indices],
                    y[stable_indices],
                    yerr=[
                        y[stable_indices] - low[stable_indices],
                        high[stable_indices] - y[stable_indices],
                    ],
                    fmt="none",
                    ecolor=METHOD_PALETTE[method],
                    elinewidth=0.65,
                    capsize=1.5,
                    alpha=0.34,
                    zorder=0,
                )
        axis.set_xscale("log")
        axis.set_xticks([1, 3, 5, 10, 20, 50], ["1", "3", "5", "10", "20", "50"])
        axis.set_ylim(0, 1)
        axis.set_xlabel("K per family–method")
        axis.set_title(label)
        axis.grid(color="#E5E7EB", linewidth=0.6)
        _panel_label(axis, chr(ord("A") + panel))
    axes[0].set_ylabel("Fraction")
    axes[-1].legend(loc="upper right", fontsize=6, ncol=2)
    fig.suptitle("Top-K sensitivity across method-specific rank definitions", fontsize=9)
    fig.text(
        0.5,
        -0.02,
        "Thin error bars: family-cluster bootstrap 95% CI. MEME uses foreground-protein-support fallback; MotifAE uses foreground-protein support; other methods use native motif numbering.",
        ha="center",
        fontsize=6,
        color="#4B5563",
    )
    _save_figure(fig, output_dir, FIGURE_NAMES[1])
    topk.to_csv(output_dir / f"{FIGURE_NAMES[1]}_source_data.csv", index=False)


def _plot_rank_trend(
    deciles: pd.DataFrame, models: pd.DataFrame, output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.1), sharex=True, sharey=True, constrained_layout=True)
    model_lookup = models.set_index("method")
    for axis, method in zip(axes.flat, METHOD_ORDER):
        group = deciles[deciles["method"] == method].sort_values("rank_decile")
        observed = group[group["status"] == "observed"]
        x = observed["rank_decile"].to_numpy(dtype=float)
        y = observed["supported_fraction_testable"].to_numpy(dtype=float)
        low = observed["supported_fraction_testable_ci_low"].to_numpy(dtype=float)
        high = observed["supported_fraction_testable_ci_high"].to_numpy(dtype=float)
        axis.plot(
            x,
            y,
            "o-",
            markersize=3,
            linewidth=1.1,
            color=METHOD_PALETTE[method],
        )
        stable = _stable_ci_mask(observed).to_numpy()
        stable_indices = np.flatnonzero(stable)
        if len(stable_indices):
            axis.errorbar(
                x[stable_indices],
                y[stable_indices],
                yerr=[
                    y[stable_indices] - low[stable_indices],
                    high[stable_indices] - y[stable_indices],
                ],
                fmt="none",
                linewidth=0.65,
                capsize=2,
                color=METHOD_PALETTE[method],
            )
        axis.set_title(method, color=METHOD_PALETTE[method], fontweight="bold")
        axis.set_xlim(0.5, 10.5)
        axis.set_ylim(0, 1)
        axis.grid(color="#E5E7EB", linewidth=0.6)
        if method in model_lookup.index:
            model = model_lookup.loc[method]
            if str(model["status"]) == "estimable":
                annotation = (
                    f"βrank={float(model['rank_coefficient']):+.2f}\n"
                    f"95% CI {float(model['rank_ci_low']):+.2f} to "
                    f"{float(model['rank_ci_high']):+.2f}\n"
                    f"BH q={float(model['rank_q_value']):.3g}"
                )
            else:
                annotation = f"Not estimable\n{model['reason']}"
            axis.text(
                0.03,
                0.04,
                annotation,
                transform=axis.transAxes,
                fontsize=5.8,
                va="bottom",
                color="#374151",
            )
    for axis in axes[-1, :]:
        axis.set_xlabel("Normalized-rank decile (1 = highest priority)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Supported / testable")
    fig.suptitle("Within-method rank trends among annotation-testable motifs", fontsize=9.5)
    fig.text(
        0.5,
        -0.015,
        "Points and error bars: decile proportion and family-cluster bootstrap 95% CI. Models include family fixed effects with family-cluster-robust standard errors.",
        ha="center",
        fontsize=6,
        color="#4B5563",
    )
    _save_figure(fig, output_dir, FIGURE_NAMES[2])
    merged = deciles.merge(
        models[
            [
                "method",
                "status",
                "reason",
                "rank_coefficient",
                "rank_ci_low",
                "rank_ci_high",
                "rank_q_value",
            ]
        ].rename(columns={"status": "model_status"}),
        on="method",
        how="left",
    )
    merged.to_csv(output_dir / f"{FIGURE_NAMES[2]}_source_data.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot provisional Exp5 precision-yield figures from audited analysis CSVs.")
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite only the known generated figure artifacts; reject unknown files.",
    )
    return parser


def _prepare_output_dir(analysis_dir: Path, *, overwrite: bool) -> Path:
    output_dir = analysis_dir.resolve() / "figures"
    if output_dir.exists():
        existing_names = {path.name for path in output_dir.iterdir()}
        unknown = sorted(existing_names - KNOWN_OUTPUT_NAMES)
        if unknown:
            raise ValueError(f"figure directory contains unknown files: {unknown}")
        if existing_names and not overwrite:
            raise ValueError(f"figure output directory is non-empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    analysis_dir = args.analysis_dir.resolve()
    input_names = [
        "method_yield_summary.csv",
        "topk_sensitivity.csv",
        "global_random_test.csv",
        "rank_decile_summary.csv",
        "rank_model_summary.csv",
    ]
    inputs = [analysis_dir / name for name in input_names]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing analysis inputs: {missing}")
    output_dir = _prepare_output_dir(analysis_dir, overwrite=args.overwrite)
    _configure_style()
    method_yield, topk, global_test, deciles, models = [pd.read_csv(path) for path in inputs]
    _plot_main(method_yield, topk, global_test, output_dir)
    _plot_topk(topk, output_dir)
    _plot_rank_trend(deciles, models, output_dir)
    generated = sorted(
        [path for path in output_dir.iterdir() if path.name != "figure_manifest.json"],
        key=lambda path: path.name,
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "python/matplotlib",
        "artifact_status": "provisional_review",
        "qa_boundary": (
            "Large review canvas and 300 dpi PNG are paired with editable PDF/SVG; "
            "final journal dimensions and any TIFF export are deferred until inclusion approval."
        ),
        "matplotlib_version": mpl.__version__,
        "canvas_inches": {
            FIGURE_NAMES[0]: [11.2, 4.1],
            FIGURE_NAMES[1]: [11.2, 3.8],
            FIGURE_NAMES[2]: [10.4, 6.1],
        },
        "png_dpi": 300,
        "fonts": ["Arial", "Helvetica", "DejaVu Sans"],
        "method_palette": METHOD_PALETTE,
        "inputs": [
            {"path": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in inputs
        ],
        "plotting_script_sha256": _sha256(Path(__file__).resolve()),
        "generated_files": [
            {"path": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in generated
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
