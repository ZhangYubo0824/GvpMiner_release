#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_RESULTS = PROJECT_ROOT / "work" / "results"
DEFAULT_METHODS = ("M2", "M3", "MEME", "GLAM2", "STREME", "MotifAE")
DEFAULT_PRIMARY_METHODS = ("M2", "M3")
ENRICHMENT_COLUMNS = {
    "ID",
    "Description",
    "family",
    "method",
    "motif_cluster",
    "p.adjust",
    "FoldEnrichment",
    "Count",
    "foreground_total",
    "universe_total",
}
MOTIF_SET_COLUMNS = {"family", "method", "motif_cluster"}
TOP_TERM_LIMIT = 20


@dataclass
class MethodEnrichmentStats:
    method: str
    motifs_with_rows: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    supported_motifs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    term_rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    significant_term_rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    min_adjusted: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: math.inf))
    top_terms: list[tuple[float, int, dict[str, object]]] = field(default_factory=list)
    _serial: int = 0

    def add_rows(self, frame: pd.DataFrame, alpha: float) -> None:
        if frame.empty:
            return
        required = {"family", "method", "motif_cluster", "p.adjust"}
        if not required.issubset(frame.columns):
            return
        frame = frame[frame["method"].astype(str).eq(self.method)].copy()
        if frame.empty:
            return
        frame["p.adjust"] = pd.to_numeric(frame["p.adjust"], errors="coerce")
        frame = frame.dropna(subset=["family", "motif_cluster", "p.adjust"])
        if frame.empty:
            return
        frame = frame.rename(columns={"p.adjust": "p_adjust"})
        for row in frame.itertuples(index=False):
            family = str(getattr(row, "family"))
            motif = str(getattr(row, "motif_cluster"))
            adjusted = float(getattr(row, "p_adjust", math.nan))
            if math.isnan(adjusted):
                continue
            self.motifs_with_rows[family].add(motif)
            self.term_rows[family] += 1
            self.min_adjusted[family] = min(self.min_adjusted[family], adjusted)
            if adjusted <= alpha:
                self.supported_motifs[family].add(motif)
                self.significant_term_rows[family] += 1
                self._push_top_term(row, adjusted)

    def _push_top_term(self, row: object, adjusted: float) -> None:
        self._serial += 1
        record = {
            "method": self.method,
            "family": str(getattr(row, "family", "")),
            "motif_cluster": str(getattr(row, "motif_cluster", "")),
            "term_id": str(getattr(row, "ID", "")),
            "description": str(getattr(row, "Description", "")),
            "p_adjust": adjusted,
            "fold_enrichment": getattr(row, "FoldEnrichment", ""),
            "count": getattr(row, "Count", ""),
        }
        heapq.heappush(self.top_terms, (-adjusted, self._serial, record))
        if len(self.top_terms) > TOP_TERM_LIMIT:
            heapq.heappop(self.top_terms)


def _read_csv_chunks(path: Path, columns: set[str], chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    try:
        yield from pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=lambda column: column in columns,
            chunksize=chunksize,
        )
    except pd.errors.EmptyDataError:
        return
    except ValueError:
        return


def _method_clusterprofiler_dir(results_root: Path, method: str) -> Path:
    nested = results_root / method / "clusterprofiler"
    if nested.is_dir():
        return nested
    return results_root / "clusterprofiler"


def _method_enrichment_files(results_root: Path, method: str) -> list[Path]:
    clusterprofiler_dir = _method_clusterprofiler_dir(results_root, method)
    batch_dir = clusterprofiler_dir / "batch_results"
    files = sorted(batch_dir.glob(f"Gvp*_{method}.csv"))
    if files:
        return files
    merged = clusterprofiler_dir / "clusterprofiler_enrichment.csv"
    return [merged] if merged.is_file() else []


def _totals_from_motif_sets(motif_sets_path: Path, method: str) -> dict[str, int]:
    if not motif_sets_path.is_file():
        return {}
    motifs: dict[str, set[str]] = defaultdict(set)
    for chunk in _read_csv_chunks(motif_sets_path, MOTIF_SET_COLUMNS):
        if not MOTIF_SET_COLUMNS.issubset(chunk.columns):
            continue
        chunk = chunk[chunk["method"].astype(str).eq(method)]
        for row in chunk[["family", "motif_cluster"]].dropna().itertuples(index=False):
            motifs[str(row.family)].add(str(row.motif_cluster))
    return {family: len(values) for family, values in motifs.items()}


def _totals_from_exp3_summary(exp3_summary: Path, method: str) -> dict[str, int]:
    if not exp3_summary.is_file():
        return {}
    frame = pd.read_csv(exp3_summary, encoding="utf-8-sig")
    required = {"family", "method", "n_motifs"}
    if not required.issubset(frame.columns):
        return {}
    frame = frame[frame["method"].astype(str).eq(method)].copy()
    frame["n_motifs"] = pd.to_numeric(frame["n_motifs"], errors="coerce").fillna(0)
    return {
        str(row.family): int(row.n_motifs)
        for row in frame[["family", "n_motifs"]].itertuples(index=False)
    }


def _load_total_motifs(
    results_root: Path,
    exp3_summary: Path,
    method: str,
    row_motif_counts: dict[str, int],
) -> tuple[dict[str, int], str]:
    clusterprofiler_dir = _method_clusterprofiler_dir(results_root, method)
    motif_set_totals = _totals_from_motif_sets(clusterprofiler_dir / "motif_sets_long.csv", method)
    if motif_set_totals:
        return motif_set_totals, "motif_sets_long"
    exp3_totals = _totals_from_exp3_summary(exp3_summary, method)
    if exp3_totals:
        return exp3_totals, "exp3_clustering_comparison"
    return row_motif_counts, "enrichment_rows_only"


def _load_method_enrichment(results_root: Path, method: str, alpha: float) -> MethodEnrichmentStats:
    stats = MethodEnrichmentStats(method=method)
    for path in _method_enrichment_files(results_root, method):
        for chunk in _read_csv_chunks(path, ENRICHMENT_COLUMNS):
            stats.add_rows(chunk, alpha=alpha)
    return stats


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _build_family_rows(
    results_root: Path,
    exp3_summary: Path,
    methods: list[str],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_rows: list[dict[str, object]] = []
    top_terms: list[dict[str, object]] = []

    for method in methods:
        enrichment = _load_method_enrichment(results_root, method, alpha)
        row_motif_counts = {
            family: len(motifs)
            for family, motifs in enrichment.motifs_with_rows.items()
        }
        total_motifs, source = _load_total_motifs(
            results_root,
            exp3_summary,
            method,
            row_motif_counts,
        )
        families = sorted(set(total_motifs) | set(row_motif_counts) | set(enrichment.supported_motifs))
        for family in families:
            total = int(total_motifs.get(family, row_motif_counts.get(family, 0)))
            with_rows = int(row_motif_counts.get(family, 0))
            supported = len(enrichment.supported_motifs.get(family, set()))
            no_rows = max(total - with_rows, 0)
            min_adjusted = enrichment.min_adjusted.get(family, math.inf)
            family_rows.append(
                {
                    "family": family,
                    "method": method,
                    "total_motifs": total,
                    "motifs_with_enrichment_rows": with_rows,
                    "motifs_without_enrichment_rows": no_rows,
                    "supported_motifs": supported,
                    "not_supported_motifs": max(total - supported, 0),
                    "all_enrichment_term_rows": int(enrichment.term_rows.get(family, 0)),
                    "significant_enrichment_term_rows": int(enrichment.significant_term_rows.get(family, 0)),
                    "supported_rate_all_motifs": _safe_rate(supported, total),
                    "supported_rate_motifs_with_rows": _safe_rate(supported, with_rows),
                    "min_adjusted_value": "" if math.isinf(min_adjusted) else min_adjusted,
                    "denominator_source": source,
                }
            )
        top_terms.extend(
            record
            for _, _, record in sorted(
                enrichment.top_terms,
                key=lambda item: item[2]["p_adjust"],
            )
        )
    return pd.DataFrame(family_rows), pd.DataFrame(top_terms)


def _build_method_summary(family_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in family_summary.groupby("method", sort=False):
        total = int(group["total_motifs"].sum())
        with_rows = int(group["motifs_with_enrichment_rows"].sum())
        supported = int(group["supported_motifs"].sum())
        rows.append(
            {
                "method": method,
                "families": int(group["family"].nunique()),
                "total_motifs": total,
                "motifs_with_enrichment_rows": with_rows,
                "motifs_without_enrichment_rows": int(group["motifs_without_enrichment_rows"].sum()),
                "supported_motifs": supported,
                "not_supported_motifs": max(total - supported, 0),
                "all_enrichment_term_rows": int(group["all_enrichment_term_rows"].sum()),
                "significant_enrichment_term_rows": int(group["significant_enrichment_term_rows"].sum()),
                "supported_rate_all_motifs": _safe_rate(supported, total),
                "supported_rate_motifs_with_rows": _safe_rate(supported, with_rows),
                "denominator_source": ";".join(sorted(set(group["denominator_source"].astype(str)))),
            }
        )
    return pd.DataFrame(rows).sort_values("supported_rate_all_motifs", ascending=False)


def _build_pairwise(family_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = family_summary.pivot_table(
        index="family",
        columns="method",
        values="supported_rate_all_motifs",
        aggfunc="first",
    )
    rows: list[dict[str, object]] = []
    for method_a, method_b in combinations(list(pivot.columns), 2):
        paired = pivot[[method_a, method_b]].dropna()
        if paired.empty:
            continue
        diff = paired[method_a] - paired[method_b]
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "families_compared": int(len(paired)),
                "method_a_mean_rate": float(paired[method_a].mean()),
                "method_b_mean_rate": float(paired[method_b].mean()),
                "mean_rate_difference_a_minus_b": float(diff.mean()),
                "median_rate_difference_a_minus_b": float(diff.median()),
                "families_a_higher": int((diff > 0).sum()),
                "families_equal": int((diff == 0).sum()),
                "families_b_higher": int((diff < 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rate_difference_a_minus_b", ascending=False)


def _build_primary_vs_baseline(
    family_summary: pd.DataFrame,
    primary_methods: set[str],
) -> pd.DataFrame:
    pivot = family_summary.pivot_table(
        index="family",
        columns="method",
        values="supported_rate_all_motifs",
        aggfunc="first",
    )
    controls = [method for method in pivot.columns if method not in primary_methods]
    rows: list[dict[str, object]] = []
    for primary in [method for method in pivot.columns if method in primary_methods]:
        for control in controls:
            paired = pivot[[primary, control]].dropna()
            if paired.empty:
                continue
            diff = paired[primary] - paired[control]
            rows.append(
                {
                    "primary_method": primary,
                    "control_method": control,
                    "families_compared": int(len(paired)),
                    "primary_mean_rate": float(paired[primary].mean()),
                    "control_mean_rate": float(paired[control].mean()),
                    "mean_rate_difference_primary_minus_control": float(diff.mean()),
                    "median_rate_difference_primary_minus_control": float(diff.median()),
                    "families_primary_higher": int((diff > 0).sum()),
                    "families_equal": int((diff == 0).sum()),
                    "families_control_higher": int((diff < 0).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["primary_method", "mean_rate_difference_primary_minus_control"],
        ascending=[True, False],
    )


def _build_family_ranks(family_summary: pd.DataFrame) -> pd.DataFrame:
    ranked = family_summary.copy()
    ranked["rank_within_family"] = ranked.groupby("family")[
        "supported_rate_all_motifs"
    ].rank(method="min", ascending=False)
    return ranked.sort_values(["family", "rank_within_family", "method"])[
        [
            "family",
            "method",
            "rank_within_family",
            "supported_rate_all_motifs",
            "supported_motifs",
            "total_motifs",
        ]
    ]


def build_comparison_tables(
    results_root: Path,
    exp3_summary: Path,
    methods: list[str] | None = None,
    primary_methods: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    selected_methods = list(methods or DEFAULT_METHODS)
    selected_primary = set(primary_methods or DEFAULT_PRIMARY_METHODS)
    family_summary, top_terms = _build_family_rows(
        results_root=Path(results_root),
        exp3_summary=Path(exp3_summary),
        methods=selected_methods,
        alpha=alpha,
    )
    method_summary = _build_method_summary(family_summary)
    pairwise = _build_pairwise(family_summary)
    primary_vs_baseline = _build_primary_vs_baseline(family_summary, selected_primary)
    family_ranks = _build_family_ranks(family_summary)
    return {
        "method_summary": method_summary,
        "family_method_summary": family_summary.sort_values(["method", "family"]),
        "pairwise_method_comparison": pairwise,
        "primary_vs_baseline": primary_vs_baseline,
        "family_method_ranks": family_ranks,
        "top_significant_terms": top_terms,
    }


def _format_percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column.endswith("rate") or column.endswith("_motifs"):
                values.append(str(value))
            elif isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _write_report(tables: dict[str, pd.DataFrame], output_dir: Path, alpha: float) -> None:
    method_summary = tables["method_summary"].copy()
    method_summary["supported_rate"] = method_summary["supported_rate_all_motifs"].map(_format_percent)
    method_summary["row_rate"] = method_summary["supported_rate_motifs_with_rows"].map(_format_percent)
    pairwise = tables["primary_vs_baseline"].copy()
    if not pairwise.empty:
        pairwise["mean_rate_diff"] = pairwise[
            "mean_rate_difference_primary_minus_control"
        ].map(_format_percent)
        pairwise["median_rate_diff"] = pairwise[
            "median_rate_difference_primary_minus_control"
        ].map(_format_percent)
    lines = [
        "# Exp5 Functional Enrichment Method Comparison",
        "",
        f"Significance threshold: FDR-adjusted p <= {alpha:g}.",
        "",
        "## Method Summary",
        "",
        _markdown_table(
            method_summary,
            [
                "method",
                "families",
                "total_motifs",
                "supported_motifs",
                "supported_rate",
                "motifs_with_enrichment_rows",
                "row_rate",
                "denominator_source",
            ],
        ),
        "",
        "## Primary-vs-Control Pairwise Family Differences",
        "",
        _markdown_table(
            pairwise,
            [
                "primary_method",
                "control_method",
                "families_compared",
                "mean_rate_diff",
                "median_rate_diff",
                "families_primary_higher",
                "families_equal",
                "families_control_higher",
            ],
        ),
        "",
        "## Interpretation Notes",
        "",
        "- `supported_rate_all_motifs` uses all motif clusters as the denominator.",
        "- `supported_rate_motifs_with_rows` uses only motif clusters that produced at least one enrichment row.",
        "- Motifs with no enrichment rows are retained in the all-motif denominator to avoid inflating support rates.",
        "- Pairwise differences are descriptive family-level comparisons, not causal or confirmatory hypothesis tests.",
    ]
    (output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_plots(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as error:  # pragma: no cover - optional plotting dependency
        (output_dir / "plot_warnings.txt").write_text(str(error), encoding="utf-8")
        return

    method_summary = tables["method_summary"].sort_values("supported_rate_all_motifs")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(method_summary["method"], method_summary["supported_rate_all_motifs"], color="#4C78A8")
    ax.set_xlabel("Supported motif fraction")
    ax.set_xlim(0, max(0.05, float(method_summary["supported_rate_all_motifs"].max()) * 1.15))
    for y, value in enumerate(method_summary["supported_rate_all_motifs"]):
        ax.text(value, y, f" {value:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "method_supported_rate_bar.png", dpi=300)
    plt.close(fig)

    family_summary = tables["family_method_summary"]
    pivot = family_summary.pivot_table(
        index="family",
        columns="method",
        values="supported_rate_all_motifs",
        aggfunc="first",
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=max(0.01, float(pivot.to_numpy().max())))
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Supported motif fraction by family and method")
    fig.colorbar(im, ax=ax, label="Supported fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "family_method_supported_rate_heatmap.png", dpi=300)
    plt.close(fig)


def write_comparison_outputs(tables: dict[str, pd.DataFrame], output_dir: Path, alpha: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    _write_report(tables, output_dir, alpha)
    _write_plots(tables, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Exp5 functional-enrichment support across GvpMiner and baseline motif methods."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=WORK_RESULTS / "exp5_external_functional_enrichment",
    )
    parser.add_argument(
        "--exp3-summary",
        type=Path,
        default=WORK_RESULTS / "exp3_domain_discovery" / "clustering_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORK_RESULTS / "exp5_external_functional_enrichment" / "comparison",
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--primary-methods", nargs="+", default=list(DEFAULT_PRIMARY_METHODS))
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tables = build_comparison_tables(
        results_root=args.results_root,
        exp3_summary=args.exp3_summary,
        methods=args.methods,
        primary_methods=args.primary_methods,
        alpha=args.alpha,
    )
    write_comparison_outputs(tables, args.output_dir, args.alpha)
    print(f"Wrote comparison outputs to {args.output_dir}")
    print(tables["method_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
