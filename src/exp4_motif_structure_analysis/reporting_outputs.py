from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _fmt(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    return f"{number:.4g}"


def _best_stat_by_cluster(
    stat_rows: Sequence[dict[str, object]], q_field: str = "q_value"
) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in stat_rows:
        cluster = str(row.get("motif_cluster", ""))
        q_value = _to_float(row.get(q_field))
        if not cluster or q_value is None:
            continue
        if cluster not in best or q_value < float(best[cluster].get(q_field, 1.0)):
            best[cluster] = row
    return best


def summarize_motif_clusters(
    *,
    feature_rows: Sequence[dict[str, object]],
    stat_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in feature_rows:
        by_cluster[str(row.get("motif_cluster", row.get("motif_id", "")))].append(row)

    q_field = "primary_q_value" if any("primary_q_value" in row for row in stat_rows) else "q_value"
    best_stats = _best_stat_by_cluster(stat_rows, q_field=q_field)
    exploratory_best = _best_stat_by_cluster(stat_rows)

    summaries = []
    for cluster, rows in sorted(by_cluster.items()):
        plddt_values = [_to_float(row.get("motif_mean_pLDDT")) for row in rows]
        plddt_values = [value for value in plddt_values if value is not None]
        best = best_stats.get(cluster, {})
        q_value = _to_float(best.get(q_field))
        mean_plddt = float(mean(plddt_values)) if plddt_values else None
        summaries.append(
            {
                "motif_cluster": cluster,
                "n_instances": len(rows),
                "n_sequences": len({str(row.get("sequence_id", "")) for row in rows}),
                "mean_motif_pLDDT": mean_plddt if mean_plddt is not None else "",
                "high_confidence_fraction": sum(str(row.get("confidence_category", "")) == "high" for row in rows) / max(1, len(rows)),
                "best_feature": best.get("feature", ""),
                "best_control_type": best.get("control_type", ""),
                "best_q_value": q_value if q_value is not None else "",
                "best_effect_median_difference": best.get("effect_median_difference", ""),
                "primary_n_proteins": best.get("n_proteins", ""),
                "primary_structural_association": q_value is not None and q_value <= 0.05,
                "low_prediction_confidence": mean_plddt is not None and mean_plddt < 70.0,
                "exploratory_best_q_value": exploratory_best.get(cluster, {}).get("q_value", ""),
                "significant_endpoints": sum(
                    1 for row in stat_rows
                    if str(row.get("motif_cluster", "")) == cluster
                    and _to_float(row.get(q_field)) is not None
                    and float(row[q_field]) <= 0.05
                ),
            }
        )
    return summaries


def write_analysis_report(
    path: Path,
    *,
    summary: dict[str, object],
    cluster_rows: Sequence[dict[str, object]],
    figure_paths: Sequence[Path],
    warnings: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = sorted(
        cluster_rows,
        key=lambda row: _to_float(row.get("best_q_value")) or 1.0,
    )[:10]
    lines = [
        "# Motif Structure Analysis Report",
        "",
        "## Run Summary",
        "",
        f"- Motif instances analyzed: {summary.get('motif_instances_analyzed', 0)}",
        f"- Control windows analyzed: {summary.get('control_windows_analyzed', 0)}",
        f"- Structural tests: {summary.get('structural_tests', 0)}",
        f"- Motif clusters: {len(cluster_rows)}",
        f"- Publication figures: {len(figure_paths)}",
        "",
        "## Lowest Primary Structural q-values",
        "",
        "| Motif cluster | Primary feature | Primary q-value | Effect | Primary significant endpoints | Low prediction confidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in top:
        lines.append(
            f"| {row.get('motif_cluster', '')} | "
            f"{row.get('best_feature', '')} | {_fmt(row.get('best_q_value'))} | "
            f"{_fmt(row.get('best_effect_median_difference'))} | "
            f"{row.get('significant_endpoints', '')} | "
            f"{row.get('low_prediction_confidence', '')} |"
        )
    lines.extend(["", "## Figure Files", ""])
    if figure_paths:
        lines.extend(f"- `{path.as_posix()}`" for path in figure_paths)
    else:
        lines.append("- No figures generated.")
    lines.extend(["", "## Notes", ""])
    lines.append("- Figures are generated from saved result tables and do not alter statistical calculations.")
    lines.append("- Interpret predicted-structure evidence as support for candidate structural relevance, not proof of function.")
    lines.append("- Headline q-values use composition-matched controls and prespecified physical endpoints; other tests are exploratory.")
    lines.append("- The position-matched group is a same-family positional reference, not a motif-free negative control.")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
