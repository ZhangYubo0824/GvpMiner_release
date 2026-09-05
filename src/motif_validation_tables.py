from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SIGNIFICANCE_THRESHOLD = 0.05
FILTER_FAILURE_TEXT = "No gene sets have size between 10 and 500"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        field_order: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    field_order.append(key)
                    seen.add(key)
        fieldnames = field_order
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return value


def _to_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_int(value: object, default: int = 0) -> int:
    number = _to_float(value)
    if math.isnan(number):
        return default
    return int(number)


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def family_method_key(family: str, method: str) -> str:
    return f"{family}_{method}"


def parse_family_method_from_name(path: Path) -> tuple[str, str]:
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]


def split_motif_cluster(motif_cluster: str) -> tuple[str, str]:
    parts = str(motif_cluster).split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def normalized_motif_id(row: Mapping[str, object]) -> str:
    return _clean(row.get("motif_cluster") or row.get("motif_uid") or row.get("motif_id"))


def normalize_catalog_row(row: Mapping[str, object]) -> dict[str, object]:
    motif_cluster = normalized_motif_id(row)
    family = _clean(row.get("family"))
    method = _clean(row.get("method"))
    if not family or not method:
        parsed_family, parsed_method = split_motif_cluster(motif_cluster)
        family = family or parsed_family
        method = method or parsed_method
    normalized = dict(row)
    normalized["motif_cluster"] = motif_cluster
    normalized["family"] = family
    normalized["method"] = method
    return normalized


def load_motif_catalogs(exp3_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(exp3_dir).glob("*/motif_catalog_*.csv")):
        family = path.parent.name
        method = path.stem.replace("motif_catalog_", "")
        for row in read_csv_rows(path):
            row.setdefault("family", family)
            row.setdefault("method", method)
            rows.append(normalize_catalog_row(row))
    return rows


def load_motif_representatives(exp3_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(exp3_dir).glob("*/motif_representative_windows_*.csv")):
        rows.extend(read_csv_rows(path))
    return rows


def attach_motif_representatives(
    rows: Sequence[Mapping[str, object]],
    representative_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    representatives = {
        normalized_motif_id(row): row
        for row in representative_rows
        if normalized_motif_id(row)
    }
    attached: list[dict[str, object]] = []
    for source_row in rows:
        row = dict(source_row)
        representative = representatives.get(normalized_motif_id(row), {})
        for key in [
            "representative_window_id",
            "representative_parent_id",
            "representative_start",
            "representative_end",
            "representative_length",
            "representative_sequence",
            "distance_to_cluster_centroid",
            "n_cluster_windows",
        ]:
            row[key] = _clean(representative.get(key))
        attached.append(row)
    return attached


def load_structural_inputs(exp4_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    enrichment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for enrichment_path in sorted(Path(exp4_dir).glob("*/*/structural_enrichment_results.csv")):
        family = enrichment_path.parent.parent.name
        method = enrichment_path.parent.name
        for row in read_csv_rows(enrichment_path):
            row.setdefault("family", family)
            row.setdefault("method", method)
            row["motif_cluster"] = normalized_motif_id(row)
            enrichment_rows.append(row)
        summary_path = enrichment_path.parent / "motif_cluster_summary.csv"
        for row in read_csv_rows(summary_path):
            row.setdefault("family", family)
            row.setdefault("method", method)
            row["motif_cluster"] = normalized_motif_id(row)
            summary_rows.append(row)
    return enrichment_rows, summary_rows


def discover_clusterprofiler_batches(
    batch_results_dir: Path,
    expected_family_methods: Iterable[tuple[str, str]] = (),
) -> list[dict[str, object]]:
    batch_results_dir = Path(batch_results_dir)
    expected = {family_method_key(family, method): (family, method) for family, method in expected_family_methods}
    for path in batch_results_dir.glob("Gvp*_M[23].csv"):
        expected[family_method_key(*parse_family_method_from_name(path))] = parse_family_method_from_name(path)
    for path in batch_results_dir.glob("Gvp*_M[23].log"):
        expected[family_method_key(*parse_family_method_from_name(path))] = parse_family_method_from_name(path)

    rows: list[dict[str, object]] = []
    for key in sorted(expected):
        family, method = expected[key]
        csv_path = batch_results_dir / f"{key}.csv"
        log_path = batch_results_dir / f"{key}.log"
        csv_rows = read_csv_rows(csv_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        has_csv = csv_path.exists()
        has_log = log_path.exists()
        log_filter_failed = FILTER_FAILURE_TEXT in log_text
        log_error = any(marker in log_text for marker in ["Error", "ERROR", "Execution halted", "Traceback"])
        if has_csv:
            coverage_status = "has_clusterprofiler_csv"
        elif has_log:
            coverage_status = "log_only_filter_failed" if log_filter_failed else "log_only_no_csv"
        else:
            coverage_status = "not_tested_external"
        rows.append(
            {
                "family": family,
                "method": method,
                "family_method": key,
                "has_clusterprofiler_csv": has_csv,
                "has_clusterprofiler_log": has_log,
                "clusterprofiler_csv_rows": len(csv_rows),
                "log_filter_failed": log_filter_failed,
                "log_error_detected": log_error,
                "coverage_status": coverage_status,
                "csv_path": str(csv_path) if has_csv else "",
                "log_path": str(log_path) if has_log else "",
            }
        )
    return rows


def build_external_motif_statuses(
    *,
    motif_sets_path: Path,
    batch_results_dir: Path,
    catalog_rows: Sequence[Mapping[str, object]] = (),
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
    scan_motif_sets: bool | None = None,
) -> list[dict[str, object]]:
    if scan_motif_sets is None:
        scan_motif_sets = not bool(catalog_rows)
    motif_set_rows = read_csv_rows(motif_sets_path) if scan_motif_sets else []
    motif_set_clusters = {normalized_motif_id(row) for row in motif_set_rows}
    motif_meta: dict[str, dict[str, object]] = {}

    for row in catalog_rows:
        normalized = normalize_catalog_row(row)
        motif_cluster = normalized_motif_id(normalized)
        if motif_cluster:
            motif_meta[motif_cluster] = _motif_base_row(normalized)
            if not scan_motif_sets:
                motif_set_clusters.add(motif_cluster)

    for row in motif_set_rows:
        motif_cluster = normalized_motif_id(row)
        if not motif_cluster:
            continue
        base = _motif_base_row(row)
        motif_meta.setdefault(motif_cluster, base)

    result_rows_by_motif: dict[str, list[dict[str, object]]] = defaultdict(list)
    result_family_methods: set[tuple[str, str]] = set()
    for path in sorted(Path(batch_results_dir).glob("Gvp*_M[23].csv")):
        path_family, path_method = parse_family_method_from_name(path)
        result_family_methods.add((path_family, path_method))
        for row in read_csv_rows(path):
            motif_cluster = normalized_motif_id(row)
            if not motif_cluster:
                continue
            row.setdefault("family", path_family)
            row.setdefault("method", path_method)
            result_rows_by_motif[motif_cluster].append(row)
            motif_meta.setdefault(motif_cluster, _motif_base_row(row))
            motif_set_clusters.add(motif_cluster)

    expected_family_methods = {
        (_clean(row.get("family")), _clean(row.get("method")))
        for row in motif_meta.values()
        if _clean(row.get("family")) and _clean(row.get("method"))
    }
    expected_family_methods.update(result_family_methods)
    batch_rows = discover_clusterprofiler_batches(batch_results_dir, expected_family_methods)
    batch_by_key = {row["family_method"]: row for row in batch_rows}

    statuses: list[dict[str, object]] = []
    for motif_cluster in sorted(motif_meta):
        base = dict(motif_meta[motif_cluster])
        family = _clean(base.get("family"))
        method = _clean(base.get("method"))
        key = family_method_key(family, method)
        batch = batch_by_key.get(key, {})
        result_rows = result_rows_by_motif.get(motif_cluster, [])
        if result_rows:
            status_row = _status_from_result_rows(result_rows, significance_threshold)
        else:
            status_row = _status_without_result_rows(
                motif_cluster=motif_cluster,
                motif_in_sets=motif_cluster in motif_set_clusters,
                batch=batch,
            )
        base.update(status_row)
        base["family_method"] = key
        base["clusterprofiler_csv_rows_for_motif"] = len(result_rows)
        base["batch_coverage_status"] = _clean(batch.get("coverage_status"))
        base["batch_log_filter_failed"] = bool(batch.get("log_filter_failed", False))
        statuses.append(base)
    return statuses


def _motif_base_row(row: Mapping[str, object]) -> dict[str, object]:
    motif_cluster = normalized_motif_id(row)
    family = _clean(row.get("family"))
    method = _clean(row.get("method"))
    if not family or not method:
        parsed_family, parsed_method = split_motif_cluster(motif_cluster)
        family = family or parsed_family
        method = method or parsed_method
    return {
        "family": family,
        "method": method,
        "motif_cluster": motif_cluster,
        "motif_consensus": _clean(row.get("motif_consensus")),
        "mean_start": _clean(row.get("mean_start")),
        "mean_end": _clean(row.get("mean_end")),
        "n_unique_sequences": _clean(row.get("n_unique_sequences") or row.get("n_sequences")),
        "support_ratio": _clean(row.get("support_ratio")),
    }


def _status_from_result_rows(
    result_rows: Sequence[Mapping[str, object]],
    significance_threshold: float,
) -> dict[str, object]:
    scored_rows = []
    for row in result_rows:
        score = _adjusted_score(row)
        if math.isnan(score):
            continue
        scored_rows.append((score, row))
    scored_rows.sort(key=lambda item: item[0])
    significant = [(score, row) for score, row in scored_rows if score <= significance_threshold]
    status = "external_enrichment_supported" if significant else "external_enrichment_not_significant"
    return {
        "external_enrichment_status": status,
        "missing_reason": "",
        "min_adjusted_value": scored_rows[0][0] if scored_rows else "",
        "n_external_terms": len(result_rows),
        "n_significant_external_terms": len(significant),
        "significant_terms": _format_terms(significant),
        "top_terms": _format_terms(scored_rows[:5], include_score=True),
        "needs_literature_validation": status != "external_enrichment_supported",
    }


def _status_without_result_rows(
    *,
    motif_cluster: str,
    motif_in_sets: bool,
    batch: Mapping[str, object],
) -> dict[str, object]:
    has_csv = _is_true(batch.get("has_clusterprofiler_csv"))
    has_log = _is_true(batch.get("has_clusterprofiler_log"))
    log_filter_failed = _is_true(batch.get("log_filter_failed"))
    if not motif_in_sets:
        status = "not_tested_external"
        reason = "motif_not_in_motif_sets_long"
    elif has_csv and log_filter_failed:
        status = "external_filter_failed"
        reason = "gene_set_size_filter_or_no_result"
    elif has_csv:
        status = "external_enrichment_no_terms"
        reason = "tested_but_no_result_rows_for_motif"
    elif has_log and log_filter_failed:
        status = "external_filter_failed"
        reason = "gene_set_size_filter_or_no_csv"
    elif has_log:
        status = "external_enrichment_unavailable"
        reason = "log_only_no_csv"
    else:
        status = "not_tested_external"
        reason = "missing_clusterprofiler_csv_and_log"
    return {
        "external_enrichment_status": status,
        "missing_reason": reason,
        "min_adjusted_value": "",
        "n_external_terms": 0,
        "n_significant_external_terms": 0,
        "significant_terms": "",
        "top_terms": "",
        "needs_literature_validation": True,
    }


def _adjusted_score(row: Mapping[str, object]) -> float:
    p_adjust = _to_float(row.get("p.adjust"))
    if not math.isnan(p_adjust):
        return p_adjust
    return _to_float(row.get("qvalue"))


def _format_terms(scored_rows: Sequence[tuple[float, Mapping[str, object]]], include_score: bool = False) -> str:
    terms: list[str] = []
    for score, row in scored_rows:
        term_id = _clean(row.get("ID") or row.get("term_id"))
        description = _clean(row.get("Description") or row.get("term_name"))
        if not term_id and not description:
            continue
        term = f"{term_id}|{description}" if description else term_id
        if include_score:
            term = f"{term}|{score:.6g}"
        terms.append(term)
    return ";".join(terms)


def summarize_structural_significance(
    *,
    enrichment_rows: Sequence[Mapping[str, object]],
    cluster_summary_rows: Sequence[Mapping[str, object]],
    catalog_rows: Sequence[Mapping[str, object]],
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
) -> list[dict[str, object]]:
    catalog_by_cluster = {
        normalized_motif_id(row): normalize_catalog_row(row)
        for row in catalog_rows
        if normalized_motif_id(row)
    }
    summary_by_cluster = {
        normalized_motif_id(row): row
        for row in cluster_summary_rows
        if normalized_motif_id(row)
    }
    enrichment_by_cluster: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in enrichment_rows:
        motif_cluster = normalized_motif_id(row)
        if motif_cluster:
            enrichment_by_cluster[motif_cluster].append(row)

    motif_ids = set(catalog_by_cluster) | set(summary_by_cluster) | set(enrichment_by_cluster)
    rows: list[dict[str, object]] = []
    for motif_cluster in sorted(motif_ids):
        catalog = catalog_by_cluster.get(motif_cluster, {})
        summary = summary_by_cluster.get(motif_cluster, {})
        family, method = split_motif_cluster(motif_cluster)
        family = _clean(catalog.get("family") or summary.get("family") or family)
        method = _clean(catalog.get("method") or summary.get("method") or method)
        sig_rows = [
            row
            for row in enrichment_by_cluster.get(motif_cluster, [])
            if _to_float(row.get("q_value")) <= significance_threshold
        ]
        min_q = _min_float([row.get("q_value") for row in sig_rows])
        n_instances = _to_int(
            _max_int(
                row.get("n_proteins") or row.get("n_instances")
                for row in sig_rows
            ),
            _to_int(summary.get("n_sequences"), _to_int(summary.get("n_instances"))),
        )
        mean_plddt = _to_float(summary.get("mean_motif_pLDDT") or summary.get("motif_mean_pLDDT"))
        endpoint_details = [_endpoint_detail(row) for row in sig_rows]
        structural_types = _infer_structural_types(sig_rows, mean_plddt)
        row = {
            "family": family,
            "method": method,
            "motif_cluster": motif_cluster,
            "motif_consensus": _clean(catalog.get("motif_consensus")),
            "mean_start": _clean(catalog.get("mean_start")),
            "mean_end": _clean(catalog.get("mean_end")),
            "n_unique_sequences": _clean(catalog.get("n_unique_sequences") or summary.get("n_sequences")),
            "support_ratio": _clean(catalog.get("support_ratio")),
            "structural_significant": bool(sig_rows),
            "significant_endpoint_count": len(sig_rows),
            "significant_endpoints": ";".join(endpoint_details),
            "min_structural_q_value": min_q if not math.isnan(min_q) else "",
            "n_structural_instances": n_instances,
            "mean_motif_pLDDT": mean_plddt if not math.isnan(mean_plddt) else "",
            "structural_type": ";".join(structural_types),
            "structural_evidence_strength": _structural_evidence_strength(sig_rows, min_q, n_instances, mean_plddt),
            "possible_biological_significance": _possible_structural_meaning(structural_types),
        }
        rows.append(row)
    return rows


def _endpoint_detail(row: Mapping[str, object]) -> str:
    feature = _clean(row.get("feature"))
    effect = _to_float(row.get("effect_median_difference"))
    direction = "neutral"
    if not math.isnan(effect):
        if effect > 0:
            direction = "positive"
        elif effect < 0:
            direction = "negative"
    q_value = _to_float(row.get("q_value"))
    if math.isnan(q_value):
        return f"{feature}:{direction}"
    effect_text = "" if math.isnan(effect) else f",effect={effect:.6g}"
    return f"{feature}:{direction}(q={q_value:.6g}{effect_text})"


def _infer_structural_types(sig_rows: Sequence[Mapping[str, object]], mean_plddt: float) -> list[str]:
    types: list[str] = []
    for feature, effect in _best_effect_by_feature(sig_rows).items():
        if math.isnan(effect):
            continue
        if feature == "contact_density" and effect > 0:
            types.append("core_or_packing_enriched")
        elif feature == "contact_density" and effect < 0:
            types.append("low_contact_density_or_flexible_context")
        elif feature in {"relative_SASA", "surface_residue_fraction"} and effect > 0:
            types.append("surface_exposed_or_display_enriched")
        elif feature in {"relative_SASA", "surface_residue_fraction"} and effect < 0:
            types.append("buried_or_less_surface_exposed")
        elif feature == "distance_to_protein_centroid" and effect < 0:
            types.append("centroid_proximal")
        elif feature == "distance_to_protein_centroid" and effect > 0:
            types.append("centroid_distal")
    if not math.isnan(mean_plddt):
        if mean_plddt >= 70:
            types.append("high_confidence_prediction")
        else:
            types.append("low_confidence_or_flexible_prediction")
    return sorted(dict.fromkeys(types))


def _best_effect_by_feature(sig_rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    best: dict[str, tuple[float, float]] = {}
    for row in sig_rows:
        feature = _clean(row.get("feature"))
        if not feature:
            continue
        q_value = _to_float(row.get("q_value"))
        effect = _to_float(row.get("effect_median_difference"))
        if math.isnan(q_value) or math.isnan(effect):
            continue
        current = best.get(feature)
        if current is None or q_value < current[0]:
            best[feature] = (q_value, effect)
    return {feature: effect for feature, (_, effect) in best.items()}


def _structural_evidence_strength(
    sig_rows: Sequence[Mapping[str, object]],
    min_q: float,
    n_instances: int,
    mean_plddt: float,
) -> str:
    if not sig_rows:
        return "not_significant"
    if not math.isnan(mean_plddt) and mean_plddt < 70:
        return "uncertain_low_plddt"
    if not math.isnan(min_q) and min_q <= 0.01 and n_instances >= 20:
        return "strong"
    if n_instances >= 10:
        return "moderate"
    return "weak"


def _possible_structural_meaning(structural_types: Sequence[str]) -> str:
    meanings = []
    types = set(structural_types)
    if "core_or_packing_enriched" in types or "centroid_proximal" in types:
        meanings.append("possible structural core, fold-stability, or shell-packing region")
    if "surface_exposed_or_display_enriched" in types or "centroid_distal" in types:
        meanings.append("possible surface interaction, display, transport, or engineering-accessible region")
    if "low_confidence_or_flexible_prediction" in types or "low_contact_density_or_flexible_context" in types:
        meanings.append("interpret cautiously as a possible flexible or low-confidence region")
    if "high_confidence_prediction" in types:
        meanings.append("structure-derived interpretation has higher pLDDT support")
    return "; ".join(dict.fromkeys(meanings))


def _min_float(values: Iterable[object]) -> float:
    parsed = [_to_float(value) for value in values]
    parsed = [value for value in parsed if not math.isnan(value)]
    return min(parsed) if parsed else math.nan


def _max_int(values: Iterable[object]) -> int:
    parsed = [_to_int(value) for value in values]
    return max(parsed) if parsed else 0


def integrate_motif_evidence(
    structural_rows: Sequence[Mapping[str, object]],
    external_rows: Sequence[Mapping[str, object]],
    literature_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    structural_by_cluster = {normalized_motif_id(row): dict(row) for row in structural_rows if normalized_motif_id(row)}
    external_by_cluster = {normalized_motif_id(row): dict(row) for row in external_rows if normalized_motif_id(row)}
    literature_by_cluster = {normalized_motif_id(row): dict(row) for row in literature_rows if normalized_motif_id(row)}
    motif_ids = set(structural_by_cluster) | set(external_by_cluster) | set(literature_by_cluster)
    rows: list[dict[str, object]] = []
    for motif_cluster in sorted(motif_ids):
        structural = structural_by_cluster.get(motif_cluster, {})
        external = external_by_cluster.get(motif_cluster, {})
        literature = literature_by_cluster.get(motif_cluster, {})
        family, method = split_motif_cluster(motif_cluster)
        row: dict[str, object] = {}
        row.update(structural)
        for key, value in external.items():
            if key not in row or row[key] in {"", None}:
                row[key] = value
            elif key not in {"family", "method", "motif_cluster"}:
                row[f"external_{key}"] = value
        for key, value in literature.items():
            if key not in row or row[key] in {"", None}:
                row[key] = value
            elif key not in {"family", "method", "motif_cluster"}:
                row[f"literature_{key}"] = value
        row["family"] = _clean(row.get("family") or family)
        row["method"] = _clean(row.get("method") or method)
        row["motif_cluster"] = motif_cluster
        row.setdefault("external_enrichment_status", "not_tested_external")
        row.setdefault("significant_terms", "")
        row.setdefault("missing_reason", "")
        row.setdefault("candidate_function", _candidate_function(row))
        row.setdefault("literature_evidence", "")
        row.setdefault("evidence_grade", "")
        row["literature_support_status"] = _literature_support_status(row)
        row["motif_validation_category"] = _motif_validation_category(row)
        rows.append(row)
    return rows


def _candidate_function(row: Mapping[str, object]) -> str:
    significant_terms = _clean(row.get("significant_terms"))
    if significant_terms:
        return f"external enrichment terms: {significant_terms}"
    structural_meaning = _clean(row.get("possible_biological_significance"))
    if structural_meaning:
        return structural_meaning
    return ""


def _literature_support_status(row: Mapping[str, object]) -> str:
    grade = _clean(row.get("evidence_grade") or row.get("literature_evidence_grade")).upper()
    if grade in {"A", "B", "C"}:
        return "literature_supported"
    if grade == "D":
        return "weak_background_only"
    evidence = _clean(row.get("literature_evidence"))
    return "pending_literature_review" if not evidence else "ungraded_literature_evidence"


def _motif_validation_category(row: Mapping[str, object]) -> str:
    structural_significant = _is_true(row.get("structural_significant"))
    external_status = _clean(row.get("external_enrichment_status"))
    literature_status = _clean(row.get("literature_support_status"))
    literature_supported = literature_status in {"literature_supported", "ungraded_literature_evidence"}
    external_supported = external_status == "external_enrichment_supported"
    external_not_significant = external_status in {
        "external_enrichment_not_significant",
        "external_enrichment_no_terms",
    }
    external_unavailable = external_status in {
        "not_tested_external",
        "external_filter_failed",
        "external_enrichment_unavailable",
    }
    if structural_significant and external_supported:
        return "structural_and_external_function_supported"
    if structural_significant and external_not_significant and literature_supported:
        return "structural_significant_external_not_significant_literature_supported"
    if structural_significant and external_not_significant:
        return "structural_significant_external_not_significant_pending_literature"
    if structural_significant and external_unavailable and literature_supported:
        return "structural_significant_external_unavailable_literature_supported"
    if structural_significant and external_unavailable:
        return "structural_significant_external_unavailable_pending_validation"
    if external_supported:
        return "external_only_function_supported"
    return "low_confidence_or_noise_candidate"


def build_literature_search_candidates(
    integrated_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in integrated_rows:
        priority = _literature_priority(row)
        if priority == "skip":
            continue
        family = _clean(row.get("family"))
        motif_cluster = _clean(row.get("motif_cluster"))
        consensus = _clean(row.get("motif_consensus"))
        position = _position_text(row)
        candidates.append(
            {
                "family": family,
                "method": _clean(row.get("method")),
                "motif_cluster": motif_cluster,
                "motif_consensus": consensus,
                "representative_window_id": _clean(row.get("representative_window_id")),
                "representative_parent_id": _clean(row.get("representative_parent_id")),
                "representative_start": _clean(row.get("representative_start")),
                "representative_end": _clean(row.get("representative_end")),
                "representative_length": _clean(row.get("representative_length")),
                "representative_sequence": _clean(row.get("representative_sequence")),
                "distance_to_cluster_centroid": _clean(row.get("distance_to_cluster_centroid")),
                "position": position,
                "external_enrichment_status": _clean(row.get("external_enrichment_status")),
                "structural_type": _clean(row.get("structural_type")),
                "literature_search_priority": priority,
                "candidate_function": _clean(row.get("candidate_function")),
                "query_primary": _primary_query(
                    family,
                    _clean(row.get("representative_sequence")),
                    position,
                ),
                "query_functional": _functional_query(
                    family,
                    _clean(row.get("representative_sequence")),
                ),
                "query_engineering": _engineering_query(
                    family,
                    _clean(row.get("representative_sequence")),
                ),
                "database_targets": "UniProt;InterPro;PDB;AlphaFold;PubMed;Google Scholar",
            }
        )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda row: (priority_order.get(str(row["literature_search_priority"]), 9), row["family"], row["motif_cluster"]))
    return candidates


def _literature_priority(row: Mapping[str, object]) -> str:
    structural_significant = _is_true(row.get("structural_significant"))
    external_status = _clean(row.get("external_enrichment_status"))
    category = _clean(row.get("motif_validation_category"))
    if category.endswith("pending_literature") or (
        structural_significant
        and external_status
        in {"external_enrichment_not_significant", "external_enrichment_no_terms", "external_filter_failed", "external_enrichment_unavailable", "not_tested_external"}
    ):
        return "high"
    if structural_significant or external_status == "external_enrichment_supported":
        return "medium"
    if external_status in {"external_enrichment_not_significant", "external_enrichment_no_terms"}:
        return "low"
    return "low"


def _position_text(row: Mapping[str, object]) -> str:
    start = _clean(row.get("representative_start"))
    end = _clean(row.get("representative_end"))
    if start and end:
        start_number = _to_float(start)
        end_number = _to_float(end)
        if not math.isnan(start_number) and not math.isnan(end_number):
            return f"{round(start_number)}-{round(end_number)}"
        return f"{start}-{end}"
    return ""


def _primary_query(family: str, representative_sequence: str, position: str) -> str:
    parts = [f'"{family}"'] if family else []
    if representative_sequence:
        parts.append(f'"{representative_sequence}"')
    parts.append('"gas vesicle"')
    parts.append("motif")
    if position:
        parts.append(f'"{position}"')
    return " ".join(parts)


def _functional_query(family: str, representative_sequence: str) -> str:
    parts = [f'"{family}"'] if family else []
    if representative_sequence:
        parts.append(f'"{representative_sequence}"')
    parts.extend(
        [
            '"gas vesicle assembly"',
            'OR "shell stability"',
            'OR "acoustic reporter"',
            'OR "ultrasound imaging"',
            "OR transport",
            "OR localization",
        ]
    )
    return " ".join(parts)


def _engineering_query(family: str, representative_sequence: str) -> str:
    parts = [f'"{family}"'] if family else []
    if representative_sequence:
        parts.append(f'"{representative_sequence}"')
    parts.extend(["mutation", "OR deletion", "OR fusion", "OR engineering", "OR display"])
    return " ".join(parts)


def load_literature_rows(path: Path | None) -> list[dict[str, object]]:
    if not path:
        return []
    return read_csv_rows(path)


def build_validation_outputs(
    *,
    result_root: Path,
    output_dir: Path,
    literature_input: Path | None = None,
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
) -> dict[str, object]:
    result_root = Path(result_root)
    output_dir = Path(output_dir)
    exp3_dir = result_root / "exp3_domain_discovery"
    exp4_dir = result_root / "exp4_motif_structure_analysis"
    external_root = result_root / "exp5_external_functional_enrichment"
    batch_dir = external_root / "clusterprofiler" / "batch_results"
    motif_sets_path = external_root / "clusterprofiler" / "motif_sets_long.csv"

    catalog_rows = load_motif_catalogs(exp3_dir)
    representative_rows = load_motif_representatives(exp3_dir)
    enrichment_rows, summary_rows = load_structural_inputs(exp4_dir)
    expected_family_methods = {
        (_clean(row.get("family")), _clean(row.get("method")))
        for row in catalog_rows
        if _clean(row.get("family")) and _clean(row.get("method"))
    }
    coverage_rows = discover_clusterprofiler_batches(batch_dir, expected_family_methods)
    structural_rows = summarize_structural_significance(
        enrichment_rows=enrichment_rows,
        cluster_summary_rows=summary_rows,
        catalog_rows=catalog_rows,
        significance_threshold=significance_threshold,
    )
    structural_rows = attach_motif_representatives(structural_rows, representative_rows)
    external_rows = build_external_motif_statuses(
        motif_sets_path=motif_sets_path,
        batch_results_dir=batch_dir,
        catalog_rows=catalog_rows,
        significance_threshold=significance_threshold,
        scan_motif_sets=False,
    )
    literature_rows = load_literature_rows(literature_input)
    integrated_rows = integrate_motif_evidence(structural_rows, external_rows, literature_rows)
    literature_validation_rows = [_literature_validation_row(row) for row in integrated_rows]
    candidate_rows = build_literature_search_candidates(integrated_rows)

    output_paths = {
        "external_enrichment_coverage_audit": output_dir / "external_enrichment_coverage_audit.csv",
        "structural_motif_biological_significance": output_dir / "structural_motif_biological_significance.csv",
        "external_functional_motif_literature_validation": output_dir / "external_functional_motif_literature_validation.csv",
        "literature_search_candidates": output_dir / "literature_search_candidates.csv",
        "motif_evidence_integration": output_dir / "motif_evidence_integration.csv",
    }
    write_csv_rows(output_paths["external_enrichment_coverage_audit"], coverage_rows)
    write_csv_rows(output_paths["structural_motif_biological_significance"], structural_rows)
    write_csv_rows(output_paths["external_functional_motif_literature_validation"], literature_validation_rows)
    write_csv_rows(output_paths["literature_search_candidates"], candidate_rows)
    write_csv_rows(output_paths["motif_evidence_integration"], integrated_rows)

    return {
        "paths": output_paths,
        "n_catalog_motifs": len(catalog_rows),
        "n_structural_rows": len(structural_rows),
        "n_external_status_rows": len(external_rows),
        "n_integrated_rows": len(integrated_rows),
        "n_literature_candidates": len(candidate_rows),
    }


def _literature_validation_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "family": _clean(row.get("family")),
        "method": _clean(row.get("method")),
        "motif_cluster": _clean(row.get("motif_cluster")),
        "motif_consensus": _clean(row.get("motif_consensus")),
        "representative_window_id": _clean(row.get("representative_window_id")),
        "representative_parent_id": _clean(row.get("representative_parent_id")),
        "representative_start": _clean(row.get("representative_start")),
        "representative_end": _clean(row.get("representative_end")),
        "representative_length": _clean(row.get("representative_length")),
        "representative_sequence": _clean(row.get("representative_sequence")),
        "distance_to_cluster_centroid": _clean(row.get("distance_to_cluster_centroid")),
        "position": _position_text(row),
        "external_enrichment_status": _clean(row.get("external_enrichment_status")),
        "significant_terms": _clean(row.get("significant_terms")),
        "top_terms": _clean(row.get("top_terms")),
        "missing_reason": _clean(row.get("missing_reason")),
        "candidate_function": _clean(row.get("candidate_function")),
        "literature_evidence": _clean(row.get("literature_evidence")),
        "evidence_grade": _clean(row.get("evidence_grade")),
        "literature_support_status": _clean(row.get("literature_support_status")),
        "motif_validation_category": _clean(row.get("motif_validation_category")),
    }


def build_external_audit_outputs(
    *,
    external_root: Path,
    output_dir: Path,
    catalog_root: Path | None = None,
    significance_threshold: float = SIGNIFICANCE_THRESHOLD,
) -> dict[str, object]:
    external_root = Path(external_root)
    output_dir = Path(output_dir)
    batch_dir = external_root / "clusterprofiler" / "batch_results"
    motif_sets_path = external_root / "clusterprofiler" / "motif_sets_long.csv"
    catalog_rows = load_motif_catalogs(catalog_root) if catalog_root else []
    motif_set_rows = read_csv_rows(motif_sets_path) if not catalog_rows else []
    expected_family_methods = {
        (_clean(row.get("family")), _clean(row.get("method")))
        for row in list(catalog_rows) + list(motif_set_rows)
        if _clean(row.get("family")) and _clean(row.get("method"))
    }
    coverage_rows = discover_clusterprofiler_batches(batch_dir, expected_family_methods)
    status_rows = build_external_motif_statuses(
        motif_sets_path=motif_sets_path,
        batch_results_dir=batch_dir,
        catalog_rows=catalog_rows,
        significance_threshold=significance_threshold,
        scan_motif_sets=not bool(catalog_rows),
    )
    coverage_path = output_dir / "external_enrichment_coverage_audit.csv"
    status_path = output_dir / "external_motif_enrichment_status.csv"
    write_csv_rows(coverage_path, coverage_rows)
    write_csv_rows(status_path, status_rows)
    return {
        "paths": {
            "external_enrichment_coverage_audit": coverage_path,
            "external_motif_enrichment_status": status_path,
        },
        "n_family_methods": len(coverage_rows),
        "n_motif_status_rows": len(status_rows),
    }


def validation_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build structural and external motif validation tables.")
    project_root = Path(__file__).resolve().parent.parent
    work_results = project_root / "work" / "results"
    parser.add_argument("--result-root", type=Path, default=work_results)
    parser.add_argument("--output-dir", type=Path, default=work_results / "motif_validation")
    parser.add_argument("--literature-input", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=SIGNIFICANCE_THRESHOLD)
    args = parser.parse_args(argv)
    summary = build_validation_outputs(
        result_root=args.result_root,
        output_dir=args.output_dir,
        literature_input=args.literature_input,
        significance_threshold=args.alpha,
    )
    for name, path in summary["paths"].items():
        print(f"{name}: {path}")
    print(
        "counts: "
        f"catalog={summary['n_catalog_motifs']} "
        f"structural={summary['n_structural_rows']} "
        f"external={summary['n_external_status_rows']} "
        f"integrated={summary['n_integrated_rows']} "
        f"literature_candidates={summary['n_literature_candidates']}"
    )
    return 0


def external_audit_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit clusterProfiler external enrichment coverage.")
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "work" / "results" / "exp5_external_functional_enrichment",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "work" / "results" / "exp3_domain_discovery",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "work" / "results" / "motif_validation",
    )
    parser.add_argument("--alpha", type=float, default=SIGNIFICANCE_THRESHOLD)
    args = parser.parse_args(argv)
    summary = build_external_audit_outputs(
        external_root=args.external_root,
        output_dir=args.output_dir,
        catalog_root=args.catalog_root,
        significance_threshold=args.alpha,
    )
    for name, path in summary["paths"].items():
        print(f"{name}: {path}")
    print(
        "counts: "
        f"family_methods={summary['n_family_methods']} "
        f"motif_status_rows={summary['n_motif_status_rows']}"
    )
    return 0
