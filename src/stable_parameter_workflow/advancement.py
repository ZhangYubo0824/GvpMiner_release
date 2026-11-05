"""Deterministic, auditable recommendations between the A/B/C stages.

The selector reads completed stage outputs and never reruns an experiment.
Each recommendation records the rule version and input digests for audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .contracts import ALL_FAMILIES, ParameterPair, REPRESENTATIVE_FAMILIES


RULE_VERSION = "local-motif-2026-10-12-v3"
BASELINE = ParameterPair(30, 5)
QUALIFIED_FRACTION_MIN = 0.80
STABILITY_TOLERANCE = 0.02
COVERAGE_TOLERANCE = 0.03
NEAR_FULL_LENGTH_FRACTION = 0.80
MAX_NEAR_FULL_FAMILY_FRACTION = 0.50
FRAGMENTATION_RATIO = 1.5


def _pair(value: Mapping[str, Any]) -> ParameterPair:
    if not isinstance(value, Mapping) or any(
        type(value.get(key)) is not int for key in ("window_size", "window_step")
    ):
        raise ValueError("candidate pair coordinates must be integers")
    return ParameterPair(value["window_size"], value["window_step"])


def _metric(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate metric {key} is missing or nonnumeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"candidate metric {key} must be finite")
    return result


def _candidate_rows(stage: str, decision: Mapping[str, Any]) -> dict[ParameterPair, Mapping[str, Any]]:
    if decision.get("stage") != stage:
        raise ValueError("decision stage does not match requested selection")
    evaluated = decision.get("evaluated_pairs")
    candidates = decision.get("candidate_metrics", decision.get("ranked_candidates"))
    if not isinstance(evaluated, list) or not isinstance(candidates, list):
        raise ValueError("decision must contain evaluated_pairs and candidate metrics")
    evaluated_pairs = [_pair(item) for item in evaluated]
    rows: dict[ParameterPair, Mapping[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, Mapping):
            raise ValueError("candidate metric row is malformed")
        pair = _pair(row.get("pair"))
        if pair in rows:
            raise ValueError("candidate metrics contain duplicate pairs")
        if row.get("window_size") != pair.window_size or row.get("window_step") != pair.window_step:
            raise ValueError("candidate metric coordinates disagree")
        for key in ("family_count", "qualified_family_fraction", "stability_floor",
                    "stable_protein_coverage", "stable_window_coverage",
                    "noise_ratio", "stable_cluster_number"):
            _metric(row, key)
        rows[pair] = row
    if len(set(evaluated_pairs)) != len(evaluated_pairs) or set(rows) != set(evaluated_pairs):
        raise ValueError("candidate metrics must cover every evaluated pair exactly once")
    if not rows:
        raise ValueError("decision has no candidates")
    return rows


def _merged_occurrence_fractions(groups: Mapping[tuple[str, int], list[tuple[int, int, int]]]) -> list[tuple[str, float]]:
    fractions: list[tuple[str, float]] = []
    for (parent_id, _), windows in groups.items():
        windows.sort()
        start, end, protein_length = windows[0]
        for next_start, next_end, next_length in windows[1:]:
            if next_length != protein_length:
                raise ValueError("protein length differs between windows")
            if next_start <= end:
                end = max(end, next_end)
            else:
                fractions.append((parent_id, (end - start) / protein_length))
                start, end = next_start, next_end
        fractions.append((parent_id, (end - start) / protein_length))
    return fractions


def summarize_pair_locality(
    pair_root: Path, families: Sequence[str], *, method: str = "M2",
) -> dict[str, dict[str, float | int]]:
    """Measure contiguous stable motif span relative to each parent protein.

    Each family is reported separately so large families cannot hide a near
    whole-protein motif problem in a smaller family.
    """
    import numpy as np

    result: dict[str, dict[str, float | int]] = {}
    for family in families:
        metadata = Path(pair_root) / "raw_exp3" / family / "windows_metadata.csv"
        labels_path = Path(pair_root) / "pilot" / "motif_stability" / family / method / "stable_cluster_labels.npy"
        if not metadata.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"locality inputs missing for {pair_root.name}/{family}/{method}")
        labels = np.load(labels_path, allow_pickle=False)
        if labels.ndim != 1:
            raise ValueError(f"stable labels must be one-dimensional: {labels_path}")
        groups: dict[tuple[str, int], list[tuple[int, int, int]]] = defaultdict(list)
        with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"parent_id", "start", "end", "window_seq_full"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"windows metadata lacks locality fields: {metadata}")
            count = 0
            for label, row in zip_longest(labels, reader, fillvalue=None):
                if label is None or row is None:
                    raise ValueError(f"stable labels and windows are misaligned: {metadata}")
                count += 1
                if int(label) < 0:
                    continue
                start, end = int(row["start"]), int(row["end"])
                protein_length = len(row["window_seq_full"])
                if not (0 <= start < end <= protein_length):
                    raise ValueError(f"invalid stable motif interval: {metadata}")
                groups[(row["parent_id"], int(label))].append((start, end, protein_length))
        if count == 0:
            raise ValueError(f"empty windows metadata: {metadata}")
        occurrences = _merged_occurrence_fractions(groups)
        fractions = [fraction for _, fraction in occurrences]
        by_protein: dict[str, list[float]] = defaultdict(list)
        for parent_id, fraction in occurrences:
            by_protein[parent_id].append(fraction)
        result[family] = {
            "occurrence_count": len(fractions),
            "protein_count": len(by_protein),
            "median_span_fraction": median(fractions) if fractions else None,
            "near_full_fraction": (
                sum(value >= NEAR_FULL_LENGTH_FRACTION for value in fractions) / len(fractions)
                if fractions else None
            ),
            "near_full_protein_fraction": (
                sum(max(values) >= NEAR_FULL_LENGTH_FRACTION for values in by_protein.values()) /
                len(by_protein) if by_protein else None
            ),
        }
    return result


def locality_input_digests(
    stage_root: Path,
    pairs: Sequence[ParameterPair],
    families: Sequence[str],
    *, method: str = "M2",
) -> dict[str, str]:
    """Hash every raw file used to judge whether stable motifs remain local."""
    root = Path(stage_root)
    digests: dict[str, str] = {}
    for pair in pairs:
        for family in families:
            paths = (
                root / pair.slug / "raw_exp3" / family / "windows_metadata.csv",
                root / pair.slug / "pilot" / "motif_stability" / family / method /
                "stable_cluster_labels.npy",
            )
            for path in paths:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digests[path.relative_to(root).as_posix()] = digest.hexdigest()
    return dict(sorted(digests.items()))


def summarize_pair_family_coverage(
    pair_root: Path, families: Sequence[str], *, method: str = "M2",
) -> dict[str, float]:
    """Read stable-protein coverage for each Stage A family-method unit."""
    result: dict[str, float] = {}
    for family in families:
        path = (Path(pair_root) / "pilot" / "motif_stability" / family /
                method / "selected_setting.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"selected setting is malformed: {path}")
        coverage = _metric(payload, "stable_protein_coverage")
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(f"stable-protein coverage is out of range: {path}")
        result[family] = coverage
    return result


def family_coverage_input_digests(
    stage_root: Path,
    pairs: Sequence[ParameterPair],
    families: Sequence[str],
    *, method: str = "M2",
) -> dict[str, str]:
    """Hash selected settings used by the family-wise coverage guard."""
    root = Path(stage_root)
    digests: dict[str, str] = {}
    for pair in pairs:
        for family in families:
            path = (root / pair.slug / "pilot" / "motif_stability" /
                    family / method / "selected_setting.json")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests[path.relative_to(root).as_posix()] = digest.hexdigest()
    return dict(sorted(digests.items()))


def summarize_confirmatory_evidence(
    stage_root: Path,
    pairs: Sequence[ParameterPair],
    *,
    families: Sequence[str] = ALL_FAMILIES,
) -> dict[ParameterPair, dict[str, Any]]:
    """Check both embedding methods and family-level material reversals."""
    if len(pairs) != 2 or len(set(pairs)) != 2:
        raise ValueError("Stage C evidence requires exactly two candidates")
    values: dict[ParameterPair, dict[tuple[str, str], dict[str, float | bool]]] = {}
    for pair in pairs:
        rows: dict[tuple[str, str], dict[str, float | bool]] = {}
        for family in families:
            for method in ("M2", "M3"):
                path = (Path(stage_root) / pair.slug / "confirmatory" /
                        "motif_stability" / family / method / "selected_setting.json")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError(f"confirmatory setting is malformed: {path}")
                floor = payload.get("stability_floor")
                if floor is None:
                    floor = min(_metric(payload, key) for key in (
                        "median_window_jaccard", "median_protein_jaccard",
                        "median_occurrence_jaccard"))
                rows[(family, method)] = {
                    "floor": _metric({"floor": floor}, "floor"),
                    "coverage": _metric(payload, "stable_protein_coverage"),
                    "qualified": str(payload.get("selection_mode", "")).lower() in {
                        "qualified", "stability_plateau", "coverage_rescue"},
                }
        values[pair] = rows
    result: dict[ParameterPair, dict[str, Any]] = {}
    first, second = pairs
    conflicts = 0
    for family in families:
        m2_delta = float(values[first][(family, "M2")]["floor"]) - float(values[second][(family, "M2")]["floor"])
        m3_delta = float(values[first][(family, "M3")]["floor"]) - float(values[second][(family, "M3")]["floor"])
        if ((m2_delta >= STABILITY_TOLERANCE and m3_delta <= -STABILITY_TOLERANCE)
                or (m2_delta <= -STABILITY_TOLERANCE and m3_delta >= STABILITY_TOLERANCE)):
            conflicts += 1
    for pair, rows in values.items():
        by_method = {}
        for method in ("M2", "M3"):
            subset = [rows[(family, method)] for family in families]
            by_method[method] = {
                "median_floor": median(float(row["floor"]) for row in subset),
                "median_protein_coverage": median(float(row["coverage"]) for row in subset),
            }
        result[pair] = {
            "qualified_unit_fraction": sum(bool(row["qualified"]) for row in rows.values()) / len(rows),
            "material_conflict_family_count": conflicts,
            "material_conflict_fraction": conflicts / len(families),
            "method_metrics": by_method,
        }
    return result


def _ranking(pair: ParameterPair, row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        _metric(row, "stability_floor"),
        _metric(row, "stable_protein_coverage"),
        _metric(row, "qualified_family_fraction"),
        -_metric(row, "noise_ratio"),
        _metric(row, "stable_window_coverage"),
        -_metric(row, "stable_cluster_number"),
        float(pair.window_step),
        -float(pair.window_size),
    )


def recommend_selection(
    stage: str,
    decision: Mapping[str, Any],
    *,
    locality: Mapping[ParameterPair, Mapping[str, Mapping[str, float | int]]] | None = None,
    family_coverage: Mapping[ParameterPair, Mapping[str, float]] | None = None,
    confirmatory_evidence: Mapping[ParameterPair, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select A→B (two or three pairs), B→C (two), or C's final pair."""
    stage = stage.upper()
    if stage not in {"A", "B", "C"}:
        raise ValueError("source stage must be A, B, or C")
    rows = _candidate_rows(stage, decision)
    if stage == "A" and BASELINE not in rows:
        raise ValueError("Stage A lacks baseline 30:5")
    if stage == "A" and (locality is None or set(locality) != set(rows)):
        raise ValueError("Stage A locality must cover every evaluated pair")
    if stage == "A" and (family_coverage is None or set(family_coverage) != set(rows)):
        raise ValueError("Stage A family coverage must cover every evaluated pair")
    if stage == "C" and confirmatory_evidence is not None and set(confirmatory_evidence) != set(rows):
        raise ValueError("Stage C method evidence must cover every evaluated pair")

    assessed: dict[ParameterPair, dict[str, Any]] = {}
    baseline = rows.get(BASELINE)
    baseline_family_coverage: Mapping[str, float] = {}
    if stage == "A":
        assert family_coverage is not None
        baseline_family_coverage = family_coverage[BASELINE]
        if not baseline_family_coverage:
            raise ValueError("Stage A family coverage is empty")
        for pair, coverage_by_family in family_coverage.items():
            if set(coverage_by_family) != set(baseline_family_coverage):
                raise ValueError(f"Stage A family coverage is incomplete for {pair.slug}")
            for family, value in coverage_by_family.items():
                coverage_value = _metric({"coverage": value}, "coverage")
                if not 0.0 <= coverage_value <= 1.0:
                    raise ValueError(f"Stage A family coverage is out of range for {pair.slug}/{family}")
    for pair, row in rows.items():
        reasons: list[str] = []
        family_coverage_shortfalls: dict[str, float] = {}
        if _metric(row, "qualified_family_fraction") < QUALIFIED_FRACTION_MIN:
            reasons.append("qualified_family_fraction_below_0.80")
        if stage == "A":
            assert baseline is not None
            if pair != BASELINE:
                if _metric(row, "stability_floor") < _metric(baseline, "stability_floor") - STABILITY_TOLERANCE:
                    reasons.append("stability_below_baseline_tolerance")
                if _metric(row, "stable_protein_coverage") < _metric(baseline, "stable_protein_coverage") - COVERAGE_TOLERANCE:
                    reasons.append("coverage_below_baseline_tolerance")
                assert family_coverage is not None
                family_coverage_shortfalls = {
                    family: float(value) - float(baseline_family_coverage[family])
                    for family, value in family_coverage[pair].items()
                    if float(value) < float(baseline_family_coverage[family]) - COVERAGE_TOLERANCE
                }
                if family_coverage_shortfalls:
                    reasons.append("family_coverage_below_baseline_tolerance")
            assert locality is not None
            empty_families = [family for family, stats in locality[pair].items()
                              if int(stats["occurrence_count"]) == 0]
            if empty_families:
                reasons.append("no_stable_motif_in_family")
            bad_families = [family for family, stats in locality[pair].items()
                            if stats["near_full_protein_fraction"] is not None
                            and float(stats["near_full_protein_fraction"]) > MAX_NEAR_FULL_FAMILY_FRACTION]
            if bad_families:
                reasons.append("whole_protein_motif")
        if stage == "C" and confirmatory_evidence is not None:
            evidence = confirmatory_evidence[pair]
            if _metric(evidence, "qualified_unit_fraction") < QUALIFIED_FRACTION_MIN:
                reasons.append("qualified_method_unit_fraction_below_0.80")
            if _metric(evidence, "material_conflict_fraction") > 0.25:
                reasons.append("material_cross_method_disagreement")
            for method in ("M2", "M3"):
                method_values = evidence["method_metrics"][method]
                best_floor = max(_metric(item["method_metrics"][method], "median_floor")
                                 for item in confirmatory_evidence.values())
                best_coverage = max(_metric(item["method_metrics"][method], "median_protein_coverage")
                                    for item in confirmatory_evidence.values())
                if _metric(method_values, "median_floor") < best_floor - STABILITY_TOLERANCE:
                    reasons.append(f"{method.lower()}_stability_below_plateau")
                if _metric(method_values, "median_protein_coverage") < best_coverage - COVERAGE_TOLERANCE:
                    reasons.append(f"{method.lower()}_coverage_below_plateau")
        assessed[pair] = {"pair": pair.as_dict(), "eligible": not reasons,
                          "reasons": reasons, "metrics": {key: row[key] for key in (
                              "qualified_family_fraction", "stability_floor",
                              "stable_protein_coverage", "stable_window_coverage",
                              "noise_ratio", "stable_cluster_number")}}
        if stage == "A":
            assessed[pair]["locality"] = locality[pair]  # type: ignore[index]
            assessed[pair]["family_coverage"] = family_coverage[pair]  # type: ignore[index]
            assessed[pair]["family_coverage_shortfalls"] = family_coverage_shortfalls
        if stage == "C" and confirmatory_evidence is not None:
            assessed[pair]["confirmatory_evidence"] = confirmatory_evidence[pair]

    if stage == "A":
        for pair, row in rows.items():
            if pair == BASELINE or not assessed[pair]["eligible"]:
                continue
            for other, other_row in rows.items():
                if other == pair or other.window_size != pair.window_size or not assessed[other]["eligible"]:
                    continue
                if (_metric(row, "stable_cluster_number") > FRAGMENTATION_RATIO * _metric(other_row, "stable_cluster_number")
                        and _metric(row, "stable_protein_coverage") < _metric(other_row, "stable_protein_coverage") + COVERAGE_TOLERANCE):
                    assessed[pair]["eligible"] = False
                    assessed[pair]["reasons"].append("fragmentation_without_coverage_gain")
                    break
        if not assessed[BASELINE]["eligible"]:
            raise ValueError("Stage A reference is nonlocal or otherwise ineligible")
        challengers = [pair for pair in rows if pair != BASELINE and assessed[pair]["eligible"]]
        if not challengers:
            raise ValueError("Stage A has no scientifically eligible challenger")
        ranked = sorted(challengers, key=lambda pair: _ranking(pair, rows[pair]), reverse=True)
        selected = [BASELINE, *ranked[:2]]
        selection_roles = {f"{BASELINE.window_size}:{BASELINE.window_step}": "reference"}
        for rank, pair in enumerate(ranked[:2], start=1):
            selection_roles[f"{pair.window_size}:{pair.window_step}"] = f"rank_{rank}"
    elif stage == "B":
        eligible = [pair for pair in rows if assessed[pair]["eligible"]]
        if len(eligible) < 2:
            raise ValueError("Stage B has fewer than two eligible candidates")
        selected = sorted(eligible, key=lambda pair: _ranking(pair, rows[pair]), reverse=True)[:2]
    else:
        eligible = [pair for pair in rows if assessed[pair]["eligible"]]
        if not eligible:
            raise ValueError("Stage C has no eligible final candidate")
        selected = sorted(eligible, key=lambda pair: _ranking(pair, rows[pair]), reverse=True)[:1]

    selected_set = set(selected)
    for pair, item in assessed.items():
        item["selected"] = pair in selected_set
        if item["eligible"] and pair not in selected_set:
            item["reasons"].append("lower_priority_than_selected_candidates")
    report = {
        "rule_version": RULE_VERSION,
        "source_stage": stage,
        "selected_pairs": [pair.as_dict() for pair in selected],
        "thresholds": {
            "qualified_family_fraction_min": QUALIFIED_FRACTION_MIN,
            "stability_tolerance": STABILITY_TOLERANCE,
            "coverage_tolerance": COVERAGE_TOLERANCE,
            "near_full_length_fraction": NEAR_FULL_LENGTH_FRACTION,
            "max_near_full_family_fraction": MAX_NEAR_FULL_FAMILY_FRACTION,
            "fragmentation_ratio": FRAGMENTATION_RATIO,
        },
        "candidates": [assessed[pair] for pair in sorted(rows)],
    }
    if stage == "A":
        report["selection_roles"] = selection_roles
    return report


def recommend_decision_file(
    decision_path: Path, *, stage_root: Path | None = None,
) -> dict[str, Any]:
    """Build a read-only recommendation from an existing decision and outputs."""
    path = Path(decision_path)
    raw = path.read_bytes()
    decision = json.loads(raw)
    if not isinstance(decision, dict):
        raise ValueError("decision JSON must be an object")
    stage = str(decision.get("stage", "")).upper()
    root = Path(stage_root) if stage_root is not None else (
        path.parent.parent / {
            "A": "stage_a_main_effects",
            "B": "stage_b_m2_all_families",
            "C": "stage_c_confirmatory",
        }.get(stage, "unknown_stage")
    )
    locality = None
    family_coverage = None
    if stage == "A":
        rows = _candidate_rows(stage, decision)
        input_digests = locality_input_digests(root, tuple(rows), REPRESENTATIVE_FAMILIES)
        coverage_input_digests = family_coverage_input_digests(
            root, tuple(rows), REPRESENTATIVE_FAMILIES)
        locality = {
            pair: summarize_pair_locality(root / pair.slug, REPRESENTATIVE_FAMILIES)
            for pair in rows
        }
        family_coverage = {
            pair: summarize_pair_family_coverage(root / pair.slug, REPRESENTATIVE_FAMILIES)
            for pair in rows
        }
        if locality_input_digests(root, tuple(rows), REPRESENTATIVE_FAMILIES) != input_digests:
            raise ValueError("Stage A locality inputs changed during recommendation")
        if (family_coverage_input_digests(root, tuple(rows), REPRESENTATIVE_FAMILIES)
                != coverage_input_digests):
            raise ValueError("Stage A family coverage inputs changed during recommendation")
    confirmatory_evidence = None
    if stage == "C":
        rows = _candidate_rows(stage, decision)
        confirmatory_evidence = summarize_confirmatory_evidence(root, tuple(rows))
    report = recommend_selection(
        stage, decision, locality=locality, family_coverage=family_coverage,
        confirmatory_evidence=confirmatory_evidence,
    )
    if stage == "A":
        report["locality_input_sha256"] = input_digests
        report["family_coverage_input_sha256"] = coverage_input_digests
    report["source_decision"] = str(path.resolve())
    report["source_decision_sha256"] = hashlib.sha256(raw).hexdigest()
    report["next_stage_args"] = (
        ["--stage", {"A": "B", "B": "C"}[stage], "--auto-select"]
        if stage in ("A", "B") else []
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Recommend auditable A→B, B→C, or final Stage C parameters")
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--stage-root", type=Path,
                        help="Stage output root; defaults to the decision's sibling stage directory")
    parser.add_argument("--output-json", type=Path,
                        help="Save the recommendation; omitted means stdout only")
    args = parser.parse_args(argv)
    report = recommend_decision_file(args.decision, stage_root=args.stage_root)
    if args.output_json is not None:
        from .manifests import atomic_write_json
        atomic_write_json(args.output_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
