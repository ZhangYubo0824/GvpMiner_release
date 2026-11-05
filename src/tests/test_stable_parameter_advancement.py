import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_parameter_workflow.advancement import (
    locality_input_digests,
    recommend_decision_file,
    recommend_selection,
    summarize_confirmatory_evidence,
    summarize_pair_locality,
)
from stable_parameter_workflow.contracts import ParameterPair
import stable_parameter_workflow.advancement as advancement


def _metric(window, step, floor, coverage, qualified=1.0, clusters=30, noise=.1):
    return {
        "pair": {"window_size": window, "window_step": step},
        "window_size": window,
        "window_step": step,
        "family_count": 6,
        "qualified_family_fraction": qualified,
        "stability_floor": floor,
        "stable_protein_coverage": coverage,
        "stable_window_coverage": coverage / 2,
        "stable_cluster_number": clusters,
        "noise_ratio": noise,
    }


def _locality_stats(protein_fraction, occurrence_fraction=None):
    return {"near_full_protein_fraction": protein_fraction,
            "near_full_fraction": protein_fraction if occurrence_fraction is None else occurrence_fraction,
            "occurrence_count": 30, "protein_count": 30}


def _family_coverage(rows, overrides=None):
    result = {}
    overrides = overrides or {}
    for row in rows:
        pair = ParameterPair(row["window_size"], row["window_step"])
        result[pair] = {"GvpF": .9, "GvpH": .9, "GvpK": .9,
                        "GvpL": .9, "GvpM": .9, "GvpN": .9}
        result[pair].update(overrides.get(pair, {}))
    return result


def test_stage_a_rejects_family_coverage_loss_without_naming_a_family():
    rows = [_metric(30, 5, .75, .92), _metric(20, 15, .896, .980),
            _metric(20, 10, .788, .998, qualified=5 / 6),
            _metric(25, 10, .777, .965, qualified=5 / 6),
            _metric(35, 10, .895, .982, qualified=5 / 6)]
    locality = {ParameterPair(r["window_size"], r["window_step"]):
                {"GvpH": _locality_stats(0.0)} for r in rows}
    coverage = _family_coverage(rows, {
        ParameterPair(30, 5): {"GvpH": .14, "GvpK": .85},
        ParameterPair(20, 15): {"GvpH": 1.0, "GvpK": .87, "GvpL": .88},
        ParameterPair(20, 10): {"GvpH": .62, "GvpK": .89, "GvpL": .99},
        ParameterPair(35, 10): {"GvpH": .088, "GvpK": .675},
    })
    decision = {"stage": "A", "evaluated_pairs": [r["pair"] for r in rows],
                "candidate_metrics": rows}
    result = recommend_selection("A", decision, locality=locality,
                                 family_coverage=coverage)
    assert result["selected_pairs"] == [rows[0]["pair"], rows[1]["pair"], rows[2]["pair"]]
    by_pair = {(r["pair"]["window_size"], r["pair"]["window_step"]): r
               for r in result["candidates"]}
    assert by_pair[(35, 10)]["eligible"] is False
    assert "family_coverage_below_baseline_tolerance" in by_pair[(35, 10)]["reasons"]
    assert by_pair[(35, 10)]["family_coverage_shortfalls"] == {
        "GvpH": pytest.approx(-.052), "GvpK": pytest.approx(-.175)}
    assert by_pair[(25, 10)]["eligible"] is True
    assert result["selection_roles"] == {"30:5": "reference", "20:15": "rank_1",
                                         "20:10": "rank_2"}


def test_stage_a_family_coverage_reads_and_hashes_selected_settings(tmp_path):
    pair = ParameterPair(20, 15)
    root = tmp_path / "stage_a_main_effects"
    selected = (root / pair.slug / "pilot" / "motif_stability" /
                "GvpH" / "M2" / "selected_setting.json")
    selected.parent.mkdir(parents=True)
    selected.write_text(json.dumps({"stable_protein_coverage": .75}), encoding="utf-8")
    coverage = advancement.summarize_pair_family_coverage(root / pair.slug, ("GvpH",))
    first = advancement.family_coverage_input_digests(root, (pair,), ("GvpH",))
    assert coverage == {"GvpH": .75}
    assert len(first) == 1
    selected.write_text(json.dumps({"stable_protein_coverage": .50}), encoding="utf-8")
    second = advancement.family_coverage_input_digests(root, (pair,), ("GvpH",))
    assert first != second


def test_stage_a_file_recommendation_uses_all_family_settings(tmp_path):
    rows = [_metric(30, 5, .75, .92), _metric(20, 15, .896, .980),
            _metric(20, 10, .788, .998, qualified=5 / 6),
            _metric(35, 10, .895, .982, qualified=5 / 6)]
    families = ("GvpF", "GvpH", "GvpK", "GvpL", "GvpM", "GvpN")
    decision_path = tmp_path / "comparisons" / "stage_a_decision.json"
    decision_path.parent.mkdir()
    decision_path.write_text(json.dumps({"stage": "A", "evaluated_pairs":
                                         [r["pair"] for r in rows],
                                         "candidate_metrics": rows}), encoding="utf-8")
    for row in rows:
        pair = ParameterPair(row["window_size"], row["window_step"])
        for family in families:
            raw = tmp_path / "stage_a_main_effects" / pair.slug / "raw_exp3" / family
            raw.mkdir(parents=True)
            with (raw / "windows_metadata.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["parent_id", "start", "end", "window_seq_full"])
                writer.writeheader()
                writer.writerow({"parent_id": "protein", "start": 0, "end": 20,
                                 "window_seq_full": "A" * 100})
            pilot = (tmp_path / "stage_a_main_effects" / pair.slug / "pilot" /
                     "motif_stability" / family / "M2")
            pilot.mkdir(parents=True)
            np.save(pilot / "stable_cluster_labels.npy", np.array([0]))
            coverage = .9
            if pair == ParameterPair(30, 5):
                coverage = {"GvpH": .14, "GvpK": .85}.get(family, .9)
            if pair == ParameterPair(35, 10):
                coverage = {"GvpH": .088, "GvpK": .675}.get(family, .9)
            (pilot / "selected_setting.json").write_text(
                json.dumps({"stable_protein_coverage": coverage}), encoding="utf-8")
    report = recommend_decision_file(decision_path, stage_root=tmp_path / "stage_a_main_effects")
    assert report["selected_pairs"] == [rows[0]["pair"], rows[1]["pair"], rows[2]["pair"]]
    assert len(report["family_coverage_input_sha256"]) == len(rows) * len(families)
    bad = next(item for item in report["candidates"] if item["pair"] == rows[3]["pair"])
    assert "family_coverage_below_baseline_tolerance" in bad["reasons"]


def test_locality_uses_contiguous_stable_occurrences_and_equal_family_guard(tmp_path):
    pair_root = tmp_path / "w40_s15"
    for family, lengths in (("GvpK", [100, 100, 100, 100]),
                            ("GvpH", [45, 45])):
        raw = pair_root / "raw_exp3" / family
        raw.mkdir(parents=True)
        with (raw / "windows_metadata.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["parent_id", "start", "end", "window_seq_full"])
            writer.writeheader()
            for index, length in enumerate(lengths):
                writer.writerow({"parent_id": f"{family}_{index}", "start": 0,
                                 "end": 40, "window_seq_full": "A" * length})
        pilot = pair_root / "pilot" / "motif_stability" / family / "M2"
        pilot.mkdir(parents=True)
        np.save(pilot / "stable_cluster_labels.npy", np.zeros(len(lengths), dtype=int))

    summary = summarize_pair_locality(pair_root, ("GvpK", "GvpH"))
    assert summary["GvpK"]["near_full_fraction"] == 0.0
    assert summary["GvpH"]["near_full_fraction"] == 1.0
    assert summary["GvpH"]["near_full_protein_fraction"] == 1.0


def test_locality_counts_proteins_once_even_with_many_stable_clusters(tmp_path):
    pair_root = tmp_path / "w40_s15"
    raw = pair_root / "raw_exp3" / "GvpH"
    raw.mkdir(parents=True)
    with (raw / "windows_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["parent_id", "start", "end", "window_seq_full"])
        writer.writeheader()
        for _ in range(10):
            writer.writerow({"parent_id": "one", "start": 0, "end": 90,
                             "window_seq_full": "A" * 100})
        for index in range(9):
            writer.writerow({"parent_id": f"local_{index}", "start": 0, "end": 20,
                             "window_seq_full": "A" * 100})
    pilot = pair_root / "pilot" / "motif_stability" / "GvpH" / "M2"
    pilot.mkdir(parents=True)
    np.save(pilot / "stable_cluster_labels.npy", np.arange(19))
    stats = summarize_pair_locality(pair_root, ("GvpH",))["GvpH"]
    assert stats["near_full_fraction"] == 10 / 19
    assert stats["protein_count"] == 10
    assert stats["near_full_protein_fraction"] == .1


def test_stage_a_rejects_whole_protein_motifs_and_fragmentation():
    rows = [
        _metric(30, 5, .75, .92, clusters=63),
        _metric(40, 15, .98, .99, clusters=23),
        _metric(20, 3, .84, .94, clusters=75),
        _metric(20, 10, .79, .998, qualified=.83, clusters=44),
        _metric(20, 15, .90, .98, clusters=39),
    ]
    decision = {"stage": "A", "evaluated_pairs": [r["pair"] for r in rows],
                "candidate_metrics": rows}
    locality = {ParameterPair(r["window_size"], r["window_step"]):
                {"GvpH": _locality_stats(1.0 if r["window_size"] == 40 else 0.0)}
                for r in rows}
    result = recommend_selection("A", decision, locality=locality,
                                 family_coverage=_family_coverage(rows))
    assert result["selected_pairs"] == [
        {"window_size": 30, "window_step": 5},
        {"window_size": 20, "window_step": 15},
        {"window_size": 20, "window_step": 10},
    ]
    by_pair = {(r["pair"]["window_size"], r["pair"]["window_step"]): r
               for r in result["candidates"]}
    assert "whole_protein_motif" in by_pair[(40, 15)]["reasons"]
    assert "fragmentation_without_coverage_gain" in by_pair[(20, 3)]["reasons"]


def test_stage_a_ranks_stability_before_window_length():
    rows = [_metric(30, 5, .75, .92), _metric(20, 15, .80, .94),
            _metric(20, 10, .70, .94), _metric(25, 5, .95, .99),
            _metric(25, 10, .90, .98)]
    locality = {ParameterPair(r["window_size"], r["window_step"]):
                {"GvpH": _locality_stats(0.0)}
                for r in rows}
    decision = {"stage": "A", "evaluated_pairs": [r["pair"] for r in rows],
                "candidate_metrics": rows}
    result = recommend_selection("A", decision, locality=locality,
                                 family_coverage=_family_coverage(rows))
    assert result["selected_pairs"] == [rows[0]["pair"], rows[3]["pair"], rows[4]["pair"]]


def test_stage_a_near_full_guard_requires_a_majority_of_proteins():
    rows = [_metric(30, 5, .75, .92), _metric(20, 15, .90, .98),
            _metric(20, 10, .79, .99), _metric(25, 10, .89, .98),
            _metric(40, 15, .98, .99)]
    locality = {ParameterPair(r["window_size"], r["window_step"]):
                {"GvpH": _locality_stats(.4, .9) if r["window_size"] == 25 else
                         _locality_stats(.9 if r["window_size"] == 40 else 0.0)}
                for r in rows}
    decision = {"stage": "A", "evaluated_pairs": [r["pair"] for r in rows],
                "candidate_metrics": rows}
    result = recommend_selection("A", decision, locality=locality,
                                 family_coverage=_family_coverage(rows))
    by_pair = {(r["pair"]["window_size"], r["pair"]["window_step"]): r
               for r in result["candidates"]}
    assert by_pair[(25, 10)]["eligible"]
    assert "whole_protein_motif" in by_pair[(40, 15)]["reasons"]
    assert result["thresholds"]["max_near_full_family_fraction"] == .5


def test_stage_a_ranks_all_eligible_challengers_without_window_preference():
    rows = [_metric(30, 5, .75, .92), _metric(20, 15, .90, .98),
            _metric(20, 10, .79, .99, qualified=.83),
            _metric(35, 10, .93, .99, qualified=.83),
            _metric(35, 15, .91, .98, qualified=.83)]
    locality = {ParameterPair(r["window_size"], r["window_step"]):
                {"GvpH": _locality_stats(0.0)}
                for r in rows}
    decision = {"stage": "A", "evaluated_pairs": [r["pair"] for r in rows],
                "candidate_metrics": rows}
    result = recommend_selection("A", decision, locality=locality,
                                 family_coverage=_family_coverage(rows))
    assert result["selected_pairs"] == [rows[0]["pair"], rows[3]["pair"], rows[4]["pair"]]
    assert result["selection_roles"] == {"30:5": "reference", "35:10": "rank_1", "35:15": "rank_2"}


def test_stage_a_can_promote_second_best_from_another_window():
    rows = [_metric(30, 5, .75, .92), _metric(20, 15, .90, .98),
            _metric(25, 10, .89, .98)]
    locality = {ParameterPair(r["window_size"], r["window_step"]):
                {"GvpH": _locality_stats(0.0)}
                for r in rows}
    decision = {"stage": "A", "evaluated_pairs": [r["pair"] for r in rows],
                "candidate_metrics": rows}
    result = recommend_selection("A", decision, locality=locality,
                                 family_coverage=_family_coverage(rows))
    assert result["selected_pairs"] == [rows[0]["pair"], rows[1]["pair"], rows[2]["pair"]]
    assert result["selection_roles"] == {"30:5": "reference", "20:15": "rank_1",
                                         "25:10": "rank_2"}


def test_stage_a_refuses_nonlocal_reference():
    rows = [_metric(30, 5, .75, .92), _metric(20, 15, .90, .98),
            _metric(20, 10, .79, .99)]
    locality = {ParameterPair(r["window_size"], r["window_step"]):
                {"GvpH": _locality_stats(.9 if r["window_size"] == 30 else 0.0)}
                for r in rows}
    decision = {"stage": "A", "evaluated_pairs": [r["pair"] for r in rows],
                "candidate_metrics": rows}
    with pytest.raises(ValueError, match="reference.*nonlocal"):
        recommend_selection("A", decision, locality=locality,
                            family_coverage=_family_coverage(rows))


def test_locality_input_digests_change_when_labels_change(tmp_path):
    root = tmp_path / "stage_a_main_effects"
    pair = ParameterPair(40, 15)
    metadata = root / pair.slug / "raw_exp3" / "GvpH" / "windows_metadata.csv"
    labels = root / pair.slug / "pilot" / "motif_stability" / "GvpH" / "M2" / "stable_cluster_labels.npy"
    for path in (metadata, labels):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"initial")
    first = locality_input_digests(root, (pair,), ("GvpH",))
    labels.write_bytes(b"changed")
    second = locality_input_digests(root, (pair,), ("GvpH",))
    assert first != second
    assert len(first) == 2


def test_stage_b_selects_stability_challengers_above_baseline():
    rows = [_metric(30, 5, .779, .929),
            _metric(20, 10, .830, .997, qualified=.9375),
            _metric(20, 15, .897, .998)]
    result = recommend_selection("B", {"stage": "B", "evaluated_pairs":
                                 [r["pair"] for r in rows], "candidate_metrics": rows})
    assert result["selected_pairs"] == [rows[2]["pair"], rows[1]["pair"]]


def test_stage_c_selects_confirmed_winner():
    rows = [_metric(20, 10, .847, .911), _metric(20, 15, .891, .964)]
    result = recommend_selection("C", {"stage": "C", "evaluated_pairs":
                                 [r["pair"] for r in rows], "candidate_metrics": rows})
    assert result["selected_pairs"] == [rows[1]["pair"]]


def test_confirmatory_evidence_rejects_method_specific_winner(tmp_path):
    pairs = (ParameterPair(20, 10), ParameterPair(20, 15))
    for pair in pairs:
        for family in ("GvpA", "GvpC"):
            for method in ("M2", "M3"):
                path = tmp_path / pair.slug / "confirmatory" / "motif_stability" / family / method / "selected_setting.json"
                path.parent.mkdir(parents=True)
                floor = .9 if (pair.window_step == 15 and method == "M2") else .7
                if pair.window_step == 10 and method == "M3":
                    floor = .9
                path.write_text(json.dumps({"selection_mode": "qualified",
                                            "stability_floor": floor,
                                            "stable_protein_coverage": .9}), encoding="utf-8")
    evidence = summarize_confirmatory_evidence(tmp_path, pairs, families=("GvpA", "GvpC"))
    assert evidence[pairs[1]]["material_conflict_fraction"] == 1.0
    rows = [_metric(20, 10, .8, .8), _metric(20, 15, .9, .9)]
    decision = {"stage": "C", "evaluated_pairs": [r["pair"] for r in rows],
                "candidate_metrics": rows}
    with pytest.raises(ValueError, match="no eligible"):
        recommend_selection("C", decision, confirmatory_evidence=evidence)


def test_missing_or_duplicate_candidate_metrics_fail_closed():
    row = _metric(30, 5, .7, .8)
    decision = {"stage": "A", "evaluated_pairs": [row["pair"],
                {"window_size": 20, "window_step": 15}], "candidate_metrics": [row]}
    with pytest.raises(ValueError, match="cover"):
        recommend_selection("A", decision, locality={})


def test_file_recommendation_records_source_hash_and_next_stage_arguments(tmp_path):
    rows = [_metric(30, 5, .779, .929),
            _metric(20, 10, .830, .997, qualified=.9375),
            _metric(20, 15, .897, .998)]
    path = tmp_path / "comparisons" / "stage_b_decision.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"stage": "B", "evaluated_pairs":
                               [r["pair"] for r in rows], "candidate_metrics": rows}),
                    encoding="utf-8")
    result = recommend_decision_file(path)
    assert len(result["source_decision_sha256"]) == 64
    assert result["next_stage_args"] == ["--stage", "C", "--auto-select"]


def test_final_stage_file_recommendation_preserves_winner_without_next_stage(tmp_path):
    rows = [_metric(20, 10, .88, .95), _metric(20, 15, .89, .96)]
    path = tmp_path / "comparisons" / "stage_c_decision.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"stage": "C", "evaluated_pairs":
                               [row["pair"] for row in rows], "candidate_metrics": rows}),
                    encoding="utf-8")
    for row in rows:
        pair = ParameterPair(**row["pair"])
        for family in advancement.ALL_FAMILIES:
            for method in ("M2", "M3"):
                setting = (tmp_path / "stage_c_confirmatory" / pair.slug /
                           "confirmatory" / "motif_stability" / family / method /
                           "selected_setting.json")
                setting.parent.mkdir(parents=True)
                setting.write_text(json.dumps({
                    "selection_mode": "qualified",
                    "stability_floor": row["stability_floor"],
                    "stable_protein_coverage": row["stable_protein_coverage"],
                }), encoding="utf-8")
    original = path.read_bytes()
    report = recommend_decision_file(path)
    assert report["selected_pairs"] == [rows[1]["pair"]]
    assert report["next_stage_args"] == []
    assert len(report["source_decision_sha256"]) == 64
    assert all("confirmatory_evidence" in candidate for candidate in report["candidates"])
    assert path.read_bytes() == original
