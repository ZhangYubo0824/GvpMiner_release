import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_parameter_workflow.pilot_reuse import inspect_pilot_source


def _source_tree(tmp_path):
    pair = tmp_path / "stage_b_m2_all_families" / "w20_s15"
    manifests = pair / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "exp3_pair.json").write_text(json.dumps({
        "status": "complete", "signature": "pair-sig", "generation_id": "gen-1",
    }), encoding="utf-8")
    (manifests / "GvpK_M2_pilot.json").write_text(json.dumps({
        "status": "complete", "signature": "pilot-sig",
        "input_fingerprints": {"dataset": "fingerprint"},
        "parameters": {"upstream_exp3": {
            "signature": "pair-sig", "generation_id": "gen-1",
        }},
    }), encoding="utf-8")
    diagnostic = pair / "pilot" / "motif_stability" / "GvpK" / "M2"
    diagnostic.mkdir(parents=True)
    (diagnostic / "selected_setting.json").write_text(
        '{"min_cluster_size": 20}', encoding="utf-8",
    )
    catalog = pair / "pilot" / "exp3_domain_discovery" / "GvpK"
    catalog.mkdir(parents=True)
    (catalog / "motif_catalog_M2.csv").write_text("motif_id\nM1\n", encoding="utf-8")
    (catalog / "motif_occurrences_M2.csv").write_text("motif_id\nM1\n", encoding="utf-8")
    return pair


def test_source_pilot_is_reusable_without_raw_embedding(tmp_path):
    pair = _source_tree(tmp_path)
    assert not (pair / "raw_exp3").exists()

    source = inspect_pilot_source(
        pair, family="GvpK", method="M2", expected_pair_signature="pair-sig",
        expected_pilot_signature="pilot-sig",
        expected_fingerprints={"dataset": "fingerprint"},
    )

    assert source is not None
    assert "selected_setting.json" in source.diagnostic_hashes
    assert source.pair_generation_id == "gen-1"


def test_missing_pilot_file_is_unavailable_for_reuse(tmp_path):
    pair = _source_tree(tmp_path)
    (pair / "pilot" / "exp3_domain_discovery" / "GvpK" /
     "motif_occurrences_M2.csv").unlink()

    assert inspect_pilot_source(
        pair, family="GvpK", method="M2", expected_pair_signature="pair-sig",
        expected_pilot_signature="pilot-sig",
        expected_fingerprints={"dataset": "fingerprint"},
    ) is None


def test_stage_a_unsuffixed_pilot_manifest_is_supported(tmp_path):
    pair = _source_tree(tmp_path)
    original = pair / "manifests" / "GvpK_M2_pilot.json"
    original.rename(pair / "manifests" / "GvpK_M2.json")

    assert inspect_pilot_source(
        pair, family="GvpK", method="M2", expected_pair_signature="pair-sig",
        expected_pilot_signature="pilot-sig",
        expected_fingerprints={"dataset": "fingerprint"},
        pilot_manifest_name="GvpK_M2.json",
    ) is not None
