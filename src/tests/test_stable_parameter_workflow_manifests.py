import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_parameter_workflow.manifests import (
    atomic_write_json,
    can_resume,
    fingerprint_inputs,
    load_manifest,
    resolve_output_root,
    unit_signature,
)


def test_output_root_must_not_overlap_existing_result_roots(tmp_path):
    baseline = tmp_path / "existing-results"
    baseline.mkdir()
    with pytest.raises(ValueError, match="existing result"):
        resolve_output_root(baseline / "new-run", baseline_roots=(baseline,))
    with pytest.raises(ValueError, match="existing result"):
        resolve_output_root(baseline, baseline_roots=(baseline,))
    assert resolve_output_root(tmp_path / "new-results", baseline_roots=(baseline,)) == (
        tmp_path / "new-results"
    ).resolve()


def test_atomic_manifest_round_trip_and_signature_match(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    signature = unit_signature({"stage": "A"}, fingerprint_inputs((source,)))
    manifest = tmp_path / "unit" / "run_manifest.json"
    atomic_write_json(manifest, {"status": "complete", "signature": signature})
    assert load_manifest(manifest)["signature"] == signature
    assert can_resume(manifest, signature)
    assert not list(manifest.parent.glob("*.tmp"))


def test_resume_rejects_changed_input_fingerprint(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("before", encoding="utf-8")
    old = unit_signature({"stage": "A"}, fingerprint_inputs((source,)))
    manifest = tmp_path / "run_manifest.json"
    atomic_write_json(manifest, {"status": "complete", "signature": old})
    source.write_text("after", encoding="utf-8")
    new = unit_signature({"stage": "A"}, fingerprint_inputs((source,)))
    assert old != new
    assert not can_resume(manifest, new)


def test_fingerprint_does_not_modify_existing_input(tmp_path):
    source = tmp_path / "baseline.json"
    source.write_text(json.dumps({"value": 1}), encoding="utf-8")
    before = source.read_bytes()
    fingerprint_inputs((source,))
    assert source.read_bytes() == before
