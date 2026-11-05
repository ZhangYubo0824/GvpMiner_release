"""Inspect completed upstream pilot outputs without requiring raw embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .manifests import load_manifest
from .reuse import source_file_hashes


@dataclass(frozen=True)
class PilotReuseSource:
    pair_root: Path
    diagnostic_dir: Path
    catalog_dir: Path
    diagnostic_hashes: dict[str, str]
    catalog_hashes: dict[str, str]
    pilot_signature: str
    pair_signature: str
    pair_generation_id: str


def inspect_pilot_source(
    pair_root: Path,
    *,
    family: str,
    method: str,
    expected_pair_signature: str,
    expected_pilot_signature: str,
    expected_fingerprints: Mapping[str, Any],
    pilot_manifest_name: str | None = None,
) -> PilotReuseSource | None:
    """Return reusable final pilot files, or None if the source is unusable."""
    pair_root = Path(pair_root)
    pair_manifest_path = pair_root / "manifests" / "exp3_pair.json"
    pilot_manifest_path = pair_root / "manifests" / (
        pilot_manifest_name or f"{family}_{method}_pilot.json"
    )
    diagnostic_dir = pair_root / "pilot" / "motif_stability" / family / method
    catalog_dir = pair_root / "pilot" / "exp3_domain_discovery" / family
    try:
        if pair_manifest_path.is_symlink() or pilot_manifest_path.is_symlink():
            return None
        pair_manifest = load_manifest(pair_manifest_path)
        pilot_manifest = load_manifest(pilot_manifest_path)
        generation_id = pair_manifest.get("generation_id")
        if (
            pair_manifest.get("status") != "complete"
            or pair_manifest.get("signature") != expected_pair_signature
            or not isinstance(generation_id, str)
            or not generation_id
            or pilot_manifest.get("status") != "complete"
            or pilot_manifest.get("signature") != expected_pilot_signature
            or pilot_manifest.get("input_fingerprints") != dict(expected_fingerprints)
            or pilot_manifest.get("parameters", {}).get("upstream_exp3") != {
                "signature": expected_pair_signature,
                "generation_id": generation_id,
            }
        ):
            return None
        diagnostic_hashes = source_file_hashes(
            diagnostic_dir, ("selected_setting.json",),
        )
        catalog_hashes = source_file_hashes(catalog_dir, (
            f"motif_catalog_{method}.csv",
            f"motif_occurrences_{method}.csv",
        ))
        selected = json.loads(
            (diagnostic_dir / "selected_setting.json").read_text(encoding="utf-8")
        )
        if not isinstance(selected, dict) or not isinstance(
            selected.get("min_cluster_size"), int
        ):
            return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return PilotReuseSource(
        pair_root=pair_root,
        diagnostic_dir=diagnostic_dir,
        catalog_dir=catalog_dir,
        diagnostic_hashes=diagnostic_hashes,
        catalog_hashes=catalog_hashes,
        pilot_signature=expected_pilot_signature,
        pair_signature=expected_pair_signature,
        pair_generation_id=generation_id,
    )
