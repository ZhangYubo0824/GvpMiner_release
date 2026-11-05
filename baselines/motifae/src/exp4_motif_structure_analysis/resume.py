from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .residues import RESIDUE_SEMANTICS_VERSION

SUPPORTED_RESUME_STAGES = frozenset(
    {
        "features_and_controls_complete",
        "local_rmsd_in_progress",
        "local_rmsd_complete",
        "complete",
    }
)
REQUIRED_COUNT_NAMES = (
    "motif_features",
    "control_features",
    "skipped_motif_instances",
)
FEATURE_REQUIRED_HEADERS = (
    "motif_instance_id",
    "sequence_id",
    "family",
    "motif_cluster",
    "motif_start",
    "motif_end",
    "core_residue_positions_0based",
    "residue_positions_0based",
    "residue_offsets_0based",
    "motif_span_length",
    "motif_observed_length",
)
SKIPPED_REQUIRED_HEADERS = ("sequence_id", "reason")


@dataclass(frozen=True)
class ResumeState:
    stage: str
    feature_rows: list[dict[str, str]]
    skipped_rows: list[dict[str, str]]
    expected_control_count: int
    feature_path: Path
    control_path: Path
    skipped_path: Path
    progress_path: Path
    progress_payload: dict[str, Any]
    counts: dict[str, int]
    family: str
    method: str


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Resume requires existing file: {path}")


def _load_progress(progress_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed resume progress file {progress_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed resume progress payload in {progress_path}: expected an object")
    return payload


def _validated_counts(payload: Mapping[str, Any], progress_path: Path) -> dict[str, int]:
    raw_counts = payload.get("counts")
    if not isinstance(raw_counts, Mapping):
        raise ValueError(f"Malformed resume counts in {progress_path}: expected an object")
    for name in REQUIRED_COUNT_NAMES:
        if name not in raw_counts:
            raise ValueError(f"Malformed resume counts in {progress_path}: missing {name}")

    counts: dict[str, int] = {}
    for name, value in raw_counts.items():
        if not isinstance(name, str) or type(value) is not int or value < 0:
            raise ValueError(
                f"Malformed resume counts in {progress_path}: {name!r} must be a non-negative integer"
            )
        counts[name] = value
    return counts


def _optional_text(payload: Mapping[str, Any], name: str, progress_path: Path) -> str:
    value = payload.get(name, "")
    if not isinstance(value, str):
        raise ValueError(f"Malformed resume progress payload in {progress_path}: {name} must be text")
    return value.strip()


def _read_resume_rows(
    path: Path,
    label: str,
    required_headers: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing_headers = [name for name in required_headers if name not in fieldnames]
            if missing_headers:
                raise ValueError(
                    f"Resume {label} CSV is missing required headers: {', '.join(missing_headers)}"
                )

            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Resume {label} CSV row {line_number} has extra column data"
                    )
                missing_cells = [name for name, value in row.items() if value is None]
                if missing_cells:
                    raise ValueError(
                        f"Resume {label} CSV row {line_number} has a missing cell: "
                        f"{', '.join(missing_cells)}"
                    )
                rows.append(dict(row))
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read resume {label} file {path}: {exc}") from exc


def load_resume_state(
    output_dir: Path,
    family: str = "",
    method: str = "",
) -> ResumeState:
    output_dir = Path(output_dir)
    feature_path = output_dir / "motif_structure_features.csv"
    control_path = output_dir / "matched_control_features.csv"
    skipped_path = output_dir / "skipped_motif_instances.csv"
    progress_path = output_dir / "analysis_progress.json"

    for path in (progress_path, feature_path, control_path, skipped_path):
        _require_file(path)

    payload = _load_progress(progress_path)
    if payload.get("residue_semantics_version") != RESIDUE_SEMANTICS_VERSION:
        raise ValueError("Resume residue semantics version mismatch; recompute legacy interval features")
    stage = payload.get("stage")
    if not isinstance(stage, str) or stage not in SUPPORTED_RESUME_STAGES:
        raise ValueError(f"Unsupported resume stage in {progress_path}: {stage!r}")

    counts = _validated_counts(payload, progress_path)
    progress_family = _optional_text(payload, "family", progress_path)
    progress_method = _optional_text(payload, "method", progress_path)
    requested_family = str(family or "").strip()
    requested_method = str(method or "").strip()

    if requested_family and requested_family != progress_family:
        raise ValueError(
            f"Resume family mismatch: requested {requested_family!r}, progress has {progress_family!r}"
        )
    if requested_method and requested_method != progress_method:
        raise ValueError(
            f"Resume method mismatch: requested {requested_method!r}, progress has {progress_method!r}"
        )

    feature_rows = _read_resume_rows(feature_path, "feature", FEATURE_REQUIRED_HEADERS)
    skipped_headers = (
        SKIPPED_REQUIRED_HEADERS if counts["skipped_motif_instances"] > 0 else ()
    )
    skipped_rows = _read_resume_rows(skipped_path, "skipped", skipped_headers)
    if len(feature_rows) != counts["motif_features"]:
        raise ValueError(
            "Resume motif_features count mismatch: "
            f"progress has {counts['motif_features']}, file has {len(feature_rows)}"
        )
    if len(skipped_rows) != counts["skipped_motif_instances"]:
        raise ValueError(
            "Resume skipped_motif_instances count mismatch: "
            f"progress has {counts['skipped_motif_instances']}, file has {len(skipped_rows)}"
        )

    if requested_family:
        wrong_families = sorted(
            {
                str(row.get("family") or "").strip()
                for row in feature_rows
                if str(row.get("family") or "").strip() != requested_family
            }
        )
        if wrong_families:
            raise ValueError(
                "Resume feature family mismatch: "
                f"requested {requested_family!r}, feature rows contain {wrong_families!r}"
            )

    return ResumeState(
        stage=stage,
        feature_rows=feature_rows,
        skipped_rows=skipped_rows,
        expected_control_count=counts["control_features"],
        feature_path=feature_path,
        control_path=control_path,
        skipped_path=skipped_path,
        progress_path=progress_path,
        progress_payload=payload,
        counts=counts,
        family=progress_family,
        method=progress_method,
    )
