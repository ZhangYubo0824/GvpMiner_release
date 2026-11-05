"""Read-only Stage A artifact reuse for an interrupted Stage B."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

from .manifests import load_manifest, sha256_file


def require_safe_parent(root: Path, target: Path) -> None:
    """Reject parent links that would redirect a Stage B write outside its pair."""
    root = Path(root).absolute()
    target = Path(target).absolute()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Stage B target escapes pair: {target}") from exc
    if target == root:
        if root.is_symlink():
            raise ValueError(f"Stage B pair link is unsafe: {root}")
        return
    current = target.parent
    while True:
        if current.is_symlink():
            raise ValueError(f"Stage B parent link is unsafe: {current}")
        if current == root:
            break
        if current == current.parent:
            raise ValueError(f"Stage B target escapes pair: {target}")
        current = current.parent
    if root.is_symlink():
        raise ValueError(f"Stage B pair link is unsafe: {root}")


def source_file_hashes(
    root: Path,
    required: Iterable[str],
    *,
    include_all: bool = True,
) -> dict[str, str]:
    """Hash a source tree, requiring the scientifically essential outputs."""
    root = Path(root)
    required_names = tuple(required)
    for name in required_names:
        if not (root / name).is_file():
            raise ValueError(f"Stage A reuse source is missing: {root / name}")
    paths = (
        sorted(root.rglob("*")) if include_all
        else [root / name for name in required_names]
    )
    hashes = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Stage A reuse source contains a link: {path}")
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def check_partial_raw_files(target: Path, hashes: dict[str, str]) -> None:
    """Do not use or overwrite conflicting interrupted Stage B raw outputs."""
    target = Path(target)
    if target.is_symlink():
        raise ValueError(f"Stage B reuse conflict: unexpected link: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Stage B reuse conflict: expected directory: {target}")
    if not target.exists():
        return
    for name, digest in hashes.items():
        path = target / name
        if path.is_symlink():
            raise ValueError(f"Stage B reuse conflict: unexpected link: {path}")
        if path.exists() and (not path.is_file() or sha256_file(path) != digest):
            raise ValueError(f"Stage B reuse conflict: {path}")


def materialize_files(
    source: Path,
    target: Path,
    hashes: dict[str, str],
    *,
    target_root: Path,
) -> None:
    """Copy verified pilot files into Stage B without following any link."""
    source = Path(source).resolve()
    target = Path(target)
    require_safe_parent(target_root, target)
    if target.is_symlink():
        raise ValueError(f"Stage B reuse conflict: unexpected link: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Stage B reuse conflict: expected directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in hashes.items():
        destination = target / name
        require_safe_parent(target_root, destination)
        if destination.is_symlink():
            raise ValueError(f"Stage B reuse conflict: unexpected link: {destination}")
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError(f"Stage B reuse conflict: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
        if sha256_file(destination) != digest:
            raise ValueError(f"Stage B reuse copy verification failed: {destination}")


def provenance_entry(
    source: Path,
    target: Path | None,
    hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "source_dir": str(Path(source).resolve()),
        "target_dir": str(Path(target).absolute()) if target is not None else None,
        "sha256": dict(hashes),
        "reuse_method": "independent_copy" if target is not None else "source_reference",
    }


def archive_interrupted_pilot(
    pair_root: Path,
    targets: Iterable[Path],
) -> list[Path]:
    """Move incomplete Stage B pilot artifacts to a recoverable pair-local backup."""
    pair_root = Path(pair_root).absolute()
    backup_parent = pair_root / "interrupted_pilot"
    backup_root = backup_parent / uuid.uuid4().hex[:12]
    while backup_root.exists():
        backup_root = backup_parent / uuid.uuid4().hex[:12]
    archived = []
    for target in targets:
        target = Path(target).absolute()
        try:
            relative = target.relative_to(pair_root)
        except ValueError as exc:
            raise ValueError(f"Stage B pilot path is outside its pair: {target}") from exc
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink():
            raise ValueError(f"Stage B reuse conflict: unexpected link: {target}")
        try:
            target.resolve().relative_to(pair_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Stage B reuse conflict: path escapes pair: {target}") from exc
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        target.replace(destination)
        archived.append(destination)
    return archived


def verify_provenance(
    entries: Iterable[dict[str, Any]], *, local_only: bool = False,
    required_local_names: set[str] | None = None,
) -> None:
    """Verify recorded files, or only materialized targets at handoff."""
    for entry in entries:
        source = Path(entry["source_dir"])
        target = (Path(entry["target_dir"])
                  if entry["target_dir"] is not None else None)
        hashes = entry["sha256"]
        if not isinstance(hashes, dict):
            raise ValueError("invalid Stage A reuse checksum map")
        for name, digest in hashes.items():
            if local_only and required_local_names is not None and name not in required_local_names:
                continue
            if local_only:
                paths = (target / name,) if target is not None else ()
            else:
                paths = ((source / name,) if target is None else
                         (source / name, target / name))
            for path in paths:
                if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError(f"Stage A reuse lineage mismatch: {path}")


def verify_reuse_record(
    record: dict[str, Any],
    *,
    stage_a_root: Path,
    target_root: Path,
    local_only: bool = False,
) -> None:
    """Check source manifests and recorded hashes at the required locations."""
    if record.get("stage") != "A":
        raise ValueError("Stage A reuse lineage has an invalid source stage")
    source_manifest_path = Path(record["source_manifest_path"]).resolve()
    stage_a_root = Path(stage_a_root).resolve()
    target_root = Path(target_root).absolute()
    try:
        source_manifest_path.relative_to(stage_a_root)
    except ValueError as exc:
        raise ValueError("Stage A reuse lineage manifest escapes source root") from exc
    if not isinstance(record.get("source_signature"), str) or not record["source_signature"]:
        raise ValueError("Stage A reuse lineage source signature is missing")
    source_manifest = load_manifest(source_manifest_path)
    if (
        source_manifest.get("status") != "complete"
        or source_manifest.get("signature") != record["source_signature"]
        or (
            "source_generation_id" in record
            and source_manifest.get("generation_id") != record["source_generation_id"]
        )
    ):
        raise ValueError(f"Stage A reuse lineage manifest changed: {source_manifest_path}")
    if "source_generation_id" in record and (
        not isinstance(record["source_generation_id"], str)
        or not record["source_generation_id"]
    ):
        raise ValueError("Stage A reuse lineage source generation is missing")
    if "family" in record:
        pair_manifest = Path(record["source_pair_manifest_path"]).resolve()
        try:
            pair_manifest.relative_to(stage_a_root)
        except ValueError as exc:
            raise ValueError("Stage A reuse pair manifest escapes source root") from exc
        if not record.get("source_pair_signature") or not record.get("source_pair_generation_id"):
            raise ValueError("Stage A reuse lineage pair generation is missing")
        pair_payload = load_manifest(pair_manifest)
        if (
            pair_payload.get("status") != "complete"
            or pair_payload.get("signature") != record["source_pair_signature"]
            or pair_payload.get("generation_id") != record["source_pair_generation_id"]
            or source_manifest.get("parameters", {}).get(
                "upstream_exp3", {}
            ).get("generation_id") != record["source_pair_generation_id"]
        ):
            raise ValueError("Stage A reuse lineage pair generation changed")
    elif source_manifest.get("generation_id") != record["source_generation_id"]:
        raise ValueError("Stage A reuse lineage source generation changed")
    entries = record["files"]
    if not entries:
        raise ValueError("Stage A reuse lineage has no files")
    for entry in entries:
        source = Path(entry["source_dir"]).resolve()
        raw_target = entry["target_dir"]
        target = Path(raw_target).absolute() if raw_target is not None else None
        try:
            source.relative_to(stage_a_root)
            if target is not None:
                target.relative_to(target_root)
        except ValueError as exc:
            raise ValueError("Stage A reuse lineage path escapes expected root") from exc
        if target is not None:
            require_safe_parent(target_root, target)
            if target.is_symlink():
                raise ValueError("Stage A reuse lineage target link is unsafe")
            try:
                target.resolve().relative_to(target_root.resolve())
            except ValueError as exc:
                raise ValueError("Stage A reuse lineage target escapes Stage B") from exc
        if provenance_entry(source, target, entry["sha256"])["reuse_method"] != entry["reuse_method"]:
            raise ValueError("Stage A reuse lineage materialization method changed")
    required_local_names = None
    if local_only and "family" in record:
        method = record["method"]
        required_local_names = {
            "selected_setting.json",
            f"motif_catalog_{method}.csv",
            f"motif_occurrences_{method}.csv",
        }
    verify_provenance(
        entries, local_only=local_only,
        required_local_names=required_local_names,
    )
