from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_output_root(
    output_root: Path,
    *,
    baseline_roots: Iterable[Path] = (),
) -> Path:
    candidate = Path(output_root).expanduser().resolve()
    for raw_baseline in baseline_roots:
        baseline = Path(raw_baseline).expanduser().resolve()
        if _is_relative_to(candidate, baseline) or _is_relative_to(baseline, candidate):
            raise ValueError(
                f"output root overlaps existing result root: {candidate} vs {baseline}"
            )
    return candidate


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_inputs(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for raw_path in sorted((Path(path).resolve() for path in paths), key=str):
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        stat = raw_path.stat()
        fingerprints[str(raw_path)] = {
            "sha256": sha256_file(raw_path),
            "size": stat.st_size,
        }
    return fingerprints


def unit_signature(
    parameters: Mapping[str, Any],
    input_fingerprints: Mapping[str, Any],
) -> str:
    canonical = json.dumps(
        {"parameters": dict(parameters), "inputs": dict(input_fingerprints)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def can_resume(path: Path, signature: str) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        manifest = load_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return manifest.get("status") == "complete" and manifest.get("signature") == signature
