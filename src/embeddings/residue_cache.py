"""Validated two-tier cache for per-residue protein embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


EXPECTED_DIMS = {"M2": 1280, "M3": 1024}
AuditCallback = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class CacheResult:
    array: np.ndarray
    path: Path
    source: str


def _normalise_method(method: str) -> str:
    normalized = str(method).upper()
    if normalized not in EXPECTED_DIMS:
        raise ValueError(f"Unsupported embedding method: {method}")
    return normalized


def sequence_hash(sequence: str) -> str:
    """Return the sequence-content digest used by validated experiment caches."""
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()


def cache_filename(method: str, sequence: str) -> str:
    method = _normalise_method(method)
    return f"{method}_{sequence_hash(sequence)}.npy"


def validate_array(
    array: np.ndarray,
    method: str,
    sequence: str,
    path: Path,
) -> None:
    method = _normalise_method(method)
    expected = (len(sequence), EXPECTED_DIMS[method])
    if array.shape != expected:
        raise ValueError(f"{path}: expected shape {expected}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{path}: residue embedding contains invalid values")


def _audit_record(
    *,
    method: str,
    sequence: str,
    path: Path,
    source: str,
    array: np.ndarray,
    outcome: str,
) -> dict[str, object]:
    return {
        "method": _normalise_method(method),
        "sequence_hash": sequence_hash(sequence),
        "path": str(path),
        "source": source,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "validation": outcome,
    }


def load_validated(
    cache_dir: Path,
    method: str,
    sequence: str,
    *,
    audit_callback: AuditCallback | None = None,
    source: str = "cache",
) -> CacheResult | None:
    path = Path(cache_dir) / cache_filename(method, sequence)
    if not path.is_file():
        return None

    # Load one protein at a time.  A persistent memmap keeps the cache file
    # locked on Windows and prevents safe cleanup/replacement.
    array = np.load(path, allow_pickle=False)
    try:
        validate_array(array, method, sequence, path)
    except (TypeError, ValueError):
        if audit_callback is not None:
            audit_callback(
                _audit_record(
                    method=method,
                    sequence=sequence,
                    path=path,
                    source=source,
                    array=array,
                    outcome="invalid",
                )
            )
        raise

    if audit_callback is not None:
        audit_callback(
            _audit_record(
                method=method,
                sequence=sequence,
                path=path,
                source=source,
                array=array,
                outcome="valid",
            )
        )
    return CacheResult(array=array, path=path, source=source)


def load_or_compute(
    *,
    method: str,
    sequence: str,
    primary_dirs: Sequence[Path],
    supplement_dir: Path,
    compute: Callable[[], np.ndarray],
    audit_callback: AuditCallback | None = None,
) -> CacheResult:
    method = _normalise_method(method)
    for directory in primary_dirs:
        hit = load_validated(
            Path(directory),
            method,
            sequence,
            audit_callback=audit_callback,
            source="primary",
        )
        if hit is not None:
            return hit

    hit = load_validated(
        Path(supplement_dir),
        method,
        sequence,
        audit_callback=audit_callback,
        source="supplement",
    )
    if hit is not None:
        return hit

    array = np.asarray(compute(), dtype=np.float32)
    computed_path = Path("computed")
    validate_array(array, method, sequence, computed_path)

    supplement_dir = Path(supplement_dir)
    supplement_dir.mkdir(parents=True, exist_ok=True)
    target = supplement_dir / cache_filename(method, sequence)
    temporary = target.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(target)

    persisted = np.load(target, allow_pickle=False)
    validate_array(persisted, method, sequence, target)
    if audit_callback is not None:
        audit_callback(
            _audit_record(
                method=method,
                sequence=sequence,
                path=target,
                source="computed",
                array=persisted,
                outcome="valid",
            )
        )
    return CacheResult(array=persisted, path=target, source="computed")
