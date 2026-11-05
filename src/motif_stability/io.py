from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REQUIRED_WINDOW_FIELDS = {
    "window_id",
    "parent_id",
    "family",
    "start",
    "end",
    "window_seq",
}


@dataclass(frozen=True)
class FamilyInput:
    family: str
    method: str
    family_dir: Path
    metadata_path: Path
    embedding_path: Path
    manifest_path: Path | None
    windows: tuple[dict[str, object], ...]
    embeddings: np.ndarray
    manifest: dict[str, object]


def _read_windows(path: Path, *, expected_family: str) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing window metadata: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_WINDOW_FIELDS - fieldnames)
        if missing:
            raise ValueError(
                f"Window metadata {path} is missing required columns: {', '.join(missing)}"
            )
        rows = []
        seen_window_ids: set[str] = set()
        for row_number, raw in enumerate(reader, start=2):
            row: dict[str, object] = dict(raw)
            for field in ("window_id", "parent_id", "family", "window_seq"):
                value = str(raw.get(field, "")).strip()
                if not value:
                    raise ValueError(f"Window metadata has empty {field} at {path}:{row_number}")
                row[field] = value
            window_id = str(row["window_id"])
            if window_id in seen_window_ids:
                raise ValueError(f"Duplicate window_id {window_id!r} at {path}:{row_number}")
            seen_window_ids.add(window_id)
            if str(row["family"]) != expected_family:
                raise ValueError(
                    f"Window family does not match directory {expected_family!r} "
                    f"at {path}:{row_number}: {row['family']!r}"
                )
            try:
                row["start"] = int(raw["start"])
                row["end"] = int(raw["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid start/end at {path}:{row_number}"
                ) from exc
            if int(row["end"]) <= int(row["start"]):
                raise ValueError(f"Non-positive window interval at {path}:{row_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Window metadata is empty: {path}")
    return tuple(rows)


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding manifest must be a JSON object: {path}")
    return payload


def load_family_input(family_dir: Path, method: str) -> FamilyInput:
    family_dir = Path(family_dir)
    method = str(method)
    metadata_path = family_dir / "windows_metadata.csv"
    embedding_path = family_dir / f"embeddings_{method}_optimized.npy"
    manifest_candidate = family_dir / f"embeddings_{method}_optimized.manifest.json"
    manifest_path = manifest_candidate if manifest_candidate.is_file() else None

    windows = _read_windows(metadata_path, expected_family=family_dir.name)
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Missing embedding array: {embedding_path}")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding array must be two-dimensional, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != len(windows):
        raise ValueError(
            "Embedding row count does not match window metadata: "
            f"{embeddings.shape[0]} != {len(windows)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Embedding array contains non-finite values: {embedding_path}")

    manifest = _load_manifest(manifest_candidate)
    if manifest:
        if manifest.get("method") not in (None, method):
            raise ValueError(
                f"Embedding manifest method mismatch: {manifest.get('method')} != {method}"
            )
        if manifest.get("window_count") not in (None, len(windows)):
            raise ValueError(
                "Embedding manifest window_count mismatch: "
                f"{manifest.get('window_count')} != {len(windows)}"
            )
        manifest_shape = manifest.get("array_shape")
        if manifest_shape is not None and list(manifest_shape) != list(embeddings.shape):
            raise ValueError(
                "Embedding manifest array_shape mismatch: "
                f"{manifest_shape} != {list(embeddings.shape)}"
            )

    return FamilyInput(
        family=family_dir.name,
        method=method,
        family_dir=family_dir,
        metadata_path=metadata_path,
        embedding_path=embedding_path,
        manifest_path=manifest_path,
        windows=windows,
        embeddings=embeddings,
        manifest=manifest,
    )


def _methods_in_family(family_dir: Path) -> tuple[str, ...]:
    methods = []
    prefix = "embeddings_"
    suffix = "_optimized.npy"
    for path in family_dir.glob(f"{prefix}*{suffix}"):
        name = path.name
        methods.append(name[len(prefix) : -len(suffix)])
    return tuple(sorted(set(methods)))


def discover_family_inputs(
    source_exp3_dir: Path,
    *,
    families: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> list[FamilyInput]:
    source_exp3_dir = Path(source_exp3_dir)
    if not source_exp3_dir.is_dir():
        raise FileNotFoundError(f"Exp3 source directory does not exist: {source_exp3_dir}")
    family_filter = set(families or [])
    method_filter = set(methods or [])
    family_dirs: Iterable[Path]
    if families:
        family_dirs = (source_exp3_dir / family for family in families)
    else:
        family_dirs = (path for path in source_exp3_dir.iterdir() if path.is_dir())

    discovered: list[FamilyInput] = []
    for family_dir in sorted(family_dirs, key=lambda path: path.name):
        if family_filter and family_dir.name not in family_filter:
            continue
        if not (family_dir / "windows_metadata.csv").is_file():
            continue
        for method in _methods_in_family(family_dir):
            if method_filter and method not in method_filter:
                continue
            discovered.append(load_family_input(family_dir, method))
    if not discovered:
        raise FileNotFoundError(
            "No complete family/method caches were found under "
            f"{source_exp3_dir}. Expected windows_metadata.csv plus "
            "embeddings_<METHOD>_optimized.npy."
        )
    return discovered
