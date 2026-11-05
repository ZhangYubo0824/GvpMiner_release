from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping


STATISTICAL_BOUNDARY = (
    "existing clusterProfiler p.adjust <= alpha; "
    "no additional multiplicity correction"
)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def existing_file_records(paths: Iterable[Path | None]) -> list[dict[str, object]]:
    return [file_record(path) for path in paths if path is not None and Path(path).is_file()]


def write_manifest(
    path: Path,
    *,
    operation: str,
    parameters: Mapping[str, Any],
    inputs: Iterable[Path | None],
    outputs: Iterable[Path | None],
    code_files: Iterable[Path | None] = (),
    extra: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "operation": operation,
        "parameters": dict(parameters),
        "inputs": existing_file_records(inputs),
        "outputs": existing_file_records(outputs),
        "code_files": existing_file_records(code_files),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "statistical_boundary": STATISTICAL_BOUNDARY,
    }
    if extra:
        payload.update(extra)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
