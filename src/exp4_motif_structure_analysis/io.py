from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


def iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield dict(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    return list(iter_csv_rows(path))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(str(key))
                    seen.add(key)
        fieldnames = keys or ["empty"]
    temp_path = path.with_name(f".{path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    os.replace(temp_path, path)


class CsvRowWriter:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self.count = 0
        self._handle = open(self.path, "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._handle.flush()

    def writerow(self, row: Mapping[str, object]) -> None:
        self._writer.writerow(row)
        self.count += 1

    def writerows(self, rows: Iterable[Mapping[str, object]]) -> int:
        written = 0
        for row in rows:
            self.writerow(row)
            written += 1
        self._handle.flush()
        return written

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> "CsvRowWriter":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def split_terms(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    for sep in [";", "|", ","]:
        text = text.replace(sep, ";")
    return [part.strip() for part in text.split(";") if part.strip()]


def stable_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
