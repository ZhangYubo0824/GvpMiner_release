#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Sequence


SUPPORTED_METHODS = (
    "M2",
    "M3",
    "MEME",
    "GLAM2",
    "STREME",
    "SLiMFinder",
    "DiMotif",
    "MotifAE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge clusterProfiler family or cluster-batch CSV results."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path)
    inputs.add_argument(
        "--input-file",
        action="append",
        type=Path,
        help=(
            "Exact result file to merge; repeat to avoid discovering stale files "
            "left by an earlier batch layout."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", default=None)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default=None)
    parser.add_argument(
        "--top-level-only",
        action="store_true",
        help="Read only family-method CSV files directly under input-dir.",
    )
    return parser.parse_args()


def discover_result_files(
    input_dir: Path,
    *,
    family: str | None = None,
    method: str | None = None,
    include_batches: bool = True,
) -> list[Path]:
    input_dir = Path(input_dir)
    if bool(family) != bool(method):
        raise ValueError("family and method must be provided together")
    if family and method:
        pattern = f"motif_sets_{family}_{method}_batch_*.csv"
        return sorted((input_dir / f"{family}_batches").glob(pattern))

    result_files = sorted(
        path
        for path in input_dir.glob("Gvp*_*.csv")
        if path.stem.rsplit("_", 1)[-1] in SUPPORTED_METHODS
    )
    if include_batches:
        result_files.extend(sorted(input_dir.glob("*_batches/*.csv")))
    return result_files


def _resolve_inputs(
    *,
    input_dir: Path | None,
    input_files: Sequence[Path] | None,
    family: str | None,
    method: str | None,
    include_batches: bool,
    output: Path,
) -> list[Path]:
    if (input_dir is None) == (not input_files):
        raise ValueError("provide exactly one of input_dir or input_files")
    if input_files:
        if family or method:
            raise ValueError("family/method filters cannot be combined with exact input_files")
        files = [Path(path) for path in input_files]
    else:
        files = discover_result_files(
            Path(input_dir),
            family=family,
            method=method,
            include_batches=include_batches,
        )
    output_resolved = output.resolve()
    files = sorted(
        {path.resolve() for path in files if path.resolve() != output_resolved},
        key=str,
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing merge inputs: {missing}")
    if not files:
        raise RuntimeError("No matching clusterProfiler CSV files found")
    return files


def _fieldnames(path: Path) -> list[str] | None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return csv.DictReader(handle).fieldnames


def merge_results(
    *,
    output: Path,
    input_dir: Path | None = None,
    input_files: Sequence[Path] | None = None,
    family: str | None = None,
    method: str | None = None,
    include_batches: bool = True,
) -> int:
    output = Path(output)
    result_files = _resolve_inputs(
        input_dir=input_dir,
        input_files=input_files,
        family=family,
        method=method,
        include_batches=include_batches,
        output=output,
    )

    fieldnames: list[str] = []
    readable_files: list[Path] = []
    for path in result_files:
        header = _fieldnames(path)
        if not header:
            print(f"EMPTY: {path}")
            continue
        readable_files.append(path)
        for field in header:
            if field and field not in fieldnames:
                fieldnames.append(field)
    if not readable_files:
        raise RuntimeError("No readable clusterProfiler result files found")
    if "result_source" not in fieldnames:
        fieldnames.append("result_source")

    duplicate_columns = [
        column
        for column in ["family", "method", "motif_cluster", "ID"]
        if column in fieldnames
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    database = output.with_name(output.name + ".merge.sqlite3")
    database_artifacts = [
        database,
        database.with_name(database.name + "-wal"),
        database.with_name(database.name + "-shm"),
    ]
    for path in [temporary, *database_artifacts]:
        path.unlink(missing_ok=True)

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute(
        "CREATE TABLE seen (deduplication_key TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    kept = 0
    removed = 0
    processed = 0
    completed = False
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            for path in readable_files:
                with path.open("r", encoding="utf-8-sig", newline="") as input_handle:
                    reader = csv.DictReader(input_handle)
                    for row in reader:
                        processed += 1
                        if duplicate_columns:
                            key = json.dumps(
                                [str(row.get(column, "") or "") for column in duplicate_columns],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            cursor = connection.execute(
                                "INSERT OR IGNORE INTO seen(deduplication_key) VALUES (?)",
                                (key,),
                            )
                            if cursor.rowcount == 0:
                                removed += 1
                                continue
                        clean = {
                            field: str(row.get(field, "") or "")
                            for field in fieldnames
                        }
                        clean["result_source"] = path.stem
                        writer.writerow(clean)
                        kept += 1
                        if processed % 100_000 == 0:
                            connection.commit()
                            print(f"Processed {processed:,} enrichment rows")
            connection.commit()
        completed = True
    finally:
        connection.close()
        if not completed:
            for path in [temporary, *database_artifacts]:
                path.unlink(missing_ok=True)

    temporary.replace(output)
    for path in database_artifacts:
        path.unlink(missing_ok=True)
    print(f"Removed {removed:,} duplicate rows")
    print(f"Wrote {kept:,} rows to {output}")
    return kept


def main() -> None:
    args = parse_args()
    merge_results(
        input_dir=args.input_dir,
        input_files=args.input_file,
        output=args.output,
        family=args.family,
        method=args.method,
        include_batches=not args.top_level_only,
    )


if __name__ == "__main__":
    main()
