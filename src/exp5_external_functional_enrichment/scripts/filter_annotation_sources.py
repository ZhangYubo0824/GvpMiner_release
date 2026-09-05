#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


EXP5_ROOT = Path(__file__).resolve().parents[1]
if str(EXP5_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP5_ROOT))

from package import reproducibility


ANNOTATION_FIELDS = [
    "source",
    "namespace",
    "term_id",
    "term_name",
    "protein_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild annotation tables while excluding selected annotation "
            "sources from an existing functional_annotations_long.csv."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="Exact source label to exclude; may be provided more than once.",
    )
    parser.add_argument(
        "--exclude-namespace",
        action="append",
        default=[],
        metavar="SOURCE:NAMESPACE",
        help=(
            "Exclude one exact source/namespace pair, for example "
            "InterProScan:Pathway; may be provided more than once."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    return parser.parse_args()


def _flush_batch(
    connection: sqlite3.Connection,
    term_gene_rows: list[tuple[str, str]],
    term_name_rows: list[tuple[str, str]],
) -> None:
    connection.executemany(
        "INSERT OR IGNORE INTO term_gene(term_id, protein_id) VALUES (?, ?)",
        term_gene_rows,
    )
    connection.executemany(
        """
        INSERT INTO term_name(term_id, term_name)
        VALUES (?, ?)
        ON CONFLICT(term_id) DO UPDATE SET term_name = excluded.term_name
        WHERE term_name.term_name = ''
          AND excluded.term_name <> ''
        """,
        term_name_rows,
    )
    connection.commit()
    term_gene_rows.clear()
    term_name_rows.clear()


def _write_term_tables(
    connection: sqlite3.Connection,
    term2gene_path: Path,
    term2name_path: Path,
) -> None:
    with term2gene_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["term_id", "protein_id"])
        writer.writerows(
            connection.execute(
                """
                SELECT term_id, protein_id
                FROM term_gene
                ORDER BY term_id, protein_id
                """
            )
        )

    with term2name_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["term_id", "term_name"])
        for term_id, term_name in connection.execute(
            """
            SELECT term_id, term_name
            FROM term_name
            ORDER BY term_id
            """
        ):
            writer.writerow([term_id, term_name or term_id])


def _write_source_counts(
    path: Path,
    input_counts: Counter[str],
    retained_counts: Counter[str],
    filtered_counts: Counter[str],
    invalid_counts: Counter[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "input_rows",
                "retained_rows",
                "filtered_rows",
                "invalid_rows",
                "excluded_rows",
            ]
        )
        for source in sorted(input_counts):
            input_rows = input_counts[source]
            retained_rows = retained_counts[source]
            writer.writerow(
                [
                    source,
                    input_rows,
                    retained_rows,
                    filtered_counts[source],
                    invalid_counts[source],
                    filtered_counts[source],
                ]
            )


def _write_namespace_counts(
    path: Path,
    input_counts: Counter[tuple[str, str]],
    retained_counts: Counter[tuple[str, str]],
    filtered_counts: Counter[tuple[str, str]],
    invalid_counts: Counter[tuple[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "namespace",
                "input_rows",
                "retained_rows",
                "filtered_rows",
                "invalid_rows",
                "excluded_rows",
            ]
        )
        for source, namespace in sorted(input_counts):
            input_rows = input_counts[(source, namespace)]
            retained_rows = retained_counts[(source, namespace)]
            writer.writerow(
                [
                    source,
                    namespace,
                    input_rows,
                    retained_rows,
                    filtered_counts[(source, namespace)],
                    invalid_counts[(source, namespace)],
                    filtered_counts[(source, namespace)],
                ]
            )


def _normalise_excluded_namespaces(
    values: Iterable[str | tuple[str, str]],
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, tuple):
            source, namespace = value
        else:
            text = str(value).strip()
            if ":" not in text:
                raise ValueError(
                    "--exclude-namespace must use SOURCE:NAMESPACE syntax"
                )
            source, namespace = text.split(":", 1)
        source = source.strip()
        namespace = namespace.strip()
        if not source or not namespace:
            raise ValueError(
                "--exclude-namespace must contain non-empty source and namespace"
            )
        result.add((source, namespace))
    return result


def rebuild_term_tables(
    *,
    input_path: Path,
    output_dir: Path,
    excluded_sources: Iterable[str],
    excluded_namespaces: Iterable[str | tuple[str, str]] = (),
    batch_size: int = 10_000,
) -> dict[str, Path]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = {source.strip() for source in excluded_sources if source.strip()}
    excluded_pairs = _normalise_excluded_namespaces(excluded_namespaces)

    long_path = output_dir / "functional_annotations_long.csv"
    term2gene_path = output_dir / "term2gene.csv"
    term2name_path = output_dir / "term2name.csv"
    counts_path = output_dir / "annotation_source_counts.csv"
    namespace_counts_path = output_dir / "annotation_namespace_counts.csv"
    database_path = output_dir / ".annotation_filter.sqlite3"

    temporary_paths = {
        "long": long_path.with_suffix(".csv.tmp"),
        "term2gene": term2gene_path.with_suffix(".csv.tmp"),
        "term2name": term2name_path.with_suffix(".csv.tmp"),
        "counts": counts_path.with_suffix(".csv.tmp"),
        "namespace_counts": namespace_counts_path.with_suffix(".csv.tmp"),
    }
    if database_path.exists():
        database_path.unlink()

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute(
        """
        CREATE TABLE term_gene (
            term_id TEXT NOT NULL,
            protein_id TEXT NOT NULL,
            PRIMARY KEY (term_id, protein_id)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE term_name (
            term_id TEXT PRIMARY KEY,
            term_name TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )

    input_counts: Counter[str] = Counter()
    retained_counts: Counter[str] = Counter()
    filtered_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    namespace_input_counts: Counter[tuple[str, str]] = Counter()
    namespace_retained_counts: Counter[tuple[str, str]] = Counter()
    namespace_filtered_counts: Counter[tuple[str, str]] = Counter()
    namespace_invalid_counts: Counter[tuple[str, str]] = Counter()
    term_gene_rows: list[tuple[str, str]] = []
    term_name_rows: list[tuple[str, str]] = []

    try:
        with input_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as input_handle, temporary_paths["long"].open(
            "w", encoding="utf-8", newline=""
        ) as long_handle:
            reader = csv.DictReader(input_handle)
            missing = set(ANNOTATION_FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Annotation table missing columns: {sorted(missing)}"
                )
            long_writer = csv.DictWriter(
                long_handle,
                fieldnames=ANNOTATION_FIELDS,
            )
            long_writer.writeheader()

            for row_number, row in enumerate(reader, start=1):
                source = str(row.get("source", "")).strip()
                namespace = str(row.get("namespace", "")).strip()
                input_counts[source] += 1
                namespace_input_counts[(source, namespace)] += 1
                if source in excluded or (source, namespace) in excluded_pairs:
                    filtered_counts[source] += 1
                    namespace_filtered_counts[(source, namespace)] += 1
                    continue

                term_id = str(row.get("term_id", "")).strip()
                protein_id = str(row.get("protein_id", "")).strip()
                if not term_id or not protein_id:
                    invalid_counts[source] += 1
                    namespace_invalid_counts[(source, namespace)] += 1
                    continue

                retained_counts[source] += 1
                namespace_retained_counts[(source, namespace)] += 1
                clean_row = {
                    field: str(row.get(field, "") or "").strip()
                    for field in ANNOTATION_FIELDS
                }
                long_writer.writerow(clean_row)
                term_gene_rows.append((term_id, protein_id))
                term_name_rows.append(
                    (term_id, clean_row["term_name"])
                )

                if len(term_gene_rows) >= batch_size:
                    _flush_batch(
                        connection,
                        term_gene_rows,
                        term_name_rows,
                    )
                if row_number % 1_000_000 == 0:
                    print(f"Processed {row_number:,} annotation rows")

        if term_gene_rows:
            _flush_batch(connection, term_gene_rows, term_name_rows)

        _write_term_tables(
            connection,
            temporary_paths["term2gene"],
            temporary_paths["term2name"],
        )
        _write_source_counts(
            temporary_paths["counts"],
            input_counts,
            retained_counts,
            filtered_counts,
            invalid_counts,
        )
        _write_namespace_counts(
            temporary_paths["namespace_counts"],
            namespace_input_counts,
            namespace_retained_counts,
            namespace_filtered_counts,
            namespace_invalid_counts,
        )
    finally:
        connection.close()

    temporary_paths["long"].replace(long_path)
    temporary_paths["term2gene"].replace(term2gene_path)
    temporary_paths["term2name"].replace(term2name_path)
    temporary_paths["counts"].replace(counts_path)
    temporary_paths["namespace_counts"].replace(namespace_counts_path)
    database_path.unlink(missing_ok=True)
    database_path.with_name(database_path.name + "-wal").unlink(
        missing_ok=True
    )
    database_path.with_name(database_path.name + "-shm").unlink(
        missing_ok=True
    )

    paths = {
        "long": long_path,
        "term2gene": term2gene_path,
        "term2name": term2name_path,
        "source_counts": counts_path,
        "namespace_counts": namespace_counts_path,
    }
    manifest_path = reproducibility.write_manifest(
        output_dir / "annotation_filter_manifest.json",
        operation="filter-annotation-sources",
        parameters={
            "excluded_sources": sorted(excluded),
            "excluded_namespaces": [
                f"{source}:{namespace}"
                for source, namespace in sorted(excluded_pairs)
            ],
            "batch_size": batch_size,
        },
        inputs=[input_path],
        outputs=paths.values(),
        code_files=[Path(__file__), Path(reproducibility.__file__)],
    )
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    args = parse_args()
    paths = rebuild_term_tables(
        input_path=args.input,
        output_dir=args.output_dir,
        excluded_sources=args.exclude_source,
        excluded_namespaces=args.exclude_namespace,
        batch_size=args.batch_size,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
