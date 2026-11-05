#!/usr/bin/env python3
"""Run clusterProfiler enrichment in bounded-memory motif-cluster batches."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping, Sequence


REQUIRED_MOTIF_FIELDS = (
    "family",
    "method",
    "motif_cluster",
    "set_type",
    "protein_id",
)
SAFE_NAME = re.compile(r"^[A-Za-z0-9.-]+$")
BATCH_STRATEGIES = ("uniform",)


@dataclass(frozen=True)
class BatchSpec:
    family: str
    method: str
    index: int
    cluster_count: int
    row_count: int
    path: Path


@dataclass(frozen=True)
class BatchRunSummary:
    total: int
    completed: int
    skipped: int
    failed: int
    state: str


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_component(value: str, label: str) -> str:
    value = str(value).strip()
    if not value or not SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe or empty {label}: {value!r}")
    return value


def split_motif_sets(
    *,
    motif_sets: Path,
    output_dir: Path,
    clusters_per_batch: int,
    batch_strategy: str = "uniform",
    progress_every_rows: int = 1_000_000,
) -> tuple[BatchSpec, ...]:
    """Stream a long motif-set CSV into deterministic family/method batches."""
    motif_sets = Path(motif_sets)
    output_dir = Path(output_dir)
    if clusters_per_batch < 1:
        raise ValueError("clusters_per_batch must be at least 1")
    if progress_every_rows < 1:
        raise ValueError("progress_every_rows must be at least 1")
    if batch_strategy not in BATCH_STRATEGIES:
        raise ValueError(f"unknown batch strategy: {batch_strategy}")
    if not motif_sets.is_file():
        raise FileNotFoundError(motif_sets)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[BatchSpec] = []
    seen_clusters: set[tuple[str, str, str]] = set()
    batch_numbers: dict[tuple[str, str], int] = {}
    current_cluster: tuple[str, str, str] | None = None
    current_pair: tuple[str, str] | None = None
    current_path: Path | None = None
    current_temporary: Path | None = None
    current_handle = None
    current_writer = None
    current_cluster_count = 0
    current_row_count = 0
    fieldnames: list[str] = []

    def close_batch() -> None:
        nonlocal current_handle, current_writer
        nonlocal current_path, current_temporary
        nonlocal current_cluster_count, current_row_count
        if current_handle is None:
            return
        current_handle.close()
        current_handle = None
        current_writer = None
        current_temporary.replace(current_path)
        family, method = current_pair
        inventory.append(
            BatchSpec(
                family=family,
                method=method,
                index=batch_numbers[current_pair],
                cluster_count=current_cluster_count,
                row_count=current_row_count,
                path=current_path,
            )
        )
        current_path = None
        current_temporary = None
        current_cluster_count = 0
        current_row_count = 0

    try:
        with motif_sets.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or [])
            missing = set(REQUIRED_MOTIF_FIELDS) - set(fieldnames)
            if missing:
                raise ValueError(f"motif sets missing columns: {sorted(missing)}")

            for total_rows, row in enumerate(reader, start=1):
                family = _safe_component(row.get("family", ""), "family")
                method = _safe_component(row.get("method", ""), "method")
                cluster = str(row.get("motif_cluster", "")).strip()
                if not cluster:
                    raise ValueError("empty motif_cluster")
                cluster_key = (family, method, cluster)
                pair = (family, method)

                if cluster_key != current_cluster:
                    if cluster_key in seen_clusters:
                        raise ValueError(
                            "motif cluster rows are not contiguous: "
                            f"{family}/{method}/{cluster}"
                        )
                    seen_clusters.add(cluster_key)
                    split_pair = True
                    needs_new_batch = (
                        current_handle is None
                        or pair != current_pair
                        or (
                            split_pair
                            and current_cluster_count >= clusters_per_batch
                        )
                    )
                    if needs_new_batch:
                        close_batch()
                        current_pair = pair
                        batch_numbers[pair] = batch_numbers.get(pair, 0) + 1
                        batch_number = batch_numbers[pair]
                        current_path = output_dir / (
                            f"motif_sets_{family}_{method}_batch_{batch_number:05d}.csv"
                        )
                        current_temporary = current_path.with_name(
                            current_path.name + ".tmp"
                        )
                        current_handle = current_temporary.open(
                            "w", encoding="utf-8", newline=""
                        )
                        current_writer = csv.DictWriter(
                            current_handle,
                            fieldnames=fieldnames,
                            extrasaction="ignore",
                        )
                        current_writer.writeheader()
                    current_cluster_count += 1
                    current_cluster = cluster_key

                current_writer.writerow(row)
                current_row_count += 1
                if total_rows % progress_every_rows == 0:
                    print(
                        f"Split progress: {total_rows:,} rows, "
                        f"{len(seen_clusters):,} clusters, "
                        f"{len(inventory):,} closed batches",
                        flush=True,
                    )
        close_batch()
    except BaseException:
        if current_handle is not None:
            current_handle.close()
        if current_temporary is not None:
            current_temporary.unlink(missing_ok=True)
        raise

    _write_json(
        output_dir / "batch_inventory.json",
        {
            "schema_version": 1,
            "source": str(motif_sets.resolve()),
            "source_size": motif_sets.stat().st_size,
            "source_mtime_ns": motif_sets.stat().st_mtime_ns,
            "clusters_per_batch": clusters_per_batch,
            "batch_strategy": batch_strategy,
            "batches": [
                {
                    "family": item.family,
                    "method": item.method,
                    "index": item.index,
                    "cluster_count": item.cluster_count,
                    "row_count": item.row_count,
                    "path": str(item.path.resolve()),
                }
                for item in inventory
            ],
        },
    )
    total_rows = sum(item.row_count for item in inventory)
    total_clusters = sum(item.cluster_count for item in inventory)
    print(
        f"Split complete: {total_rows:,} rows, {total_clusters:,} clusters, "
        f"{len(inventory):,} batches",
        flush=True,
    )
    return tuple(inventory)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_identity(path: Path) -> dict[str, object]:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _result_path(output_dir: Path, batch: BatchSpec) -> Path:
    return Path(output_dir) / batch.family / f"{batch.path.stem}.csv"


def _result_manifest_path(result_path: Path) -> Path:
    return result_path.with_suffix(".manifest.json")


def _matching_complete_manifest(path: Path, expected: Mapping[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return payload.get("state") == "complete" and all(
        payload.get(key) == value for key, value in expected.items()
    )


def run_batches(
    *,
    inventory: Sequence[BatchSpec],
    rscript: str,
    r_script: Path,
    family_term2gene: Mapping[str, Path],
    term2name: Path,
    output_dir: Path,
    progress_path: Path,
    resume: bool,
    runner: Callable[..., object] = subprocess.run,
) -> BatchRunSummary:
    """Run the existing per-cluster R analysis on small serial batches."""
    r_script = Path(r_script)
    term2name = Path(term2name)
    output_dir = Path(output_dir)
    progress_path = Path(progress_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    total = len(inventory)

    def write_progress(
        state: str,
        *,
        current: str | None = None,
        failed: int = 0,
    ) -> None:
        _write_json(
            progress_path,
            {
                "schema_version": 1,
                "state": state,
                "total_batches": total,
                "completed_batches": completed,
                "skipped_batches": skipped,
                "failed_batches": failed,
                "pending_batches": max(0, total - completed),
                "current_batch": current,
                "updated_at": _utc_now(),
            },
        )

    write_progress("running")
    for position, batch in enumerate(inventory, 1):
        term2gene = family_term2gene.get(batch.family)
        if term2gene is None or not Path(term2gene).is_file():
            raise FileNotFoundError(f"missing family TERM2GENE for {batch.family}")
        term2gene = Path(term2gene)
        result_path = _result_path(output_dir, batch)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = result_path.with_suffix(".log")
        manifest_path = _result_manifest_path(result_path)
        argv = [
            rscript,
            str(r_script),
            f"--term2gene={term2gene}",
            f"--term2name={term2name}",
            f"--motif-sets={batch.path}",
            f"--output={result_path}",
        ]
        expected = {
            "command": argv,
            "batch_input": _file_identity(batch.path),
            "r_script": _file_identity(r_script),
            "term2gene": _file_identity(term2gene),
            "term2name": _file_identity(term2name),
        }
        if (
            resume
            and result_path.is_file()
            and result_path.stat().st_size > 0
            and _matching_complete_manifest(manifest_path, expected)
        ):
            completed += 1
            skipped += 1
            print(f"SKIP batch {position}/{total}: {batch.path.name}", flush=True)
            write_progress("running")
            continue

        manifest = {
            "schema_version": 1,
            "state": "running",
            "started_at": _utc_now(),
            "ended_at": None,
            "error": None,
            **expected,
        }
        _write_json(manifest_path, manifest)
        write_progress("running", current=batch.path.name)
        print(f"START batch {position}/{total}: {batch.path.name}", flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                runner(
                    argv,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                )
            if not result_path.is_file() or result_path.stat().st_size == 0:
                raise RuntimeError(f"batch result is missing or empty: {result_path}")
        except BaseException as exc:
            manifest.update(
                state="failed",
                ended_at=_utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_json(manifest_path, manifest)
            write_progress("failed", current=batch.path.name, failed=1)
            raise
        manifest.update(state="complete", ended_at=_utc_now())
        _write_json(manifest_path, manifest)
        completed += 1
        print(f"DONE batch {position}/{total}: {batch.path.name}", flush=True)
        write_progress("running")

    write_progress("complete")
    return BatchRunSummary(total, completed, skipped, 0, "complete")


def merge_batch_results(
    *,
    inventory: Sequence[BatchSpec],
    results_dir: Path,
    output: Path,
) -> int:
    """Concatenate disjoint batch results without changing their schema."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    fieldnames: list[str] | None = None
    empty_header: list[str] | None = None
    row_count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = None
            for batch in inventory:
                result_path = _result_path(results_dir, batch)
                if not result_path.is_file() or result_path.stat().st_size == 0:
                    raise FileNotFoundError(f"missing batch result: {result_path}")
                with result_path.open("r", encoding="utf-8-sig", newline="") as source:
                    reader = csv.DictReader(source)
                    header = list(reader.fieldnames or [])
                    if not header:
                        raise ValueError(f"batch result has no header: {result_path}")
                    first_row = next(reader, None)
                    if first_row is None:
                        if empty_header is None:
                            empty_header = header
                        continue
                    if fieldnames is None:
                        fieldnames = header
                        writer = csv.DictWriter(target, fieldnames=fieldnames)
                        writer.writeheader()
                    elif header != fieldnames:
                        raise ValueError(f"batch result schema mismatch: {result_path}")
                    writer.writerow(first_row)
                    row_count += 1
                    for row in reader:
                        writer.writerow(row)
                        row_count += 1
        if fieldnames is None:
            if empty_header is None:
                raise RuntimeError("no batch results to merge")
            with temporary.open("w", encoding="utf-8", newline="") as target:
                csv.DictWriter(target, fieldnames=empty_header).writeheader()
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return row_count


def split_term2gene_by_family(
    *,
    sequence_index: Path,
    term2gene: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Stream TERM2GENE links into files selected by sequence-index family."""
    sequence_index = Path(sequence_index)
    term2gene = Path(term2gene)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_to_family: dict[str, str] = {}
    with sequence_index.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"protein_id", "family"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"sequence index missing columns: {sorted(missing)}")
        for row in reader:
            protein_id = str(row.get("protein_id", "")).strip()
            family = str(row.get("family", "")).strip()
            if protein_id and family:
                protein_to_family[protein_id] = _safe_component(family, "family")

    paths = {
        family: output_dir / f"term2gene_{family}.csv"
        for family in sorted(set(protein_to_family.values()))
    }
    handles = {}
    writers = {}
    temporary_paths = {
        family: path.with_name(path.name + ".tmp")
        for family, path in paths.items()
    }
    try:
        for family, temporary in temporary_paths.items():
            handle = temporary.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=["term_id", "protein_id"])
            writer.writeheader()
            handles[family] = handle
            writers[family] = writer
        with term2gene.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {"term_id", "protein_id"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"TERM2GENE missing columns: {sorted(missing)}")
            for row in reader:
                protein_id = str(row.get("protein_id", "")).strip()
                family = protein_to_family.get(protein_id)
                if family is not None:
                    writers[family].writerow(
                        {
                            "term_id": str(row.get("term_id", "")),
                            "protein_id": protein_id,
                        }
                    )
    except BaseException:
        for handle in handles.values():
            handle.close()
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()
        for family, path in paths.items():
            temporary_paths[family].replace(path)
    return paths


def _load_inventory(
    *,
    motif_sets: Path,
    output_dir: Path,
    clusters_per_batch: int,
    batch_strategy: str,
) -> tuple[BatchSpec, ...] | None:
    manifest_path = Path(output_dir) / "batch_inventory.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = _file_identity(motif_sets)
    except (OSError, ValueError, TypeError):
        return None
    if (
        payload.get("source") != identity["path"]
        or payload.get("source_size") != identity["size"]
        or payload.get("source_mtime_ns") != identity["mtime_ns"]
        or payload.get("clusters_per_batch") != clusters_per_batch
        or payload.get("batch_strategy") != batch_strategy
    ):
        return None
    try:
        inventory = tuple(
            BatchSpec(
                family=str(item["family"]),
                method=str(item["method"]),
                index=int(item["index"]),
                cluster_count=int(item["cluster_count"]),
                row_count=int(item["row_count"]),
                path=Path(item["path"]),
            )
            for item in payload.get("batches", [])
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not inventory or not all(item.path.is_file() for item in inventory):
        return None
    return inventory


def _family_term_manifest_payload(
    sequence_index: Path,
    term2gene: Path,
) -> dict[str, object]:
    return {
        "sequence_index": _file_identity(sequence_index),
        "term2gene": _file_identity(term2gene),
    }


def ensure_family_term2gene(
    *,
    sequence_index: Path,
    term2gene: Path,
    output_dir: Path,
    resume: bool,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "family_term2gene_manifest.json"
    expected = _family_term_manifest_payload(sequence_index, term2gene)
    if resume and manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = {
                family: Path(path)
                for family, path in payload.get("families", {}).items()
            }
        except (OSError, ValueError, TypeError):
            paths = {}
            payload = {}
        if all(payload.get(key) == value for key, value in expected.items()) and paths:
            if all(path.is_file() for path in paths.values()):
                return paths
    paths = split_term2gene_by_family(
        sequence_index=sequence_index,
        term2gene=term2gene,
        output_dir=output_dir,
    )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            **expected,
            "families": {key: str(value.resolve()) for key, value in paths.items()},
        },
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motif-sets", required=True, type=Path)
    parser.add_argument("--r-script", required=True, type=Path)
    parser.add_argument("--sequence-index", required=True, type=Path)
    parser.add_argument("--term2gene", required=True, type=Path)
    parser.add_argument("--term2name", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--family-term-dir", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clusters-per-batch", type=int, default=10)
    parser.add_argument(
        "--batch-strategy",
        choices=BATCH_STRATEGIES,
        default="uniform",
    )
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.clusters_per_batch < 1:
        raise ValueError("--clusters-per-batch must be at least 1")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    family_term_dir = args.family_term_dir or (work_dir / "family_term2gene")
    family_terms = ensure_family_term2gene(
        sequence_index=args.sequence_index,
        term2gene=args.term2gene,
        output_dir=family_term_dir,
        resume=args.resume,
    )
    batch_dir = work_dir / "cluster_batches"
    inventory = None
    if args.resume:
        inventory = _load_inventory(
            motif_sets=args.motif_sets,
            output_dir=batch_dir,
            clusters_per_batch=args.clusters_per_batch,
            batch_strategy=args.batch_strategy,
        )
    if inventory is None:
        inventory = split_motif_sets(
            motif_sets=args.motif_sets,
            output_dir=batch_dir,
            clusters_per_batch=args.clusters_per_batch,
            batch_strategy=args.batch_strategy,
        )
    results_dir = work_dir / "batch_results"
    run_batches(
        inventory=inventory,
        rscript=args.rscript,
        r_script=args.r_script,
        family_term2gene=family_terms,
        term2name=args.term2name,
        output_dir=results_dir,
        progress_path=work_dir / "batch_progress.json",
        resume=args.resume,
    )
    rows = merge_batch_results(
        inventory=inventory,
        results_dir=results_dir,
        output=args.output,
    )
    print(f"Merged {rows:,} enrichment rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
