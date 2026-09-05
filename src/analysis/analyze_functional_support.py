#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_RESULTS = PROJECT_ROOT / "work" / "results"
DEFAULT_METHODS = ("M2", "M3", "MEME", "GLAM2", "STREME", "MotifAE")
DEFAULT_PRIMARY_METHODS = ("M2", "M3")
DEFAULT_TOP_KS = (10, 20, 50)
MOTIF_SET_REQUIRED_COLUMNS = {"family", "method", "motif_cluster", "protein_id"}
ENRICHMENT_REQUIRED_COLUMNS = {"family", "method", "motif_cluster", "p.adjust"}
TERM2GENE_REQUIRED_COLUMNS = {"term_id", "protein_id"}
RANK_RE = re.compile(r"(\d+)(?!.*\d)")
MOTIF_SET_CHUNKSIZE = 1_000_000


@dataclass
class MotifAccumulator:
    protein_count: int = 0
    annotated_protein_count: int = 0
    first_order: int = 0
    source_file: str = ""


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{number:.2%}"


def _parse_rank(value: object) -> int | None:
    match = RANK_RE.search(str(value))
    if not match:
        return None
    return int(match.group(1))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _validate_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _normalise_strings(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


def load_motif_sets(
    results_root: Path,
    primary_motif_sets: Path,
    methods: Iterable[str],
    primary_methods: Iterable[str],
) -> pd.DataFrame:
    methods = list(methods)
    primary_methods = set(primary_methods)
    frames: list[pd.DataFrame] = []
    for method in methods:
        if method in primary_methods:
            path = primary_motif_sets
        else:
            path = results_root / method / "clusterprofiler" / "motif_sets_long.csv"
        if not path.exists():
            raise FileNotFoundError(f"Motif set file not found for {method}: {path}")
        frame = _read_csv(path)
        _validate_columns(frame, MOTIF_SET_REQUIRED_COLUMNS, path)
        frame = _normalise_strings(frame, MOTIF_SET_REQUIRED_COLUMNS)
        frame = frame[frame["method"].eq(method)].copy()
        if "set_type" in frame.columns:
            frame = frame[frame["set_type"].astype(str).str.lower().eq("foreground")].copy()
        frame["_source_file"] = str(path)
        frame["_row_order"] = range(len(frame))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "family",
                "method",
                "motif_cluster",
                "proteins",
                "protein_count",
                "motif_rank",
                "source_file",
            ]
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["family", "method", "motif_cluster", "protein_id"])
    combined = combined.drop_duplicates(["family", "method", "motif_cluster", "protein_id"])
    grouped = (
        combined.groupby(["family", "method", "motif_cluster"], sort=False)
        .agg(
            proteins=("protein_id", lambda values: tuple(sorted(set(map(str, values))))),
            protein_count=("protein_id", "nunique"),
            first_order=("_row_order", "min"),
            source_file=("_source_file", "first"),
        )
        .reset_index()
    )
    grouped["parsed_rank"] = grouped["motif_cluster"].map(_parse_rank)
    grouped["rank_sort"] = grouped["parsed_rank"].fillna(grouped["first_order"] + 1)
    grouped = grouped.sort_values(
        ["family", "method", "rank_sort", "first_order", "motif_cluster"],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped["motif_rank"] = grouped.groupby(["family", "method"]).cumcount() + 1
    return grouped[
        [
            "family",
            "method",
            "motif_cluster",
            "proteins",
            "protein_count",
            "motif_rank",
            "source_file",
        ]
    ]


def _motif_set_paths_by_file(
    results_root: Path,
    primary_motif_sets: Path,
    methods: Iterable[str],
    primary_methods: Iterable[str],
) -> dict[Path, list[str]]:
    primary_methods = set(primary_methods)
    path_to_methods: dict[Path, list[str]] = defaultdict(list)
    for method in methods:
        if method in primary_methods:
            path = primary_motif_sets
        else:
            path = results_root / method / "clusterprofiler" / "motif_sets_long.csv"
        if not path.exists():
            raise FileNotFoundError(f"Motif set file not found for {method}: {path}")
        path_to_methods[path].append(method)
    return dict(path_to_methods)


def _motif_set_usecols(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    _validate_columns(header, MOTIF_SET_REQUIRED_COLUMNS, path)
    usecols = ["family", "method", "motif_cluster", "protein_id"]
    if "set_type" in header.columns:
        usecols.append("set_type")
    return usecols


def load_motif_level_from_sets(
    *,
    results_root: Path,
    primary_motif_sets: Path,
    methods: Iterable[str],
    primary_methods: Iterable[str],
    family_term2gene: dict[str, pd.DataFrame],
    supported_keys: set[tuple[str, str, str]],
    row_keys: set[tuple[str, str, str]],
    chunksize: int = MOTIF_SET_CHUNKSIZE,
) -> pd.DataFrame:
    annotated_by_family = {
        family: set(frame["protein_id"].astype(str))
        for family, frame in family_term2gene.items()
    }
    path_to_methods = _motif_set_paths_by_file(
        results_root,
        primary_motif_sets,
        methods,
        primary_methods,
    )
    accumulators: dict[tuple[str, str, str], MotifAccumulator] = {}
    first_order_counter = 0
    for path, target_methods in path_to_methods.items():
        target_methods = set(target_methods)
        usecols = _motif_set_usecols(path)
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            chunksize=chunksize,
            dtype=str,
        ):
            chunk = chunk.dropna(subset=["family", "method", "motif_cluster", "protein_id"])
            if chunk.empty:
                continue
            chunk = chunk[chunk["method"].isin(target_methods)].copy()
            if chunk.empty:
                continue
            if "set_type" in chunk.columns:
                chunk = chunk[chunk["set_type"].str.lower().eq("foreground")].copy()
                if chunk.empty:
                    continue
            chunk["_one"] = 1
            counts = (
                chunk.groupby(["family", "method", "motif_cluster"], sort=False)["_one"]
                .sum()
                .reset_index(name="protein_count")
            )
            for row in counts.itertuples(index=False):
                key = (str(row.family), str(row.method), str(row.motif_cluster))
                accumulator = accumulators.get(key)
                if accumulator is None:
                    accumulator = MotifAccumulator(
                        first_order=first_order_counter,
                        source_file=str(path),
                    )
                    accumulators[key] = accumulator
                    first_order_counter += 1
                accumulator.protein_count += int(row.protein_count)
            for family, family_chunk in chunk.groupby("family", sort=False):
                annotated_set = annotated_by_family.get(str(family))
                if not annotated_set:
                    continue
                annotated_chunk = family_chunk[
                    family_chunk["protein_id"].isin(annotated_set)
                ]
                if annotated_chunk.empty:
                    continue
                annotated_counts = (
                    annotated_chunk.groupby(
                        ["family", "method", "motif_cluster"],
                        sort=False,
                    )["_one"]
                    .sum()
                    .reset_index(name="annotated_protein_count")
                )
                for row in annotated_counts.itertuples(index=False):
                    key = (str(row.family), str(row.method), str(row.motif_cluster))
                    accumulator = accumulators.get(key)
                    if accumulator is None:
                        accumulator = MotifAccumulator(
                            first_order=first_order_counter,
                            source_file=str(path),
                        )
                        accumulators[key] = accumulator
                        first_order_counter += 1
                    accumulator.annotated_protein_count += int(
                        row.annotated_protein_count
                    )
    records: list[dict[str, object]] = []
    for (family, method, motif), accumulator in accumulators.items():
        key = (family, method, motif)
        parsed_rank = _parse_rank(motif)
        records.append(
            {
                "family": family,
                "method": method,
                "motif_cluster": motif,
                "parsed_rank": parsed_rank,
                "rank_sort": parsed_rank
                if parsed_rank is not None
                else accumulator.first_order + 1,
                "first_order": accumulator.first_order,
                "protein_count": accumulator.protein_count,
                "annotated_protein_count": accumulator.annotated_protein_count,
                "has_enrichment_rows": key in row_keys,
                "is_supported": key in supported_keys,
                "source_file": accumulator.source_file,
            }
        )
    motif_level = pd.DataFrame.from_records(records)
    if motif_level.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "method",
                "motif_cluster",
                "motif_rank",
                "protein_count",
                "annotated_protein_count",
                "has_enrichment_rows",
                "is_supported",
                "source_file",
            ]
        )
    motif_level = motif_level.sort_values(
        ["family", "method", "rank_sort", "first_order", "motif_cluster"],
        kind="mergesort",
    ).reset_index(drop=True)
    motif_level["motif_rank"] = motif_level.groupby(["family", "method"]).cumcount() + 1
    return motif_level[
        [
            "family",
            "method",
            "motif_cluster",
            "motif_rank",
            "protein_count",
            "annotated_protein_count",
            "has_enrichment_rows",
            "is_supported",
            "source_file",
        ]
    ]


def load_primary_motif_level_from_exp3(
    *,
    primary_exp3_root: Path,
    primary_methods: Iterable[str],
    family_term2gene: dict[str, pd.DataFrame],
    supported_keys: set[tuple[str, str, str]],
    row_keys: set[tuple[str, str, str]],
) -> tuple[pd.DataFrame, set[str]]:
    if not primary_exp3_root or not primary_exp3_root.exists():
        return pd.DataFrame(), set()
    annotated_by_family = {
        family: set(frame["protein_id"].astype(str))
        for family, frame in family_term2gene.items()
    }
    records: list[dict[str, object]] = []
    loaded_methods: set[str] = set()
    for method in primary_methods:
        occurrence_files = sorted(
            primary_exp3_root.glob(f"*/motif_occurrences_{method}.csv")
        )
        if not occurrence_files:
            continue
        loaded_methods.add(method)
        rank_map: dict[tuple[str, str, str], int] = {}
        for catalog_path in sorted(primary_exp3_root.glob(f"*/motif_catalog_{method}.csv")):
            catalog = pd.read_csv(catalog_path, dtype=str)
            required = {"family", "method", "motif_uid", "rank_within_method"}
            if not required.issubset(catalog.columns):
                continue
            catalog["rank_within_method"] = pd.to_numeric(
                catalog["rank_within_method"],
                errors="coerce",
            )
            catalog = catalog.dropna(
                subset=["family", "method", "motif_uid", "rank_within_method"]
            )
            for row in catalog.itertuples(index=False):
                rank_map[
                    (str(row.family), str(row.method), str(row.motif_uid))
                ] = int(row.rank_within_method)
        method_frames: list[pd.DataFrame] = []
        for occurrence_path in occurrence_files:
            header = pd.read_csv(occurrence_path, nrows=0)
            required = {"motif_uid", "family", "method", "parent_id"}
            if not required.issubset(header.columns):
                continue
            frame = pd.read_csv(
                occurrence_path,
                usecols=["motif_uid", "family", "method", "parent_id"],
                dtype=str,
            )
            frame = frame.dropna(subset=["motif_uid", "family", "method", "parent_id"])
            frame = frame[frame["method"].eq(method)].copy()
            if frame.empty:
                continue
            frame = frame.rename(
                columns={"motif_uid": "motif_cluster", "parent_id": "protein_id"}
            )
            frame = frame.drop_duplicates(
                ["family", "method", "motif_cluster", "protein_id"]
            )
            method_frames.append(frame)
        if not method_frames:
            loaded_methods.discard(method)
            continue
        combined = pd.concat(method_frames, ignore_index=True)
        combined["_is_annotated"] = False
        for family, family_frame in combined.groupby("family", sort=False):
            annotated_set = annotated_by_family.get(str(family), set())
            if not annotated_set:
                continue
            combined.loc[family_frame.index, "_is_annotated"] = family_frame[
                "protein_id"
            ].isin(annotated_set)
        grouped = (
            combined.groupby(["family", "method", "motif_cluster"], sort=False)
            .agg(
                protein_count=("protein_id", "nunique"),
                annotated_protein_count=("_is_annotated", "sum"),
            )
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            family = str(row.family)
            motif_method = str(row.method)
            motif = str(row.motif_cluster)
            key = (family, motif_method, motif)
            records.append(
                {
                    "family": family,
                    "method": motif_method,
                    "motif_cluster": motif,
                    "motif_rank": rank_map.get(key) or _parse_rank(motif) or 0,
                    "protein_count": int(row.protein_count),
                    "annotated_protein_count": int(row.annotated_protein_count),
                    "has_enrichment_rows": key in row_keys,
                    "is_supported": key in supported_keys,
                    "source_file": str(primary_exp3_root),
                }
            )
    motif_level = pd.DataFrame.from_records(records)
    if motif_level.empty:
        return motif_level, loaded_methods
    motif_level = motif_level.sort_values(
        ["family", "method", "motif_rank", "motif_cluster"],
        kind="mergesort",
    ).reset_index(drop=True)
    missing_rank = motif_level["motif_rank"].eq(0)
    if missing_rank.any():
        motif_level.loc[missing_rank, "motif_rank"] = (
            motif_level[missing_rank].groupby(["family", "method"]).cumcount() + 1
        )
    return motif_level, loaded_methods


def _enrichment_files_for_method(
    results_root: Path,
    method: str,
    primary_methods: set[str],
) -> list[Path]:
    if method in primary_methods:
        batch_dir = results_root / "clusterprofiler" / "batch_results"
    else:
        batch_dir = results_root / method / "clusterprofiler" / "batch_results"
    if not batch_dir.exists():
        return []
    return sorted(
        path
        for path in batch_dir.glob("*.csv")
        if path.is_file() and not path.name.startswith("motif_sets_")
    )


def load_enrichment_results(
    results_root: Path,
    methods: Iterable[str],
    primary_methods: Iterable[str],
) -> pd.DataFrame:
    primary_methods = set(primary_methods)
    frames: list[pd.DataFrame] = []
    for method in methods:
        for path in _enrichment_files_for_method(results_root, method, primary_methods):
            frame = _read_csv(path)
            if frame.empty:
                continue
            if not ENRICHMENT_REQUIRED_COLUMNS.issubset(frame.columns):
                continue
            frame = _normalise_strings(frame, ["family", "method", "motif_cluster"])
            frame = frame[frame["method"].eq(method)].copy()
            if frame.empty:
                continue
            frame["p.adjust"] = pd.to_numeric(frame["p.adjust"], errors="coerce")
            frame = frame.dropna(subset=["family", "method", "motif_cluster", "p.adjust"])
            frame["_source_file"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["family", "method", "motif_cluster", "p.adjust"])
    return pd.concat(frames, ignore_index=True)


def supported_motif_keys(enrichment: pd.DataFrame, alpha: float) -> set[tuple[str, str, str]]:
    if enrichment.empty:
        return set()
    significant = enrichment[enrichment["p.adjust"].le(alpha)]
    return set(
        zip(
            significant["family"].astype(str),
            significant["method"].astype(str),
            significant["motif_cluster"].astype(str),
        )
    )


def motif_keys_with_rows(enrichment: pd.DataFrame) -> set[tuple[str, str, str]]:
    if enrichment.empty:
        return set()
    return set(
        zip(
            enrichment["family"].astype(str),
            enrichment["method"].astype(str),
            enrichment["motif_cluster"].astype(str),
        )
    )


def load_family_term2gene(results_root: Path) -> dict[str, pd.DataFrame]:
    term_dir = results_root / "clusterprofiler" / "family_term2gene"
    if not term_dir.exists():
        raise FileNotFoundError(f"Family term2gene directory not found: {term_dir}")
    family_frames: dict[str, pd.DataFrame] = {}
    for path in sorted(term_dir.glob("term2gene_*.csv")):
        family = path.stem.replace("term2gene_", "", 1)
        frame = _read_csv(path)
        _validate_columns(frame, TERM2GENE_REQUIRED_COLUMNS, path)
        frame = _normalise_strings(frame, TERM2GENE_REQUIRED_COLUMNS)
        frame = frame.dropna(subset=["term_id", "protein_id"])
        frame = frame.drop_duplicates(["term_id", "protein_id"])
        family_frames[family] = frame
    if not family_frames:
        raise FileNotFoundError(f"No family term2gene CSV files found in {term_dir}")
    return family_frames


def build_motif_level_table(
    motif_sets: pd.DataFrame,
    family_term2gene: dict[str, pd.DataFrame],
    supported_keys: set[tuple[str, str, str]],
    row_keys: set[tuple[str, str, str]],
) -> pd.DataFrame:
    annotated_proteins_by_family = {
        family: set(frame["protein_id"].astype(str))
        for family, frame in family_term2gene.items()
    }
    records: list[dict[str, object]] = []
    for row in motif_sets.itertuples(index=False):
        family = str(row.family)
        method = str(row.method)
        motif = str(row.motif_cluster)
        proteins = tuple(row.proteins)
        annotated = annotated_proteins_by_family.get(family, set()).intersection(proteins)
        key = (family, method, motif)
        records.append(
            {
                "family": family,
                "method": method,
                "motif_cluster": motif,
                "motif_rank": int(row.motif_rank),
                "protein_count": int(row.protein_count),
                "annotated_protein_count": len(annotated),
                "has_enrichment_rows": key in row_keys,
                "is_supported": key in supported_keys,
                "source_file": row.source_file,
            }
        )
    return pd.DataFrame.from_records(records)


def build_topk_tables(
    motif_level: pd.DataFrame,
    top_ks: Iterable[int],
    primary_methods: Iterable[str],
) -> dict[str, pd.DataFrame]:
    top_ks = [int(k) for k in top_ks]
    family_records: list[dict[str, object]] = []
    for (family, method), group in motif_level.groupby(["family", "method"], sort=True):
        group = group.sort_values("motif_rank")
        available = int(len(group))
        for k in top_ks:
            selected = group.head(k)
            selected_count = int(len(selected))
            supported = int(selected["is_supported"].sum())
            testable_selected = int(selected["is_testable"].sum()) if "is_testable" in selected else 0
            supported_testable = int(
                (selected["is_testable"] & selected["is_supported"]).sum()
            ) if "is_testable" in selected else 0
            family_records.append(
                {
                    "family": family,
                    "method": method,
                    "k": k,
                    "available_motifs": available,
                    "selected_motifs": selected_count,
                    "eligible_full_k": available >= k,
                    "supported_selected_motifs": supported,
                    "support_rate": _safe_divide(supported, selected_count),
                    "selected_testable_motifs": testable_selected,
                    "supported_selected_testable_motifs": supported_testable,
                    "selected_testable_support_rate": _safe_divide(
                        supported_testable, testable_selected
                    ),
                    "median_selected_protein_count": float(selected["protein_count"].median())
                    if selected_count
                    else float("nan"),
                    "median_selected_annotated_protein_count": float(
                        selected["annotated_protein_count"].median()
                    )
                    if selected_count
                    else float("nan"),
                }
            )
    family_summary = pd.DataFrame.from_records(family_records)
    if family_summary.empty:
        method_summary = pd.DataFrame()
        pairwise = pd.DataFrame()
    else:
        method_summary = (
            family_summary.groupby(["method", "k"], as_index=False)
            .agg(
                families=("family", "nunique"),
                families_with_full_k=("eligible_full_k", "sum"),
                selected_motifs=("selected_motifs", "sum"),
                supported_selected_motifs=("supported_selected_motifs", "sum"),
                selected_testable_motifs=("selected_testable_motifs", "sum"),
                supported_selected_testable_motifs=(
                    "supported_selected_testable_motifs",
                    "sum",
                ),
                median_family_support_rate=("support_rate", "median"),
                mean_family_support_rate=("support_rate", "mean"),
            )
        )
        method_summary["support_rate"] = method_summary.apply(
            lambda row: _safe_divide(
                row["supported_selected_motifs"], row["selected_motifs"]
            ),
            axis=1,
        )
        method_summary["selected_testable_support_rate"] = method_summary.apply(
            lambda row: _safe_divide(
                row["supported_selected_testable_motifs"],
                row["selected_testable_motifs"],
            ),
            axis=1,
        )
        pairwise = build_topk_primary_vs_baseline(family_summary, primary_methods)
    return {
        "topk_family_method_summary": family_summary,
        "topk_method_summary": method_summary,
        "topk_primary_vs_baseline": pairwise,
    }


def build_topk_primary_vs_baseline(
    family_summary: pd.DataFrame,
    primary_methods: Iterable[str],
) -> pd.DataFrame:
    primary_methods = list(primary_methods)
    methods = sorted(set(family_summary["method"]))
    records: list[dict[str, object]] = []
    for k in sorted(family_summary["k"].unique()):
        k_frame = family_summary[family_summary["k"].eq(k)]
        for primary in primary_methods:
            if primary not in methods:
                continue
            primary_frame = k_frame[k_frame["method"].eq(primary)]
            for control in methods:
                if control == primary:
                    continue
                control_frame = k_frame[k_frame["method"].eq(control)]
                merged = primary_frame.merge(
                    control_frame,
                    on="family",
                    suffixes=("_primary", "_control"),
                )
                merged = merged[
                    merged["eligible_full_k_primary"] & merged["eligible_full_k_control"]
                ].copy()
                if merged.empty:
                    records.append(
                        {
                            "k": k,
                            "primary_method": primary,
                            "control_method": control,
                            "families_compared": 0,
                            "mean_rate_difference_primary_minus_control": float("nan"),
                            "median_rate_difference_primary_minus_control": float("nan"),
                            "families_primary_higher": 0,
                            "families_equal": 0,
                            "families_control_higher": 0,
                        }
                    )
                    continue
                diff = merged["support_rate_primary"] - merged["support_rate_control"]
                records.append(
                    {
                        "k": k,
                        "primary_method": primary,
                        "control_method": control,
                        "families_compared": int(len(merged)),
                        "mean_rate_difference_primary_minus_control": float(diff.mean()),
                        "median_rate_difference_primary_minus_control": float(diff.median()),
                        "families_primary_higher": int((diff > 0).sum()),
                        "families_equal": int((diff == 0).sum()),
                        "families_control_higher": int((diff < 0).sum()),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_testable_tables(
    motif_level: pd.DataFrame,
    testable_min_annotated_proteins: int,
) -> dict[str, pd.DataFrame]:
    motif_level = motif_level.copy()
    motif_level["is_testable"] = motif_level["annotated_protein_count"].ge(
        testable_min_annotated_proteins
    )
    motif_level["is_supported_testable"] = motif_level["is_testable"] & motif_level[
        "is_supported"
    ]
    family_summary = (
        motif_level.groupby(["family", "method"], as_index=False)
        .agg(
            total_motifs=("motif_cluster", "nunique"),
            testable_motifs=("is_testable", "sum"),
            supported_testable_motifs=("is_supported_testable", "sum"),
            motifs_with_enrichment_rows=("has_enrichment_rows", "sum"),
            supported_all_motifs=("is_supported", "sum"),
            median_protein_count=("protein_count", "median"),
            median_annotated_protein_count=("annotated_protein_count", "median"),
        )
    )
    family_summary["non_testable_motifs"] = (
        family_summary["total_motifs"] - family_summary["testable_motifs"]
    )
    family_summary["testable_support_rate"] = family_summary.apply(
        lambda row: _safe_divide(
            row["supported_testable_motifs"], row["testable_motifs"]
        ),
        axis=1,
    )
    family_summary["all_motif_support_rate"] = family_summary.apply(
        lambda row: _safe_divide(row["supported_all_motifs"], row["total_motifs"]),
        axis=1,
    )
    method_summary = (
        family_summary.groupby("method", as_index=False)
        .agg(
            families=("family", "nunique"),
            total_motifs=("total_motifs", "sum"),
            testable_motifs=("testable_motifs", "sum"),
            non_testable_motifs=("non_testable_motifs", "sum"),
            supported_testable_motifs=("supported_testable_motifs", "sum"),
            supported_all_motifs=("supported_all_motifs", "sum"),
            median_family_testable_support_rate=("testable_support_rate", "median"),
            mean_family_testable_support_rate=("testable_support_rate", "mean"),
        )
    )
    method_summary["testable_support_rate"] = method_summary.apply(
        lambda row: _safe_divide(
            row["supported_testable_motifs"], row["testable_motifs"]
        ),
        axis=1,
    )
    method_summary["all_motif_support_rate"] = method_summary.apply(
        lambda row: _safe_divide(row["supported_all_motifs"], row["total_motifs"]),
        axis=1,
    )
    return {
        "motif_level_fairness": motif_level,
        "testable_family_method_summary": family_summary,
        "testable_method_summary": method_summary,
    }


@lru_cache(maxsize=None)
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@lru_cache(maxsize=200_000)
def _hypergeom_sf(k: int, population: int, successes: int, draws: int) -> float:
    if k <= 0:
        return 1.0
    max_x = min(successes, draws)
    min_x = max(0, draws - (population - successes), k)
    if min_x > max_x:
        return 0.0
    denominator = _log_comb(population, draws)
    probabilities = [
        math.exp(
            _log_comb(successes, x)
            + _log_comb(population - successes, draws - x)
            - denominator
        )
        for x in range(min_x, max_x + 1)
    ]
    return min(1.0, math.fsum(probabilities))


@lru_cache(maxsize=200_000)
def _critical_overlap_bonferroni(
    population: int,
    successes: int,
    draws: int,
    term_count: int,
    alpha_scaled: int,
) -> int | None:
    alpha = alpha_scaled / 1_000_000_000
    if draws <= 0 or successes <= 0 or term_count <= 0:
        return None
    threshold = alpha / term_count
    for overlap in range(1, min(successes, draws) + 1):
        if _hypergeom_sf(overlap, population, successes, draws) <= threshold:
            return overlap
    return None


class FamilyEnrichmentUniverse:
    def __init__(self, term2gene: pd.DataFrame) -> None:
        self.universe = tuple(sorted(set(term2gene["protein_id"].astype(str))))
        self.population = len(self.universe)
        self.term_to_proteins: dict[str, set[str]] = {
            str(term): set(group["protein_id"].astype(str))
            for term, group in term2gene.groupby("term_id", sort=False)
        }
        self.term_sizes = {
            term: len(proteins) for term, proteins in self.term_to_proteins.items()
        }
        self.term_count = len(self.term_to_proteins)
        protein_to_terms: dict[str, list[str]] = defaultdict(list)
        for term, proteins in self.term_to_proteins.items():
            for protein in proteins:
                protein_to_terms[protein].append(term)
        self.protein_to_terms = dict(protein_to_terms)
        self._threshold_cache: dict[tuple[int, int], dict[str, int]] = {}

    def is_supported(self, proteins: Iterable[str], alpha: float) -> bool:
        sample = set(proteins).intersection(self.universe)
        draws = len(sample)
        if draws == 0 or self.population == 0 or self.term_count == 0:
            return False
        thresholds = self._critical_thresholds(draws, alpha)
        if not thresholds:
            return False
        counts: Counter[str] = Counter()
        for protein in sample:
            counts.update(self.protein_to_terms.get(protein, []))
        if not counts:
            return False
        for term, overlap in counts.items():
            threshold = thresholds.get(term)
            if threshold is not None and overlap >= threshold:
                return True
        return False

    def _critical_thresholds(self, draws: int, alpha: float) -> dict[str, int]:
        alpha_scaled = int(round(alpha * 1_000_000_000))
        cache_key = (draws, alpha_scaled)
        cached = self._threshold_cache.get(cache_key)
        if cached is not None:
            return cached
        thresholds: dict[str, int] = {}
        for term, successes in self.term_sizes.items():
            threshold = _critical_overlap_bonferroni(
                int(self.population),
                int(successes),
                int(draws),
                int(self.term_count),
                alpha_scaled,
            )
            if threshold is not None:
                thresholds[term] = threshold
        self._threshold_cache[cache_key] = thresholds
        return thresholds

    def sample(self, size: int, rng: random.Random) -> tuple[str, ...]:
        size = max(0, min(int(size), self.population))
        if size == 0:
            return tuple()
        return tuple(rng.sample(self.universe, size))


def build_protein_set_random_background_summary(
    motif_level: pd.DataFrame,
    family_term2gene: dict[str, pd.DataFrame],
    *,
    alpha: float,
    random_iterations: int,
    random_top_k: int,
    random_seed: int,
    testable_min_annotated_proteins: int,
) -> pd.DataFrame:
    if random_iterations <= 0 or random_top_k <= 0:
        return pd.DataFrame()
    universes = {
        family: FamilyEnrichmentUniverse(frame)
        for family, frame in family_term2gene.items()
    }
    rng = random.Random(random_seed)
    records: list[dict[str, object]] = []
    for (family, method), group in motif_level.groupby(["family", "method"], sort=True):
        universe = universes.get(family)
        if universe is None:
            continue
        selected = group.sort_values("motif_rank").head(random_top_k).copy()
        if selected.empty:
            continue
        sizes = [
            int(value)
            for value in selected["annotated_protein_count"].tolist()
            if int(value) >= testable_min_annotated_proteins
        ]
        observed_supported = int(selected["is_supported"].sum())
        observed_rate = _safe_divide(observed_supported, len(selected))
        if not sizes:
            random_rates = [0.0 for _ in range(random_iterations)]
        else:
            random_rates = []
            for _ in range(random_iterations):
                supported = 0
                for size in sizes:
                    proteins = universe.sample(size, rng)
                    if universe.is_supported(proteins, alpha):
                        supported += 1
                random_rates.append(_safe_divide(supported, len(selected)))
        random_mean = float(sum(random_rates) / len(random_rates)) if random_rates else float("nan")
        random_sorted = sorted(random_rates)
        if random_sorted:
            lower_index = int(math.floor(0.025 * (len(random_sorted) - 1)))
            upper_index = int(math.ceil(0.975 * (len(random_sorted) - 1)))
            ci_low = float(random_sorted[lower_index])
            ci_high = float(random_sorted[upper_index])
            empirical_p = (1 + sum(rate >= observed_rate for rate in random_rates)) / (
                random_iterations + 1
            )
        else:
            ci_low = ci_high = empirical_p = float("nan")
        records.append(
            {
                "family": family,
                "method": method,
                "random_top_k": random_top_k,
                "selected_motifs": int(len(selected)),
                "randomized_testable_motifs": int(len(sizes)),
                "observed_supported_motifs": observed_supported,
                "observed_supported_rate": observed_rate,
                "random_mean_supported_rate": random_mean,
                "random_ci_low": ci_low,
                "random_ci_high": ci_high,
                "observed_minus_random_mean": observed_rate - random_mean,
                "empirical_p_value_greater_equal": empirical_p,
                "random_iterations": random_iterations,
                "random_seed": random_seed,
                "random_enrichment_screen": "bonferroni_hypergeometric",
            }
        )
    return pd.DataFrame.from_records(records)


def build_motif_permutation_background_summary(
    motif_level: pd.DataFrame,
    *,
    random_iterations: int,
    random_top_k: int,
    random_seed: int,
) -> pd.DataFrame:
    if random_iterations <= 0 or random_top_k <= 0:
        return pd.DataFrame()
    rng = random.Random(random_seed)
    records: list[dict[str, object]] = []
    for (family, method), group in motif_level.groupby(["family", "method"], sort=True):
        group = group.sort_values("motif_rank").reset_index(drop=True)
        if group.empty:
            continue
        selected = group.head(random_top_k)
        selected_count = int(len(selected))
        observed_supported = int(selected["is_supported"].sum())
        observed_rate = _safe_divide(observed_supported, selected_count)
        support_labels = [bool(value) for value in group["is_supported"].tolist()]
        indices = list(range(len(support_labels)))
        random_rates: list[float] = []
        for _ in range(random_iterations):
            if len(indices) <= selected_count:
                sampled = indices
            else:
                sampled = rng.sample(indices, selected_count)
            supported = sum(1 for index in sampled if support_labels[index])
            random_rates.append(_safe_divide(supported, selected_count))
        random_mean = float(sum(random_rates) / len(random_rates)) if random_rates else float("nan")
        random_sorted = sorted(random_rates)
        if random_sorted:
            lower_index = int(math.floor(0.025 * (len(random_sorted) - 1)))
            upper_index = int(math.ceil(0.975 * (len(random_sorted) - 1)))
            ci_low = float(random_sorted[lower_index])
            ci_high = float(random_sorted[upper_index])
            empirical_p = (1 + sum(rate >= observed_rate for rate in random_rates)) / (
                random_iterations + 1
            )
        else:
            ci_low = ci_high = empirical_p = float("nan")
        records.append(
            {
                "family": family,
                "method": method,
                "random_top_k": random_top_k,
                "selected_motifs": selected_count,
                "randomized_testable_motifs": int(selected["is_testable"].sum())
                if "is_testable" in selected
                else selected_count,
                "observed_supported_motifs": observed_supported,
                "observed_supported_rate": observed_rate,
                "random_mean_supported_rate": random_mean,
                "random_ci_low": ci_low,
                "random_ci_high": ci_high,
                "observed_minus_random_mean": observed_rate - random_mean,
                "empirical_p_value_greater_equal": empirical_p,
                "random_iterations": random_iterations,
                "random_seed": random_seed,
                "random_enrichment_screen": "motif_permutation_fdr_labels",
            }
        )
    return pd.DataFrame.from_records(records)


def build_fair_comparison_tables(
    *,
    results_root: Path,
    primary_motif_sets: Path,
    primary_exp3_root: Path | None = None,
    methods: Iterable[str] = DEFAULT_METHODS,
    primary_methods: Iterable[str] = DEFAULT_PRIMARY_METHODS,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    alpha: float = 0.05,
    testable_min_annotated_proteins: int = 3,
    random_iterations: int = 1000,
    random_top_k: int = 10,
    random_seed: int = 20260714,
) -> dict[str, pd.DataFrame]:
    methods = list(methods)
    primary_methods = list(primary_methods)
    enrichment = load_enrichment_results(results_root, methods, primary_methods)
    family_term2gene = load_family_term2gene(results_root)
    supported = supported_motif_keys(enrichment, alpha)
    with_rows = motif_keys_with_rows(enrichment)
    primary_motif_level, loaded_primary_methods = load_primary_motif_level_from_exp3(
        primary_exp3_root=primary_exp3_root,
        primary_methods=primary_methods,
        family_term2gene=family_term2gene,
        supported_keys=supported,
        row_keys=with_rows,
    ) if primary_exp3_root is not None else (pd.DataFrame(), set())
    motif_set_methods = [
        method
        for method in methods
        if method not in loaded_primary_methods
    ]
    motif_set_level = load_motif_level_from_sets(
        results_root=results_root,
        primary_motif_sets=primary_motif_sets,
        methods=motif_set_methods,
        primary_methods=primary_methods,
        family_term2gene=family_term2gene,
        supported_keys=supported,
        row_keys=with_rows,
    ) if motif_set_methods else pd.DataFrame()
    motif_level = pd.concat(
        [frame for frame in [primary_motif_level, motif_set_level] if not frame.empty],
        ignore_index=True,
    )
    testable_tables = build_testable_tables(
        motif_level,
        testable_min_annotated_proteins=testable_min_annotated_proteins,
    )
    motif_level = testable_tables["motif_level_fairness"]
    topk_tables = build_topk_tables(motif_level, top_ks, primary_methods)
    random_summary = build_motif_permutation_background_summary(
        motif_level,
        random_iterations=random_iterations,
        random_top_k=random_top_k,
        random_seed=random_seed,
    )
    tables = {
        **testable_tables,
        **topk_tables,
        "random_background_summary": random_summary,
    }
    return tables


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    available = [column for column in columns if column in frame.columns]
    rows = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join(["---"] * len(available)) + " |",
    ]
    for _, row in frame[available].iterrows():
        values = []
        for column in available:
            value = row[column]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("NA")
                elif column.endswith("rate") or "rate_" in column:
                    values.append(_format_percent(value))
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _write_report(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    alpha: float,
    top_ks: Iterable[int],
    testable_min_annotated_proteins: int,
    random_iterations: int,
    random_top_k: int,
) -> None:
    topk = tables["topk_method_summary"].copy()
    if not topk.empty:
        topk["support_rate_fmt"] = topk["support_rate"].map(_format_percent)
        topk["testable_rate_fmt"] = topk["selected_testable_support_rate"].map(_format_percent)
    testable = tables["testable_method_summary"].copy()
    if not testable.empty:
        testable["testable_rate_fmt"] = testable["testable_support_rate"].map(_format_percent)
        testable["all_rate_fmt"] = testable["all_motif_support_rate"].map(_format_percent)
    random_summary = tables["random_background_summary"].copy()
    random_method = pd.DataFrame()
    if not random_summary.empty:
        random_method = (
            random_summary.groupby("method", as_index=False)
            .agg(
                families=("family", "nunique"),
                selected_motifs=("selected_motifs", "sum"),
                observed_supported_motifs=("observed_supported_motifs", "sum"),
                observed_supported_rate=("observed_supported_rate", "mean"),
                random_mean_supported_rate=("random_mean_supported_rate", "mean"),
                observed_minus_random_mean=("observed_minus_random_mean", "mean"),
                median_empirical_p_value=(
                    "empirical_p_value_greater_equal",
                    "median",
                ),
            )
        )
        random_method["observed_rate_fmt"] = random_method[
            "observed_supported_rate"
        ].map(_format_percent)
        random_method["random_rate_fmt"] = random_method[
            "random_mean_supported_rate"
        ].map(_format_percent)
        random_method["delta_fmt"] = random_method["observed_minus_random_mean"].map(
            _format_percent
        )
    lines = [
        "# Fair Exp5 Functional Support Comparison",
        "",
        f"Significance threshold: FDR-adjusted p <= {alpha:g}.",
        f"Top-K settings: {', '.join(map(str, top_ks))}.",
        f"Testable motif definition: at least {testable_min_annotated_proteins} motif proteins present in the filtered family-specific term2gene background.",
        f"Random background: Top-{random_top_k} motif permutation within each family-method, {random_iterations} iterations, using the observed FDR-supported motif labels.",
        "",
        "## Top-K method summary",
        "",
        _markdown_table(
            topk,
            [
                "method",
                "k",
                "families",
                "families_with_full_k",
                "selected_motifs",
                "supported_selected_motifs",
                "support_rate_fmt",
                "selected_testable_motifs",
                "supported_selected_testable_motifs",
                "testable_rate_fmt",
            ],
        ),
        "",
        "## Testable-denominator method summary",
        "",
        _markdown_table(
            testable,
            [
                "method",
                "families",
                "total_motifs",
                "testable_motifs",
                "supported_testable_motifs",
                "testable_rate_fmt",
                "supported_all_motifs",
                "all_rate_fmt",
            ],
        ),
        "",
        "## Random-background method summary",
        "",
        _markdown_table(
            random_method,
            [
                "method",
                "families",
                "selected_motifs",
                "observed_supported_motifs",
                "observed_rate_fmt",
                "random_rate_fmt",
                "delta_fmt",
                "median_empirical_p_value",
            ],
        ),
        "",
        "## Interpretation guardrails",
        "",
        "- The all-motif support rate is conservative but penalizes methods that deliberately discover many fine-grained motifs.",
        "- Top-K is the fairest cross-method comparison because each family contributes the same number of submitted motifs per method when `eligible_full_k` is true.",
        "- The testable-denominator analysis asks whether motifs with enough filtered annotation support are enriched, reducing annotation-missingness bias.",
        "- The random-background analysis checks whether the ranked Top-K motifs are more functionally supported than random motifs drawn from the same family and method; it reuses the observed FDR-supported motif labels and therefore avoids rerunning enrichment on synthetic protein sets.",
        "- These analyses evaluate functional-support evidence only; they should be interpreted together with motif quality, structural consistency, and biological plausibility.",
    ]
    (output_dir / "fair_comparison_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_fair_comparison_outputs(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    alpha: float,
    top_ks: Iterable[int],
    testable_min_annotated_proteins: int,
    random_iterations: int,
    random_top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    _write_report(
        tables,
        output_dir,
        alpha=alpha,
        top_ks=top_ks,
        testable_min_annotated_proteins=testable_min_annotated_proteins,
        random_iterations=random_iterations,
        random_top_k=random_top_k,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fair functional-support analyses for filtered Exp5 enrichment results: "
            "Top-K, testable-denominator, and size-matched random background."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=WORK_RESULTS / "exp5_external_functional_enrichment",
    )
    parser.add_argument(
        "--primary-motif-sets",
        type=Path,
        default=WORK_RESULTS
        / "exp5_external_functional_enrichment"
        / "clusterprofiler"
        / "motif_sets_long.csv",
    )
    parser.add_argument(
        "--primary-exp3-root",
        type=Path,
        default=WORK_RESULTS / "exp3_domain_discovery",
        help=(
            "Preferred source for M2/M3 motif protein membership. "
            "When present, motif_occurrences_<method>.csv and motif_catalog_<method>.csv "
            "are used instead of scanning the large primary motif_sets_long.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORK_RESULTS
        / "exp5_external_functional_enrichment"
        / "fair_comparison",
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--primary-methods", nargs="+", default=list(DEFAULT_PRIMARY_METHODS))
    parser.add_argument("--top-ks", nargs="+", type=int, default=list(DEFAULT_TOP_KS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--testable-min-annotated-proteins", type=int, default=3)
    parser.add_argument("--random-iterations", type=int, default=1000)
    parser.add_argument("--random-top-k", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260714)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tables = build_fair_comparison_tables(
        results_root=args.results_root,
        primary_motif_sets=args.primary_motif_sets,
        primary_exp3_root=args.primary_exp3_root,
        methods=args.methods,
        primary_methods=args.primary_methods,
        top_ks=args.top_ks,
        alpha=args.alpha,
        testable_min_annotated_proteins=args.testable_min_annotated_proteins,
        random_iterations=args.random_iterations,
        random_top_k=args.random_top_k,
        random_seed=args.random_seed,
    )
    write_fair_comparison_outputs(
        tables,
        args.output_dir,
        alpha=args.alpha,
        top_ks=args.top_ks,
        testable_min_annotated_proteins=args.testable_min_annotated_proteins,
        random_iterations=args.random_iterations,
        random_top_k=args.random_top_k,
    )
    print(f"Wrote fair comparison outputs to {args.output_dir}")
    print(tables["topk_method_summary"].to_string(index=False))
    print(tables["testable_method_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
