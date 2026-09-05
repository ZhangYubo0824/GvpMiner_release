#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


EXP5_ROOT = Path(__file__).resolve().parents[1]
if str(EXP5_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP5_ROOT))

from package import exploratory_analysis as exploratory
from package.reproducibility import file_record, sha256_file


OUTPUT_NAMES = (
    "method_yield_summary.csv",
    "topk_sensitivity.csv",
    "rank_decile_summary.csv",
    "rank_model_summary.csv",
    "global_random_test.csv",
    "evidence_chain_inventory.csv",
    "analysis_inclusion_decisions.csv",
    "exploratory_analysis_report.md",
)
KNOWN_OUTPUT_NAMES = frozenset((*OUTPUT_NAMES, "analysis_manifest.json", "figures"))
FIGURE_STEMS = (
    "provisional_figure6_precision_yield",
    "provisional_figureS3_topk_sensitivity",
    "provisional_figureS4_rank_trend",
)
KNOWN_FIGURE_OUTPUT_NAMES = frozenset(
    [
        f"{stem}.{extension}"
        for stem in FIGURE_STEMS
        for extension in ("png", "pdf", "svg")
    ]
    + [f"{stem}_source_data.csv" for stem in FIGURE_STEMS]
    + ["figure_manifest.json"]
)
PUBLICATION_FAMILIES = frozenset(
    (
        "GvpA",
        "GvpC",
        "GvpD",
        "GvpF",
        "GvpG",
        "GvpH",
        "GvpJ",
        "GvpK",
        "GvpL",
        "GvpM",
        "GvpN",
        "GvpO",
        "GvpQ",
        "GvpT",
        "GvpU",
        "GvpV",
    )
)
PUBLICATION_RANDOM_ITERATIONS = 1_000
PUBLICATION_RANDOM_SEED = 20_260_714
PUBLICATION_SAMPLING_CONTRACT = (
    "without replacement; matched on floor(log2(protein_count)) and five "
    "annotation-coverage bins"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run downstream Exp5 candidate-yield, rank-sensitivity and matched-"
            "background analyses without modifying corrected enrichment inputs."
        )
    )
    parser.add_argument("--motif-level-csv", required=True, type=Path)
    parser.add_argument("--existing-random-summary-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k-values", default="1,3,5,10,20,50")
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--global-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_812)
    parser.add_argument("--global-seed", type=int, default=20_260_813)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known generated analysis/figure artifacts; reject unknown files.",
    )
    parser.add_argument(
        "--allow-nonpublication-audit-inventory",
        action="store_true",
        help=(
            "Permit a self-consistent existing random audit that is not the exact "
            "16-family x 6-method publication inventory; intended only for tests "
            "or separate datasets."
        ),
    )
    return parser


def _parse_k_values(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("--k-values must be comma-separated positive integers") from error
    if not values or any(item < 1 for item in values) or len(set(values)) != len(values):
        raise ValueError("--k-values must be unique positive integers")
    return values


def _corrected_root(path: Path) -> Path | None:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "exp5_external_functional_enrichment_corrected":
            return candidate
    return None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_existing_output(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    existing_names = {path.name for path in output_dir.iterdir()}
    if existing_names and not overwrite:
        raise ValueError(f"output directory is non-empty: {output_dir}")
    unknown = sorted(existing_names - KNOWN_OUTPUT_NAMES)
    if unknown:
        raise ValueError(f"output directory contains unknown files: {unknown}")
    for child in output_dir.iterdir():
        if child.name == "figures":
            if child.is_symlink() or not child.is_dir():
                raise ValueError("output directory contains an invalid figures entry")
            unknown_figures = sorted(
                nested.name
                for nested in child.iterdir()
                if nested.name not in KNOWN_FIGURE_OUTPUT_NAMES
                or nested.is_symlink()
                or not nested.is_file()
            )
            if unknown_figures:
                raise ValueError(
                    f"output directory contains unknown figure files: {unknown_figures}"
                )
        elif child.is_symlink() or not child.is_file():
            raise ValueError(f"output directory contains invalid generated file: {child}")


def _validate_paths(
    motif_path: Path, random_path: Path, output_dir: Path, *, overwrite: bool
) -> None:
    for path in (motif_path, random_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = output_dir.resolve()
    for input_path in (motif_path, random_path):
        corrected = _corrected_root(input_path)
        if corrected is not None and _is_within(output, corrected):
            raise ValueError(
                "output directory cannot be inside the corrected Exp5 input tree"
            )
    _validate_existing_output(output, overwrite=overwrite)


def _audit_existing_random(
    path: Path,
    motif_rows: pd.DataFrame,
    *,
    enforce_publication_contract: bool = True,
) -> dict[str, object]:
    rows = pd.read_csv(path)
    required = {
        "family",
        "method",
        "top_k_requested",
        "observed_selected_count",
        "observed_testable_count",
        "observed_supported_count",
        "empirical_p_ge_observed",
        "random_iterations",
        "global_seed",
        "sampling_contract",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"existing random summary missing columns: {missing}")
    p_values = pd.to_numeric(rows["empirical_p_ge_observed"], errors="coerce")
    if p_values.isna().any() or ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("existing random summary contains invalid empirical p values")
    keys = ["family", "method"]
    if rows[keys].isna().any().any():
        raise ValueError("existing random summary contains missing family-method keys")
    if rows.duplicated(keys).any():
        raise ValueError("existing random summary contains duplicate family-method rows")

    integer_columns = [
        "top_k_requested",
        "observed_selected_count",
        "observed_testable_count",
        "observed_supported_count",
        "random_iterations",
        "global_seed",
    ]
    numeric: dict[str, pd.Series] = {}
    for column in integer_columns:
        values = pd.to_numeric(rows[column], errors="coerce")
        if (
            values.isna().any()
            or (~np.isfinite(values)).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError(
                f"existing random summary contains invalid integer values in {column}"
            )
        numeric[column] = values.astype(int)
    if not numeric["top_k_requested"].eq(10).all():
        raise ValueError("existing random summary must use Top-K=10 throughout")
    if (numeric["random_iterations"] < 1).any():
        raise ValueError("existing random summary iteration count must be positive")
    if numeric["random_iterations"].nunique() != 1:
        raise ValueError("existing random summary has inconsistent iteration counts")
    if numeric["global_seed"].nunique() != 1:
        raise ValueError("existing random summary has inconsistent seeds")
    sampling_contracts = rows["sampling_contract"].astype(str).str.strip()
    if sampling_contracts.eq("").any() or sampling_contracts.nunique() != 1:
        raise ValueError("existing random summary has inconsistent sampling contracts")

    ranked = exploratory.add_normalized_rank(motif_rows)
    selected = (
        ranked.sort_values(
            ["family", "method", "motif_rank", "motif_cluster"],
            kind="mergesort",
        )
        .groupby(keys, sort=False, group_keys=False)
        .head(10)
        .copy()
    )
    selected["_selected"] = 1
    selected["_testable"] = selected["status"].isin(
        ["tested_supported", "tested_not_significant"]
    ).astype(int)
    selected["_supported"] = selected["status"].eq("tested_supported").astype(int)
    expected = (
        selected.groupby(keys, as_index=False)[
            ["_selected", "_testable", "_supported"]
        ]
        .sum()
        .rename(
            columns={
                "_selected": "expected_selected_count",
                "_testable": "expected_testable_count",
                "_supported": "expected_supported_count",
            }
        )
    )
    observed = rows[keys].copy()
    for column in (
        "observed_selected_count",
        "observed_testable_count",
        "observed_supported_count",
    ):
        observed[column] = numeric[column]
    reconciled = expected.merge(observed, on=keys, how="outer", indicator=True)
    if not reconciled["_merge"].eq("both").all():
        raise ValueError(
            "existing random summary family-method inventory does not match motif table"
        )
    count_pairs = (
        ("expected_selected_count", "observed_selected_count"),
        ("expected_testable_count", "observed_testable_count"),
        ("expected_supported_count", "observed_supported_count"),
    )
    if any(
        not reconciled[expected_column].eq(reconciled[observed_column]).all()
        for expected_column, observed_column in count_pairs
    ):
        raise ValueError(
            "existing random summary observed counts do not match motif Top-10 inventory"
        )
    if enforce_publication_contract:
        observed_families = frozenset(rows["family"].astype(str))
        observed_methods = frozenset(rows["method"].astype(str))
        expected_methods = frozenset(exploratory.METHOD_ORDER)
        if (
            observed_families != PUBLICATION_FAMILIES
            or observed_methods != expected_methods
            or len(rows) != len(PUBLICATION_FAMILIES) * len(expected_methods)
        ):
            raise ValueError(
                "existing random summary does not match the publication inventory "
                "(16 families x 6 methods = 96 comparisons)"
            )
        if numeric["random_iterations"].iloc[0] != PUBLICATION_RANDOM_ITERATIONS:
            raise ValueError(
                "existing random summary does not match the publication inventory "
                "iteration count (1000)"
            )
        if numeric["global_seed"].iloc[0] != PUBLICATION_RANDOM_SEED:
            raise ValueError(
                "existing random summary does not match the publication inventory "
                "seed (20260714)"
            )
        if sampling_contracts.iloc[0] != PUBLICATION_SAMPLING_CONTRACT:
            raise ValueError(
                "existing random summary does not match the publication inventory "
                "sampling contract"
            )
    return {
        "comparison_count": int(len(rows)),
        "p_lt_0_05_count": int((p_values < 0.05).sum()),
        "method_count": int(rows["method"].nunique()),
        "family_count": int(rows["family"].nunique()),
        "iteration_values": sorted(
            numeric["random_iterations"].unique().tolist()
        ),
        "seed_values": sorted(
            numeric["global_seed"].unique().tolist()
        ),
        "top_k_requested": 10,
        "inventory_reconciled": True,
        "sampling_contract": sampling_contracts.iloc[0],
        "publication_contract_enforced": enforce_publication_contract,
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _evidence_inventory(method_yield: pd.DataFrame) -> pd.DataFrame:
    candidate_count = int(method_yield["candidate_motifs"].sum())
    testable_count = int(method_yield["testable_motifs"].sum())
    supported_count = int(method_yield["supported_motifs"].sum())
    records = [
        {
            "evidence_layer": "sequence_candidates",
            "unit": "protein sequence",
            "count": math.nan,
            "source": "not supplied to this downstream analysis",
            "linkage_status": "not_integrated",
            "interpretation_boundary": "No sequence-level total is inferred from motif rows.",
        },
        {
            "evidence_layer": "family_resolved_candidate_motifs",
            "unit": "family-method candidate motif unit",
            "count": candidate_count,
            "source": "motif_level_fairness.csv",
            "linkage_status": "directly_derived",
            "interpretation_boundary": "Method-specific candidate units are not unique biological sites.",
        },
        {
            "evidence_layer": "annotation_testable_motifs",
            "unit": "family-method candidate motif unit",
            "count": testable_count,
            "source": "motif_level_fairness.csv",
            "linkage_status": "directly_derived",
            "interpretation_boundary": "Testability reflects eligible annotation coverage, not motif function.",
        },
        {
            "evidence_layer": "functionally_supported_motifs",
            "unit": "family-method candidate motif unit",
            "count": supported_count,
            "source": "motif_level_fairness.csv",
            "linkage_status": "directly_derived",
            "interpretation_boundary": "Support is annotation enrichment evidence, not causal validation.",
        },
        {
            "evidence_layer": "structurally_characterized_subset",
            "unit": "candidate motif unit",
            "count": math.nan,
            "source": "not supplied to this downstream analysis",
            "linkage_status": "not_integrated",
            "interpretation_boundary": "Structural counts require a separately hashed motif-level linkage source.",
        },
        {
            "evidence_layer": "literature_audited_subset",
            "unit": "candidate motif unit",
            "count": math.nan,
            "source": "not supplied to this downstream analysis",
            "linkage_status": "not_integrated",
            "interpretation_boundary": "Historical manuscript counts are not reclassified as newly computed results.",
        },
    ]
    return pd.DataFrame(records)


def _inclusion_decisions(
    method_yield: pd.DataFrame,
    rank_models: pd.DataFrame,
    global_test: pd.DataFrame,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    six_methods = set(method_yield["method"]) == set(exploratory.METHOD_ORDER)
    yield_decomposition_valid = bool(
        (method_yield["candidate_motifs"] > 0).all()
        and (
            method_yield["candidate_motifs"]
            == method_yield[
                [
                    "supported_motifs",
                    "tested_nonsignificant_motifs",
                    "no_annotation_family_motifs",
                    "no_eligible_term_motifs",
                    "insufficient_family_background_motifs",
                ]
            ].sum(axis=1)
        ).all()
    )
    yield_main_eligible = yield_decomposition_valid and six_methods
    any_rank_estimable = bool((rank_models["status"] == "estimable").any())
    global_valid = bool((global_test["status"] == "valid").all())
    direction = ";".join(
        f"{row.method}:{float(row.observed_minus_null_percentage_points):+.3f}pp"
        for row in global_test.itertuples()
    )
    raw_global = float(global_test["empirical_p_ge_observed"].min())
    adjusted_global = float(global_test["adjusted_q_value"].min())
    records = [
        {
            "analysis_id": "candidate_yield",
            "destination": "main_candidate" if yield_main_eligible else "report_only",
            "validity_status": (
                "valid"
                if yield_main_eligible
                else ("incomplete_method_inventory" if yield_decomposition_valid else "invalid")
            ),
            "decision_basis": "complete status decomposition; six-method completeness disclosed",
            "result_direction": "descriptive scale and conversion",
            "raw_p": math.nan,
            "adjusted_q": math.nan,
            "result_used_for_destination": False,
            "notes": f"six_method_inventory={six_methods}",
        },
        {
            "analysis_id": "precision_yield_map",
            "destination": "main_candidate" if yield_main_eligible else "report_only",
            "validity_status": (
                "valid"
                if yield_main_eligible
                else ("incomplete_method_inventory" if yield_decomposition_valid else "invalid")
            ),
            "decision_basis": "finite count-derived precision and yield measures",
            "result_direction": "trade-off visualization",
            "raw_p": math.nan,
            "adjusted_q": math.nan,
            "result_used_for_destination": False,
            "notes": "no method is declared best",
        },
        {
            "analysis_id": "evidence_chain",
            "destination": "main_narrative_candidate",
            "validity_status": "partial",
            "decision_basis": "evidence units remain separate with explicit provenance",
            "result_direction": "architecture only",
            "raw_p": math.nan,
            "adjusted_q": math.nan,
            "result_used_for_destination": False,
            "notes": "structural/literature layers not integrated without machine-readable sources",
        },
        {
            "analysis_id": "topk_sensitivity",
            "destination": "supplement_candidate",
            "validity_status": "valid",
            "decision_basis": "all declared K values computed; rank-source heterogeneity retained",
            "result_direction": f"K={','.join(map(str, k_values))}",
            "raw_p": math.nan,
            "adjusted_q": math.nan,
            "result_used_for_destination": False,
            "notes": "sensitivity analysis, not a cross-method score",
        },
        {
            "analysis_id": "rank_trend",
            "destination": "supplement_candidate" if any_rank_estimable else "report_only",
            "validity_status": "partly_estimable" if any_rank_estimable else "not_estimable",
            "decision_basis": "destination depends only on numerical identifiability and stability",
            "result_direction": "method-specific rank association",
            "raw_p": (
                float(rank_models.loc[rank_models["status"] == "estimable", "rank_p_value"].min())
                if any_rank_estimable
                else math.nan
            ),
            "adjusted_q": (
                float(rank_models.loc[rank_models["status"] == "estimable", "rank_q_value"].min())
                if any_rank_estimable
                else math.nan
            ),
            "result_used_for_destination": False,
            "notes": "family fixed effects and family-cluster-robust SE",
        },
        {
            "analysis_id": "global_matched_background",
            "destination": "supplement_candidate" if global_valid else "report_only",
            "validity_status": "valid" if global_valid else "invalid",
            "decision_basis": "matched draws conserved and null distribution reproducible",
            "result_direction": direction,
            "raw_p": raw_global,
            "adjusted_q": adjusted_global,
            "result_used_for_destination": False,
            "notes": "effect direction and p/q do not determine destination",
        },
    ]
    return pd.DataFrame(records)


def _format_number(value: object, digits: int = 4) -> str:
    number = float(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}g}"


def _build_report(
    *,
    motif_path: Path,
    random_path: Path,
    method_yield: pd.DataFrame,
    topk: pd.DataFrame,
    rank_models: pd.DataFrame,
    global_test: pd.DataFrame,
    decisions: pd.DataFrame,
    existing_audit: dict[str, object],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Exp5 precision–yield exploratory analysis report",
        "",
        "## Scope and immutable inputs",
        "",
        f"- Motif-level source SHA-256: `{sha256_file(motif_path)}`",
        f"- Existing random-summary SHA-256: `{sha256_file(random_path)}`",
        "- This downstream analysis did not rerun clusterProfiler and did not change its multiplicity handling.",
        "- Protein family is the uncertainty cluster for bootstrap intervals and robust model standard errors.",
        "",
        "## Method-level candidate conversion",
        "",
        "| Method | All | Testable | Supported | Supported/all | Supported/testable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_yield.itertuples():
        lines.append(
            f"| {row.method} | {row.candidate_motifs:,} | {row.testable_motifs:,} | "
            f"{row.supported_motifs:,} | {row.supported_fraction_all:.3f} | "
            f"{row.supported_fraction_testable:.3f} |"
        )
    lines.extend(
        [
            "",
            "These quantities describe scale, annotation testability and support yield. They do not establish that the method with the largest count has the best enrichment performance.",
            "",
            "## Top-K sensitivity and rank semantics",
            "",
            f"Top-K values: {', '.join(map(str, sorted(topk['top_k_requested'].unique())))}. "
            f"Family-cluster bootstrap iterations: {args.bootstrap_iterations}; seed: {args.bootstrap_seed}.",
        ]
    )
    for method, group in topk.groupby("method", sort=False):
        sources = ";".join(sorted(set(group["rank_source_set"].astype(str))))
        lines.append(f"- {method}: `{sources}`.")
    lines.extend(["", "Rank measures are normalized only within family–method groups; normalized ranks are not treated as commensurable native scores across methods.", "", "## Rank models", ""])
    for row in rank_models.itertuples():
        if row.status == "estimable":
            lines.append(
                f"- {row.method}: coefficient {_format_number(row.rank_coefficient)}, "
                f"family-cluster-robust 95% CI [{_format_number(row.rank_ci_low)}, "
                f"{_format_number(row.rank_ci_high)}], p={_format_number(row.rank_p_value)}, "
                f"BH q={_format_number(row.rank_q_value)}, n={row.sample_size}, "
                f"families={row.family_cluster_count}."
            )
        else:
            lines.append(f"- {row.method}: not estimable ({row.reason}).")
    lines.extend(
        [
            "",
            "## Method-level matched-background test",
            "",
            f"Joint matched-background iterations: {args.global_iterations}; seed: {args.global_seed}. The observed Top-10 motifs remained eligible for random sampling, matching the corrected original workflow.",
            "",
            "| Method | Observed supported | Null mean | Difference | p | BH q |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in global_test.itertuples():
        lines.append(
            f"| {row.method} | {row.observed_supported_count} | {row.null_supported_mean:.3f} | "
            f"{row.observed_minus_null_percentage_points:+.3f} pp | "
            f"{row.empirical_p_ge_observed:.4g} | {row.adjusted_q_value:.4g} |"
        )
    lines.extend(
        [
            "",
            f"Existing family–method audit: {existing_audit['p_lt_0_05_count']}/{existing_audit['comparison_count']} comparisons had empirical p < 0.05. This is retained as a separate {existing_audit['iteration_values'][0]:,}-draw family-level result.",
            "",
            "## Outcome-independent inclusion gate",
            "",
            "| Analysis | Provisional destination | Validity basis |",
            "|---|---|---|",
        ]
    )
    for row in decisions.itertuples():
        lines.append(f"| {row.analysis_id} | {row.destination} | {row.decision_basis} |")
    lines.extend(
        [
            "",
            "Destinations were assigned from identifiability, comparability, stability and redundancy. Effect direction and p/q thresholds were not used to decide inclusion.",
            "",
            "## Permitted interpretation",
            "",
            "- GvpMiner may be described as expanding the candidate search space while retaining an annotation-supported subset for prioritization.",
            "- Equal-size Top-K and matched-background results may be described as prioritization diagnostics with method-specific uncertainty.",
            "- Null or negative findings must be reported when the corresponding valid analysis is presented.",
            "",
            "## Prohibited interpretation",
            "",
            "- Do not claim that GvpMiner is the functionally best enrichment method.",
            "- Do not interpret annotation enrichment as direct proof that an exact 30-aa candidate has a causal function.",
            "- Do not treat heterogeneous native motif ranks as a common continuous score across methods.",
            "- Do not merge the historical 2,584/465 literature audit with motif-level counts without its machine-readable linkage source and hash.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.bootstrap_iterations < 1 or args.global_iterations < 1:
        raise ValueError("iteration counts must be positive")
    k_values = _parse_k_values(args.k_values)
    motif_path = args.motif_level_csv.resolve()
    random_path = args.existing_random_summary_csv.resolve()
    output_dir = args.output_dir.resolve()
    _validate_paths(motif_path, random_path, output_dir, overwrite=args.overwrite)

    motif_rows = pd.read_csv(motif_path)
    exploratory.validate_motif_level_table(motif_rows)
    existing_audit = _audit_existing_random(
        random_path,
        motif_rows,
        enforce_publication_contract=not args.allow_nonpublication_audit_inventory,
    )
    method_yield = exploratory.summarise_method_yield(motif_rows)
    topk = exploratory.summarise_topk_sensitivity(
        motif_rows,
        k_values=k_values,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    rank_deciles = exploratory.summarise_rank_deciles(
        motif_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    rank_models = exploratory.fit_all_method_rank_models(motif_rows)
    global_test = exploratory.global_matched_background_test(
        motif_rows,
        iterations=args.global_iterations,
        seed=args.global_seed,
    )
    inventory = _evidence_inventory(method_yield)
    decisions = _inclusion_decisions(
        method_yield, rank_models, global_test, k_values
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    ).resolve()
    try:
        frames = {
            "method_yield_summary.csv": method_yield,
            "topk_sensitivity.csv": topk,
            "rank_decile_summary.csv": rank_deciles,
            "rank_model_summary.csv": rank_models,
            "global_random_test.csv": global_test,
            "evidence_chain_inventory.csv": inventory,
            "analysis_inclusion_decisions.csv": decisions,
        }
        for name, frame in frames.items():
            _write_csv(frame, temporary_dir / name)
        report = _build_report(
            motif_path=motif_path,
            random_path=random_path,
            method_yield=method_yield,
            topk=topk,
            rank_models=rank_models,
            global_test=global_test,
            decisions=decisions,
            existing_audit=existing_audit,
            args=args,
        )
        (temporary_dir / "exploratory_analysis_report.md").write_text(
            report, encoding="utf-8"
        )
        output_records = []
        for name in OUTPUT_NAMES:
            temporary_path = temporary_dir / name
            output_records.append(
                {
                    "path": str((output_dir / name).resolve()),
                    "size_bytes": temporary_path.stat().st_size,
                    "sha256": sha256_file(temporary_path),
                }
            )
        manifest = {
            "schema_version": 1,
            "operation": "exp5_precision_yield_exploratory_analysis",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "k_values": list(k_values),
                "bootstrap_iterations": args.bootstrap_iterations,
                "global_iterations": args.global_iterations,
                "bootstrap_seed": args.bootstrap_seed,
                "global_seed": args.global_seed,
                "rank_model": (
                    "unpenalized logistic IRLS with family fixed effects and "
                    "family-cluster-robust standard errors"
                ),
                "multiplicity": (
                    "BH across six method-level rank tests and separately across "
                    "six method-level global matched-background tests; no change to "
                    "upstream clusterProfiler p.adjust"
                ),
            },
            "inputs": [file_record(motif_path), file_record(random_path)],
            "code_files": [
                file_record(Path(__file__).resolve()),
                file_record((EXP5_ROOT / "package" / "exploratory_analysis.py").resolve()),
                file_record((EXP5_ROOT / "package" / "support_analysis.py").resolve()),
            ],
            "outputs": output_records,
            "row_counts": {
                "motif_level": int(len(motif_rows)),
                **{name: int(len(frame)) for name, frame in frames.items()},
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "existing_random_audit": existing_audit,
            "input_read_only_contract": True,
        }
        (temporary_dir / "analysis_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            _validate_existing_output(output_dir, overwrite=True)
            for child in output_dir.iterdir():
                if child.name not in KNOWN_OUTPUT_NAMES:
                    raise ValueError(f"refusing to replace unknown output: {child}")
                if child.name == "figures":
                    for figure_file in child.iterdir():
                        figure_file.unlink()
                    child.rmdir()
                else:
                    child.unlink()
            output_dir.rmdir()
        os.replace(temporary_dir, output_dir)
    except Exception:
        if temporary_dir.exists() and _is_within(temporary_dir, output_dir.parent):
            shutil.rmtree(temporary_dir)
        raise
    print(output_dir / "exploratory_analysis_report.md")


if __name__ == "__main__":
    main()
