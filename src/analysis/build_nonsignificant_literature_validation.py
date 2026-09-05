#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import heapq
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motif_function_literature_candidates import select_literature_candidates
from motif_validation_tables import (
    FILTER_FAILURE_TEXT,
    load_motif_catalogs,
    load_motif_representatives,
)


EXP3 = ROOT / "result6" / "exp3_domain_discovery"
BATCH = (
    ROOT
    / "result6"
    / "exp5_external_functional_enrichment"
    / "clusterprofiler"
    / "batch_results"
)
OUTDIR = (
    ROOT
    / "result6"
    / "motif_validation"
    / "cluster_commonality_validation"
)
EVIDENCE_SOURCES = (
    ROOT
    / "result6"
    / "motif_validation"
    / "literature_validation"
    / "evidence_sources_expanded5.csv"
)


FUNCTION_EVIDENCE = {
    "GvpA": (
        "shell assembly; shell stability; acoustic reporter systems",
        "S01;S10;S11;S18;S19",
        "B",
        "Homologous GvpA is the major shell protein and is used in functional gas-vesicle reporter systems.",
    ),
    "GvpC": (
        "shell reinforcement; collapse-pressure tuning; protein display/engineering",
        "S01;S03;S12;S18;S19",
        "B",
        "Homologous GvpC directly reinforces shells and has been used for display engineering.",
    ),
    "GvpD": (
        "regulation of gas-vesicle gene expression",
        "S08;S18;S19",
        "B",
        "Homologous GvpD has genetic evidence for regulation of gas-vesicle gene expression.",
    ),
    "GvpF": (
        "accessory-complex assembly and gas-vesicle biogenesis",
        "S02;S04;S14;S16;S18;S19",
        "B",
        "Homologous GvpF has a solved structure and participates in accessory assembly complexes.",
    ),
    "GvpG": (
        "accessory-protein interaction during gas-vesicle assembly",
        "S04;S05;S14;S18;S19",
        "B",
        "Homologous GvpG participates in experimentally mapped assembly interactions.",
    ),
    "GvpH": (
        "assembly support and possible chaperone-like stabilization",
        "S04;S05;S18;S19",
        "B",
        "Homologous GvpH participates in accessory interactions; some representatives have HSP20-like annotation.",
    ),
    "GvpJ": (
        "gas-vesicle assembly or nucleation",
        "S04;S05;S06;S18;S19",
        "B",
        "Homologous GvpJ mutation and interaction studies support an assembly role.",
    ),
    "GvpK": (
        "accessory assembly-network interaction",
        "S04;S05;S14;S18;S19",
        "B",
        "Homologous GvpK participates in experimentally mapped gas-vesicle assembly networks.",
    ),
    "GvpL": (
        "accessory-complex scaffold during assembly",
        "S04;S05;S14;S18;S19",
        "B",
        "Homologous GvpL forms central accessory-protein interactions.",
    ),
    "GvpM": (
        "minor shell or assembly component affecting gas-vesicle formation",
        "S04;S06;S18;S19",
        "B",
        "Homologous GvpM mutation and interaction studies support a role in vesicle formation.",
    ),
    "GvpN": (
        "ATPase-assisted gas-vesicle assembly",
        "S07;S14;S18;S19",
        "B",
        "Homologous GvpN participates in gas-vesicle assembly and carries an AAA+ ATPase domain.",
    ),
    "GvpO": (
        "GvpA-associated shell assembly or maturation",
        "S07;S18;S19",
        "B",
        "Homologous GvpO interacts with GvpA.",
    ),
    "GvpQ": (
        "gas-vesicle accessory role; molecular function unresolved",
        "S10;S17;S18;S19",
        "C",
        "Database classification supports GvpQ identity, but direct molecular-function evidence is limited.",
    ),
    "GvpT": (
        "accessory assembly-network interaction",
        "S10;S14;S18;S19",
        "B",
        "Homologous GvpT is part of a functional Bacillus gas-vesicle cluster and assembly network.",
    ),
    "GvpU": (
        "gas-vesicle clustering and intracellular spatial organization",
        "S10;S14;S15;S18;S19",
        "B",
        "Homologous GvpU directly regulates gas-vesicle clustering and spatial organization.",
    ),
    "GvpV": (
        "buoyancy-associated gas-vesicle gene-cluster function",
        "S13;S18;S19",
        "B",
        "A Microcystis rearrangement interrupting gvpV is associated with loss of buoyancy.",
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild motif-level external enrichment status tables."
    )
    parser.add_argument("--exp3-dir", type=Path, default=EXP3)
    parser.add_argument("--batch-results", type=Path, default=BATCH)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--evidence-sources",
        type=Path,
        default=EVIDENCE_SOURCES,
    )
    return parser


def adjusted_score(row: dict[str, str]) -> float:
    for key in ["p.adjust", "qvalue"]:
        text = str(row.get(key, "")).strip()
        if text:
            try:
                return float(text)
            except ValueError:
                pass
    return math.nan


def term_text(row: dict[str, str], score: float) -> str:
    term_id = str(row.get("ID", "")).strip()
    description = str(row.get("Description", "")).strip()
    label = f"{term_id}|{description}" if description else term_id
    return f"{label}|{score:.6g}"


def scan_clusterprofiler_csv(
    path: Path,
) -> tuple[dict[str, dict[str, object]], int]:
    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "n_external_terms": 0,
            "n_significant_external_terms": 0,
            "min_adjusted_value": math.inf,
            "top_terms_heap": [],
        }
    )
    row_count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            motif = str(row.get("motif_cluster", "")).strip()
            if not motif:
                continue
            score = adjusted_score(row)
            if math.isnan(score):
                continue
            item = stats[motif]
            item["n_external_terms"] = int(item["n_external_terms"]) + 1
            item["min_adjusted_value"] = min(
                float(item["min_adjusted_value"]), score
            )
            if score <= 0.05:
                item["n_significant_external_terms"] = (
                    int(item["n_significant_external_terms"]) + 1
                )
            heap = item["top_terms_heap"]
            heapq.heappush(heap, (-score, term_text(row, score)))
            if len(heap) > 5:
                heapq.heappop(heap)
    return stats, row_count


def build_statuses(
    exp3_dir: Path = EXP3,
    batch_results: Path = BATCH,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    catalogs = load_motif_catalogs(exp3_dir)
    catalog_by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in catalogs:
        catalog_by_method[f"{row['family']}_{row['method']}"].append(dict(row))

    statuses: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    family_methods = sorted(catalog_by_method)
    for family_method in family_methods:
        family, method = family_method.rsplit("_", 1)
        csv_path = batch_results / f"{family_method}.csv"
        log_path = batch_results / f"{family_method}.log"
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        filter_failed = FILTER_FAILURE_TEXT in log_text
        stats: dict[str, dict[str, object]] = {}
        csv_rows = 0
        if csv_path.exists():
            stats, csv_rows = scan_clusterprofiler_csv(csv_path)
            coverage_status = "has_clusterprofiler_csv"
        elif log_path.exists():
            coverage_status = (
                "log_only_filter_failed"
                if filter_failed
                else "external_enrichment_unavailable"
            )
        else:
            coverage_status = "not_tested_external"
        coverage.append(
            {
                "family": family,
                "method": method,
                "family_method": family_method,
                "has_clusterprofiler_csv": csv_path.exists(),
                "has_clusterprofiler_log": log_path.exists(),
                "clusterprofiler_csv_rows": csv_rows,
                "log_filter_failed": filter_failed,
                "coverage_status": coverage_status,
                "csv_path": str(csv_path) if csv_path.exists() else "",
                "log_path": str(log_path) if log_path.exists() else "",
            }
        )

        for catalog in catalog_by_method[family_method]:
            motif = str(catalog["motif_cluster"])
            catalog_base = {
                key: catalog.get(key, "")
                for key in [
                    "family",
                    "method",
                    "motif_cluster",
                    "motif_consensus",
                    "min_start",
                    "max_end",
                    "mean_start",
                    "mean_end",
                    "n_windows",
                    "n_unique_sequences",
                    "support_ratio",
                ]
            }
            stat = stats.get(motif)
            if stat is not None:
                n_significant = int(stat["n_significant_external_terms"])
                status = (
                    "external_enrichment_supported"
                    if n_significant
                    else "external_enrichment_not_significant"
                )
                heap = stat["top_terms_heap"]
                top_terms = ";".join(
                    text
                    for _, text in sorted(
                        [(-negative_score, text) for negative_score, text in heap],
                        key=lambda item: item[0],
                    )
                )
                missing_reason = ""
                min_adjusted = float(stat["min_adjusted_value"])
                n_terms = int(stat["n_external_terms"])
            elif csv_path.exists() and filter_failed:
                status = "external_filter_failed"
                missing_reason = "gene_set_size_filter_or_no_result"
                min_adjusted = ""
                n_terms = 0
                n_significant = 0
                top_terms = ""
            elif csv_path.exists():
                status = "external_enrichment_no_terms"
                missing_reason = "tested_but_no_result_rows_for_motif"
                min_adjusted = ""
                n_terms = 0
                n_significant = 0
                top_terms = ""
            elif log_path.exists() and filter_failed:
                status = "external_filter_failed"
                missing_reason = "gene_set_size_filter_or_no_csv"
                min_adjusted = ""
                n_terms = 0
                n_significant = 0
                top_terms = ""
            elif log_path.exists():
                status = "external_enrichment_unavailable"
                missing_reason = "log_only_no_csv"
                min_adjusted = ""
                n_terms = 0
                n_significant = 0
                top_terms = ""
            else:
                status = "not_tested_external"
                missing_reason = "missing_clusterprofiler_csv_and_log"
                min_adjusted = ""
                n_terms = 0
                n_significant = 0
                top_terms = ""
            statuses.append(
                {
                    **catalog_base,
                    "family_method": family_method,
                    "external_enrichment_status": status,
                    "missing_reason": missing_reason,
                    "min_adjusted_value": min_adjusted,
                    "n_external_terms": n_terms,
                    "n_significant_external_terms": n_significant,
                    "top_terms": top_terms,
                    "batch_coverage_status": coverage_status,
                    "batch_log_filter_failed": filter_failed,
                }
            )
    return statuses, coverage


def attach_representatives_and_evidence(
    candidates: list[dict[str, object]],
    exp3_dir: Path = EXP3,
) -> list[dict[str, object]]:
    representatives = {
        str(row.get("motif_cluster") or row.get("motif_uid")): row
        for row in load_motif_representatives(exp3_dir)
    }
    output = []
    for row in candidates:
        motif = str(row["motif_cluster"])
        representative = representatives.get(motif, {})
        family = str(row["family"])
        function, source_ids, _, scope = FUNCTION_EVIDENCE[family]
        accession = str(representative.get("representative_parent_id", ""))
        start = str(representative.get("representative_start", ""))
        end = str(representative.get("representative_end", ""))
        sequence = str(representative.get("representative_sequence", ""))
        output.append(
            {
                "family_method": row["family_method"],
                "family": family,
                "method": row["method"],
                "motif": motif,
                "motif_consensus": row.get("motif_consensus", ""),
                "n_unique_sequences": row.get("n_unique_sequences", ""),
                "representative_window_id": representative.get(
                    "representative_window_id", ""
                ),
                "representative_accession": accession,
                "representative_start": start,
                "representative_end": end,
                "representative_length": representative.get(
                    "representative_length", ""
                ),
                "representative_sequence": sequence,
                "external_enrichment_status": row[
                    "external_enrichment_status"
                ],
                "min_adjusted_value": row["min_adjusted_value"],
                "n_external_terms": row["n_external_terms"],
                "top_nonsignificant_terms": row["top_terms"],
                "missing_reason": row["missing_reason"],
                "related_literature_supported_function": function,
                "literature_evidence": scope,
                "evidence_scope": (
                    "Related function is supported at homologous protein, gene-cluster, "
                    "or family level; exact representative motif sequence has not yet "
                    "been demonstrated as causal."
                ),
                "source_ids": source_ids,
                "exact_sequence_query": f'"{sequence}"',
                "motif_direct_literature_support": "not_identified",
                "limitations": (
                    f"No direct experiment was identified for {accession} residues "
                    f"{start}-{end}; external nonsignificance does not establish absence "
                    "of biological function."
                ),
            }
        )
    return output


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statuses, coverage = build_statuses(
        exp3_dir=args.exp3_dir,
        batch_results=args.batch_results,
    )
    candidates = select_literature_candidates(statuses)
    validation_rows = attach_representatives_and_evidence(
        candidates,
        exp3_dir=args.exp3_dir,
    )

    pd.DataFrame(statuses).to_csv(
        args.output_dir / "external_enrichment_status_recomputed.csv",
        index=False,
    )
    pd.DataFrame(coverage).to_csv(
        args.output_dir / "external_enrichment_coverage_recomputed.csv",
        index=False,
    )
    validation_path = (
        args.output_dir
        / "external_nonsignificant_motif_function_literature_validation.csv"
    )
    validation_frame = pd.DataFrame(validation_rows)
    audit_path = (
        args.output_dir / "functional_motif_exact_sequence_search_audit.csv"
    )
    if audit_path.exists():
        audit = pd.read_csv(audit_path, encoding="utf-8-sig")
        audit = audit[["motif", "hit_count", "verified"]].rename(
            columns={
                "hit_count": "exact_sequence_hit_count",
                "verified": "exact_sequence_search_verified",
            }
        )
        validation_frame = validation_frame.merge(
            audit, on="motif", how="left", validate="one_to_one"
        )
        validation_frame["motif_direct_literature_support"] = validation_frame[
            "exact_sequence_hit_count"
        ].map(
            lambda value: (
                "not_identified"
                if value == 0
                else "candidate_hit_requires_manual_review"
            )
        )
    validation_frame.to_csv(validation_path, index=False)
    evidence_output = args.output_dir / "evidence_sources.csv"
    if (
        args.evidence_sources.exists()
        and not evidence_output.exists()
    ):
        shutil.copyfile(args.evidence_sources, evidence_output)

    status_counts = pd.Series(
        [row["external_enrichment_status"] for row in statuses]
    ).value_counts()
    candidate_counts = pd.Series(
        [row["family"] for row in validation_rows]
    ).value_counts()
    print(f"status_rows={len(statuses)}")
    print(f"candidate_rows={len(validation_rows)}")
    print("status_counts=" + repr(status_counts.to_dict()))
    print("candidate_family_counts=" + repr(candidate_counts.to_dict()))


if __name__ == "__main__":
    main()
