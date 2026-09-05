#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in [SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pydssp

from motif_cluster_secondary_structure import (
    cluster_secondary_structure_summary,
    interval_secondary_structure_features,
)


RESULTS = ROOT / "work" / "results"
EXP3 = RESULTS / "exp3_domain_discovery"
EXP4 = RESULTS / "exp4_motif_structure_analysis"
STRUCTURES = RESULTS / "structures" / "esmfold"
OUTDIR = (
    RESULTS
    / "motif_validation"
    / "cluster_commonality_validation"
)

INSTANCE_COLUMNS = [
    "motif_instance_id",
    "sequence_id",
    "family",
    "method",
    "motif_cluster",
    "motif_start",
    "motif_end",
    "motif_length",
    "motif_sequence",
    "structure_path",
    "motif_mean_pLDDT",
    "secondary_structure",
    "helix_fraction",
    "sheet_fraction",
    "coil_fraction",
    "dominant_secondary_structure",
    "longest_helix_run",
    "longest_sheet_run",
    "longest_coil_run",
    "helix_segment_count",
    "sheet_segment_count",
    "coil_segment_count",
    "secondary_structure_topology",
    "secondary_structure_transition_count",
]


FAMILY_SOURCE_IDS = {
    "GvpA": "S01;S20;S21",
    "GvpC": "S01;S03;S21",
    "GvpD": "S08",
    "GvpF": "S02;S04;S14;S16",
    "GvpG": "S04;S05;S14",
    "GvpH": "S04;S05",
    "GvpJ": "S04;S05;S06",
    "GvpK": "S04;S05;S14",
    "GvpL": "S04;S05;S14",
    "GvpM": "S04;S06",
    "GvpN": "S07;S14",
    "GvpO": "S07",
    "GvpQ": "S10;S17",
    "GvpT": "S10;S14",
}


FAMILY_CONTEXT = {
    "GvpA": (
        "major shell protein with experimentally supported coil-alpha-beta-beta-alpha-coil "
        "topology; helices participate in exterior/rib contacts while beta structure "
        "contributes to the shell core and hydrophobic interior"
    ),
    "GvpC": (
        "external shell-strengthening protein with repeated alpha-helical architecture; "
        "conserved helices may support binding along GvpA ribs and reinforcement"
    ),
    "GvpD": "regulatory P-loop protein; conserved local structure may support the regulatory ATPase fold",
    "GvpF": (
        "assembly protein with an experimental two-domain alpha+beta fold; conserved "
        "mixed alpha/beta structure may form the stable scaffold or interaction surface"
    ),
    "GvpG": "accessory assembly protein; conserved local structure may support protein-protein interaction",
    "GvpH": "accessory protein with possible chaperone-like annotation; conserved structure may support assembly stabilization",
    "GvpJ": "GvpA-like accessory protein; conserved structure may support nucleation or early assembly",
    "GvpK": "accessory assembly-network protein; conserved structure may support interaction or packing",
    "GvpL": "central accessory-complex protein; conserved structure may provide an assembly scaffold",
    "GvpM": "minor GvpA-like assembly component; conserved structure may support shell-associated assembly",
    "GvpN": "AAA+ assembly ATPase; conserved structure may support ATPase organization or assembly contacts",
    "GvpO": "GvpA-interacting accessory protein; conserved structure may support shell association or maturation",
    "GvpQ": "GvpQ-like accessory protein with unresolved molecular function",
    "GvpT": "accessory-network protein; conserved structure may support interaction while low-complexity regions may remain regulatory",
}


STRUCTURE_CLASS_MEANING = {
    "alpha_helix_dominant": (
        "A conserved alpha-helical segment is compatible with a stable helical "
        "scaffold, oligomerization surface, or elongated protein-interaction element."
    ),
    "beta_strand_dominant": (
        "A conserved beta-strand-rich segment is compatible with a sheet-forming "
        "structural core, shell-facing surface, or beta-edge interaction region."
    ),
    "coil_loop_dominant": (
        "A conserved coil/loop-rich segment may provide flexibility, a connecting "
        "turn, or an accessible recognition region; low pLDDT requires caution."
    ),
    "helix_coil_mixed": (
        "A conserved helix-plus-loop architecture may combine a stable interaction "
        "segment with flexible flanks or a helix-terminating recognition region."
    ),
    "beta_coil_mixed": (
        "A conserved beta-plus-loop architecture may mark a strand-turn boundary, "
        "sheet edge, or adaptable interaction region."
    ),
    "alpha_beta_mixed": (
        "A conserved mixed alpha/beta architecture is compatible with a compact "
        "folded unit rather than an isolated flexible peptide."
    ),
    "mixed_or_heterogeneous": (
        "The cluster does not share one dominant local topology; biological meaning "
        "should not be assigned from secondary structure alone."
    ),
}


def load_representatives() -> dict[str, dict[str, str]]:
    representatives: dict[str, dict[str, str]] = {}
    for path in EXP3.glob("*/motif_representative_windows_*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                motif = row.get("motif_cluster") or row.get("motif_uid")
                if motif:
                    representatives[motif] = row
    return representatives


def assign_structure(path: Path) -> str:
    coordinates = pydssp.read_pdbtext(
        path.read_text(encoding="utf-8", errors="ignore")
    )
    return "".join(pydssp.assign(coordinates, out_type="c3")).replace("-", "C")


def commonality_strength(summary: dict[str, object], coverage: float) -> str:
    n_instances = int(summary["n_structure_instances"])
    consistency = float(summary["position_consistency_score"])
    plddt = float(summary["mean_motif_pLDDT"])
    if n_instances >= 10 and coverage >= 0.8 and consistency >= 0.8 and plddt >= 70:
        return "high"
    if n_instances >= 5 and coverage >= 0.6 and consistency >= 0.65 and plddt >= 50:
        return "moderate"
    return "low_or_uncertain"


def structural_meaning(family: str, structure_class: str) -> str:
    generic = STRUCTURE_CLASS_MEANING[structure_class]
    context = FAMILY_CONTEXT.get(
        family,
        "gas-vesicle-associated protein; family-specific molecular function requires validation",
    )
    return f"{generic} Family context: {context}."


def process_feature_file(
    path: Path,
    instance_writer: csv.DictWriter,
    structure_cache: dict[tuple[str, str], str | None],
) -> list[dict[str, object]]:
    family = path.parent.parent.name
    method = path.parent.name
    usecols = [
        "motif_instance_id",
        "sequence_id",
        "motif_cluster",
        "motif_start",
        "motif_end",
        "motif_length",
        "motif_sequence",
        "motif_mean_pLDDT",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    input_counts = frame.groupby("motif_cluster").size().to_dict()
    sequence_counts = frame.groupby("motif_cluster")["sequence_id"].nunique().to_dict()
    grouped_instances: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in frame.itertuples(index=False):
        key = (family, str(row.sequence_id))
        if key not in structure_cache:
            structure_path = STRUCTURES / family / f"{row.sequence_id}.pdb"
            if not structure_path.exists():
                structure_cache[key] = None
            else:
                try:
                    structure_cache[key] = assign_structure(structure_path)
                except (AssertionError, ValueError, IndexError):
                    structure_cache[key] = None
        full_secondary = structure_cache[key]
        if full_secondary is None:
            continue
        start = int(row.motif_start)
        end = int(row.motif_end)
        if start < 0 or end > len(full_secondary) or end <= start:
            continue
        secondary = full_secondary[start:end]
        if len(secondary) != int(row.motif_length):
            continue
        features = interval_secondary_structure_features(secondary)
        output = {
            "motif_instance_id": row.motif_instance_id,
            "sequence_id": row.sequence_id,
            "family": family,
            "method": method,
            "motif_cluster": row.motif_cluster,
            "motif_start": start,
            "motif_end": end,
            "motif_length": int(row.motif_length),
            "motif_sequence": row.motif_sequence,
            "structure_path": str(STRUCTURES / family / f"{row.sequence_id}.pdb"),
            "motif_mean_pLDDT": float(row.motif_mean_pLDDT),
            **features,
        }
        instance_writer.writerow({column: output.get(column, "") for column in INSTANCE_COLUMNS})
        grouped_instances[str(row.motif_cluster)].append(output)

    summaries = []
    for motif_cluster, rows in grouped_instances.items():
        summary = cluster_secondary_structure_summary(rows)
        n_input = int(input_counts.get(motif_cluster, 0))
        coverage = (
            int(summary["n_structure_instances"]) / n_input if n_input else 0.0
        )
        summary.update(
            {
                "family_method": f"{family}_{method}",
                "family": family,
                "method": method,
                "motif": motif_cluster,
                "n_input_instances": n_input,
                "n_unique_sequences": int(sequence_counts.get(motif_cluster, 0)),
                "structure_mapping_coverage": coverage,
                "commonality_strength": commonality_strength(summary, coverage),
                "possible_biological_significance": structural_meaning(
                    family, str(summary["common_structure_class"])
                ),
                "evidence_scope": (
                    "Secondary-structure commonality is calculated from all mapped "
                    "cluster windows; biological interpretation is family-level or "
                    "homolog-level unless a direct region experiment is available."
                ),
                "source_ids": FAMILY_SOURCE_IDS.get(family, ""),
                "evidence_grade": "B" if family != "GvpQ" else "C",
                "limitations": (
                    "ESMFold/PyDSSP-derived local structure; no comparison with "
                    "matched controls and no direct validation of the exact motif."
                ),
            }
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    instance_path = OUTDIR / "motif_window_secondary_structure.csv"
    summary_path = OUTDIR / "motif_cluster_secondary_structure_commonality.csv"
    significance_path = OUTDIR / "structural_commonality_biological_significance.csv"
    representatives = load_representatives()
    structure_cache: dict[tuple[str, str], str | None] = {}
    summaries: list[dict[str, object]] = []

    with instance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INSTANCE_COLUMNS)
        writer.writeheader()
        for path in sorted(EXP4.glob("*/*/motif_structure_features.csv")):
            summaries.extend(process_feature_file(path, writer, structure_cache))

    summary_frame = pd.DataFrame(summaries)
    for column in [
        "representative_window_id",
        "representative_parent_id",
        "representative_start",
        "representative_end",
        "representative_length",
        "representative_sequence",
    ]:
        summary_frame[column] = summary_frame["motif"].map(
            lambda motif: representatives.get(motif, {}).get(column, "")
        )
    summary_frame = summary_frame.sort_values(["family", "method", "motif"])
    summary_frame.to_csv(summary_path, index=False)
    summary_frame[
        summary_frame["commonality_strength"].isin(["high", "moderate"])
    ].to_csv(significance_path, index=False)

    print(f"instance_rows={sum(int(row['n_structure_instances']) for row in summaries)}")
    print(f"cluster_rows={len(summary_frame)}")
    print(f"structures_cached={len(structure_cache)}")
    print(
        "commonality_strength_counts="
        + repr(summary_frame["commonality_strength"].value_counts().to_dict())
    )
    print(
        "structure_class_counts="
        + repr(summary_frame["common_structure_class"].value_counts().to_dict())
    )


if __name__ == "__main__":
    main()
