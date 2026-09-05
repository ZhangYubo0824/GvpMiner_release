from __future__ import annotations

from collections import Counter
from itertools import groupby
from statistics import mean, pstdev
from typing import Iterable, Mapping


SECONDARY_TYPES = ("H", "E", "C")


def normalize_secondary_structure(value: str) -> str:
    return "".join("C" if item in {"-", "C", "T", "S", "G", "I", "B"} else item for item in value)


def compressed_topology(value: str) -> str:
    normalized = normalize_secondary_structure(value)
    return "-".join(key for key, _ in groupby(normalized))


def _longest_run(value: str, target: str) -> int:
    return max(
        (sum(1 for _ in run) for key, run in groupby(value) if key == target),
        default=0,
    )


def _segment_count(value: str, target: str) -> int:
    return sum(1 for key, _ in groupby(value) if key == target)


def interval_secondary_structure_features(value: str) -> dict[str, object]:
    secondary = normalize_secondary_structure(value)
    if not secondary:
        raise ValueError("secondary structure cannot be empty")
    length = len(secondary)
    counts = Counter(secondary)
    fractions = {item: counts[item] / length for item in SECONDARY_TYPES}
    dominant = max(SECONDARY_TYPES, key=lambda item: (fractions[item], item == "H", item == "E"))
    return {
        "secondary_structure": secondary,
        "helix_fraction": fractions["H"],
        "sheet_fraction": fractions["E"],
        "coil_fraction": fractions["C"],
        "dominant_secondary_structure": dominant,
        "longest_helix_run": _longest_run(secondary, "H"),
        "longest_sheet_run": _longest_run(secondary, "E"),
        "longest_coil_run": _longest_run(secondary, "C"),
        "helix_segment_count": _segment_count(secondary, "H"),
        "sheet_segment_count": _segment_count(secondary, "E"),
        "coil_segment_count": _segment_count(secondary, "C"),
        "secondary_structure_topology": compressed_topology(secondary),
        "secondary_structure_transition_count": sum(
            left != right for left, right in zip(secondary, secondary[1:])
        ),
    }


def classify_cluster_structure(
    *,
    mean_helix_fraction: float,
    mean_sheet_fraction: float,
    mean_coil_fraction: float,
    helix_dominant_fraction: float,
    sheet_dominant_fraction: float,
    coil_dominant_fraction: float,
) -> str:
    if mean_helix_fraction >= 0.5 and helix_dominant_fraction >= 0.6:
        return "alpha_helix_dominant"
    if mean_sheet_fraction >= 0.5 and sheet_dominant_fraction >= 0.6:
        return "beta_strand_dominant"
    if mean_coil_fraction >= 0.65 and coil_dominant_fraction >= 0.6:
        return "coil_loop_dominant"
    if mean_helix_fraction >= 0.25 and mean_coil_fraction >= 0.25:
        return "helix_coil_mixed"
    if mean_sheet_fraction >= 0.25 and mean_coil_fraction >= 0.25:
        return "beta_coil_mixed"
    if mean_helix_fraction >= 0.2 and mean_sheet_fraction >= 0.2:
        return "alpha_beta_mixed"
    return "mixed_or_heterogeneous"


def cluster_secondary_structure_summary(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("at least one structure instance is required")
    instance_features = [
        interval_secondary_structure_features(str(row["secondary_structure"]))
        for row in records
    ]
    length_counts = Counter(
        len(str(row["secondary_structure"])) for row in instance_features
    )
    motif_length = max(length_counts, key=lambda length: (length_counts[length], length))
    consensus_instances = [
        row
        for row in instance_features
        if len(str(row["secondary_structure"])) == motif_length
    ]

    consensus = []
    consistency_values = []
    for position in range(motif_length):
        counts = Counter(
            str(row["secondary_structure"])[position] for row in consensus_instances
        )
        dominant, count = max(
            counts.items(),
            key=lambda item: (item[1], item[0] == "H", item[0] == "E"),
        )
        consensus.append(dominant)
        consistency_values.append(count / len(consensus_instances))

    dominant_counts = Counter(
        str(row["dominant_secondary_structure"]) for row in instance_features
    )
    helix_values = [float(row["helix_fraction"]) for row in instance_features]
    sheet_values = [float(row["sheet_fraction"]) for row in instance_features]
    coil_values = [float(row["coil_fraction"]) for row in instance_features]
    mean_helix = mean(helix_values)
    mean_sheet = mean(sheet_values)
    mean_coil = mean(coil_values)
    helix_dominant = dominant_counts["H"] / len(instance_features)
    sheet_dominant = dominant_counts["E"] / len(instance_features)
    coil_dominant = dominant_counts["C"] / len(instance_features)
    consensus_string = "".join(consensus)

    return {
        "n_structure_instances": len(instance_features),
        "motif_length": motif_length,
        "minimum_instance_length": min(length_counts),
        "maximum_instance_length": max(length_counts),
        "n_consensus_length_instances": len(consensus_instances),
        "consensus_length": motif_length,
        "consensus_length_instance_fraction": len(consensus_instances)
        / len(instance_features),
        "mean_helix_fraction": mean_helix,
        "sd_helix_fraction": pstdev(helix_values),
        "mean_sheet_fraction": mean_sheet,
        "sd_sheet_fraction": pstdev(sheet_values),
        "mean_coil_fraction": mean_coil,
        "sd_coil_fraction": pstdev(coil_values),
        "helix_dominant_instance_fraction": helix_dominant,
        "sheet_dominant_instance_fraction": sheet_dominant,
        "coil_dominant_instance_fraction": coil_dominant,
        "mean_longest_helix_run": mean(
            float(row["longest_helix_run"]) for row in instance_features
        ),
        "mean_longest_sheet_run": mean(
            float(row["longest_sheet_run"]) for row in instance_features
        ),
        "mean_longest_coil_run": mean(
            float(row["longest_coil_run"]) for row in instance_features
        ),
        "mean_secondary_structure_transition_count": mean(
            float(row["secondary_structure_transition_count"])
            for row in instance_features
        ),
        "consensus_secondary_structure": consensus_string,
        "consensus_topology": compressed_topology(consensus_string),
        "position_consistency_score": mean(consistency_values),
        "minimum_position_consistency": min(consistency_values),
        "common_structure_class": classify_cluster_structure(
            mean_helix_fraction=mean_helix,
            mean_sheet_fraction=mean_sheet,
            mean_coil_fraction=mean_coil,
            helix_dominant_fraction=helix_dominant,
            sheet_dominant_fraction=sheet_dominant,
            coil_dominant_fraction=coil_dominant,
        ),
        "mean_motif_pLDDT": mean(
            float(row.get("motif_mean_pLDDT", 0.0)) for row in records
        ),
    }
