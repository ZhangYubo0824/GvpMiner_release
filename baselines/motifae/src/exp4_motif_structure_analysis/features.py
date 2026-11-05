from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .parsing import ParsedStructure
from .residues import observed_positions, positions_json


AA_GROUPS = {
    "hydrophobic": set("AILMFWYV"),
    "charged": set("DEKRH"),
    "aromatic": set("FWYH"),
    "polar": set("STNQCY"),
}


@dataclass(frozen=True)
class StructureFeatureCache:
    parsed: ParsedStructure
    all_coords: np.ndarray
    protein_center: np.ndarray
    all_distances_to_center: np.ndarray
    surface_cutoff: float
    global_mean_plddt: float
    contact_tree: cKDTree | None


def build_structure_feature_cache(parsed: ParsedStructure) -> StructureFeatureCache:
    all_coords = np.asarray(parsed.ca_coords, dtype=float)
    if len(all_coords):
        protein_center = all_coords.mean(axis=0)
        all_distances_to_center = np.sqrt(np.sum((all_coords - protein_center) ** 2, axis=1))
        surface_cutoff = float(np.quantile(all_distances_to_center, 0.75))
        contact_tree = cKDTree(all_coords)
    else:
        protein_center = np.asarray([0.0, 0.0, 0.0], dtype=float)
        all_distances_to_center = np.asarray([], dtype=float)
        surface_cutoff = 0.0
        contact_tree = None
    return StructureFeatureCache(
        parsed=parsed,
        all_coords=all_coords,
        protein_center=protein_center,
        all_distances_to_center=all_distances_to_center,
        surface_cutoff=surface_cutoff,
        global_mean_plddt=float(np.mean(parsed.plddt)) if len(parsed.plddt) else 0.0,
        contact_tree=contact_tree,
    )


def _pairwise_distances(coords: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    n = coords.shape[0]
    if n < 2:
        return np.asarray([], dtype=float), (np.asarray([], dtype=int), np.asarray([], dtype=int))
    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    indices = np.triu_indices(n, k=1)
    return distances[indices], indices


def _safe_fraction(mask: Sequence[bool]) -> float:
    if not mask:
        return 0.0
    return float(np.sum(mask)) / float(len(mask))


def _entropy(sequence: str) -> float:
    if not sequence:
        return 0.0
    counts = {}
    for aa in sequence:
        counts[aa] = counts.get(aa, 0) + 1
    total = float(len(sequence))
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def amino_acid_composition_features(sequence: str) -> dict[str, float]:
    sequence = str(sequence or "").upper()
    n = max(1, len(sequence))
    return {
        "hydrophobic_fraction": sum(aa in AA_GROUPS["hydrophobic"] for aa in sequence) / n,
        "charged_fraction": sum(aa in AA_GROUPS["charged"] for aa in sequence) / n,
        "aromatic_fraction": sum(aa in AA_GROUPS["aromatic"] for aa in sequence) / n,
        "polar_fraction": sum(aa in AA_GROUPS["polar"] for aa in sequence) / n,
        "motif_entropy": _entropy(sequence),
    }


def confidence_category(mean_plddt: float) -> str:
    if mean_plddt >= 70.0:
        return "high"
    if mean_plddt >= 50.0:
        return "medium"
    return "low"


def compute_interval_features(
    parsed: ParsedStructure,
    start: int,
    end: int,
    *,
    residue_positions: Sequence[int | None] | None = None,
    contact_threshold: float = 8.0,
    long_range_min_separation: int = 12,
    low_plddt_cutoff: float = 70.0,
) -> dict[str, object]:
    return compute_interval_features_cached(
        build_structure_feature_cache(parsed),
        start,
        end,
        residue_positions=residue_positions,
        contact_threshold=contact_threshold,
        long_range_min_separation=long_range_min_separation,
        low_plddt_cutoff=low_plddt_cutoff,
    )


def compute_interval_features_cached(
    cache: StructureFeatureCache,
    start: int,
    end: int,
    *,
    residue_positions: Sequence[int | None] | None = None,
    contact_threshold: float = 8.0,
    long_range_min_separation: int = 12,
    low_plddt_cutoff: float = 70.0,
) -> dict[str, object]:
    parsed = cache.parsed
    sequence_length = len(parsed.sequence)
    start = max(0, int(start))
    end = min(sequence_length, int(end))
    if end <= start:
        raise ValueError(f"Invalid interval: start={start}, end={end}")

    positions = observed_positions(start, end, residue_positions)
    indices = np.asarray(positions, dtype=int)
    interval_sequence = "".join(parsed.sequence[p] for p in positions)
    interval_plddt = np.asarray(parsed.plddt, dtype=float)[indices]
    interval_coords = cache.all_coords[indices]
    secondary = [parsed.secondary_structure[p] for p in positions]

    pairwise, pair_indices = _pairwise_distances(interval_coords)
    motif_center = interval_coords.mean(axis=0)
    distances_to_protein_center = np.sqrt(np.sum((interval_coords - cache.protein_center) ** 2, axis=1))
    surface_mask = distances_to_protein_center >= cache.surface_cutoff

    contact_mask = pairwise <= contact_threshold if pairwise.size else np.asarray([], dtype=bool)
    long_range_contact_count = 0
    if pairwise.size:
        seq_sep = np.abs(indices[pair_indices[0]] - indices[pair_indices[1]])
        long_range_contact_count = int(np.sum((pairwise <= contact_threshold) & (seq_sep >= long_range_min_separation)))

    interval_sasa = (
        np.asarray(parsed.relative_sasa, dtype=float)[indices]
        if parsed.relative_sasa is not None else np.asarray([], dtype=float)
    )
    relative_sasa: float | str = (
        float(np.mean(interval_sasa))
        if interval_sasa.size == len(positions) and np.all(np.isfinite(interval_sasa))
        else ""
    )
    secondary_available = bool(secondary) and all(item in {"H", "E", "C"} for item in secondary)

    radius = float(np.sqrt(np.mean(np.sum((interval_coords - motif_center) ** 2, axis=1)))) if len(interval_coords) else 0.0
    mean_plddt = float(np.mean(interval_plddt))
    min_plddt = float(np.min(interval_plddt))

    features: dict[str, object] = {
        "motif_start": start,
        "motif_end": end,
        "motif_length": len(positions),
        "motif_span_length": end - start,
        "motif_observed_length": len(positions),
        "residue_positions_0based": positions_json(positions),
        "core_residue_positions_0based": positions_json(residue_positions) if residue_positions is not None else "",
        "residue_offsets_0based": positions_json([None if p is None else p - start for p in residue_positions] if residue_positions is not None else [p - start for p in positions]),
        "motif_sequence": interval_sequence,
        "global_mean_pLDDT": cache.global_mean_plddt,
        "motif_mean_pLDDT": mean_plddt,
        "motif_min_pLDDT": min_plddt,
        "confidence_category": confidence_category(mean_plddt),
        "low_confidence_fraction": _safe_fraction(list(interval_plddt < low_plddt_cutoff)),
        "contact_density": float(np.mean(contact_mask)) if pairwise.size else 0.0,
        "relative_SASA": relative_sasa,
        "long_range_contact_count": long_range_contact_count,
        "local_RMSD_or_structural_similarity": "",
        "helix_fraction": _safe_fraction([item == "H" for item in secondary]) if secondary_available else "",
        "sheet_fraction": _safe_fraction([item == "E" for item in secondary]) if secondary_available else "",
        "coil_fraction": _safe_fraction([item == "C" for item in secondary]) if secondary_available else "",
        "radius_of_gyration": radius,
        "distance_to_protein_centroid": float(np.linalg.norm(motif_center - cache.protein_center)),
        "surface_residue_fraction": _safe_fraction(list(surface_mask)),
    }
    features.update(amino_acid_composition_features(interval_sequence))
    return features
