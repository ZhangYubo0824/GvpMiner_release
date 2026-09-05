from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class ParsedStructure:
    sequence: str
    plddt: np.ndarray
    ca_coords: np.ndarray
    secondary_structure: list[str]
    source_path: Path


def _validate_structure(structure: ParsedStructure) -> ParsedStructure:
    n = len(structure.sequence)
    if len(structure.plddt) != n:
        raise ValueError(f"pLDDT length mismatch for {structure.source_path}: expected {n}, got {len(structure.plddt)}")
    if structure.ca_coords.shape != (n, 3):
        raise ValueError(f"CA coordinate shape mismatch for {structure.source_path}: expected {(n, 3)}, got {structure.ca_coords.shape}")
    if len(structure.secondary_structure) != n:
        raise ValueError(
            f"secondary-structure length mismatch for {structure.source_path}: expected {n}, got {len(structure.secondary_structure)}"
        )
    return structure


def _parse_json_structure(path: Path) -> ParsedStructure:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequence = str(payload.get("sequence", ""))
    plddt = np.asarray(payload.get("plddt", []), dtype=float)
    ca_coords = np.asarray(payload.get("ca_coords", []), dtype=float)
    secondary_structure = [str(item) for item in payload.get("secondary_structure", ["C"] * len(sequence))]
    return _validate_structure(
        ParsedStructure(
            sequence=sequence,
            plddt=plddt,
            ca_coords=ca_coords,
            secondary_structure=secondary_structure,
            source_path=path,
        )
    )


def _parse_pdb_structure(path: Path) -> ParsedStructure:
    residues: dict[tuple[str, int, str], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        res_name = line[17:20].strip().upper()
        chain_id = line[21].strip() or "A"
        res_seq_text = line[22:26].strip()
        insertion_code = line[26].strip()
        try:
            res_seq = int(res_seq_text)
        except ValueError:
            continue
        residues[(chain_id, res_seq, insertion_code)] = {
            "aa": THREE_TO_ONE.get(res_name, "X"),
            "coord": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            "plddt": float(line[60:66].strip() or 0.0),
        }
    ordered = [residues[key] for key in sorted(residues.keys(), key=lambda item: (item[0], item[1], item[2]))]
    sequence = "".join(str(item["aa"]) for item in ordered)
    plddt = np.asarray([float(item["plddt"]) for item in ordered], dtype=float)
    ca_coords = np.asarray([item["coord"] for item in ordered], dtype=float)
    secondary_structure = ["C"] * len(sequence)
    return _validate_structure(
        ParsedStructure(
            sequence=sequence,
            plddt=plddt,
            ca_coords=ca_coords,
            secondary_structure=secondary_structure,
            source_path=path,
        )
    )


def parse_structure_file(path: Path) -> ParsedStructure:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _parse_json_structure(path)
    if suffix in {".pdb", ".ent"}:
        return _parse_pdb_structure(path)
    raise ValueError(f"Unsupported structure file format: {path}")


def resolve_structure_path(structure_dir: Path, sequence_id: str) -> Path | None:
    structure_dir = Path(structure_dir)
    for suffix in (".json", ".pdb", ".ent"):
        candidate = structure_dir / f"{sequence_id}{suffix}"
        if candidate.exists():
            return candidate
    return None
