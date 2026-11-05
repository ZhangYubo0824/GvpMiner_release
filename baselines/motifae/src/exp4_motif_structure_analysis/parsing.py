from __future__ import annotations

import json
import shutil
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
    relative_sasa: np.ndarray | None = None


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
    if structure.relative_sasa is not None and structure.relative_sasa.shape != (n,):
        raise ValueError(
            f"relative SASA length mismatch for {structure.source_path}: expected {n}, got {structure.relative_sasa.shape}"
        )
    return structure


def _parse_json_structure(path: Path) -> ParsedStructure:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sequence = str(payload.get("sequence", ""))
    plddt = np.asarray(payload.get("plddt", []), dtype=float)
    ca_coords = np.asarray(payload.get("ca_coords", []), dtype=float)
    secondary_structure = [str(item) for item in payload.get("secondary_structure", [""] * len(sequence))]
    relative_sasa = payload.get("relative_sasa")
    return _validate_structure(
        ParsedStructure(
            sequence=sequence,
            plddt=plddt,
            ca_coords=ca_coords,
            secondary_structure=secondary_structure,
            source_path=path,
            relative_sasa=np.asarray(relative_sasa, dtype=float) if relative_sasa is not None else None,
        )
    )


def _pdb_residue_annotations(
    path: Path,
    residue_keys: list[tuple[str, int, str]],
) -> tuple[np.ndarray, list[str]]:
    """Measure full-atom RSA; use DSSP assignments only when it is installed."""
    from Bio.PDB import PDBParser, ShrakeRupley
    from Bio.PDB.DSSP import DSSP, residue_max_acc

    model = next(PDBParser(QUIET=True).get_structure("model", str(path)).get_models())
    residues = {
        (chain.id, residue.id[1], residue.id[2].strip()): residue
        for chain in model
        for residue in chain
        if residue.id[0] == " "
    }
    ShrakeRupley(n_points=100).compute(model, level="R")
    rsa = []
    for key in residue_keys:
        residue = residues.get(key)
        if residue is None or not {"N", "CA", "C", "O"}.issubset(residue.child_dict):
            rsa.append(float("nan"))
            continue
        maximum = residue_max_acc["Sander"].get(residue.resname.upper())
        rsa.append(
            min(1.0, max(0.0, float(residue.sasa) / maximum))
            if maximum else float("nan")
        )

    secondary = [""] * len(residue_keys)
    executable = shutil.which("mkdssp") or shutil.which("dssp")
    if executable:
        try:
            dssp = DSSP(model, str(path), dssp=executable)
            for index, key in enumerate(residue_keys):
                residue = residues.get(key)
                if residue is None or (key[0], residue.id) not in dssp:
                    continue
                assignment = dssp[(key[0], residue.id)][2]
                secondary[index] = (
                    "H" if assignment in {"H", "G", "I"}
                    else "E" if assignment in {"E", "B"}
                    else "C"
                )
        except Exception:
            # DSSP is an optional annotation; its failures must not erase valid
            # coordinates or SASA measured directly from the PDB atoms.
            pass
    return np.asarray(rsa, dtype=float), secondary


def _parse_pdb_structure(path: Path) -> ParsedStructure:
    residues: dict[tuple[str, int, str], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        res_name = line[17:20].strip().upper()
        chain_id = line[21]
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
    residue_keys = sorted(residues.keys(), key=lambda item: (item[0], item[1], item[2]))
    relative_sasa, secondary_structure = _pdb_residue_annotations(path, residue_keys)
    return _validate_structure(
        ParsedStructure(
            sequence=sequence,
            plddt=plddt,
            ca_coords=ca_coords,
            secondary_structure=secondary_structure,
            source_path=path,
            relative_sasa=relative_sasa,
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
