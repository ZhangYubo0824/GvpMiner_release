#!/usr/bin/env python3
"""Turn frozen InterPLM SAE features into Exp4/Exp5 motif candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from adapter_common import (
    canonical_sequence_id,
    read_fasta_records,
    write_standard_outputs,
)


METHOD = "InterPLM"
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
SAE_MODEL = "esm2-650m"
LAYER = 33
MAX_PROTEIN_LENGTH = 1022  # ESM-2's 1024 tokens include CLS and EOS.
SOURCE_MODEL = "InterPLM ESM-2-650M layer 33 pretrained normalized SAE"


@dataclass(frozen=True)
class Protein:
    sequence_id: str
    sequence: str


@dataclass(frozen=True)
class Candidate:
    feature_id: int
    sequence_id: str
    start: int
    end: int
    score: float
    motif_sequence: str


def load_sequence_table(path: Path | None) -> dict[tuple[str, str], str] | None:
    if path is None:
        return None
    result: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence_id = str(row.get("protein_id") or row.get("sequence_id") or "").strip()
            if not sequence_id:
                continue
            family = str(row.get("family") or "").strip()
            sequence = str(row.get("sequence") or "").strip().upper()
            key = (family, sequence_id)
            if key in result and result[key] != sequence:
                raise ValueError(f"Conflicting sequence-table rows for {family}/{sequence_id}")
            result[key] = sequence
    return result


def load_default_families(stats_path: Path, min_family_size: int) -> list[str]:
    families: list[str] = []
    with stats_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            family = str(row.get("family") or "").strip()
            count = int(float(row.get("count") or row.get("cleaned_count") or 0))
            is_small = str(row.get("is_small") or "").strip().upper() in {"TRUE", "1", "YES"}
            if family and count >= min_family_size and not is_small:
                families.append(family)
    return list(dict.fromkeys(families))


def load_family(
    fasta_path: Path,
    family: str,
    table: dict[tuple[str, str], str] | None,
    max_records: int | None,
) -> list[Protein]:
    records = read_fasta_records(fasta_path)
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError(f"No proteins found in {fasta_path}")
    proteins: list[Protein] = []
    seen: set[str] = set()
    for header, raw_sequence in records:
        sequence_id = canonical_sequence_id(header)
        sequence = raw_sequence.upper()
        if not sequence_id or not sequence or not sequence.isalpha():
            raise ValueError(f"Invalid FASTA record {header!r} in {fasta_path}")
        if sequence_id in seen:
            raise ValueError(f"Duplicate canonical protein ID {sequence_id!r} in {fasta_path}")
        seen.add(sequence_id)
        if table is not None:
            reference = table.get((family, sequence_id))
            if reference is None:
                raise ValueError(f"Protein {family}/{sequence_id} is absent from the sequence table")
            if reference != sequence:
                raise ValueError(f"Sequence mismatch for {family}/{sequence_id} between FASTA and sequence table")
        proteins.append(Protein(sequence_id, sequence))
    return proteins


def candidates_from_topk(
    protein: Protein,
    values,
    indices,
    *,
    min_activation: float,
    min_motif_length: int,
    max_motif_length: int,
) -> list[Candidate]:
    """Merge adjacent residues assigned to the same SAE feature."""
    positions: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for pos in range(len(protein.sequence)):
        for rank in range(indices.shape[1]):
            score = float(values[pos, rank])
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite SAE activation for {protein.sequence_id} at {pos}")
            if score > min_activation:
                positions[int(indices[pos, rank])].append((pos, score))

    candidates: list[Candidate] = []

    def emit(feature_id: int, start: int, end: int, peak: float) -> None:
        if min_motif_length <= end - start <= max_motif_length:
            motif_sequence = protein.sequence[start:end]
            if len(motif_sequence) != end - start:
                raise ValueError(f"Invalid motif coordinates for {protein.sequence_id}: {start}-{end}")
            candidates.append(Candidate(feature_id, protein.sequence_id, start, end, peak, motif_sequence))

    for feature_id, hits in positions.items():
        start, peak = hits[0]
        previous = start
        for pos, score in hits[1:]:
            if pos != previous + 1:
                emit(feature_id, start, previous + 1, peak)
                start, peak = pos, score
            else:
                peak = max(peak, score)
            previous = pos
        emit(feature_id, start, previous + 1, peak)
    return candidates


def select_candidates(
    candidates: list[Candidate], *, min_protein_support: int, max_features: int
) -> tuple[list[Candidate], dict[int, int]]:
    proteins_by_feature: dict[int, set[str]] = defaultdict(set)
    for candidate in candidates:
        proteins_by_feature[candidate.feature_id].add(candidate.sequence_id)
    support = {feature_id: len(proteins) for feature_id, proteins in proteins_by_feature.items()}
    eligible = [feature_id for feature_id, count in support.items() if count >= min_protein_support]
    retained = set(sorted(eligible, key=lambda feature_id: (-support[feature_id], feature_id))[:max_features])
    selected = [candidate for candidate in candidates if candidate.feature_id in retained]
    selected.sort(key=lambda item: (item.feature_id, item.sequence_id, item.start, item.end))
    return selected, {feature_id: support[feature_id] for feature_id in sorted(retained)}


def output_rows(family: str, selected: list[Candidate]):
    occurrences: list[dict[str, object]] = []
    scores: dict[int, list[float]] = defaultdict(list)
    for candidate in selected:
        motif_uid = f"{family}_{METHOD}_L{LAYER}_F{candidate.feature_id:05d}"
        occurrences.append(
            {
                "family": family,
                "method": METHOD,
                "motif_uid": motif_uid,
                "sequence_id": candidate.sequence_id,
                "parent_id": candidate.sequence_id,
                "occ_start": candidate.start,
                "occ_end": candidate.end,
                "occ_core_start": candidate.start,
                "occ_core_end": candidate.end,
                "score": f"{candidate.score:.6g}",
                "significance": "",
                "motif_sequence": candidate.motif_sequence,
                "consensus_or_label": f"SAE feature {candidate.feature_id}",
                "source_model": SOURCE_MODEL,
            }
        )
        scores[candidate.feature_id].append(candidate.score)

    catalog: list[dict[str, object]] = []
    for feature_id in sorted(scores):
        catalog.append(
            {
                "family": family,
                "method": METHOD,
                "motif_uid": f"{family}_{METHOD}_L{LAYER}_F{feature_id:05d}",
                "consensus_or_label": f"SAE feature {feature_id}",
                "n_occurrences": len(scores[feature_id]),
                "score": f"{sum(scores[feature_id]) / len(scores[feature_id]):.6g}",
                "significance": "",
                "source_model": SOURCE_MODEL,
            }
        )
    return occurrences, catalog


def check_output_paths(output_dir: Path, overwrite: bool) -> None:
    paths = [
        output_dir / f"motif_occurrences_{METHOD}.csv",
        output_dir / f"motif_catalog_{METHOD}.csv",
        output_dir / "interplm_manifest.json",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Outputs already exist under {output_dir}; pass --overwrite to replace them")


def write_family_outputs(
    output_dir: Path,
    family: str,
    selected: list[Candidate],
    support: dict[int, int],
    manifest: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    occurrences, catalog = output_rows(family, selected)
    occurrence_path = output_dir / f"motif_occurrences_{METHOD}.csv"
    catalog_path = output_dir / f"motif_catalog_{METHOD}.csv"
    if occurrences:
        write_standard_outputs(output_dir=output_dir, method=METHOD, occurrences=occurrences, catalog=catalog)
    else:
        # An empty family has no Exp4/Exp5 unit to discover; the manifest records why.
        for path in (occurrence_path, catalog_path):
            if path.exists():
                path.unlink()
    manifest["n_motifs"] = len(catalog)
    manifest["n_occurrences"] = len(occurrences)
    manifest["feature_protein_support"] = {str(key): value for key, value in support.items()}
    (output_dir / "interplm_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sequence-table", type=Path, default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--family-statistics", type=Path, default=None)
    parser.add_argument("--min-family-size", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None, help="Pilot only: process at most this many FASTA records per family")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sae-chunk-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-activation", type=float, default=0.0)
    parser.add_argument("--min-motif-length", type=int, default=3)
    parser.add_argument("--max-motif-length", type=int, default=50)
    parser.add_argument("--min-protein-support", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=200)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    if not args.fasta_dir.is_dir():
        raise FileNotFoundError(f"FASTA directory not found: {args.fasta_dir}")
    for name in ("batch_size", "sae_chunk_size", "top_k", "min_motif_length", "max_motif_length", "min_protein_support", "max_features"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_motif_length < args.min_motif_length:
        raise ValueError("--max-motif-length must be at least --min-motif-length")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be positive")
    if args.min_family_size < 1:
        raise ValueError("--min-family-size must be positive")
    if not math.isfinite(args.min_activation) or args.min_activation < 0:
        raise ValueError("--min-activation must be a finite, nonnegative number")

    stats_path = args.family_statistics or args.fasta_dir.parent / "family_statistics.csv"
    families = list(dict.fromkeys(args.families)) if args.families else load_default_families(
        stats_path, args.min_family_size
    )
    if not families:
        raise ValueError("No eligible families found; check --families or family_statistics.csv")
    print(f"Selected {len(families)} families: {', '.join(families)}", flush=True)
    table = load_sequence_table(args.sequence_table)
    proteins_by_family: dict[str, list[Protein]] = {}
    for family in families:
        fasta_path = args.fasta_dir / f"{family}_cleaned.fasta"
        proteins_by_family[family] = load_family(fasta_path, family, table, args.max_records)
        check_output_paths(args.output_root / family / METHOD, args.overwrite)

    try:
        import torch
        from interplm.embedders.esm import ESM
        from interplm.sae.inference import get_sae_feats_in_batches, load_sae_from_hf
    except ImportError as exc:
        raise RuntimeError("Run this adapter in the installed InterPLM environment") from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a small diagnostic run")

    print("Loading frozen ESM-2 and InterPLM SAE models", flush=True)
    embedder = ESM(model_name=ESM_MODEL, device=args.device, max_length=1024)
    sae = load_sae_from_hf(plm_model=SAE_MODEL, plm_layer=LAYER).to(args.device).eval()
    if args.top_k > sae.dict_size:
        raise ValueError(f"--top-k exceeds SAE dictionary size {sae.dict_size}")

    for family in families:
        proteins = proteins_by_family[family]
        skipped = [
            {"sequence_id": protein.sequence_id, "length": len(protein.sequence), "reason": "exceeds 1022 residues"}
            for protein in proteins if len(protein.sequence) > MAX_PROTEIN_LENGTH
        ]
        usable = [protein for protein in proteins if len(protein.sequence) <= MAX_PROTEIN_LENGTH]
        raw_candidates: list[Candidate] = []
        print(f"{family}: processing {len(usable)} proteins; skipping {len(skipped)} overlength proteins", flush=True)
        for offset in range(0, len(usable), args.batch_size):
            batch = usable[offset : offset + args.batch_size]
            try:
                result = embedder.extract_embeddings_with_boundaries(
                    [protein.sequence for protein in batch], layer=LAYER, batch_size=len(batch)
                )
                embeddings = result["embeddings"]
                boundaries = result["boundaries"]
                if len(boundaries) != len(batch):
                    raise ValueError("ESM returned the wrong number of protein boundaries")
                with torch.inference_mode():
                    for protein, (start, end) in zip(batch, boundaries):
                        embedding = embeddings[start:end]
                        if len(protein.sequence) != len(embedding):
                            raise ValueError(f"Embedding length mismatch for {protein.sequence_id}")
                        features = get_sae_feats_in_batches(
                            sae, args.device, embedding, chunk_size=args.sae_chunk_size
                        )
                        if len(features) != len(protein.sequence):
                            raise ValueError(f"SAE feature length mismatch for {protein.sequence_id}")
                        values, indices = torch.topk(features, k=args.top_k, dim=1)
                        raw_candidates.extend(candidates_from_topk(
                            protein, values.cpu().numpy(), indices.cpu().numpy(),
                            min_activation=args.min_activation,
                            min_motif_length=args.min_motif_length,
                            max_motif_length=args.max_motif_length,
                        ))
            except Exception as exc:
                ids = ", ".join(protein.sequence_id for protein in batch)
                raise RuntimeError(f"InterPLM inference failed for {family} batch containing {ids}") from exc
            if (offset // args.batch_size + 1) % 25 == 0 or offset + len(batch) == len(usable):
                print(f"{family}: {offset + len(batch)}/{len(usable)} proteins processed", flush=True)

        selected, support = select_candidates(
            raw_candidates, min_protein_support=args.min_protein_support, max_features=args.max_features
        )
        manifest: dict[str, object] = {
            "family": family,
            "method": METHOD,
            "esm_model": ESM_MODEL,
            "sae_model": SAE_MODEL,
            "sae_layer": LAYER,
            "sae_variant": "normalized pretrained",
            "coordinate_system": "zero-based, end-exclusive",
            "parameters": {
                "top_k": args.top_k,
                "min_activation": args.min_activation,
                "min_motif_length": args.min_motif_length,
                "max_motif_length": args.max_motif_length,
                "min_protein_support": args.min_protein_support,
                "max_features": args.max_features,
                "max_records": args.max_records,
                "min_family_size": args.min_family_size if args.families is None else None,
            },
            "family_selection": "explicit" if args.families else str(stats_path),
            "fasta": str(args.fasta_dir / f"{family}_cleaned.fasta"),
            "n_input_proteins": len(proteins),
            "n_processed_proteins": len(usable),
            "n_skipped_proteins": len(skipped),
            "skipped_proteins": skipped,
            "n_candidate_intervals_before_support_filter": len(raw_candidates),
        }
        output_dir = args.output_root / family / METHOD
        write_family_outputs(output_dir, family, selected, support, manifest)
        print(f"{family}: {len(support)} features, {len(selected)} occurrences -> {output_dir}", flush=True)


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
