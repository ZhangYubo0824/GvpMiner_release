"""Build all frozen candidate profiles using existing verified family MSAs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .core import AA, project, profile, consensus
from scripts.paths import ROOT, table_path, content_sha256, output_directory
from scripts.candidate_evidence.msa_alignment import read_fasta, validate_alignment, sha256


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_gzip(path, text):
    path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))


def load_family(root, family, provenance):
    manifest_path = root/"data/reference/alignments"/family/"alignment_manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = root/"data/reference/alignments"/family
    for name, field in [("input.fasta", "input_sha256"), ("alignment.fasta", "alignment_sha256")]:
        actual = content_sha256(table_path(cache/name))
        if actual != m[field]:
            raise ValueError(f"Frozen alignment hash mismatch: {family}/{name}")
        provenance[str(table_path(cache/name).resolve())] = sha256(table_path(cache/name))
    provenance[str(manifest_path.resolve())] = sha256(manifest_path)
    ids_path = cache/"identifiers.csv"
    provenance[str(ids_path.resolve())] = sha256(ids_path)
    inputs, aln = read_fasta(table_path(cache/"input.fasta")), read_fasta(table_path(cache/"alignment.fasta"))
    validate_alignment(inputs, aln)
    ids = pd.read_csv(ids_path)
    if set(ids.alignment_id) != set(aln) or ids.protein_id.duplicated().any():
        raise ValueError("Alignment identity table is not one-to-one")
    return {str(r.protein_id): aln[r.alignment_id] for r in ids.itertuples()}, {
        "family": family, "n_sequences": len(aln), "alignment_columns": len(next(iter(aln.values()))),
        "ungapped_identity_checked": True, "alignment_sha256": m["alignment_sha256"]}


def build(root, output):
    root, output = root.resolve(), output.resolve()
    output=output_directory(root,output)
    for name in ["profiles", "alignments", "sequences"]:
        (output/name).mkdir(exist_ok=True)
    source = root/"results/evidence"
    provenance = {}
    for path in [root/"results/catalogues/GvpMiner_candidate_evidence_latest.csv", source/"evidence/occurrence_details.csv"]:
        provenance[str(path.resolve())] = sha256(path)
    catalogue = pd.read_csv(root/"results/catalogues/GvpMiner_candidate_evidence_latest.csv")
    occurrences = pd.read_csv(source/"evidence/occurrence_details.csv")
    if len(catalogue) != 522 or catalogue.motif_uid.duplicated().any() or len(occurrences) != 84012:
        raise ValueError("Unexpected frozen catalogue")
    key = ["motif_uid", "parent_id", "occ_start", "occ_end"]
    if occurrences.duplicated(key).any() or set(occurrences.motif_uid) != set(catalogue.motif_uid):
        raise ValueError("Occurrence keys inconsistent")
    all_summaries, all_map, families, all_consensus = [], [], [], []
    for family, cats in catalogue.groupby("family", sort=True):
        alignment, fm = load_family(root, family, provenance)
        families.append(fm)
        for candidate in cats.sort_values("motif_uid").itertuples():
            uid = candidate.motif_uid
            occ = occurrences[occurrences.motif_uid == uid].sort_values(["parent_id", "occ_start", "occ_end"])
            rows, mapping = [], []
            fasta = []
            for i, r in enumerate(occ.itertuples(), 1):
                lo, hi, fragment = project(alignment[str(r.parent_id)], int(r.occ_start), int(r.occ_end), r.sequence)
                row_id = f"{uid}__{i:06d}"
                rows.append((str(r.parent_id), lo, fragment))
                mapping.append(dict(motif_uid=uid, row_id=row_id, parent_id=r.parent_id,
                                    start_0based=int(r.occ_start), end_exclusive=int(r.occ_end),
                                    msa_start_0based=lo, msa_end_exclusive=hi,
                                    sequence_sha256=hashlib.sha256(r.sequence.encode()).hexdigest()))
                fasta.append(f">{row_id} protein={r.parent_id} interval=[{r.occ_start},{r.occ_end})\n{r.sequence}\n")
            p = profile(rows)
            first, last = int(p["columns"][0]), int(p["columns"][-1])+1
            text = consensus(p)
            visible = p["coverage"] >= .5-1e-12
            segments = int((visible & ~np.r_[False, visible[:-1]]).sum())
            weights = Counter(r[0] for r in rows)
            canonical_mass, retained_mass = 0., 0.
            overlaps = []
            masked = []
            for (protein, lo, seq), m in zip(rows, mapping):
                pos = [j+lo-first for j, a in enumerate(seq) if a in AA]
                covered = sum(bool(visible[j]) for j in pos)
                w = 1/weights[protein]
                canonical_mass += len(pos)*w
                retained_mass += covered*w
                overlaps.append((protein, covered/len(pos) if pos else 0.))
                m.update(protein_weight=w, canonical_residues=len(pos),
                         displayed_residues=covered, displayed_fraction=covered/len(pos) if pos else 0.)
                masked.append(f">{m['row_id']}\n{'.'*(lo-first)}{seq}{'.'*(last-lo-len(seq))}\n")
            # Arithmetic average within protein, then equally across proteins.
            by_protein = {}
            for protein, value in overlaps:
                by_protein.setdefault(protein, []).append(value)
            protein_fraction = np.array([np.mean(v) for v in by_protein.values()])
            frame = pd.DataFrame(dict(msa_column_0based=p["columns"],
                                      canonical_coverage=p["coverage"], internal_gap_fraction=p["gap"],
                                      ambiguous_fraction=p["ambiguous"], outside_fraction=p["outside"],
                                      n_proteins_observed=p["n_proteins_observed"],
                                      occurrence_weighting_tv=p["tv"], display_column=visible,
                                      consensus_symbol=list(text)))
            for j, aa in enumerate(AA):
                frame["weighted_count_"+aa] = p["weighted_counts"][:, j]
                frame["raw_count_"+aa] = p["raw_counts"][:, j]
                frame["p_"+aa] = p["probabilities"][:, j]
                frame["occ_p_"+aa] = p["occurrence_probabilities"][:, j]
            write_gzip(output/"profiles"/f"{uid}.csv.gz", frame.to_csv(index=False))
            write_gzip(output/"alignments"/f"{uid}.masked.fasta.gz", "".join(masked))
            write_gzip(output/"sequences"/f"{uid}.fasta.gz", "".join(fasta))
            numeric = p["tv"][visible]
            summary = dict(motif_uid=uid, family=family, n_proteins=p["n_proteins"],
                           n_occurrences=len(rows), n_unique_fragments=occ.sequence.nunique(),
                           min_length=int(occ.occ_length.min()), max_length=int(occ.occ_length.max()),
                           msa_start_0based=first, msa_end_exclusive=last, span_columns=last-first,
                           display_columns=int(visible.sum()), display_runs=segments,
                           specific_consensus_columns=sum(a in AA for a in text),
                           variable_consensus_columns=text.count("X"),
                           retained_residue_mass=retained_mass/canonical_mass if canonical_mass else 0.,
                           proteins_half_residues_displayed=int((protein_fraction >= .5-1e-12).sum()),
                           fraction_proteins_half_residues_displayed=float((protein_fraction >= .5-1e-12).mean()),
                           mean_protein_displayed_fraction=float(protein_fraction.mean()),
                           maximum_coverage=float(p["coverage"].max()),
                           mean_display_tv=float(np.mean(numeric)) if len(numeric) else None,
                           max_display_tv=float(np.max(numeric)) if len(numeric) else None,
                           representation_status="shared_columns_present" if visible.any() else "no_majority_covered_column",
                           representative_accession=candidate.representative_accession,
                           representative_start_0based=int(candidate.representative_start_0based),
                           representative_end_exclusive=int(candidate.representative_end_exclusive),
                           representative_sequence=candidate.representative_sequence)
            if summary["n_proteins"] != candidate.n_unique_sequences:
                raise ValueError("Member count changed")
            all_summaries.append(summary)
            all_map.extend(mapping)
            all_consensus.append(f">{uid} msa_columns_0based=[{first},{last}) dot=low_coverage X=no_majority\n{text}\n")
        print(f"{family}: {len(cats)} profiles completed", flush=True)
    pd.DataFrame(all_summaries).to_csv(output/"candidate_summary.csv", index=False)
    pd.DataFrame(all_map).to_csv(output/"occurrence_mapping.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    (output/"coordinate_consensus.fasta").write_text("".join(all_consensus), encoding="ascii")
    json_write(output/"input_hashes.json", provenance)
    json_write(output/"build_manifest.json", dict(version="coordinate_profile_v1", candidates=len(all_summaries),
               occurrences=len(all_map), families=families, weight="one unit per protein, 1/k per occurrence",
               display_minimum_canonical_coverage=.5, consensus_strict_majority=.5,
               smoothing="none", phylogenetic_weighting=False, new_alignment=False,
               source_code_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")}))
    print(pd.DataFrame(all_summaries).groupby("representation_status").size().to_dict(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT/"work/sequence_representation")
    args = parser.parse_args()
    build(args.root, args.output)
