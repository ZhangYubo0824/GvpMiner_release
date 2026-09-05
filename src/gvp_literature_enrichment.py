from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from motif_region_evidence import (
    SequenceAlignment,
    align_query_to_reference,
    map_query_interval,
    project_reference_interval_to_query,
    reference_region_exact_match_fraction,
    region_overlap,
)


FUNCTION_TRACKS = {
    "natural_structure",
    "natural_assembly",
    "natural_regulation",
    "natural_mechanics",
    "engineering_imaging",
    "engineering_display",
}

MATCH_GRADE_ORDER = {
    "A_direct_exact": 0,
    "B_homologous_residue": 1,
    "C_homologous_region": 2,
    "D_context_only": 3,
    "no_localized_support": 4,
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "y"}


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _join_unique(values: Iterable[object]) -> str:
    output: list[str] = []
    for value in values:
        for item in _clean(value).split(";"):
            item = item.strip()
            if item and item not in output:
                output.append(item)
    return ";".join(output)


def normalize_doi(value: object) -> str:
    doi = _clean(value).lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi.strip()


def normalize_accession(value: object) -> str:
    accession = _clean(value).split()[0]
    parts = accession.split("|")
    if len(parts) >= 3 and parts[1]:
        accession = parts[1]
    return re.sub(r"\.\d+$", "", accession)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str] = (),
) -> None:
    records = [dict(row) for row in rows]
    columns = list(fieldnames)
    for row in records:
        for key in row:
            if key not in columns:
                columns.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    identifier = ""
    chunks: list[str] = []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier:
                    records[identifier] = "".join(chunks)
                token = line[1:].split()[0]
                identifier = token.split("|")[1] if "|" in token else token
                chunks = []
            else:
                chunks.append(line)
    if identifier:
        records[identifier] = "".join(chunks)
    return records


def pdb_seqres(path: Path, chain: str) -> str:
    three_to_one = {
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
    residues: list[str] = []
    for line in Path(path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if line.startswith("SEQRES") and line[11:12] == chain:
            residues.extend(
                three_to_one.get(code, "X") for code in line[19:].split()
            )
    return "".join(residues)


def load_reference_sequences(reference_dir: Path) -> dict[str, str]:
    reference_dir = Path(reference_dir)
    references: dict[str, str] = {}
    for path in sorted(reference_dir.glob("*.fasta")):
        if path.stat().st_size:
            references.update(read_fasta(path))
    pdb_path = reference_dir / "7R1C.pdb"
    if pdb_path.exists():
        references["7R1C_N"] = pdb_seqres(pdb_path, "N")
    return references


def deduplicate_sources(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        doi = normalize_doi(row.get("DOI"))
        row["DOI"] = doi
        key = f"doi:{doi}" if doi else f"id:{_clean(row.get('source_id'))}"
        grouped[key].append(row)

    deduplicated: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        primary = dict(group[0])
        primary["source_ids"] = _join_unique(
            row.get("source_id", "") for row in group
        )
        years = {_clean(row.get("year")) for row in group if row.get("year")}
        if len(years) > 1:
            conflicts.append(
                {
                    "conflict_type": "duplicate_source_metadata_mismatch",
                    "source_ids": primary["source_ids"],
                    "DOI": primary.get("DOI", ""),
                    "titles": _join_unique(row.get("title", "") for row in group),
                    "years": ";".join(sorted(years)),
                }
            )
        deduplicated.append(primary)
    return deduplicated, conflicts


def load_literature_sources(
    path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_csv_rows(path)
    required = {"source_id", "title", "source_type", "verified"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"literature_sources missing columns: {sorted(missing)}")
    return deduplicate_sources(rows)


def load_experimental_regions(path: Path) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    required = {
        "evidence_id",
        "source_id",
        "family",
        "reference_accession",
        "region_start",
        "region_end",
        "function_track",
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"experimental_regions missing columns: {sorted(missing)}")
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in rows:
        row: dict[str, object] = dict(raw)
        evidence_id = _clean(row.get("evidence_id"))
        if not evidence_id or evidence_id in seen:
            raise ValueError(f"duplicate or empty evidence_id: {evidence_id!r}")
        seen.add(evidence_id)
        track = _clean(row.get("function_track"))
        if track not in FUNCTION_TRACKS:
            raise ValueError(f"unknown function_track: {track}")
        start = _int(row.get("region_start"))
        end = _int(row.get("region_end"))
        if start < 1 or end < start:
            raise ValueError(f"invalid evidence coordinates for {evidence_id}")
        row["region_start"] = start
        row["region_end"] = end
        row["reference_id"] = _clean(row.get("reference_accession"))
        row["source_ids"] = _clean(row.get("source_id"))
        row["requires_exact_residue_match"] = _truthy(
            row.get("requires_exact_residue_match")
        )
        output.append(row)
    return output


def classify_literature_hit(
    hit: Mapping[str, object],
    representative_accession: str,
) -> str:
    mapping_fraction = _float(hit.get("motif_mapping_fraction"))
    motif_identity = _float(hit.get("motif_alignment_identity"))
    motif_overlap = _float(hit.get("motif_overlap_fraction"))
    region_overlap = _float(hit.get("region_overlap_fraction"))
    exact_fraction = _float(hit.get("reference_region_exact_match_fraction"))
    region_mapping_fraction = _float(
        hit.get("projected_query_region_mapping_fraction"),
        1.0,
    )
    overlap_count = _int(hit.get("overlap_residue_count"))
    reference_id = _clean(
        hit.get("reference_accession") or hit.get("reference_id")
    )

    if mapping_fraction < 0.8 or overlap_count == 0:
        return "D_context_only"
    if (
        normalize_accession(representative_accession)
        == normalize_accession(reference_id)
        and region_overlap >= 1.0
        and exact_fraction >= 1.0
    ):
        return "A_direct_exact"
    if (
        motif_identity >= 0.70
        and region_overlap >= 1.0
        and exact_fraction >= 1.0
    ):
        return "B_homologous_residue"
    if region_mapping_fraction < 0.50:
        return "D_context_only"
    if motif_overlap >= 0.20 or region_overlap >= 0.50:
        return "C_homologous_region"
    return "D_context_only"


def best_match_grade(hits: Sequence[Mapping[str, object]]) -> str:
    if not hits:
        return "no_localized_support"
    return min(
        (_clean(hit.get("match_grade")) for hit in hits),
        key=lambda grade: MATCH_GRADE_ORDER.get(grade, 99),
    )


def build_mapping_audit(
    motifs: Iterable[Mapping[str, object]],
    hits_by_motif: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in motifs:
        motif = dict(raw)
        motif_id = _clean(motif.get("motif_cluster"))
        hits = list(hits_by_motif.get(motif_id, ()))
        natural_tracks = sorted(
            {
                _clean(hit.get("function_track"))
                for hit in hits
                if _clean(hit.get("function_track")).startswith("natural_")
            }
        )
        engineering_tracks = sorted(
            {
                _clean(hit.get("function_track"))
                for hit in hits
                if _clean(hit.get("function_track")).startswith("engineering_")
            }
        )
        motif.update(
            {
                "best_match_grade": best_match_grade(hits),
                "literature_hit_count": len(hits),
                "evidence_ids": _join_unique(
                    hit.get("evidence_id", "") for hit in hits
                ),
                "source_ids": _join_unique(
                    hit.get("source_id") or hit.get("source_ids", "")
                    for hit in hits
                ),
                "best_grade_source_ids": "",
                "best_grade_primary_source_ids": "",
                "best_grade_function_tracks": "",
                "candidate_functions": _join_unique(
                    hit.get("candidate_function")
                    or hit.get("supported_function", "")
                    for hit in hits
                ),
                "natural_function_tracks": ";".join(natural_tracks),
                "engineering_function_tracks": ";".join(engineering_tracks),
                "literature_interpretation": (
                    "No localized literature evidence was identified; this is "
                    "not evidence that the motif lacks function."
                    if not hits
                    else "Localized literature evidence mapped to this motif."
                ),
            }
        )
        grade = _clean(motif["best_match_grade"])
        motif["best_grade_source_ids"] = _join_unique(
            hit.get("source_id") or hit.get("source_ids", "")
            for hit in hits
            if _clean(hit.get("match_grade")) == grade
        )
        motif["best_grade_primary_source_ids"] = _join_unique(
            hit.get("primary_source_ids", "")
            for hit in hits
            if _clean(hit.get("match_grade")) == grade
        )
        motif["best_grade_function_tracks"] = _join_unique(
            hit.get("function_track", "")
            for hit in hits
            if _clean(hit.get("match_grade")) == grade
        )
        rows.append(motif)
    return sorted(
        rows,
        key=lambda row: (
            _clean(row.get("family")),
            _clean(row.get("method")),
            _int(row.get("representative_start"), 10**9),
            _int(row.get("representative_end"), 10**9),
            _clean(row.get("motif_cluster")),
        ),
    )


def map_motifs_to_regions(
    motifs: Iterable[Mapping[str, object]],
    *,
    full_sequences: Mapping[str, str],
    reference_sequences: Mapping[str, str],
    regions: Sequence[Mapping[str, object]],
    minimum_global_identity: float = 0.25,
    minimum_query_coverage: float = 0.60,
    minimum_mapping_fraction: float = 0.80,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    motif_rows = [dict(row) for row in motifs]
    regions_by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for region in regions:
        regions_by_family[_clean(region.get("family"))].append(region)

    alignment_cache: dict[tuple[str, str], SequenceAlignment] = {}
    hits_by_motif: dict[str, list[dict[str, object]]] = defaultdict(list)
    conflicts: list[dict[str, object]] = []

    for motif in motif_rows:
        motif_id = _clean(motif.get("motif_cluster"))
        family = _clean(motif.get("family"))
        accession = _clean(
            motif.get("representative_parent_id")
            or motif.get("representative_accession")
        )
        sequence = full_sequences.get(accession, "")
        if not sequence:
            conflicts.append(
                {
                    "motif_cluster": motif_id,
                    "conflict_type": "representative_full_sequence_missing",
                    "details": accession,
                }
            )
            continue
        start = _int(motif.get("representative_start"), -1)
        end = _int(motif.get("representative_end"), -1)
        if start < 0 or end <= start:
            conflicts.append(
                {
                    "motif_cluster": motif_id,
                    "conflict_type": "invalid_representative_coordinates",
                    "details": f"{start}-{end}",
                }
            )
            continue

        for region in regions_by_family.get(family, ()):
            reference_id = _clean(
                region.get("reference_accession") or region.get("reference_id")
            )
            reference = reference_sequences.get(reference_id, "")
            if not reference:
                continue
            cache_key = (accession, reference_id)
            if cache_key not in alignment_cache:
                alignment_cache[cache_key] = align_query_to_reference(
                    sequence, reference
                )
            alignment = alignment_cache[cache_key]
            if (
                alignment.identity < minimum_global_identity
                or alignment.query_coverage < minimum_query_coverage
            ):
                continue
            mapped = map_query_interval(alignment, start=start, end=end)
            if mapped.mapped_fraction < minimum_mapping_fraction:
                continue
            overlap = region_overlap(
                mapped.reference_start,
                mapped.reference_end,
                _int(region.get("region_start")),
                _int(region.get("region_end")),
            )
            if overlap.overlap_residue_count == 0:
                continue
            exact_fraction = reference_region_exact_match_fraction(
                alignment,
                _int(region.get("region_start")),
                _int(region.get("region_end")),
            )
            projected_region = project_reference_interval_to_query(
                alignment,
                _int(region.get("region_start")),
                _int(region.get("region_end")),
            )
            if (
                _truthy(region.get("requires_exact_residue_match"))
                and exact_fraction < 1.0
            ):
                continue
            hit: dict[str, object] = {
                **dict(region),
                "motif_cluster": motif_id,
                "family": family,
                "method": _clean(motif.get("method")),
                "representative_accession": accession,
                "reference_id": reference_id,
                "global_alignment_identity": alignment.identity,
                "global_query_coverage": alignment.query_coverage,
                "mapped_reference_start": mapped.reference_start,
                "mapped_reference_end": mapped.reference_end,
                "motif_mapping_fraction": mapped.mapped_fraction,
                "motif_alignment_identity": mapped.identity,
                "overlap_residue_count": overlap.overlap_residue_count,
                "motif_overlap_fraction": overlap.motif_overlap_fraction,
                "region_overlap_fraction": overlap.region_overlap_fraction,
                "reference_region_exact_match_fraction": exact_fraction,
                "projected_query_region_start": (
                    projected_region.query_start
                    if projected_region.query_start is not None
                    else ""
                ),
                "projected_query_region_end": (
                    projected_region.query_end
                    if projected_region.query_end is not None
                    else ""
                ),
                "projected_query_region_mapping_fraction": (
                    projected_region.mapped_fraction
                ),
                "projected_query_region_status": (
                    "reference_region_deleted_in_query"
                    if projected_region.mapped_residue_count == 0
                    else (
                        "projected"
                        if projected_region.mapped_fraction >= 1.0
                        else "partially_projected"
                    )
                ),
                "candidate_function": _clean(
                    region.get("supported_function") or region.get("phenotype")
                ),
            }
            hit["match_grade"] = classify_literature_hit(hit, accession)
            hits_by_motif[motif_id].append(hit)

    for motif_id in hits_by_motif:
        hits_by_motif[motif_id].sort(
            key=lambda hit: (
                MATCH_GRADE_ORDER.get(_clean(hit.get("match_grade")), 99),
                -_float(hit.get("motif_overlap_fraction")),
                -_float(hit.get("motif_alignment_identity")),
                _clean(hit.get("evidence_id")),
            )
        )
    mappings = build_mapping_audit(motif_rows, hits_by_motif)
    hits = [
        hit
        for mapping in mappings
        for hit in hits_by_motif.get(_clean(mapping.get("motif_cluster")), ())
    ]
    return mappings, hits, conflicts


def load_interpro_annotations(
    path: Path,
) -> dict[str, list[dict[str, object]]]:
    annotations: dict[str, list[dict[str, object]]] = defaultdict(list)
    path = Path(path)
    if not path.exists():
        return annotations
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            try:
                start = int(fields[6])
                end = int(fields[7])
            except ValueError:
                continue
            annotations[normalize_accession(fields[0])].append(
                {
                    "analysis": fields[3],
                    "signature": fields[4],
                    "signature_name": fields[5],
                    "start": start,
                    "end": end,
                    "interpro_id": fields[11] if len(fields) > 11 else "",
                    "interpro_name": fields[12] if len(fields) > 12 else "",
                }
            )
    return annotations


def build_database_crosscheck(
    mappings: Iterable[Mapping[str, object]],
    interpro_path: Path,
    assessed_fasta: Path | None = None,
) -> list[dict[str, object]]:
    annotations = load_interpro_annotations(interpro_path)
    output: list[dict[str, object]] = []
    available = Path(interpro_path).exists()
    if assessed_fasta is None:
        interpro_path = Path(interpro_path)
        candidates = (
            interpro_path.with_suffix(""),
            interpro_path.parent.parent
            / "input"
            / interpro_path.with_suffix("").name,
        )
        assessed_fasta = next(
            (candidate for candidate in candidates if candidate.exists()),
            None,
        )
    assessed_accessions = (
        {
            normalize_accession(accession)
            for accession in read_fasta(assessed_fasta)
        }
        if assessed_fasta and Path(assessed_fasta).exists()
        else None
    )
    for mapping in mappings:
        motif_id = _clean(mapping.get("motif_cluster"))
        accession = _clean(
            mapping.get("representative_parent_id")
            or mapping.get("representative_accession")
        )
        motif_start = _int(mapping.get("representative_start")) + 1
        motif_end = _int(mapping.get("representative_end"))
        overlapping: list[dict[str, object]] = []
        for annotation in annotations.get(normalize_accession(accession), ()):
            if min(motif_end, _int(annotation["end"])) >= max(
                motif_start, _int(annotation["start"])
            ):
                overlapping.append(annotation)
        if not available:
            status = "database_unavailable"
        elif (
            assessed_accessions is not None
            and normalize_accession(accession) not in assessed_accessions
        ):
            status = "database_sequence_not_assessed"
        elif overlapping:
            status = "localized_annotation_present"
        else:
            status = "no_local_annotation"
        labels = [
            ":".join(
                part
                for part in [
                    _clean(annotation.get("analysis")),
                    _clean(annotation.get("signature")),
                    _clean(annotation.get("signature_name")),
                    f"{annotation['start']}-{annotation['end']}",
                ]
                if part
            )
            for annotation in overlapping
        ]
        output.append(
            {
                "motif_cluster": motif_id,
                "family": _clean(mapping.get("family")),
                "method": _clean(mapping.get("method")),
                "representative_accession": accession,
                "representative_start": mapping.get("representative_start", ""),
                "representative_end": mapping.get("representative_end", ""),
                "database_local_annotation_status": status,
                "overlapping_annotation_count": len(overlapping),
                "overlapping_annotations": " | ".join(labels),
                "database_sources_checked": "InterProScan",
                "database_sequence_assessed": (
                    ""
                    if assessed_accessions is None
                    else normalize_accession(accession) in assessed_accessions
                ),
                "database_assessed_fasta": (
                    str(assessed_fasta) if assessed_fasta else ""
                ),
            }
        )
    return output


def attach_family_context(
    mappings: Iterable[Mapping[str, object]],
    sources: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    context_by_family: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        if not _truthy(source.get("verified")):
            continue
        if not _clean(source.get("source_type")).startswith("primary_"):
            continue
        source_ids = _clean(
            source.get("source_ids") or source.get("source_id")
        ).split(";")
        for family in _clean(source.get("families")).split(";"):
            family = family.strip()
            if not family:
                continue
            for source_id in source_ids:
                if source_id and source_id not in context_by_family[family]:
                    context_by_family[family].append(source_id)

    output: list[dict[str, object]] = []
    for raw in mappings:
        row = dict(raw)
        if _clean(row.get("best_match_grade")) == "no_localized_support":
            family_sources = context_by_family.get(_clean(row.get("family")), [])
            if family_sources:
                row["best_match_grade"] = "D_context_only"
                row["source_ids"] = ";".join(family_sources)
                row["best_grade_source_ids"] = ";".join(family_sources)
                row["literature_interpretation"] = (
                    "Only family- or protein-level literature context was found; "
                    "no localized motif support was identified. This is not "
                    "evidence that the motif lacks function."
                )
        output.append(row)
    return output


def load_occurrence_samples(
    occurrence_root: Path,
    target_motifs: set[str],
    max_instances_per_motif: int = 100,
) -> dict[str, list[dict[str, str]]]:
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_parents: dict[str, set[str]] = defaultdict(set)
    occurrence_root = Path(occurrence_root)
    if not occurrence_root.exists() or not target_motifs:
        return samples
    for path in sorted(occurrence_root.glob("*/motif_occurrences_M[23].csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                motif_id = _clean(row.get("motif_uid"))
                if motif_id not in target_motifs:
                    continue
                parent_id = _clean(row.get("parent_id"))
                if (
                    not parent_id
                    or parent_id in seen_parents[motif_id]
                    or len(samples[motif_id]) >= max_instances_per_motif
                ):
                    continue
                seen_parents[motif_id].add(parent_id)
                samples[motif_id].append(row)
    return samples


def augment_full_sequences_from_windows(
    occurrence_root: Path,
    full_sequences: Mapping[str, str],
    required_accessions: set[str],
) -> dict[str, str]:
    output = dict(full_sequences)
    missing = {item for item in required_accessions if item not in output}
    if not missing:
        return output
    for path in sorted(Path(occurrence_root).glob("*/windows_metadata.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                parent_id = _clean(row.get("parent_id"))
                if parent_id not in missing:
                    continue
                sequence = _clean(
                    row.get("full_sequence") or row.get("window_seq_full")
                )
                if sequence:
                    output[parent_id] = sequence
                    missing.remove(parent_id)
                if not missing:
                    return output
    return output


def annotate_hits_with_source_provenance(
    hits: Iterable[Mapping[str, object]],
    sources: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_by_id = _source_index(sources)
    output: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    for raw in hits:
        hit = dict(raw)
        primary_ids: list[str] = []
        reference_ids: list[str] = []
        primary_dois: list[str] = []
        primary_pmids: list[str] = []
        for source_id in _clean(
            hit.get("source_id") or hit.get("source_ids")
        ).split(";"):
            if not source_id:
                continue
            source = source_by_id.get(source_id)
            if source is None:
                continue
            if _clean(source.get("source_type")).startswith("primary_"):
                primary_ids.append(source_id)
                doi = normalize_doi(source.get("DOI"))
                pmid = _clean(source.get("PMID"))
                if doi:
                    primary_dois.append(doi)
                if pmid:
                    primary_pmids.append(pmid)
            else:
                reference_ids.append(source_id)
        hit["primary_source_ids"] = _join_unique(primary_ids)
        hit["reference_source_ids"] = _join_unique(reference_ids)
        hit["primary_dois"] = _join_unique(primary_dois)
        hit["primary_pmids"] = _join_unique(primary_pmids)
        if (
            _clean(hit.get("match_grade"))
            in {
                "A_direct_exact",
                "B_homologous_residue",
                "C_homologous_region",
            }
            and not primary_ids
        ):
            conflicts.append(
                {
                    "motif_cluster": _clean(hit.get("motif_cluster")),
                    "evidence_id": _clean(hit.get("evidence_id")),
                    "conflict_type": "localized_hit_without_primary_source",
                    "details": _clean(
                        hit.get("source_id") or hit.get("source_ids")
                    ),
                }
            )
        output.append(hit)
    return output, conflicts


def assess_cluster_instance_support(
    mappings: Iterable[Mapping[str, object]],
    hits: Iterable[Mapping[str, object]],
    occurrences_by_motif: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    full_sequences: Mapping[str, str],
    reference_sequences: Mapping[str, str],
) -> list[dict[str, object]]:
    hits_by_motif: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for hit in hits:
        if _clean(hit.get("match_grade")) in {
            "A_direct_exact",
            "B_homologous_residue",
            "C_homologous_region",
        }:
            hits_by_motif[_clean(hit.get("motif_cluster"))].append(hit)
    for motif_hits in hits_by_motif.values():
        motif_hits.sort(
            key=lambda hit: (
                MATCH_GRADE_ORDER.get(_clean(hit.get("match_grade")), 99),
                -_float(hit.get("motif_overlap_fraction")),
                _clean(hit.get("evidence_id")),
            )
        )

    alignment_cache: dict[tuple[str, str], SequenceAlignment] = {}
    output: list[dict[str, object]] = []
    for raw in mappings:
        row = dict(raw)
        motif_id = _clean(row.get("motif_cluster"))
        localized_hits = hits_by_motif.get(motif_id, [])
        if not localized_hits:
            row["cluster_instance_review_status"] = "not_applicable"
            row["cluster_instance_sample_n"] = 0
            row["cluster_instance_supported_n"] = 0
            row["cluster_region_support_fraction"] = ""
            output.append(row)
            continue
        occurrences = list(occurrences_by_motif.get(motif_id, ()))
        if not occurrences:
            row["cluster_instance_review_status"] = "occurrences_unavailable"
            row["cluster_instance_sample_n"] = 0
            row["cluster_instance_supported_n"] = 0
            row["cluster_region_support_fraction"] = ""
            output.append(row)
            continue

        best = localized_hits[0]
        reference_id = _clean(
            best.get("reference_accession") or best.get("reference_id")
        )
        reference = reference_sequences.get(reference_id, "")
        supported = 0
        assessed = 0
        for occurrence in occurrences:
            parent_id = _clean(occurrence.get("parent_id"))
            sequence = full_sequences.get(parent_id, "")
            if not sequence or not reference:
                continue
            key = (parent_id, reference_id)
            if key not in alignment_cache:
                alignment_cache[key] = align_query_to_reference(
                    sequence, reference
                )
            alignment = alignment_cache[key]
            if alignment.identity < 0.25 or alignment.query_coverage < 0.60:
                continue
            start = _int(occurrence.get("occ_start"), -1)
            end = _int(occurrence.get("occ_end"), -1)
            if start < 0 or end <= start:
                continue
            mapped = map_query_interval(alignment, start, end)
            if mapped.mapped_fraction < 0.80:
                continue
            overlap = region_overlap(
                mapped.reference_start,
                mapped.reference_end,
                _int(best.get("region_start")),
                _int(best.get("region_end")),
            )
            exact_fraction = reference_region_exact_match_fraction(
                alignment,
                _int(best.get("region_start")),
                _int(best.get("region_end")),
            )
            assessed += 1
            grade = _clean(best.get("match_grade"))
            if grade in {"A_direct_exact", "B_homologous_residue"}:
                is_supported = (
                    overlap.region_overlap_fraction >= 1.0
                    and exact_fraction >= 1.0
                )
            else:
                is_supported = (
                    overlap.motif_overlap_fraction >= 0.20
                    or overlap.region_overlap_fraction >= 0.50
                )
            supported += int(is_supported)

        row["cluster_instance_review_status"] = (
            "assessed" if assessed else "no_mappable_occurrences"
        )
        row["cluster_instance_sample_n"] = assessed
        row["cluster_instance_supported_n"] = supported
        row["cluster_region_support_fraction"] = (
            supported / assessed if assessed else ""
        )
        output.append(row)
    return output


def validate_region_sources(
    regions: Iterable[Mapping[str, object]],
    sources: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    source_by_id = _source_index(sources)
    conflicts: list[dict[str, object]] = []
    for region in regions:
        evidence_id = _clean(region.get("evidence_id"))
        for source_id in _clean(
            region.get("source_id") or region.get("source_ids")
        ).split(";"):
            if not source_id:
                continue
            source = source_by_id.get(source_id)
            if source is None:
                conflicts.append(
                    {
                        "evidence_id": evidence_id,
                        "conflict_type": "region_source_missing",
                        "source_id": source_id,
                    }
                )
            elif not _truthy(source.get("verified")):
                conflicts.append(
                    {
                        "evidence_id": evidence_id,
                        "conflict_type": "region_source_unverified",
                        "source_id": source_id,
                    }
                )
    return conflicts


def build_n_to_c_plot_tracks(
    mappings: Iterable[Mapping[str, object]],
    hits: Iterable[Mapping[str, object]],
    database_annotations: Mapping[
        str, Sequence[Mapping[str, object]]
    ],
) -> dict[tuple[str, str], dict[str, list[dict[str, object]]]]:
    grouped: dict[
        tuple[str, str], dict[str, list[dict[str, object]]]
    ] = defaultdict(
        lambda: {"motifs": [], "literature": [], "database": []}
    )
    mapping_by_motif: dict[str, dict[str, object]] = {}
    database_seen: set[tuple[object, ...]] = set()
    for raw in mappings:
        mapping = dict(raw)
        motif_id = _clean(mapping.get("motif_cluster"))
        family = _clean(mapping.get("family"))
        method = _clean(mapping.get("method"))
        key = (family, method)
        mapping_by_motif[motif_id] = mapping
        grouped[key]["motifs"].append(
            {
                **mapping,
                "start": _int(mapping.get("representative_start")),
                "end": _int(mapping.get("representative_end")),
            }
        )
        accession = _clean(
            mapping.get("representative_parent_id")
            or mapping.get("representative_accession")
        )
        for annotation in database_annotations.get(
            normalize_accession(accession),
            (),
        ):
            annotation_key = (
                family,
                method,
                accession,
                _int(annotation.get("start")),
                _int(annotation.get("end")),
                _clean(annotation.get("analysis")),
                _clean(annotation.get("signature")),
            )
            if annotation_key in database_seen:
                continue
            database_seen.add(annotation_key)
            grouped[key]["database"].append(
                {
                    **dict(annotation),
                    "representative_accession": accession,
                    "start": _int(annotation.get("start")) - 1,
                    "end": _int(annotation.get("end")),
                }
            )

    literature_seen: set[tuple[object, ...]] = set()
    for raw in hits:
        hit = dict(raw)
        if _clean(hit.get("match_grade")) not in {
            "A_direct_exact",
            "B_homologous_residue",
            "C_homologous_region",
        }:
            continue
        mapping = mapping_by_motif.get(_clean(hit.get("motif_cluster")))
        if not mapping:
            continue
        start = _int(hit.get("projected_query_region_start"), -1)
        end = _int(hit.get("projected_query_region_end"), -1)
        if start < 0 or end <= start:
            continue
        key = (
            _clean(mapping.get("family")),
            _clean(mapping.get("method")),
        )
        literature_key = (
            key,
            _clean(hit.get("representative_accession")),
            _clean(hit.get("evidence_id")),
            start,
            end,
        )
        if literature_key in literature_seen:
            continue
        literature_seen.add(literature_key)
        grouped[key]["literature"].append(
            {**hit, "start": start, "end": end}
        )
    return dict(grouped)


def generate_n_to_c_figures(
    mappings: Iterable[Mapping[str, object]],
    output_dir: Path,
    *,
    hits: Iterable[Mapping[str, object]] = (),
    database_annotations: Mapping[
        str, Sequence[Mapping[str, object]]
    ]
    | None = None,
) -> list[Path]:
    try:
        cache_dir = Path(output_dir).parent / ".matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    grouped = build_n_to_c_plot_tracks(
        mappings,
        hits,
        database_annotations or {},
    )
    colors = {
        "A_direct_exact": "#b2182b",
        "B_homologous_residue": "#ef8a62",
        "C_homologous_region": "#67a9cf",
        "D_context_only": "#bdbdbd",
        "no_localized_support": "#e5e5e5",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (family, method), tracks in sorted(grouped.items()):
        if not family or not method:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.6))
        for row in tracks["database"]:
            ax.plot(
                [row["start"], row["end"]],
                [0, 0],
                color="#4daf4a",
                alpha=0.35,
                linewidth=3.0,
            )
        for row in tracks["literature"]:
            ax.plot(
                [row["start"], row["end"]],
                [1, 1],
                color="#984ea3",
                alpha=0.45,
                linewidth=3.0,
            )
        for row in tracks["motifs"]:
            start = _int(row.get("start"))
            end = _int(row.get("end"))
            grade = _clean(row.get("best_match_grade"))
            ax.plot(
                [start, end],
                [2, 2],
                color=colors.get(grade, "#e5e5e5"),
                alpha=0.65 if grade.startswith(("A_", "B_", "C_")) else 0.18,
                linewidth=2.2 if grade.startswith(("A_", "B_", "C_")) else 1.0,
            )
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(
            ["Database annotations", "Literature regions", "Motifs"]
        )
        ax.set_xlabel("Protein coordinate (N to C)")
        ax.set_title(
            f"{family} {method}: motif, literature and database overlap"
        )
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        path = output_dir / f"{family}_{method}_n_to_c.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_search_protocol(
    path: Path,
    *,
    motif_count: int,
    source_count: int,
    region_count: int,
    localized_count: int,
    strict_count: int,
) -> None:
    text = f"""# GVP motif literature enrichment search protocol

## Search boundary

- Search coverage: database inception through 2026-06-14.
- Primary bibliographic sources: PubMed/MEDLINE, Europe PMC and Crossref.
- Citation tracking: Web of Science, Scopus and Google Scholar.
- Database cross-checks: UniProt, InterPro, Pfam, PROSITE, NCBI CDD and RCSB PDB where available; the automated local run uses InterProScan coordinates.
- Chinese-language supplementation: CNKI and Wanfang may identify leads but do not replace an original primary paper.
- Exact public-API hit counts and execution timestamps are recorded in
  `literature_search_log.csv`; subscription or login-dependent platforms remain
  `manual_required` until a researcher records their execution.

## Evidence policy

- Reviews are discovery aids only.
- Region evidence requires a primary mutation, deletion, truncation, interaction, complementation, structure, pressure, localization or engineering experiment.
- Natural and engineering evidence tracks are reported separately.
- Engineering papers remain in the source table for discovery and context, but
  are not promoted to localized motif evidence until an experimentally tested
  residue or interval has been verified from the primary paper.
- Match grades are `A_direct_exact`, `B_homologous_residue`, `C_homologous_region`, `D_context_only` and `no_localized_support`.
- B-grade transfer requires motif identity >= 0.70, mapping fraction >= 0.80, complete coverage of the tested region and exact conservation of tested residues.
- C-grade transfer requires a valid homologous mapping and either motif overlap >= 0.20 or experimental-region overlap >= 0.50.
- Exact 30-aa text search is an audit aid, not the primary evidence method.

## Strict novel-candidate rule

A strict candidate must have A/B evidence, at least 10 unique sequences, verified DOI/PMID metadata, no overlapping local database annotation and no unresolved conflict. C-grade rows remain exploratory. D-grade rows never establish motif function.

## Run summary

- Motifs audited: {motif_count}
- Verified/deduplicated sources: {source_count}
- Experimental regions: {region_count}
- Motifs with A-C localized support: {localized_count}
- Strict database-unannotated literature candidates: {strict_count}

## Interpretation limitation

`no_localized_support` means that no verified local experiment was mapped by this evidence set. It must not be interpreted as evidence that a motif lacks function.
"""
    Path(path).write_text(text, encoding="utf-8")


def run_literature_enrichment(
    *,
    motif_table: Path,
    full_fasta: Path,
    literature_sources: Path,
    experimental_regions: Path,
    reference_dir: Path,
    interpro_path: Path,
    search_log: Path,
    output_dir: Path,
    occurrence_root: Path | None = None,
    max_cluster_instances: int = 100,
) -> dict[str, object]:
    motifs = read_csv_rows(motif_table)
    full_sequences = read_fasta(full_fasta)
    if occurrence_root:
        required_accessions = {
            _clean(
                row.get("representative_parent_id")
                or row.get("representative_accession")
            )
            for row in motifs
        }
        full_sequences = augment_full_sequences_from_windows(
            occurrence_root,
            full_sequences,
            required_accessions,
        )
    references = load_reference_sequences(reference_dir)
    sources, source_conflicts = load_literature_sources(literature_sources)
    regions = load_experimental_regions(experimental_regions)
    region_conflicts = validate_region_sources(regions, sources)

    mappings, hits, mapping_conflicts = map_motifs_to_regions(
        motifs,
        full_sequences=full_sequences,
        reference_sequences=references,
        regions=regions,
    )
    hits, hit_source_conflicts = annotate_hits_with_source_provenance(
        hits, sources
    )
    hits_by_motif: dict[str, list[dict[str, object]]] = defaultdict(list)
    for hit in hits:
        hits_by_motif[_clean(hit.get("motif_cluster"))].append(hit)
    mappings = build_mapping_audit(motifs, hits_by_motif)
    mappings = attach_family_context(mappings, sources)
    target_motifs = {
        _clean(row.get("motif_cluster"))
        for row in mappings
        if _clean(row.get("best_match_grade"))
        in {"A_direct_exact", "B_homologous_residue", "C_homologous_region"}
    }
    occurrence_samples = (
        load_occurrence_samples(
            occurrence_root,
            target_motifs,
            max_instances_per_motif=max_cluster_instances,
        )
        if occurrence_root
        else {}
    )
    mappings = assess_cluster_instance_support(
        mappings,
        hits,
        occurrence_samples,
        full_sequences=full_sequences,
        reference_sequences=references,
    )
    database_rows = build_database_crosscheck(mappings, interpro_path)
    database_by_motif = {
        _clean(row.get("motif_cluster")): row for row in database_rows
    }
    conflicts = (
        source_conflicts
        + region_conflicts
        + mapping_conflicts
        + hit_source_conflicts
    )
    strict = select_novel_literature_supported_motifs(
        mappings,
        database_by_motif,
        sources,
        conflicts,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_rows = read_csv_rows(search_log)
    write_csv_rows(output_dir / "literature_search_log.csv", search_rows)
    write_csv_rows(output_dir / "literature_sources.csv", sources)
    write_csv_rows(output_dir / "experimental_regions.csv", regions)
    write_csv_rows(output_dir / "motif_literature_mapping_all.csv", mappings)
    write_csv_rows(
        output_dir / "motif_literature_hits_long.csv",
        hits,
        fieldnames=(
            "motif_cluster",
            "evidence_id",
            "source_id",
            "match_grade",
        ),
    )
    write_csv_rows(output_dir / "database_crosscheck.csv", database_rows)
    write_csv_rows(
        output_dir / "novel_literature_supported_motifs.csv",
        strict,
        fieldnames=("motif_cluster", "best_match_grade"),
    )
    write_csv_rows(
        output_dir / "evidence_conflicts.csv",
        conflicts,
        fieldnames=("motif_cluster", "evidence_id", "conflict_type", "details"),
    )
    localized_count = sum(
        _clean(row.get("best_match_grade"))
        in {"A_direct_exact", "B_homologous_residue", "C_homologous_region"}
        for row in mappings
    )
    write_search_protocol(
        output_dir / "search_protocol.md",
        motif_count=len(mappings),
        source_count=len(sources),
        region_count=len(regions),
        localized_count=localized_count,
        strict_count=len(strict),
    )
    figure_paths = generate_n_to_c_figures(
        mappings,
        output_dir / "n_to_c_maps",
        hits=hits,
        database_annotations=load_interpro_annotations(interpro_path),
    )
    return {
        "motif_count": len(mappings),
        "source_count": len(sources),
        "region_count": len(regions),
        "hit_count": len(hits),
        "localized_motif_count": localized_count,
        "strict_candidate_count": len(strict),
        "conflict_count": len(conflicts),
        "figure_count": len(figure_paths),
        "output_dir": output_dir,
    }


def _source_index(
    sources: Mapping[str, Mapping[str, object]]
    | Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if isinstance(sources, Mapping):
        return dict(sources)
    output: dict[str, Mapping[str, object]] = {}
    for source in sources:
        ids = _clean(
            source.get("source_ids") or source.get("source_id")
        ).split(";")
        for source_id in ids:
            if source_id:
                output[source_id] = source
    return output


def select_novel_literature_supported_motifs(
    mappings: Iterable[Mapping[str, object]],
    database_by_motif: Mapping[str, Mapping[str, object]],
    sources: Mapping[str, Mapping[str, object]]
    | Iterable[Mapping[str, object]],
    conflicts: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    source_by_id = _source_index(sources)
    conflicted = {
        _clean(row.get("motif_cluster"))
        for row in conflicts
        if row.get("motif_cluster")
    }
    selected: list[dict[str, object]] = []
    for raw in mappings:
        row = dict(raw)
        motif_id = _clean(row.get("motif_cluster"))
        grade = _clean(row.get("best_match_grade"))
        if grade not in {"A_direct_exact", "B_homologous_residue"}:
            continue
        if _int(row.get("n_unique_sequences")) < 10:
            continue
        if motif_id in conflicted:
            continue
        database = database_by_motif.get(motif_id, {})
        if (
            _clean(database.get("database_local_annotation_status"))
            != "no_local_annotation"
        ):
            continue
        source_ids = [
            item
            for item in _clean(
                row.get("best_grade_primary_source_ids")
                or row.get("best_grade_source_ids")
                or row.get("source_ids")
            ).split(";")
            if item
        ]
        if not source_ids or any(
            source_id not in source_by_id
            or not _truthy(source_by_id[source_id].get("verified"))
            or not (
                normalize_doi(source_by_id[source_id].get("DOI"))
                or _clean(source_by_id[source_id].get("PMID"))
            )
            for source_id in source_ids
        ):
            continue
        row["verified_primary_source_ids"] = ";".join(source_ids)
        row["primary_dois"] = _join_unique(
            normalize_doi(source_by_id[source_id].get("DOI"))
            for source_id in source_ids
        )
        row["primary_pmids"] = _join_unique(
            source_by_id[source_id].get("PMID", "")
            for source_id in source_ids
        )
        row["direct_motif_validation_status"] = (
            "exact_accession_direct_local_evidence"
            if grade == "A_direct_exact"
            else "homologous_residue_evidence_transfer"
        )
        if grade == "A_direct_exact":
            row["interpretation"] = (
                "same reference accession; directly tested residues or "
                "interval fall within the motif"
            )
        else:
            best_tracks = {
                item
                for item in _clean(
                    row.get("best_grade_function_tracks")
                ).split(";")
                if item
            }
            if best_tracks == {"natural_structure"}:
                row["interpretation"] = (
                    "homologous experimentally resolved structural region "
                    "transferred by alignment"
                )
            elif best_tracks and all(
                item.startswith("engineering_") for item in best_tracks
            ):
                row["interpretation"] = (
                    "homologous experimentally characterized engineering "
                    "region transferred by alignment"
                )
            else:
                row["interpretation"] = (
                    "homologous experimentally tested residues transferred "
                    "by alignment"
                )
        selected.append(row)
    return sorted(
        selected,
        key=lambda row: (
            _clean(row.get("family")),
            _clean(row.get("method")),
            _int(row.get("representative_start"), 10**9),
            _clean(row.get("motif_cluster")),
        ),
    )
