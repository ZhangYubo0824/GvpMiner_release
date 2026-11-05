from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .controls import (
    CompositionFeatureIndex,
    CompositionWindowIndex,
    PositionSequenceIndex,
    build_composition_feature_index,
    build_composition_window_index,
    build_position_sequence_index,
    candidate_within_protein_windows,
    sample_composition_matched_windows,
    sample_composition_matched_windows_from_feature_index,
    sample_position_matched_windows_from_index,
    sample_windows_from_candidates,
)
from .control_audit import ControlAudit
from .features import StructureFeatureCache, build_structure_feature_cache, compute_interval_features_cached
from .figures import generate_publication_figures_from_summaries

from .io import CsvRowWriter, iter_csv_rows, read_csv_rows, write_csv_rows, write_json
from .parsing import ParsedStructure, parse_structure_file, resolve_structure_path
from .reporting_outputs import summarize_motif_clusters, write_analysis_report
from .resume import load_resume_state
from .residues import RESIDUE_SEMANTICS_VERSION, row_residue_positions, residue_offsets, selection_sequence
from .rmsd import backfill_local_rmsd_resumable
from .stats import (
    ControlFeatureAccumulator,
    PRIMARY_CONTROL_TYPE,
    PRIMARY_ENDPOINTS,
    SECONDARY_ENDPOINTS,
    assign_primary_q_values,
    summarize_structural_enrichment_from_control_accumulator,
)
from .structure_backends import get_structure_backend


MOTIF_FEATURE_FIELDS = [
    "motif_instance_id",
    "sequence_id",
    "family",
    "motif_id",
    "motif_cluster",
    "motif_start",
    "motif_end",
    "motif_length",
    "motif_sequence",
    "localization_scope",
    "sampler_kind",
    "guard_provenance",
    "core_residue_positions_0based",
    "residue_positions_0based",
    "residue_offsets_0based",
    "motif_span_length",
    "motif_observed_length",
    "aligned_core",
    "prediction_backend",
    "structure_path",
    "global_mean_pLDDT",
    "motif_mean_pLDDT",
    "motif_min_pLDDT",
    "confidence_category",
    "contact_density",
    "relative_SASA",
    "long_range_contact_count",
    "local_RMSD_or_structural_similarity",
    "helix_fraction",
    "sheet_fraction",
    "coil_fraction",
    "radius_of_gyration",
    "distance_to_protein_centroid",
    "surface_residue_fraction",
    "low_confidence_fraction",
    "hydrophobic_fraction",
    "charged_fraction",
    "aromatic_fraction",
    "polar_fraction",
    "motif_entropy",
]


CONTROL_FEATURE_FIELDS = [
    "control_id",
    "motif_instance_id",
    "motif_cluster",
    "control_type",
    "control_sequence_id",
    "control_start",
    "control_end",
    "prediction_backend",
] + [field for field in MOTIF_FEATURE_FIELDS if field not in {"motif_instance_id", "sequence_id", "family", "motif_id", "motif_cluster", "motif_start", "motif_end", "prediction_backend", "structure_path"}]


def stable_seed_offset(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _wrap_progress(iterable, *, show_progress: bool, desc: str, unit: str):
    if not show_progress or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, dynamic_ncols=True)


def _progress_step(message: str, *, show_progress: bool) -> None:
    if show_progress:
        print(message, flush=True)


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def _write_progress(
    output_dir: Path,
    *,
    stage: str,
    counts: dict[str, int],
    family: str = "",
    method: str = "",
    complete: bool = False,
    warnings: Sequence[str] = (),
) -> None:
    from datetime import datetime, timezone

    write_json(
        output_dir / "analysis_progress.json",
        {
            "family": family,
            "method": method,
            "stage": stage,
            "residue_semantics_version": RESIDUE_SEMANTICS_VERSION,
            "complete": complete,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts,
            "warnings": list(warnings),
        },
    )


def _require_columns(rows: Sequence[dict[str, str]], required: Sequence[str], table_name: str) -> None:
    if not rows:
        raise ValueError(f"{table_name} is empty")
    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def _normalize_sequence_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        if not normalized.get("sequence_id"):
            normalized["sequence_id"] = str(
                normalized.get("protein_id") or normalized.get("seq_id") or ""
            )
        normalized_rows.append(normalized)
    return normalized_rows


def _int_value(row: dict[str, object], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return int(float(value))


def _sequence_maps(
    sequence_rows: Sequence[dict[str, str]],
) -> dict[str, dict[str, object]]:
    result = {}
    for row in sequence_rows:
        sequence_id = str(row.get("sequence_id", ""))
        if not sequence_id:
            continue
        sequence = str(row.get("sequence", ""))
        normalized = dict(row)
        normalized["sequence_length"] = _int_value(row, "sequence_length", len(sequence)) or len(sequence)
        if sequence_id in result and (result[sequence_id].get('sequence') != sequence or result[sequence_id].get('family') != normalized.get('family')):
            raise ValueError(f'Ambiguous sequence/family for {sequence_id}; analyze shared accessions by family separately')
        result[sequence_id] = normalized
    return result


def _structure_inventory(structure_dir, sequence_rows):
    result = {}
    for protein in sorted({str(r['sequence_id']) for r in sequence_rows}):
        path = resolve_structure_path(Path(structure_dir), protein)
        if path is None:
            result[protein] = None
        else:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            result[protein] = dict(path=str(path.resolve()), sha256=digest.hexdigest())
    return result


def _motif_interval(row: dict[str, object]) -> tuple[int, int]:
    return _int_value(row, "motif_start"), _int_value(row, "motif_end")


def _motif_instance_id(row: dict[str, object], index: int) -> str:
    if row.get("motif_instance_id"):
        return str(row["motif_instance_id"])
    if row.get("occurrence_id"):
        return str(row["occurrence_id"])
    motif_id = str(row.get("motif_id", row.get("motif_cluster", "motif")))
    sequence_id = str(row.get("sequence_id", row.get("parent_id", f"seq{index}")))
    start, end = _motif_interval(row)
    return f"{motif_id}:{sequence_id}:{start}-{end}:{index}"


def _validate_unique_motif_instance_ids(motif_rows: Sequence[dict[str, object]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for index, row in enumerate(motif_rows):
        instance_id = _motif_instance_id(row, index)
        row["motif_instance_id"] = instance_id
        if instance_id in seen:
            duplicates.add(instance_id)
        seen.add(instance_id)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"Duplicate motif_instance_id values detected: {preview}")


def _load_structure_cache(
    structure_dir: Path,
    sequence_ids: Sequence[str],
    *,
    required_sequence_ids: Sequence[str] = (),
    sequences: dict[str, dict[str, object]] | None = None,
    predict_missing: bool = False,
    prediction_backend: str = "esmfold",
    show_progress: bool = True,
    progress_desc: str = "Loading structures",
) -> dict[str, ParsedStructure]:
    cache = {}
    required = set(required_sequence_ids)
    missing = []
    backend = get_structure_backend(prediction_backend) if predict_missing else None
    for sequence_id in _wrap_progress(
        sorted(set(sequence_ids)),
        show_progress=show_progress,
        desc=progress_desc,
        unit="structure",
    ):
        structure_path = resolve_structure_path(structure_dir, sequence_id)
        if structure_path is None:
            if predict_missing and sequences is not None and sequence_id in sequences:
                sequence = str(sequences[sequence_id].get("sequence", ""))
                if sequence:
                    structure_path = backend.predict(sequence_id, sequence, structure_dir)
            if structure_path is not None and not Path(structure_path).exists():
                structure_path = None
        if structure_path is None:
            if sequence_id in required:
                missing.append(sequence_id)
            continue
        parsed = parse_structure_file(structure_path)
        _verify_structure_sequence(sequence_id, parsed, sequences)
        cache[sequence_id] = parsed
    if missing:
        skipped_path = structure_dir / "missing_structures.txt"
        skipped_path.write_text("\n".join(missing))
        print(f"[warning] {len(missing)} structures missing (e.g. {missing[:5]}), "
              f"wrote to {skipped_path}, related motif instances will be skipped")
    return cache


def _verify_structure_sequence(sequence_id, parsed, sequences):
    if sequences is None or sequence_id not in sequences:
        return
    expected = str(sequences[sequence_id].get("sequence", "")).upper()
    if expected and parsed.sequence.upper() != expected:
        raise ValueError(f"Structure sequence mismatch for {sequence_id}: cached structure must match the full input protein")


def _get_structure(
    sequence_id: str,
    *,
    structure_dir: Path,
    structures: dict[str, ParsedStructure],
    sequences: dict[str, dict[str, object]],
    predict_missing: bool = False,
    prediction_backend: str = "esmfold",
) -> ParsedStructure | None:
    if sequence_id in structures:
        _verify_structure_sequence(sequence_id, structures[sequence_id], sequences)
        return structures[sequence_id]
    structure_path = resolve_structure_path(structure_dir, sequence_id)
    if structure_path is None and predict_missing and sequence_id in sequences:
        sequence = str(sequences[sequence_id].get("sequence", ""))
        if sequence:
            structure_path = get_structure_backend(prediction_backend).predict(sequence_id, sequence, structure_dir)
    if structure_path is None or not Path(structure_path).exists():
        return None
    parsed = parse_structure_file(structure_path)
    _verify_structure_sequence(sequence_id, parsed, sequences)
    structures[sequence_id] = parsed
    return parsed


def _get_feature_cache(
    sequence_id: str,
    *,
    structures: dict[str, ParsedStructure],
    feature_caches: dict[str, StructureFeatureCache],
) -> StructureFeatureCache | None:
    if sequence_id in feature_caches:
        return feature_caches[sequence_id]
    parsed = structures.get(sequence_id)
    if parsed is None:
        return None
    cache = build_structure_feature_cache(parsed)
    feature_caches[sequence_id] = cache
    return cache


def _excluded_intervals_by_sequence(motif_rows: Sequence[dict[str, object]]) -> dict[str, list[tuple[int, int]]]:
    excluded: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in motif_rows:
        sequence_id = str(row.get("sequence_id", row.get("parent_id", "")))
        start, end = _motif_interval(row)
        if sequence_id and end > start:
            excluded[sequence_id].append((start, end))
    return dict(excluded)


def _feature_row_for_interval(
    *,
    cache: StructureFeatureCache,
    start: int,
    end: int,
    contact_threshold: float,
    long_range_min_separation: int,
    low_plddt_cutoff: float,
    residue_positions: tuple[int | None, ...] | None = None,
) -> dict[str, object]:
    return compute_interval_features_cached(
        cache,
        start,
        end,
        residue_positions=residue_positions,
        contact_threshold=contact_threshold,
        long_range_min_separation=long_range_min_separation,
        low_plddt_cutoff=low_plddt_cutoff,
    )


def _cached_feature_row_for_interval(
    *,
    sequence_id: str,
    cache: StructureFeatureCache,
    start: int,
    end: int,
    contact_threshold: float,
    long_range_min_separation: int,
    low_plddt_cutoff: float,
    interval_feature_cache: dict[tuple, dict[str, object]],
    residue_positions: tuple[int | None, ...] | None = None,
) -> dict[str, object]:
    normalized_start = max(0, int(start))
    normalized_end = min(len(cache.parsed.sequence), int(end))
    key = (
        str(sequence_id),
        normalized_start,
        normalized_end,
        float(contact_threshold),
        int(long_range_min_separation),
        float(low_plddt_cutoff),
        residue_positions,
    )
    if key not in interval_feature_cache:
        interval_feature_cache[key] = _feature_row_for_interval(
            cache=cache,
            residue_positions=residue_positions,
            start=normalized_start,
            end=normalized_end,
            contact_threshold=contact_threshold,
            long_range_min_separation=long_range_min_separation,
            low_plddt_cutoff=low_plddt_cutoff,
        )
    return dict(interval_feature_cache[key])


def _build_controls_for_motif(
    *,
    motif_row: dict[str, object],
    motif_instance_id: str,
    motif_cluster: str,
    sequences: dict[str, dict[str, object]],
    structures: dict[str, ParsedStructure],
    feature_caches: dict[str, StructureFeatureCache],
    interval_feature_cache: dict[tuple[str, int, int, float, int, float], dict[str, object]],
    within_candidate_cache: dict[tuple[str, int], list[tuple[int, int]]],
    position_index: PositionSequenceIndex,
    position_control_cache: dict[tuple[str, str, int, int, int], list[tuple[str, tuple[int, int]]]],
    composition_control_cache: dict[tuple[str, str, str, int, int, str], list[tuple[str, tuple[int, int]]]],
    structure_dir: Path,
    excluded_intervals: dict[str, list[tuple[int, int]]],
    composition_index: CompositionWindowIndex | None,
    composition_feature_index: CompositionFeatureIndex | None,
    num_random_controls: int,
    seed: int,
    contact_threshold: float,
    long_range_min_separation: int,
    low_plddt_cutoff: float,
) -> list[dict[str, object]]:
    sequence_id = str(motif_row["sequence_id"])
    sequence_row = sequences[sequence_id]
    family = str(motif_row.get("family", sequence_row.get("family", "")))
    start, end = _motif_interval(motif_row)
    motif_length = end - start
    positions = row_residue_positions(motif_row)
    offsets = residue_offsets(start, end, positions)
    layout = offsets if positions is not None else None
    motif_sequence = selection_sequence(str(sequence_row.get("sequence", "")), start, end, positions)
    sequence_length = int(sequence_row["sequence_length"])
    control_rows = []

    within_key = (sequence_id, motif_length, layout)
    if within_key not in within_candidate_cache:
        within_candidate_cache[within_key] = candidate_within_protein_windows(
            sequence_length=sequence_length,
            window_length=motif_length,
            excluded_intervals=excluded_intervals.get(sequence_id, []),
        )
    within_windows = sample_windows_from_candidates(
        within_candidate_cache[within_key],
        num_controls=num_random_controls,
        rng_seed=seed + stable_seed_offset(motif_instance_id) % 100000,
    )
    control_specs: list[tuple[str, str, tuple[int, int]]] = [
        ("within_random", sequence_id, window) for window in within_windows
    ]
    position_key = (sequence_id, family, start, end, num_random_controls, layout)
    if position_key not in position_control_cache:
        position_control_cache[position_key] = sample_position_matched_windows_from_index(
            motif_sequence_id=sequence_id,
            motif_family=family,
            motif_start=start,
            motif_end=end,
            index=position_index,
            num_controls=num_random_controls,
            rng_seed=seed + stable_seed_offset(f"{sequence_id}:{start}-{end}") % 100000,
        )
    control_specs.extend(
        ("position_matched", control_sequence_id, window)
        for control_sequence_id, window in position_control_cache[position_key]
    )
    composition_mode = "feature_indexed" if composition_feature_index is not None else "scan"
    composition_key = (
        sequence_id,
        family,
        motif_sequence,
        motif_length,
        num_random_controls,
        composition_mode,
        layout,
    )
    if composition_key not in composition_control_cache:
        composition_control_cache[composition_key] = (
            sample_composition_matched_windows_from_feature_index(
                motif_sequence=motif_sequence,
                residue_offsets=layout,
                motif_family=family,
                window_length=motif_length,
                index=composition_feature_index,
                num_controls=num_random_controls,
            )
            if composition_feature_index is not None
            else sample_composition_matched_windows(
                motif_sequence=motif_sequence,
                residue_offsets=layout,
                motif_sequence_id=sequence_id,
                motif_family=family,
                window_length=motif_length,
                sequences=sequences,
                excluded_by_sequence=excluded_intervals,
                num_controls=num_random_controls,
            )
        )
    control_specs.extend(
        ("composition_matched", control_sequence_id, window)
        for control_sequence_id, window in composition_control_cache[composition_key]
    )

    for idx, (control_type, control_sequence_id, (control_start, control_end)) in enumerate(control_specs):
        parsed = _get_structure(
            control_sequence_id,
            structure_dir=structure_dir,
            structures=structures,
            sequences=sequences,
        )
        if parsed is None or control_end > len(parsed.sequence) or parsed.sequence != str(sequences[control_sequence_id].get("sequence", "")):
            continue
        cache = _get_feature_cache(
            control_sequence_id,
            structures=structures,
            feature_caches=feature_caches,
        )
        if cache is None:
            continue
        features = _cached_feature_row_for_interval(
            sequence_id=control_sequence_id,
            cache=cache,
            start=control_start,
            end=control_end,
            residue_positions=tuple(None if p is None else control_start + p for p in offsets) if positions is not None else None,
            contact_threshold=contact_threshold,
            long_range_min_separation=long_range_min_separation,
            low_plddt_cutoff=low_plddt_cutoff,
            interval_feature_cache=interval_feature_cache,
        )
        control_rows.append(
            {
                "control_id": f"{motif_instance_id}:{control_type}:{idx}",
                "motif_instance_id": motif_instance_id,
                "motif_cluster": motif_cluster,
                "control_type": control_type,
                "control_sequence_id": control_sequence_id,
                "control_start": control_start,
                "control_end": control_end,
                "prediction_backend": "precomputed",
                **features,
            }
        )
    return control_rows


def run_exp4_motif_structure_analysis(
    *,
    sequence_table: Path,
    motif_table: Path | None = None,
    structure_dir: Path,
    output_dir: Path,
    motif_rows_override: Sequence[dict[str, object]] | None = None,
    predict_missing_structures: bool = False,
    prediction_backend: str = "esmfold",
    num_random_controls: int = 50,
    seed: int = 42,
    contact_threshold: float = 8.0,
    long_range_min_separation: int = 12,
    low_plddt_cutoff: float = 70.0,
    exp3_profile: str = "unknown",
    exp3_run_signature: str = "",
    resume: bool = False,
    rmsd_workers: int = 1,
    show_progress: bool = True,
    generate_figures: bool = True,
    figure_formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, object]:
    sequence_rows = _normalize_sequence_rows(read_csv_rows(Path(sequence_table)))
    motif_rows = [dict(row) for row in motif_rows_override] if motif_rows_override is not None else read_csv_rows(Path(motif_table))
    _require_columns(sequence_rows, ["sequence_id", "family", "sequence"], "sequence table")
    _require_columns(motif_rows, ["sequence_id", "motif_start", "motif_end"], "motif table")
    _validate_unique_motif_instance_ids(motif_rows)
    motif_families = {str(row.get("family", "")) for row in motif_rows if row.get("family")}
    analysis_family = sorted(motif_families)[0] if len(motif_families) == 1 else ""
    analysis_methods = {str(row.get("method", "")) for row in motif_rows if row.get("method")}
    analysis_method = sorted(analysis_methods)[0] if len(analysis_methods) == 1 else ""
    motif_sequence_ids = {str(row.get("sequence_id", "")) for row in motif_rows if row.get("sequence_id")}
    if motif_families:
        sequence_rows = [
            row for row in sequence_rows
            if str(row.get("family", "")) in motif_families or
            (analysis_method != 'MotifAE' and str(row.get("sequence_id", "")) in motif_sequence_ids)
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_payload = dict(
        residue_semantics_version=RESIDUE_SEMANTICS_VERSION,
        sequences=sequence_rows, motifs=motif_rows,
        num_random_controls=num_random_controls, seed=seed,
        contact_threshold=contact_threshold, long_range_min_separation=long_range_min_separation,
        low_plddt_cutoff=low_plddt_cutoff, exp3_run_signature=exp3_run_signature,
        prediction_backend=prediction_backend,predict_missing_structures=predict_missing_structures,
        structures=_structure_inventory(structure_dir,sequence_rows),
    )
    input_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, default=str).encode("utf8")).hexdigest()
    signature_path = output_dir / "analysis_input_signature.json"
    if resume:
        if not signature_path.is_file() or json.loads(signature_path.read_text(encoding="utf8")).get("signature") != input_signature:
            raise ValueError("Resume input signature mismatch: exact core residue layout, sequence or analysis settings changed")
    else:
        write_json(signature_path, {"signature": input_signature, "residue_semantics_version": RESIDUE_SEMANTICS_VERSION})
    resume_state = None
    if resume:
        resume_state = load_resume_state(
            output_dir,
            family=analysis_family,
            method=analysis_method,
        )
        analysis_family = resume_state.family
        analysis_method = resume_state.method
        _progress_step(
            f"Resuming from {resume_state.stage}: "
            f"{len(resume_state.feature_rows)} saved motif features...",
            show_progress=show_progress,
        )
    else:
        _progress_step(
            f"Preparing {len(sequence_rows)} sequences for {len(motif_rows)} motif instances...",
            show_progress=show_progress,
        )

    sequences = _sequence_maps(sequence_rows)
    for row in motif_rows:
        if "motif_cluster" not in row or not row.get("motif_cluster"):
            row["motif_cluster"] = row.get("motif_id", "motif")
        if "family" not in row or not row.get("family"):
            row["family"] = sequences.get(str(row.get("sequence_id", "")), {}).get("family", "")

    if resume_state is None:
        required_structure_ids = sorted(
            {str(row.get("sequence_id", "")) for row in motif_rows if row.get("sequence_id")}
        )
    else:
        required_structure_ids = sorted(
            {
                str(row.get("sequence_id", ""))
                for row in resume_state.feature_rows
                if row.get("sequence_id")
            }
        )
    structures = (
        {}
        if resume_state is not None and resume_state.stage in ("local_rmsd_complete", "complete")
        else _load_structure_cache(
            Path(structure_dir),
            required_structure_ids,
            required_sequence_ids=required_structure_ids,
            sequences=sequences,
            predict_missing=predict_missing_structures,
            prediction_backend=prediction_backend,
            show_progress=show_progress,
            progress_desc="Loading required structures",
        )
    )
    if predict_missing_structures and resume_state is None:
        signature_payload['structures'] = _structure_inventory(structure_dir, sequence_rows)
        input_signature = hashlib.sha256(json.dumps(signature_payload,sort_keys=True,default=str).encode('utf8')).hexdigest()
        write_json(signature_path,dict(signature=input_signature,residue_semantics_version=RESIDUE_SEMANTICS_VERSION))

    control_accumulator = ControlFeatureAccumulator(PRIMARY_ENDPOINTS + SECONDARY_ENDPOINTS)
    control_audit = ControlAudit(motif_rows)
    control_contact_density: list[float] = []
    if resume_state is not None:
        feature_rows: list[dict[str, object]] = [dict(row) for row in resume_state.feature_rows]
        skipped_rows: list[dict[str, object]] = [dict(row) for row in resume_state.skipped_rows]
        control_count = resume_state.expected_control_count

        if resume_state.stage not in ("local_rmsd_complete", "complete"):
            def record_rmsd_progress(
                completed_clusters: int,
                total_clusters: int,
                completed_rows: int,
                total_rows: int,
            ) -> None:
                _write_progress(
                    output_dir,
                    stage="local_rmsd_in_progress",
                    family=analysis_family,
                    method=analysis_method,
                    counts={
                        "motif_features": len(feature_rows),
                        "control_features": control_count,
                        "skipped_motif_instances": len(skipped_rows),
                        "structural_tests": 0,
                        "rmsd_clusters_complete": completed_clusters,
                        "rmsd_clusters_total": total_clusters,
                        "rmsd_rows_complete": completed_rows,
                        "rmsd_rows_total": total_rows,
                    },
                )

            _progress_step("Backfilling local RMSD from saved checkpoints...", show_progress=show_progress)
            backfill_local_rmsd_resumable(
                feature_rows,
                structures,
                output_dir=output_dir,
                workers=rmsd_workers,
                show_progress=show_progress,
                progress_callback=record_rmsd_progress,
                reuse_checkpoints=True,
            )
            write_csv_rows(output_dir / "motif_structure_features.csv", feature_rows, MOTIF_FEATURE_FIELDS)
            _write_progress(
                output_dir,
                stage="local_rmsd_complete",
                family=analysis_family,
                method=analysis_method,
                counts={
                    "motif_features": len(feature_rows),
                    "control_features": control_count,
                    "skipped_motif_instances": len(skipped_rows),
                    "structural_tests": 0,
                },
            )

        _progress_step("Loading saved control features...", show_progress=show_progress)
        control_rows = iter_csv_rows(resume_state.control_path)
        if show_progress and tqdm is not None:
            control_rows = tqdm(
                control_rows,
                total=resume_state.expected_control_count,
                desc="Loading saved controls",
                unit="control",
                dynamic_ncols=True,
            )
        actual_control_count = 0
        for control_row in control_rows:
            control_accumulator.add(control_row)
            control_audit.add(control_row)
            actual_control_count += 1
            if generate_figures and control_row.get("control_type") == "composition_matched":
                contact_density = _to_float(control_row.get("contact_density"))
                if contact_density is not None:
                    control_contact_density.append(contact_density)
        if actual_control_count != resume_state.expected_control_count:
            raise ValueError(
                "Resume control feature count mismatch: "
                f"progress has {resume_state.expected_control_count}, "
                f"file has {actual_control_count}"
            )
        control_count = actual_control_count
    else:
        feature_caches: dict[str, StructureFeatureCache] = {}
        interval_feature_cache: dict[tuple[str, int, int, float, int, float], dict[str, object]] = {}
        within_candidate_cache: dict[tuple[str, int], list[tuple[int, int]]] = {}
        position_index = build_position_sequence_index(sequences)
        position_control_cache: dict[tuple[str, str, int, int, int], list[tuple[str, tuple[int, int]]]] = {}
        composition_control_cache: dict[tuple[str, str, str, int, int, str], list[tuple[str, tuple[int, int]]]] = {}
        excluded = _excluded_intervals_by_sequence(motif_rows)
        _progress_step("Building composition-matched control index...", show_progress=show_progress)
        composition_index = (
            build_composition_window_index(
                sequences=sequences,
                excluded_by_sequence=excluded,
                window_lengths=[_motif_interval(row)[1] - _motif_interval(row)[0] for row in motif_rows if row_residue_positions(row) is None],
                residue_offset_layouts=[residue_offsets(*_motif_interval(row), row_residue_positions(row)) for row in motif_rows if row_residue_positions(row) is not None],
            )
            if num_random_controls > 0
            else None
        )
        composition_feature_index = build_composition_feature_index(composition_index) if composition_index is not None else None
        if composition_feature_index is not None:
            composition_index = None

        feature_rows = []
        skipped_rows = []
        with CsvRowWriter(output_dir / "matched_control_features.csv", CONTROL_FEATURE_FIELDS) as control_writer:
            for index, motif_row in _wrap_progress(
                enumerate(motif_rows),
                show_progress=show_progress,
                desc="Analyzing motif instances",
                unit="motif",
            ):
                sequence_id = str(motif_row.get("sequence_id", ""))
                sequence = sequences.get(sequence_id)
                parsed = structures.get(sequence_id)
                if sequence is None or parsed is None:
                    skipped_rows.append({"sequence_id": sequence_id, "reason": "missing sequence or structure"})
                    continue
                start, end = _motif_interval(motif_row)
                if start < 0 or end <= start or end > len(parsed.sequence):
                    skipped_rows.append({"sequence_id": sequence_id, "reason": f"invalid motif interval {start}-{end}"})
                    continue
                if parsed.sequence != str(sequence.get("sequence", "")):
                    skipped_rows.append({"sequence_id": sequence_id, "reason": "structure sequence does not match source sequence"})
                    continue
                positions = row_residue_positions(motif_row)
                try:
                    residue_offsets(start, end, positions)
                except ValueError as exc:
                    skipped_rows.append({"sequence_id": sequence_id, "reason": str(exc)})
                    continue
                cache = _get_feature_cache(
                    sequence_id,
                    structures=structures,
                    feature_caches=feature_caches,
                )
                if cache is None:
                    skipped_rows.append({"sequence_id": sequence_id, "reason": "missing feature cache"})
                    continue
                motif_instance_id = _motif_instance_id(motif_row, index)
                motif_cluster = str(motif_row.get("motif_cluster", motif_row.get("motif_id", "")))
                features = _cached_feature_row_for_interval(
                    sequence_id=sequence_id,
                    cache=cache,
                    start=start,
                    end=end,
                    residue_positions=positions,
                    contact_threshold=contact_threshold,
                    long_range_min_separation=long_range_min_separation,
                    low_plddt_cutoff=low_plddt_cutoff,
                    interval_feature_cache=interval_feature_cache,
                )
                row = {
                    "motif_instance_id": motif_instance_id,
                    "sequence_id": sequence_id,
                    "family": motif_row.get("family", sequence.get("family", "")),
                    "motif_id": motif_row.get("motif_id", motif_instance_id),
                    "motif_cluster": motif_cluster,
                    "prediction_backend": "precomputed",
                    "structure_path": str(parsed.source_path),
                    "aligned_core": motif_row.get("aligned_core", ""),
                    "localization_scope": motif_row.get("localization_scope", ""),
                    "sampler_kind": motif_row.get("sampler_kind", ""),
                    "guard_provenance": motif_row.get("guard_provenance", ""),
                    "cluster_score": motif_row.get("cluster_score", ""),
                    "cluster_frequency": motif_row.get("cluster_frequency", motif_row.get("n_support_windows", "")),
                    **features,
                }
                feature_rows.append(row)
                control_batch = _build_controls_for_motif(
                    motif_row=motif_row,
                    motif_instance_id=motif_instance_id,
                    motif_cluster=motif_cluster,
                    sequences=sequences,
                    structures=structures,
                    feature_caches=feature_caches,
                    interval_feature_cache=interval_feature_cache,
                    within_candidate_cache=within_candidate_cache,
                    position_index=position_index,
                    position_control_cache=position_control_cache,
                    composition_control_cache=composition_control_cache,
                    structure_dir=Path(structure_dir),
                    excluded_intervals=excluded,
                    composition_index=composition_index,
                    composition_feature_index=composition_feature_index,
                    num_random_controls=num_random_controls,
                    seed=seed,
                    contact_threshold=contact_threshold,
                    long_range_min_separation=long_range_min_separation,
                    low_plddt_cutoff=low_plddt_cutoff,
                )
                control_writer.writerows(control_batch)
                for control_row in control_batch:
                    control_accumulator.add(control_row)
                    control_audit.add(control_row)
                    if generate_figures and control_row.get("control_type") == "composition_matched":
                        contact_density = _to_float(control_row.get("contact_density"))
                        if contact_density is not None:
                            control_contact_density.append(contact_density)
        control_count = control_writer.count

        write_csv_rows(output_dir / "motif_structure_features.csv", feature_rows, MOTIF_FEATURE_FIELDS)
        write_csv_rows(output_dir / "skipped_motif_instances.csv", skipped_rows)
        _write_progress(
            output_dir,
            stage="features_and_controls_complete",
            family=analysis_family,
            method=analysis_method,
            counts={
                "motif_features": len(feature_rows),
                "control_features": control_count,
                "skipped_motif_instances": len(skipped_rows),
                "structural_tests": 0,
            },
        )

        def record_fresh_rmsd_progress(
            completed_clusters: int,
            total_clusters: int,
            completed_rows: int,
            total_rows: int,
        ) -> None:
            _write_progress(
                output_dir,
                stage="local_rmsd_in_progress",
                family=analysis_family,
                method=analysis_method,
                counts={
                    "motif_features": len(feature_rows),
                    "control_features": control_count,
                    "skipped_motif_instances": len(skipped_rows),
                    "structural_tests": 0,
                    "rmsd_clusters_complete": completed_clusters,
                    "rmsd_clusters_total": total_clusters,
                    "rmsd_rows_complete": completed_rows,
                    "rmsd_rows_total": total_rows,
                },
            )

        _progress_step("Backfilling local RMSD...", show_progress=show_progress)
        backfill_local_rmsd_resumable(
            feature_rows,
            structures,
            output_dir=output_dir,
            workers=rmsd_workers,
            show_progress=show_progress,
            progress_callback=record_fresh_rmsd_progress,
            reuse_checkpoints=False,
        )
        write_csv_rows(output_dir / "motif_structure_features.csv", feature_rows, MOTIF_FEATURE_FIELDS)
        _write_progress(
            output_dir,
            stage="local_rmsd_complete",
            family=analysis_family,
            method=analysis_method,
            counts={
                "motif_features": len(feature_rows),
                "control_features": control_count,
                "skipped_motif_instances": len(skipped_rows),
                "structural_tests": 0,
            },
        )

    endpoints = [field for field in PRIMARY_ENDPOINTS + SECONDARY_ENDPOINTS if any(field in row and row[field] != "" for row in feature_rows)]

    # --- Fix 7: Multi-control-type statistics ---
    stat_rows: list[dict[str, object]] = []
    for ct in ("within_random", "position_matched", "composition_matched"):
        _progress_step(f"Running structural enrichment for {ct} controls...", show_progress=show_progress)
        if not control_accumulator.has_control_type(ct):
            continue
        ct_stats = summarize_structural_enrichment_from_control_accumulator(
            feature_rows,
            control_accumulator,
            endpoints=endpoints,
            control_type=ct,
            seed=seed,
        )
        stat_rows.extend(ct_stats)
        write_csv_rows(output_dir / "structural_enrichment_results.csv", stat_rows)
        _write_progress(
            output_dir,
            stage=f"structural_enrichment_{ct}_complete",
            family=analysis_family,
            method=analysis_method,
            counts={
                "motif_features": len(feature_rows),
                "control_features": control_count,
                "skipped_motif_instances": len(skipped_rows),
                "structural_tests": len(stat_rows),
            },
        )

    assign_primary_q_values(stat_rows)
    write_csv_rows(output_dir / "structural_enrichment_results.csv", stat_rows)
    _write_progress(
        output_dir,
        stage="structural_enrichment_all_complete",
        family=analysis_family,
        method=analysis_method,
        counts={
            "motif_features": len(feature_rows),
            "control_features": control_count,
            "skipped_motif_instances": len(skipped_rows),
            "structural_tests": len(stat_rows),
        },
    )

    _progress_step("Writing outputs...", show_progress=show_progress)
    write_csv_rows(output_dir / "motif_structure_features.csv", feature_rows, MOTIF_FEATURE_FIELDS)
    write_csv_rows(output_dir / "structural_enrichment_results.csv", stat_rows)
    write_csv_rows(output_dir / "skipped_motif_instances.csv", skipped_rows)

    cluster_summary_rows = summarize_motif_clusters(
        feature_rows=feature_rows,
        stat_rows=stat_rows,
    )
    write_csv_rows(output_dir / "motif_cluster_summary.csv", cluster_summary_rows)
    control_diagnostic_rows = control_audit.summaries(feature_rows)
    write_csv_rows(output_dir / "control_diagnostics.csv", control_diagnostic_rows)

    figure_paths: list[Path] = []
    output_warnings: list[str] = []
    if generate_figures:
        _progress_step("Generating publication figures...", show_progress=show_progress)
        figure_paths, output_warnings = generate_publication_figures_from_summaries(
            output_dir=output_dir / "figures",
            feature_rows=feature_rows,
            control_type_counts=control_accumulator.control_type_counts,
            control_contact_density=control_contact_density,
            stat_rows=stat_rows,
            formats=figure_formats,
        )
    if exp3_profile == "pilot":
        output_warnings.append(
            "Exp3 motifs came from the pilot stability profile; use a confirmatory motif set for manuscript conclusions."
        )

    summary = {
        "localization_scopes": sorted({str(r.get('localization_scope')) for r in motif_rows if r.get('localization_scope')}),
        "sampler_kinds": sorted({str(r.get('sampler_kind')) for r in motif_rows if r.get('sampler_kind')}),
        "contains_guarded_sampler_results": any('guard' in str(r.get('sampler_kind','')).lower() for r in motif_rows),
        "biological_function_validated": False,
        "sequence_table": str(sequence_table),
        "motif_table": str(motif_table) if motif_table is not None else "exp3_adapter",
        "structure_dir": str(structure_dir),
        "prediction_backend": prediction_backend,
        "exp3_profile": exp3_profile,
        "exp3_run_signature": exp3_run_signature,
        "predict_missing_structures": bool(predict_missing_structures),
        "output_dir": str(output_dir),
        "motif_instances_requested": len(motif_rows),
        "motif_instances_analyzed": len(feature_rows),
        "control_windows_analyzed": control_count,
        "structural_tests": len(stat_rows),
        "motif_cluster_summaries": len(cluster_summary_rows),
        "control_diagnostic_rows": len(control_diagnostic_rows),
        "publication_figures": len(figure_paths),
        "output_warnings": output_warnings,
        "skipped_motif_instances": len(skipped_rows),
        "primary_endpoints": PRIMARY_ENDPOINTS,
        "primary_control_type": PRIMARY_CONTROL_TYPE,
        "secondary_endpoints": SECONDARY_ENDPOINTS,
    }
    if _structure_inventory(structure_dir, sequence_rows) != signature_payload['structures']:
        raise ValueError('Structure inputs changed during analysis; preserve partial results and use a new output')
    write_analysis_report(
        output_dir / "analysis_report.md",
        summary=summary,
        cluster_rows=cluster_summary_rows,
        figure_paths=[path.relative_to(output_dir) for path in figure_paths],
        warnings=output_warnings,
    )
    write_json(output_dir / "analysis_summary.json", summary)
    _write_progress(
        output_dir,
        stage="complete",
        family=analysis_family,
        method=analysis_method,
        complete=True,
        warnings=output_warnings,
        counts={
            "motif_features": len(feature_rows),
            "control_features": control_count,
            "skipped_motif_instances": len(skipped_rows),
            "structural_tests": len(stat_rows),
            "motif_cluster_summaries": len(cluster_summary_rows),
            "publication_figures": len(figure_paths),
        },
    )
    return summary
