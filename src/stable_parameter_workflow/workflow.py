from __future__ import annotations

import csv
import json
import math
import re
import shlex
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from .commands import (
    CommandContext,
    CommandSpec,
    build_confirmatory_command,
    build_grouped_exp3_command,
    build_stability_selection_command,
)
from .contracts import (
    ALL_FAMILIES, ParameterPair, StagePlan, WorkUnit, REPRESENTATIVE_FAMILIES,
)
from .manifests import (
    atomic_write_json,
    can_resume,
    fingerprint_inputs,
    load_manifest,
    resolve_output_root,
    sha256_file,
    unit_signature,
)
from .planning import (
    build_stage_a_plan,
    build_stage_b_plan,
    build_stage_c_plan,
)
from .progress import CommandResult, StreamingCommandRunner, TerminalProgress
from .pilot_reuse import PilotReuseSource, inspect_pilot_source
from .results import (
    annotate_candidate_uncertainty,
    summarize_candidate,
)
from .reuse import (
    archive_interrupted_pilot,
    check_partial_raw_files,
    materialize_files,
    provenance_entry,
    require_safe_parent,
    source_file_hashes,
    verify_reuse_record,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Runner(Protocol):
    def run(self, command: CommandSpec, *, stage: str, current: int,
            total: int) -> CommandResult: ...


class WorkflowExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowContext:
    commands: CommandContext
    output_root: Path
    baseline_roots: tuple[Path, ...] = ()
    resume: bool = False
    force: bool = False
    reuse_stage_a: bool = True
    dry_run: bool = False
    minimum_free_gb: float = 5.0
    reporter: TerminalProgress | None = None
    exp3_workers: int = 1
    stability_workers: int = 1
    confirmatory_workers: int = 1

    def resolved_output_root(self) -> Path:
        return resolve_output_root(self.output_root, baseline_roots=self.baseline_roots)


def _nearest_existing(path: Path) -> Path:
    current = Path(path).resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def preflight(context: WorkflowContext) -> None:
    context.resolved_output_root()
    dataset = Path(context.commands.dataset_csv)
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset CSV not found: {dataset}")
    if context.minimum_free_gb < 0:
        raise ValueError("minimum_free_gb cannot be negative")
    free = shutil.disk_usage(_nearest_existing(context.output_root)).free / (1024 ** 3)
    if free < context.minimum_free_gb:
        raise OSError(
            f"insufficient free disk space: {free:.2f} GiB available, "
            f"{context.minimum_free_gb:.2f} GiB required"
        )


def _unit_manifest(unit: WorkUnit, *, phase: str | None = None) -> Path:
    suffix = f"_{phase}" if phase else ""
    return (
        unit.output_dir / "manifests" /
        f"{unit.family}_{unit.method}{suffix}.json"
    )


def _unit_parameters(unit: WorkUnit) -> dict[str, Any]:
    return {
        "stage": unit.stage,
        "window_size": unit.pair.window_size,
        "window_step": unit.pair.window_step,
        "family": unit.family,
        "method": unit.method,
        "profile": unit.profile,
    }


def _selection_path(unit: WorkUnit) -> Path:
    return (
        unit.output_dir / "pilot" / "motif_stability" /
        unit.family / unit.method / "selected_setting.json"
    )


@dataclass(frozen=True)
class _ParallelTask:
    index: int
    slug: str
    command: CommandSpec
    manifest_path: Path
    signature: str
    parameters: Mapping[str, Any]
    reuse_provenance: Mapping[str, Any] | None = None


def _outputs_exist(command: CommandSpec) -> bool:
    return bool(command.expected_outputs) and all(
        path.is_file() for path in command.expected_outputs
    )


def _pair_outputs_exist(
    plan: StagePlan,
    context: WorkflowContext,
    unit: WorkUnit,
    full_command: CommandSpec,
) -> bool:
    """A reused B pair has ten local outputs and six validated A sources."""
    manifest_path = _pair_manifest(unit)
    if not manifest_path.is_file():
        return False
    manifest = load_manifest(manifest_path)
    if "reused_from_stage_a" not in manifest:
        return _outputs_exist(full_command)
    _verify_reused_manifest(manifest_path, context, unit.output_dir)
    record = manifest["reused_from_stage_a"]
    entries = record["files"]
    source_pair = context.output_root / "stage_a_main_effects" / unit.pair.slug
    expected_dirs = {
        (source_pair / "raw_exp3" / family).resolve()
        for family in REPRESENTATIVE_FAMILIES
    }
    actual_dirs = {Path(entry["source_dir"]).resolve() for entry in entries}
    if (
        record.get("pair") != unit.pair.slug
        or len(entries) != len(REPRESENTATIVE_FAMILIES)
        or actual_dirs != expected_dirs
        or any(entry["target_dir"] is not None for entry in entries)
    ):
        raise WorkflowExecutionError(
            f"Stage A reuse lineage requires all six source families for "
            f"{unit.pair.slug}"
        )
    families = tuple(dict.fromkeys(
        item.family for item in plan.units
        if item.pair == unit.pair and item.family not in REPRESENTATIVE_FAMILIES
    ))
    methods = tuple(dict.fromkeys(
        item.method for item in plan.units if item.pair == unit.pair
    ))
    partial_command = build_grouped_exp3_command(
        context.commands, unit, families=families, methods=methods
    )
    return _outputs_exist(partial_command)


def _verify_reused_pair_handoff(
    manifest_path: Path,
    context: WorkflowContext,
    unit: WorkUnit,
) -> None:
    manifest = load_manifest(manifest_path)
    record = manifest.get("reused_from_stage_a")
    if record is None:
        return
    _verify_reused_manifest(
        manifest_path, context, unit.output_dir, local_only=True,
    )
    source_pair = context.output_root / "stage_a_main_effects" / unit.pair.slug
    expected_dirs = {
        (source_pair / "raw_exp3" / family).resolve()
        for family in REPRESENTATIVE_FAMILIES
    }
    expected_hash_keys = {
        "windows_metadata.csv",
        f"embeddings_{unit.method}_optimized.npy",
    }
    entries = record["files"]
    if (
        record.get("pair") != unit.pair.slug
        or Path(record["source_manifest_path"]).resolve() !=
        (source_pair / "manifests" / "exp3_pair.json").resolve()
        or len(entries) != len(REPRESENTATIVE_FAMILIES)
        or {Path(entry["source_dir"]).resolve() for entry in entries} != expected_dirs
        or any(entry["target_dir"] is not None for entry in entries)
        or any(
            not isinstance(entry.get("sha256"), dict)
            or set(entry["sha256"]) != expected_hash_keys
            for entry in entries
        )
    ):
        raise WorkflowExecutionError(
            f"Stage A reuse lineage requires all six source families for {unit.pair.slug}"
        )


def _verify_reused_manifest(
    manifest_path: Path,
    context: WorkflowContext,
    target_root: Path,
    *,
    local_only: bool = False,
) -> None:
    try:
        payload = load_manifest(manifest_path)
        record = payload.get("reused_from_stage_a")
        if record is not None:
            verify_reuse_record(
                record,
                stage_a_root=context.output_root / "stage_a_main_effects",
                target_root=target_root,
                local_only=local_only,
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise WorkflowExecutionError(
            f"Stage A reuse lineage is invalid for {manifest_path}: {exc}"
        ) from exc


def _verify_reused_pilot_manifest(
    manifest_path: Path,
    context: WorkflowContext,
    unit: WorkUnit,
    *,
    handoff: bool = False,
) -> None:
    _verify_reused_manifest(
        manifest_path, context, unit.output_dir, local_only=handoff,
    )
    record = load_manifest(manifest_path).get("reused_from_stage_a")
    if record is None:
        return
    source_pair = context.output_root / "stage_a_main_effects" / unit.pair.slug
    expected_sources = {
        (source_pair / "pilot" / "motif_stability" /
         unit.family / unit.method).resolve(),
        (source_pair / "pilot" / "exp3_domain_discovery" /
         unit.family).resolve(),
    }
    expected_targets = {
        (unit.output_dir / "pilot" / "motif_stability" /
         unit.family / unit.method).absolute(),
        (unit.output_dir / "pilot" / "exp3_domain_discovery" /
         unit.family).absolute(),
    }
    entries = record["files"]
    if (
        record.get("pair") != unit.pair.slug
        or record.get("family") != unit.family
        or record.get("method") != unit.method
        or Path(record["source_manifest_path"]).resolve() !=
        (source_pair / "manifests" /
         f"{unit.family}_{unit.method}.json").resolve()
        or len(entries) != 2
        or {Path(entry["source_dir"]).resolve() for entry in entries} !=
        expected_sources
        or {Path(entry["target_dir"]).absolute() for entry in entries} !=
        expected_targets
    ):
        raise WorkflowExecutionError(
            f"Stage A reuse lineage pilot source differs for {unit.slug}"
        )
    if handoff:
        pair_record = load_manifest(
            unit.output_dir / "manifests" / "exp3_pair.json"
        ).get("reused_from_stage_a")
        source_pair_manifest = source_pair / "manifests" / "exp3_pair.json"
        required = (
            {"selected_setting.json"},
            {f"motif_catalog_{unit.method}.csv",
             f"motif_occurrences_{unit.method}.csv"},
        )
        if (
            Path(record["source_pair_manifest_path"]).resolve() !=
            source_pair_manifest.resolve()
            or (pair_record is not None and (
                record["source_pair_signature"] != pair_record.get("source_signature")
                or record["source_pair_generation_id"] !=
                pair_record.get("source_generation_id")
            ))
            or any(not names.issubset(entry["sha256"]) for names, entry in zip(required, entries))
        ):
            raise WorkflowExecutionError(
                f"Stage A reuse lineage pilot generation or final files differ for {unit.slug}"
            )


def _stage_b_pilot_copy_valid(manifest_path: Path, context: WorkflowContext,
                              unit: WorkUnit) -> bool:
    """Check an independent B→C pilot copy without reading B raw inputs."""
    try:
        manifest = load_manifest(manifest_path)
        record = manifest.get("reused_from_stage_b")
        if record is None:
            return True
        source_pair = context.output_root / "stage_b_m2_all_families" / unit.pair.slug
        target_pair = unit.output_dir
        expected = (
            (
                source_pair / "pilot" / "motif_stability" / unit.family / unit.method,
                target_pair / "pilot" / "motif_stability" / unit.family / unit.method,
                {"selected_setting.json"},
            ),
            (
                source_pair / "pilot" / "exp3_domain_discovery" / unit.family,
                target_pair / "pilot" / "exp3_domain_discovery" / unit.family,
                {f"motif_catalog_{unit.method}.csv",
                 f"motif_occurrences_{unit.method}.csv"},
            ),
        )
        if (
            record.get("stage") != "B"
            or record.get("pair") != unit.pair.slug
            or record.get("family") != unit.family
            or record.get("method") != unit.method
            or len(record.get("files", [])) != 2
            or Path(record["source_manifest_path"]).resolve() !=
            (source_pair / "manifests" /
             f"{unit.family}_{unit.method}_pilot.json").resolve()
        ):
            return False
        source_pair_manifest = load_manifest(
            source_pair / "manifests" / "exp3_pair.json"
        )
        source_pilot_manifest = load_manifest(
            source_pair / "manifests" /
            f"{unit.family}_{unit.method}_pilot.json"
        )
        if (
            source_pair_manifest.get("status") != "complete"
            or source_pair_manifest.get("signature") !=
            record.get("source_pair_signature")
            or source_pair_manifest.get("generation_id") !=
            record.get("source_pair_generation_id")
            or source_pilot_manifest.get("status") != "complete"
            or source_pilot_manifest.get("signature") !=
            record.get("source_signature")
            or Path(record["source_pair_manifest_path"]).resolve() !=
            (source_pair / "manifests" / "exp3_pair.json").resolve()
        ):
            return False
        for entry, (source_dir, target_dir, required) in zip(
            record["files"], expected
        ):
            if (
                Path(entry["source_dir"]).resolve() != source_dir.resolve()
                or Path(entry["target_dir"]).absolute() != target_dir.absolute()
                or entry.get("reuse_method") != "independent_copy"
                or not required.issubset(entry["sha256"])
            ):
                return False
            for name, digest in entry["sha256"].items():
                path = target_dir / name
                if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                    return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _run_parallel_task(
    task: _ParallelTask,
    *,
    runner: Runner,
    stage_label: str,
    total: int,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    running = {
        "status": "running",
        "signature": task.signature,
        "started_at": _now(),
        "parameters": dict(task.parameters),
        "input_fingerprints": fingerprints,
        "commands": [list(task.command.argv)],
    }
    if task.reuse_provenance is not None:
        running["reused_from_stage_a"] = dict(task.reuse_provenance)
    atomic_write_json(task.manifest_path, running)
    try:
        outcome = runner.run(
            task.command,
            stage=stage_label,
            current=task.index + 1,
            total=total,
        )
    except Exception as exc:
        atomic_write_json(task.manifest_path, {
            **running,
            "status": "failed",
            "failed_at": _now(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        })
        raise WorkflowExecutionError(
            f"{task.slug} failed with {type(exc).__name__}: {exc}"
        ) from exc
    if outcome.returncode != 0:
        atomic_write_json(task.manifest_path, {
            **running,
            "status": "failed",
            "failed_at": _now(),
            "exit_code": int(outcome.returncode),
            "failed_command": list(task.command.argv),
            "log_path": str(outcome.log_path),
        })
        raise WorkflowExecutionError(
            f"{task.slug} failed with exit code {outcome.returncode}; "
            f"see {outcome.log_path}"
        )
    try:
        missing = [
            str(path) for path in task.command.expected_outputs if not path.is_file()
        ]
    except OSError as exc:
        atomic_write_json(task.manifest_path, {
            **running,
            "status": "failed",
            "failed_at": _now(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "log_path": str(outcome.log_path),
        })
        raise WorkflowExecutionError(
            f"{task.slug} output verification failed with "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if missing:
        atomic_write_json(task.manifest_path, {
            **running,
            "status": "failed",
            "failed_at": _now(),
            "exit_code": 0,
            "failed_command": list(task.command.argv),
            "log_path": str(outcome.log_path),
            "missing_outputs": missing,
        })
        raise WorkflowExecutionError(
            f"{task.slug} exited successfully but required outputs are missing: "
            f"{', '.join(missing)}"
        )
    completed = {
        **running,
        "status": "complete",
        "completed_at": _now(),
        "elapsed_seconds": float(outcome.elapsed_seconds),
    }
    if task.parameters.get("phase") == "exp3":
        completed["generation_id"] = uuid.uuid4().hex
    atomic_write_json(task.manifest_path, completed)
    return {"unit": task.slug, "status": "complete", "signature": task.signature}


def _execute_parallel_tasks(
    tasks: list[_ParallelTask],
    *,
    runner: Runner,
    stage_label: str,
    max_workers: int,
    fingerprints: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    first_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _run_parallel_task,
                task,
                runner=runner,
                stage_label=stage_label,
                total=len(tasks),
                fingerprints=fingerprints,
            ): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                results[task.index] = future.result()
            except Exception as exc:  # preserve the first child failure
                if first_error is None:
                    first_error = exc
                    for pending in future_to_task:
                        if pending is not future:
                            pending.cancel()
    if first_error is not None:
        raise first_error
    return [row for row in results if row is not None]


def _pair_unit(plan: StagePlan, pair: ParameterPair) -> WorkUnit:
    return next(unit for unit in plan.units if unit.pair == pair)


def _pair_manifest(unit: WorkUnit) -> Path:
    return unit.output_dir / "manifests" / "exp3_pair.json"


def _pair_parameters(
    unit: WorkUnit,
    *,
    families: tuple[str, ...],
    methods: tuple[str, ...],
    command: CommandSpec,
) -> dict[str, Any]:
    return {
        "stage": unit.stage,
        "window_size": unit.pair.window_size,
        "window_step": unit.pair.window_step,
        "families": list(families),
        "methods": list(methods),
        "phase": "exp3",
        "command_contract": list(command.argv[2:]),
    }


def _pair_generation_identity(
    unit: WorkUnit,
    *,
    expected_signature: str,
) -> dict[str, str]:
    manifest = load_manifest(_pair_manifest(unit))
    generation_id = manifest.get("generation_id")
    if (
        manifest.get("status") != "complete"
        or manifest.get("signature") != expected_signature
        or not isinstance(generation_id, str)
        or not generation_id
    ):
        raise WorkflowExecutionError(
            f"grouped Exp3 generation identity is invalid for {unit.pair.slug}"
        )
    return {
        "signature": expected_signature,
        "generation_id": generation_id,
    }


def _has_pair_generation_identity(
    unit: WorkUnit,
    *,
    expected_signature: str,
) -> bool:
    try:
        _pair_generation_identity(unit, expected_signature=expected_signature)
    except (OSError, ValueError, KeyError, WorkflowExecutionError):
        return False
    return True


def _pilot_parameters(
    unit: WorkUnit,
    command: CommandSpec,
    upstream: Mapping[str, str],
) -> dict[str, Any]:
    return {
        **_unit_parameters(unit),
        "phase": "stability",
        "command_contract": list(command.argv[2:]),
        "upstream_exp3": dict(upstream),
    }


def _upstream_pilot_source(
    context: WorkflowContext,
    unit: WorkUnit,
    fingerprints: Mapping[str, Any],
) -> PilotReuseSource | None:
    """Inspect the matching upstream M2 pilot without its raw embeddings."""
    if unit.method != "M2" or unit.stage not in ("B", "C"):
        return None
    source_stage = "A" if unit.stage == "B" else "B"
    source_root = context.output_root / (
        "stage_a_main_effects" if source_stage == "A" else "stage_b_m2_all_families"
    )
    source_unit = WorkUnit(
        source_stage, unit.pair, unit.family, unit.method, "pilot", source_root,
    )
    families = (REPRESENTATIVE_FAMILIES if source_stage == "A" else ALL_FAMILIES)
    pair_command = build_grouped_exp3_command(
        context.commands, source_unit, families=families, methods=("M2",),
    )
    pair_signature = unit_signature(
        _pair_parameters(
            source_unit, families=families, methods=("M2",),
            command=pair_command,
        ),
        fingerprints,
    )
    source_pair_root = source_root / unit.pair.slug
    try:
        source_pair_manifest = load_manifest(
            source_pair_root / "manifests" / "exp3_pair.json"
        )
        generation_id = source_pair_manifest["generation_id"]
        if not isinstance(generation_id, str) or not generation_id:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    pilot_command = build_stability_selection_command(context.commands, source_unit)
    pilot_signature = unit_signature(
        _pilot_parameters(source_unit, pilot_command, {
            "signature": pair_signature, "generation_id": generation_id,
        }),
        fingerprints,
    )
    source = inspect_pilot_source(
        source_pair_root, family=unit.family, method=unit.method,
        expected_pair_signature=pair_signature,
        expected_pilot_signature=pilot_signature,
        expected_fingerprints=fingerprints,
        pilot_manifest_name=(
            f"{unit.family}_{unit.method}.json" if source_stage == "A" else None
        ),
    )
    if source is None:
        return None
    try:
        decision = load_manifest(decision_path(context.output_root, source_stage))
        if decision.get("stage") != source_stage or unit.pair not in _evaluated_pairs(decision):
            return None
        if decision.get("schema_version") == 2:
            expected = decision["final_output_sha256"]
            required = (
                (source.diagnostic_dir, "selected_setting.json"),
                (source.catalog_dir, f"motif_catalog_{unit.method}.csv"),
                (source.catalog_dir, f"motif_occurrences_{unit.method}.csv"),
            )
            for directory, name in required:
                relative = (directory / name).relative_to(source_root).as_posix()
                actual = (
                    source.diagnostic_hashes if directory == source.diagnostic_dir
                    else source.catalog_hashes
                )[name]
                if expected.get(relative) != actual:
                    return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return source


def execute_stage_a(
    plan: StagePlan,
    context: WorkflowContext,
    *,
    runner: Runner | None = None,
) -> list[dict[str, Any]]:
    if plan.stage != "A":
        raise ValueError("execute_stage_a requires a Stage A plan")
    preflight(context)
    reporter = context.reporter or TerminalProgress()
    command_runner: Runner = runner or StreamingCommandRunner(reporter)
    reporter.emit(
        "A Plan",
        0,
        2,
        "CONFIG",
        f"exp3_workers={context.exp3_workers} "
        f"stability_workers={context.stability_workers}",
    )
    fingerprints = fingerprint_inputs((context.commands.dataset_csv,))
    exp3_rows: list[dict[str, Any]] = []
    exp3_tasks: list[_ParallelTask] = []
    pair_signatures: dict[ParameterPair, str] = {}

    for pair_index, pair in enumerate(plan.pairs):
        unit = _pair_unit(plan, pair)
        families = tuple(dict.fromkeys(
            item.family for item in plan.units if item.pair == pair
        ))
        methods = tuple(dict.fromkeys(
            item.method for item in plan.units if item.pair == pair
        ))
        command = build_grouped_exp3_command(
            context.commands, unit, families=families, methods=methods
        )
        parameters = _pair_parameters(
            unit, families=families, methods=methods, command=command
        )
        signature = unit_signature(parameters, fingerprints)
        pair_signatures[pair] = signature
        manifest_path = _pair_manifest(unit)
        if (
            not context.force
            and can_resume(manifest_path, signature)
            and _outputs_exist(command)
            and _has_pair_generation_identity(
                unit, expected_signature=signature
            )
        ):
            reporter.emit(
                "A Exp3", pair_index + 1, len(plan.pairs), "SKIP",
                f"{pair.slug} signature matched",
            )
            exp3_rows.append({
                "unit": pair.slug, "status": "skipped", "signature": signature,
            })
            continue
        if context.dry_run:
            reporter.emit(
                "A Exp3", pair_index + 1, len(plan.pairs), "DRY-RUN",
                shlex.join(command.argv),
            )
            exp3_rows.append({
                "unit": pair.slug, "status": "dry-run", "signature": signature,
            })
            continue
        exp3_tasks.append(_ParallelTask(
            len(exp3_tasks), pair.slug, command, manifest_path, signature, parameters
        ))

    if exp3_tasks:
        exp3_rows.extend(_execute_parallel_tasks(
            exp3_tasks,
            runner=command_runner,
            stage_label="A Exp3",
            max_workers=context.exp3_workers,
            fingerprints=fingerprints,
        ))
    reporter.emit(
        "A Plan",
        1,
        2,
        "BARRIER",
        f"20 Exp3 pairs {'planned' if context.dry_run else 'complete'}; "
        "stability phase may start",
    )

    stability_rows: list[dict[str, Any]] = []
    stability_tasks: list[_ParallelTask] = []
    for unit_index, unit in enumerate(plan.units):
        command = build_stability_selection_command(context.commands, unit)
        upstream = (
            {
                "signature": pair_signatures[unit.pair],
                "generation_id": "dry-run",
            }
            if context.dry_run
            else _pair_generation_identity(
                _pair_unit(plan, unit.pair),
                expected_signature=pair_signatures[unit.pair],
            )
        )
        parameters = _pilot_parameters(unit, command, upstream)
        signature = unit_signature(parameters, fingerprints)
        manifest_path = _unit_manifest(unit)
        if (
            not context.force
            and can_resume(manifest_path, signature)
            and _outputs_exist(command)
        ):
            reporter.emit(
                "A Stability", unit_index + 1, len(plan.units), "SKIP",
                f"{unit.slug} signature matched",
            )
            stability_rows.append({
                "unit": unit.slug, "status": "skipped", "signature": signature,
            })
            continue
        if context.dry_run:
            reporter.emit(
                "A Stability", unit_index + 1, len(plan.units), "DRY-RUN",
                shlex.join(command.argv),
            )
            stability_rows.append({
                "unit": unit.slug, "status": "dry-run", "signature": signature,
            })
            continue
        stability_tasks.append(_ParallelTask(
            len(stability_tasks), unit.slug, command, manifest_path,
            signature, parameters,
        ))

    if stability_tasks:
        stability_rows.extend(_execute_parallel_tasks(
            stability_tasks,
            runner=command_runner,
            stage_label="A Stability",
            max_workers=context.stability_workers,
            fingerprints=fingerprints,
        ))
    return exp3_rows + stability_rows


def _pilot_unit(unit: WorkUnit) -> WorkUnit:
    return WorkUnit(
        unit.stage,
        unit.pair,
        unit.family,
        unit.method,
        "pilot",
        unit.output_root,
    )


def _execute_grouped_exp3_phase(
    plan: StagePlan,
    context: WorkflowContext,
    *,
    runner: Runner,
    reporter: TerminalProgress,
    fingerprints: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[ParameterPair, str]]:
    rows: list[dict[str, Any]] = []
    tasks: list[_ParallelTask] = []
    pair_signatures: dict[ParameterPair, str] = {}
    for pair_index, pair in enumerate(plan.pairs):
        unit = _pair_unit(plan, pair)
        families = tuple(dict.fromkeys(
            item.family for item in plan.units if item.pair == pair
        ))
        methods = tuple(dict.fromkeys(
            item.method for item in plan.units if item.pair == pair
        ))
        command = build_grouped_exp3_command(
            context.commands, unit, families=families, methods=methods
        )
        parameters = _pair_parameters(
            unit, families=families, methods=methods, command=command
        )
        signature = unit_signature(parameters, fingerprints)
        pair_signatures[pair] = signature
        manifest_path = _pair_manifest(unit)
        stage_a_root = context.output_root / "stage_a_main_effects"
        source_unit = WorkUnit(
            "A", pair, unit.family, "M2", "pilot", stage_a_root,
        )
        source_command = build_grouped_exp3_command(
            context.commands, source_unit,
            families=REPRESENTATIVE_FAMILIES, methods=("M2",),
        )
        source_signature = unit_signature(
            _pair_parameters(
                source_unit, families=REPRESENTATIVE_FAMILIES,
                methods=("M2",), command=source_command,
            ),
            fingerprints,
        )
        reuse_raw = (
            plan.stage == "B" and context.reuse_stage_a and not context.force
            and _has_pair_generation_identity(
                source_unit, expected_signature=source_signature,
            )
            and all(
                (context.output_root / "stage_a_main_effects" / pair.slug /
                 "raw_exp3" / family / name).is_file()
                for family in REPRESENTATIVE_FAMILIES
                for name in (
                    "windows_metadata.csv",
                    f"embeddings_{methods[0]}_optimized.npy",
                )
            )
        )
        try:
            pair_outputs_valid = _pair_outputs_exist(plan, context, unit, command)
        except WorkflowExecutionError:
            pair_outputs_valid = False
        if (
            not context.force
            and can_resume(manifest_path, signature)
            and pair_outputs_valid
            and _has_pair_generation_identity(unit, expected_signature=signature)
        ):
            reporter.emit(
                f"{plan.stage} Exp3", pair_index + 1, len(plan.pairs), "SKIP",
                f"{pair.slug} signature matched",
            )
            rows.append({
                "unit": pair.slug, "status": "skipped", "signature": signature,
            })
            continue
        if context.dry_run:
            if reuse_raw:
                reporter.emit(
                    f"{plan.stage} Exp3", pair_index + 1, len(plan.pairs),
                    "REUSE", f"{pair.slug}: {len(REPRESENTATIVE_FAMILIES)} "
                    "Stage A families; 10 families to calculate",
                )
                remaining = tuple(
                    family for family in families
                    if family not in REPRESENTATIVE_FAMILIES
                )
                command = build_grouped_exp3_command(
                    context.commands, unit, families=remaining, methods=methods
                )
            reporter.emit(
                f"{plan.stage} Exp3", pair_index + 1, len(plan.pairs), "DRY-RUN",
                shlex.join(command.argv),
            )
            rows.append({
                "unit": pair.slug, "status": "dry-run", "signature": signature,
            })
            continue
        reuse_provenance = None
        if reuse_raw:
            source_pair = context.output_root / "stage_a_main_effects" / pair.slug
            entries = []
            for family in REPRESENTATIVE_FAMILIES:
                source_dir = source_pair / "raw_exp3" / family
                target_dir = unit.output_dir / "raw_exp3" / family
                try:
                    hashes = source_file_hashes(source_dir, (
                        "windows_metadata.csv",
                        f"embeddings_{methods[0]}_optimized.npy",
                    ), include_all=False)
                    require_safe_parent(unit.output_dir, target_dir)
                    check_partial_raw_files(target_dir, hashes)
                except (OSError, ValueError) as exc:
                    raise WorkflowExecutionError(
                        f"Stage A Exp3 reuse failed for {pair.slug}/{family}: {exc}"
                    ) from exc
                entries.append(provenance_entry(source_dir, None, hashes))
            source_manifest = load_manifest(
                source_pair / "manifests" / "exp3_pair.json"
            )
            reuse_provenance = {
                "stage": "A", "pair": pair.slug,
                "source_manifest_path": str(
                    (source_pair / "manifests" / "exp3_pair.json").resolve()
                ),
                "source_signature": source_manifest["signature"],
                "source_generation_id": source_manifest["generation_id"],
                "files": entries,
            }
            reporter.emit(
                f"{plan.stage} Exp3", pair_index + 1, len(plan.pairs),
                "REUSE", f"{pair.slug}: reference {len(entries)} Stage A families; "
                f"calculate {len(families) - len(entries)} others",
            )
            remaining = tuple(
                family for family in families
                if family not in REPRESENTATIVE_FAMILIES
            )
            partial_command = build_grouped_exp3_command(
                context.commands, unit, families=remaining, methods=methods
            )
            command = partial_command
        tasks.append(_ParallelTask(
            len(tasks), pair.slug, command, manifest_path, signature, parameters,
            reuse_provenance,
        ))
    if tasks:
        rows.extend(_execute_parallel_tasks(
            tasks,
            runner=runner,
            stage_label=f"{plan.stage} Exp3",
            max_workers=context.exp3_workers,
            fingerprints=fingerprints,
        ))
    return rows, pair_signatures


def _execute_pilot_phase(
    plan: StagePlan,
    context: WorkflowContext,
    *,
    runner: Runner,
    reporter: TerminalProgress,
    fingerprints: Mapping[str, Any],
    pair_signatures: Mapping[ParameterPair, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks: list[_ParallelTask] = []
    pair_upstreams: dict[ParameterPair, dict[str, str]] = {}
    for pair in plan.pairs:
        pair_unit = _pair_unit(plan, pair)
        pair_signature = pair_signatures[pair]
        pair_families = tuple(dict.fromkeys(
            item.family for item in plan.units if item.pair == pair
        ))
        pair_methods = tuple(dict.fromkeys(
            item.method for item in plan.units if item.pair == pair
        ))
        pair_command = build_grouped_exp3_command(
            context.commands, pair_unit,
            families=pair_families, methods=pair_methods,
        )
        pair_complete = (
            can_resume(_pair_manifest(pair_unit), pair_signature)
            and _pair_outputs_exist(plan, context, pair_unit, pair_command)
            and _has_pair_generation_identity(
                pair_unit, expected_signature=pair_signature
            )
        )
        pair_upstreams[pair] = (
            {"signature": pair_signature, "generation_id": "dry-run"}
            if context.dry_run and not pair_complete
            else _pair_generation_identity(
                pair_unit, expected_signature=pair_signature
            )
        )
    for unit_index, source_unit in enumerate(plan.units):
        unit = _pilot_unit(source_unit)
        command = build_stability_selection_command(context.commands, unit)
        upstream = pair_upstreams[unit.pair]
        parameters = _pilot_parameters(unit, command, upstream)
        signature = unit_signature(parameters, fingerprints)
        manifest_path = _unit_manifest(unit, phase="pilot")
        if (
            not context.force
            and can_resume(manifest_path, signature)
            and _outputs_exist(command)
            and (plan.stage != "C" or _stage_b_pilot_copy_valid(
                manifest_path, context, unit,
            ))
        ):
            if plan.stage == "B":
                _verify_reused_pilot_manifest(
                    manifest_path, context, unit, handoff=True,
                )
            reporter.emit(
                f"{plan.stage} Pilot", unit_index + 1, len(plan.units), "SKIP",
                f"{unit.slug} signature matched",
            )
            rows.append({
                "unit": unit.slug, "status": "skipped", "signature": signature,
            })
            continue
        if context.dry_run:
            if plan.stage == "C" and unit.method == "M2" and not context.force:
                source = _upstream_pilot_source(context, unit, fingerprints)
                if source is not None:
                    reporter.emit(
                        f"{plan.stage} Pilot", unit_index + 1, len(plan.units),
                        "REUSE", f"{unit.slug} from Stage B",
                    )
                    rows.append({
                        "unit": unit.slug, "status": "would-reuse",
                        "signature": signature,
                    })
                    continue
            if (
                plan.stage == "B" and context.reuse_stage_a and not context.force
                and unit.family in REPRESENTATIVE_FAMILIES
                and _upstream_pilot_source(context, unit, fingerprints) is not None
            ):
                reporter.emit(
                    f"{plan.stage} Pilot", unit_index + 1, len(plan.units),
                    "REUSE", f"{unit.slug} from Stage A",
                )
                rows.append({
                    "unit": unit.slug, "status": "would-reuse",
                    "signature": signature,
                })
                continue
            reporter.emit(
                f"{plan.stage} Pilot", unit_index + 1, len(plan.units), "DRY-RUN",
                shlex.join(command.argv),
            )
            rows.append({
                "unit": unit.slug, "status": "dry-run", "signature": signature,
            })
            continue
        if plan.stage == "C" and unit.method == "M2" and not context.force:
            source = _upstream_pilot_source(context, unit, fingerprints)
            if source is not None:
                target_diagnostic = (
                    unit.output_dir / "pilot" / "motif_stability" /
                    unit.family / unit.method
                )
                target_catalog = (
                    unit.output_dir / "pilot" / "exp3_domain_discovery" /
                    unit.family
                )
                try:
                    for target in (target_diagnostic, target_catalog, manifest_path):
                        require_safe_parent(unit.output_dir, target)
                    archived = archive_interrupted_pilot(
                        unit.output_dir,
                        (target_diagnostic, target_catalog, manifest_path),
                    )
                    materialize_files(
                        source.diagnostic_dir, target_diagnostic,
                        source.diagnostic_hashes, target_root=unit.output_dir,
                    )
                    materialize_files(
                        source.catalog_dir, target_catalog,
                        source.catalog_hashes, target_root=unit.output_dir,
                    )
                except (OSError, ValueError) as exc:
                    raise WorkflowExecutionError(
                        f"Stage B pilot reuse failed for {unit.slug}: {exc}"
                    ) from exc
                source_manifest = (
                    source.pair_root / "manifests" /
                    f"{unit.family}_{unit.method}_pilot.json"
                )
                provenance = {
                    "stage": "B", "pair": unit.pair.slug,
                    "family": unit.family, "method": unit.method,
                    "source_manifest_path": str(source_manifest.resolve()),
                    "source_signature": source.pilot_signature,
                    "source_pair_manifest_path": str(
                        (source.pair_root / "manifests" / "exp3_pair.json").resolve()
                    ),
                    "source_pair_signature": source.pair_signature,
                    "source_pair_generation_id": source.pair_generation_id,
                    "files": [
                        provenance_entry(
                            source.diagnostic_dir, target_diagnostic,
                            source.diagnostic_hashes,
                        ),
                        provenance_entry(
                            source.catalog_dir, target_catalog,
                            source.catalog_hashes,
                        ),
                    ],
                    "archived_interrupted_paths": [str(path) for path in archived],
                }
                atomic_write_json(manifest_path, {
                    "status": "complete", "signature": signature,
                    "completed_at": _now(), "parameters": parameters,
                    "input_fingerprints": fingerprints, "commands": [],
                    "reused_from_stage_b": provenance,
                })
                reporter.emit(
                    f"{plan.stage} Pilot", unit_index + 1, len(plan.units),
                    "REUSE", f"{unit.slug} from Stage B",
                )
                rows.append({
                    "unit": unit.slug, "status": "reused", "signature": signature,
                })
                continue
        if (
            plan.stage == "B" and context.reuse_stage_a and not context.force
            and unit.family in REPRESENTATIVE_FAMILIES
            and _upstream_pilot_source(context, unit, fingerprints) is not None
        ):
            source_pair = context.output_root / "stage_a_main_effects" / unit.pair.slug
            source_diagnostic = (
                source_pair / "pilot" / "motif_stability" /
                unit.family / unit.method
            )
            target_diagnostic = (
                unit.output_dir / "pilot" / "motif_stability" /
                unit.family / unit.method
            )
            source_catalog = source_pair / "pilot" / "exp3_domain_discovery" / unit.family
            target_catalog = unit.output_dir / "pilot" / "exp3_domain_discovery" / unit.family
            try:
                diagnostic_hashes = source_file_hashes(
                    source_diagnostic, ("selected_setting.json",)
                )
                catalog_hashes = source_file_hashes(source_catalog, (
                    f"motif_catalog_{unit.method}.csv",
                    f"motif_occurrences_{unit.method}.csv",
                ))
                source_manifest = load_manifest(
                    source_pair / "manifests" / f"{unit.family}_{unit.method}.json"
                )
                source_pair_manifest = load_manifest(
                    source_pair / "manifests" / "exp3_pair.json"
                )
                if source_manifest.get("status") != "complete":
                    raise ValueError(f"Stage A pilot is incomplete: {unit.slug}")
                if source_pair_manifest.get("status") != "complete":
                    raise ValueError(f"Stage A Exp3 is incomplete: {unit.pair.slug}")
                require_safe_parent(unit.output_dir, target_diagnostic)
                require_safe_parent(unit.output_dir, target_catalog)
                require_safe_parent(unit.output_dir, manifest_path)
                archived = archive_interrupted_pilot(
                    unit.output_dir,
                    (target_diagnostic, target_catalog, manifest_path),
                )
                materialize_files(
                    source_diagnostic, target_diagnostic, diagnostic_hashes,
                    target_root=unit.output_dir,
                )
                materialize_files(
                    source_catalog, target_catalog, catalog_hashes,
                    target_root=unit.output_dir,
                )
            except (OSError, ValueError, KeyError) as exc:
                raise WorkflowExecutionError(
                    f"Stage A pilot reuse failed for {unit.slug}: {exc}"
                ) from exc
            provenance = {
                "stage": "A", "pair": unit.pair.slug,
                "family": unit.family, "method": unit.method,
                "source_manifest_path": str(
                    (source_pair / "manifests" /
                     f"{unit.family}_{unit.method}.json").resolve()
                ),
                "source_signature": source_manifest["signature"],
                "source_pair_manifest_path": str(
                    (source_pair / "manifests" / "exp3_pair.json").resolve()
                ),
                "source_pair_signature": source_pair_manifest["signature"],
                "source_pair_generation_id": source_pair_manifest["generation_id"],
                "files": [
                    provenance_entry(
                        source_diagnostic, target_diagnostic, diagnostic_hashes
                    ),
                    provenance_entry(source_catalog, target_catalog, catalog_hashes),
                ],
                "archived_interrupted_paths": [str(path) for path in archived],
            }
            atomic_write_json(manifest_path, {
                "status": "complete", "signature": signature,
                "completed_at": _now(), "parameters": parameters,
                "input_fingerprints": fingerprints,
                "commands": [],
                "reused_from_stage_a": provenance,
            })
            reporter.emit(
                f"{plan.stage} Pilot", unit_index + 1, len(plan.units),
                "REUSE", f"{unit.slug} from Stage A",
            )
            rows.append({
                "unit": unit.slug, "status": "reused", "signature": signature,
            })
            continue
        tasks.append(_ParallelTask(
            len(tasks), unit.slug, command, manifest_path, signature, parameters
        ))
    if tasks:
        rows.extend(_execute_parallel_tasks(
            tasks,
            runner=runner,
            stage_label=f"{plan.stage} Pilot",
            max_workers=context.stability_workers,
            fingerprints=fingerprints,
        ))
    return rows


def execute_stage_b(
    plan: StagePlan,
    context: WorkflowContext,
    *,
    runner: Runner | None = None,
) -> list[dict[str, Any]]:
    if plan.stage != "B":
        raise ValueError("execute_stage_b requires a Stage B plan")
    preflight(context)
    _guard_stage_b_layout(plan)
    _guard_stage_b_force_links(plan, context)
    reporter = context.reporter or TerminalProgress()
    command_runner: Runner = runner or StreamingCommandRunner(reporter)
    reporter.emit(
        "B Plan", 0, 2, "CONFIG",
        f"exp3_workers={context.exp3_workers} "
        f"stability_workers={context.stability_workers}",
    )
    fingerprints = fingerprint_inputs((context.commands.dataset_csv,))
    exp3_rows, pair_signatures = _execute_grouped_exp3_phase(
        plan,
        context,
        runner=command_runner,
        reporter=reporter,
        fingerprints=fingerprints,
    )
    reporter.emit(
        "B Plan", 1, 2, "BARRIER",
        f"{len(plan.pairs)} Exp3 pairs "
        f"{'planned' if context.dry_run else 'complete'}; pilot phase may start",
    )
    pilot_rows = _execute_pilot_phase(
        plan,
        context,
        runner=command_runner,
        reporter=reporter,
        fingerprints=fingerprints,
        pair_signatures=pair_signatures,
    )
    return exp3_rows + pilot_rows


def _guard_stage_b_force_links(plan: StagePlan, context: WorkflowContext) -> None:
    for pair in plan.pairs:
        raw_root = plan.output_root / pair.slug / "raw_exp3"
        roots = (
            root
            for family in REPRESENTATIVE_FAMILIES
            for root in (
                raw_root / family,
                plan.output_root / pair.slug / "pilot" /
                "motif_stability" / family / "M2",
                plan.output_root / pair.slug / "pilot" /
                "exp3_domain_discovery" / family,
            )
        )
        if any(
            root.is_symlink() or (
                root.exists() and any(path.is_symlink() for path in root.rglob("*"))
            )
            for root in roots
        ):
            if context.force:
                raise WorkflowExecutionError(
                    "Stage B --force cannot overwrite reused Stage A links"
                )
            raise WorkflowExecutionError(
                "Stage B contains unsafe links; remove them before continuing"
            )


def _guard_stage_b_layout(plan: StagePlan) -> None:
    """Never let a Stage B child process follow a parent or new-family link."""
    if plan.output_root.is_symlink():
        raise WorkflowExecutionError(
            f"Stage B output root must not be a link: {plan.output_root}"
        )
    for pair in plan.pairs:
        pair_root = plan.output_root / pair.slug
        for path in (
            pair_root,
            pair_root / "raw_exp3",
            pair_root / "pilot",
            pair_root / "pilot" / "motif_stability",
            pair_root / "pilot" / "exp3_domain_discovery",
            pair_root / "manifests",
        ):
            if path.is_symlink():
                raise WorkflowExecutionError(
                    f"Stage B parent link is unsafe: {path}"
                )
        for unit in (item for item in plan.units if item.pair == pair):
            if unit.family in REPRESENTATIVE_FAMILIES:
                continue
            family_dir = pair_root / "raw_exp3" / unit.family
            if family_dir.is_symlink() or (
                family_dir.exists() and any(
                    path.is_symlink() for path in family_dir.rglob("*")
                )
            ):
                raise WorkflowExecutionError(
                    f"Stage B new-family link is unsafe: {family_dir}"
                )
            for pilot_dir in (
                pair_root / "pilot" / "motif_stability" / unit.family,
                pair_root / "pilot" / "exp3_domain_discovery" / unit.family,
            ):
                if pilot_dir.is_symlink() or (
                    pilot_dir.exists() and any(
                        path.is_symlink() for path in pilot_dir.rglob("*")
                    )
                ):
                    raise WorkflowExecutionError(
                        f"Stage B new-family pilot link is unsafe: {pilot_dir}"
                    )


def _confirmatory_parameters(
    unit: WorkUnit,
    command: CommandSpec,
    upstream: Mapping[str, str],
    *,
    selected_mcs: int,
) -> dict[str, Any]:
    return {
        **_unit_parameters(unit),
        "phase": "confirmatory",
        "selected_mcs": selected_mcs,
        "command_contract": list(command.argv[2:]),
        "upstream_exp3": dict(upstream),
    }


def _execute_confirmatory_phase(
    plan: StagePlan,
    context: WorkflowContext,
    *,
    runner: Runner,
    reporter: TerminalProgress,
    fingerprints: Mapping[str, Any],
    pair_signatures: Mapping[ParameterPair, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks: list[_ParallelTask] = []
    for unit_index, unit in enumerate(plan.units):
        if context.dry_run:
            reporter.emit(
                "C Confirmatory", unit_index + 1, len(plan.units), "PENDING",
                f"{unit.slug} selected MCS will be read from the completed pilot",
            )
            rows.append({"unit": unit.slug, "status": "pending-pilot"})
            continue
        selection_path = _selection_path(unit)
        if not selection_path.is_file():
            raise WorkflowExecutionError(
                f"pilot selection missing for {unit.slug}: {selection_path}"
            )
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_mcs = int(selected["min_cluster_size"])
        command = build_confirmatory_command(
            context.commands, unit, selected_mcs=selected_mcs
        )
        upstream = _pair_generation_identity(
            _pair_unit(plan, unit.pair),
            expected_signature=pair_signatures[unit.pair],
        )
        parameters = _confirmatory_parameters(
            unit, command, upstream, selected_mcs=selected_mcs
        )
        signature = unit_signature(parameters, fingerprints)
        manifest_path = _unit_manifest(unit, phase="confirmatory")
        if (
            not context.force
            and can_resume(manifest_path, signature)
            and _outputs_exist(command)
        ):
            reporter.emit(
                "C Confirmatory", unit_index + 1, len(plan.units), "SKIP",
                f"{unit.slug} signature matched",
            )
            rows.append({
                "unit": unit.slug, "status": "skipped", "signature": signature,
            })
            continue
        tasks.append(_ParallelTask(
            len(tasks), unit.slug, command, manifest_path, signature, parameters
        ))
    if tasks:
        rows.extend(_execute_parallel_tasks(
            tasks,
            runner=runner,
            stage_label="C Confirmatory",
            max_workers=context.confirmatory_workers,
            fingerprints=fingerprints,
        ))
    return rows


def execute_stage_c(
    plan: StagePlan,
    context: WorkflowContext,
    *,
    runner: Runner | None = None,
) -> list[dict[str, Any]]:
    if plan.stage != "C":
        raise ValueError("execute_stage_c requires a Stage C plan")
    preflight(context)
    reporter = context.reporter or TerminalProgress()
    command_runner: Runner = runner or StreamingCommandRunner(reporter)
    reporter.emit(
        "C Plan", 0, 3, "CONFIG",
        f"exp3_workers={context.exp3_workers} "
        f"stability_workers={context.stability_workers} "
        f"confirmatory_workers={context.confirmatory_workers}",
    )
    fingerprints = fingerprint_inputs((context.commands.dataset_csv,))
    exp3_rows, pair_signatures = _execute_grouped_exp3_phase(
        plan,
        context,
        runner=command_runner,
        reporter=reporter,
        fingerprints=fingerprints,
    )
    reporter.emit(
        "C Plan", 1, 3, "BARRIER",
        f"{len(plan.pairs)} Exp3 pairs "
        f"{'planned' if context.dry_run else 'complete'}; pilot phase may start",
    )
    pilot_rows = _execute_pilot_phase(
        plan,
        context,
        runner=command_runner,
        reporter=reporter,
        fingerprints=fingerprints,
        pair_signatures=pair_signatures,
    )
    reporter.emit(
        "C Plan", 2, 3, "BARRIER",
        f"{len(plan.units)} pilot selections "
        f"{'planned' if context.dry_run else 'complete'}; "
        "confirmatory phase may start",
    )
    confirmatory_rows = _execute_confirmatory_phase(
        plan,
        context,
        runner=command_runner,
        reporter=reporter,
        fingerprints=fingerprints,
        pair_signatures=pair_signatures,
    )
    return exp3_rows + pilot_rows + confirmatory_rows


def require_decision(path: Path, *, prior_stage: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"complete Stage {prior_stage} first; decision file is missing: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid decision file: {path}")
    return payload


def decision_path(output_root: Path, stage: str) -> Path:
    return Path(output_root) / "comparisons" / f"stage_{stage.lower()}_decision.json"


def _selected_rows(
    stage_roots: tuple[Path, ...],
    pair: ParameterPair,
    *,
    phase: str = "pilot",
) -> list[dict[str, Any]]:
    rows = []
    for root in stage_roots:
        pair_root = root / pair.slug
        for path in pair_root.glob(f"{phase}/motif_stability/*/*/selected_setting.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = dict(payload)
            row["family"] = path.parent.parent.name
            row["method"] = path.parent.name
            rows.append(row)
    return rows


def _completed_stage_lineage(
    plan: StagePlan, context: WorkflowContext, *, handoff: bool = False,
    allow_unavailable_pilot: bool = False,
) -> str:
    """Validate completed manifests and required outputs for this purpose."""
    fingerprints = fingerprint_inputs((context.commands.dataset_csv,))
    pair_signatures: dict[ParameterPair, str] = {}
    generations: list[dict[str, str]] = []
    unit_signatures: list[dict[str, str]] = []
    for pair in plan.pairs:
        unit = _pair_unit(plan, pair)
        families = tuple(dict.fromkeys(
            item.family for item in plan.units if item.pair == pair
        ))
        methods = tuple(dict.fromkeys(
            item.method for item in plan.units if item.pair == pair
        ))
        command = build_grouped_exp3_command(
            context.commands, unit, families=families, methods=methods
        )
        signature = unit_signature(
            _pair_parameters(
                unit,
                families=families,
                methods=methods,
                command=command,
            ),
            fingerprints,
        )
        pair_signatures[pair] = signature
        if (
            not can_resume(_pair_manifest(unit), signature)
            or not _has_pair_generation_identity(
                unit, expected_signature=signature
            )
            or (not handoff and not _pair_outputs_exist(plan, context, unit, command))
        ):
            raise WorkflowExecutionError(
                f"Stage {plan.stage} has incomplete grouped Exp3 outputs "
                f"for {pair.slug}"
            )
        if handoff and plan.stage == "B":
            _verify_reused_pair_handoff(_pair_manifest(unit), context, unit)
        generation = _pair_generation_identity(
            unit, expected_signature=signature
        )
        generations.append({"pair": pair.slug, **generation})
    incomplete_units = []
    for unit in plan.units:
        pilot_unit = _pilot_unit(unit)
        pilot_command = build_stability_selection_command(
            context.commands, pilot_unit
        )
        upstream = _pair_generation_identity(
            _pair_unit(plan, unit.pair),
            expected_signature=pair_signatures[unit.pair],
        )
        pilot_parameters = _pilot_parameters(
            pilot_unit, pilot_command, upstream
        )
        pilot_manifest = _unit_manifest(
            unit, phase=None if plan.stage == "A" else "pilot"
        )
        pilot_signature = unit_signature(pilot_parameters, fingerprints)
        pilot_complete = (
            can_resume(pilot_manifest, pilot_signature)
            and _outputs_exist(pilot_command)
            and (plan.stage != "C" or _stage_b_pilot_copy_valid(
                pilot_manifest, context, unit,
            ))
        )
        if not pilot_complete and not allow_unavailable_pilot:
            incomplete_units.append(f"{unit.slug}:pilot")
        elif pilot_complete and plan.stage == "B" and not allow_unavailable_pilot:
            _verify_reused_pilot_manifest(
                pilot_manifest, context, unit, handoff=handoff,
            )
        unit_signatures.append({
            "unit": unit.slug, "phase": "pilot", "signature": pilot_signature,
        })
        if plan.stage == "C":
            selection_path = _selection_path(unit)
            if not selection_path.is_file():
                incomplete_units.append(f"{unit.slug}:selection")
                continue
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            selected_mcs = int(selected["min_cluster_size"])
            confirm_command = build_confirmatory_command(
                context.commands, unit, selected_mcs=selected_mcs
            )
            confirm_parameters = _confirmatory_parameters(
                unit, confirm_command, upstream, selected_mcs=selected_mcs
            )
            confirm_signature = unit_signature(
                confirm_parameters, fingerprints
            )
            if (
                not can_resume(
                    _unit_manifest(unit, phase="confirmatory"),
                    confirm_signature,
                )
                or not _outputs_exist(confirm_command)
            ):
                incomplete_units.append(f"{unit.slug}:confirmatory")
            unit_signatures.append({
                "unit": unit.slug, "phase": "confirmatory",
                "signature": confirm_signature,
            })
    if incomplete_units:
        raise WorkflowExecutionError(f"Stage {plan.stage} has incomplete units")
    return unit_signature({
        "stage": plan.stage,
        "pairs": [pair.as_dict() for pair in plan.pairs],
        "generations": generations,
        "unit_signatures": unit_signatures,
    }, fingerprints)


def _final_output_paths(plan: StagePlan, context: WorkflowContext) -> dict[str, Path]:
    """Enumerate every final selected setting and motif file for a stage."""
    paths: dict[str, Path] = {}
    for unit in plan.units:
        commands = [build_stability_selection_command(
            context.commands, _pilot_unit(unit)
        )]
        if plan.stage == "C":
            selected = json.loads(_selection_path(unit).read_text(encoding="utf-8"))
            commands.append(build_confirmatory_command(
                context.commands, unit,
                selected_mcs=int(selected["min_cluster_size"]),
            ))
        for command in commands:
            for path in command.expected_outputs:
                paths[path.relative_to(plan.output_root).as_posix()] = path
    return dict(sorted(paths.items()))


def _validate_candidate_metrics(
    payload: Mapping[str, Any], evaluated: tuple[ParameterPair, ...],
) -> None:
    rows = payload.get("candidate_metrics")
    if not isinstance(rows, list) or len(rows) != len(evaluated):
        raise ValueError("candidate_metrics must cover every evaluated pair")
    numeric_fields = (
        "family_count", "qualified_family_count", "qualified_family_fraction",
        "stability_floor", "stable_protein_coverage", "stable_window_coverage",
        "noise_ratio", "stable_cluster_number", "runtime_seconds",
    )
    if payload["stage"] == "A":
        numeric_fields += (
            "family_compared_count", "family_improved_count",
            "family_worsened_count", "interaction_reversal_count",
        )
    found = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("pair"), dict):
            raise ValueError("candidate_metrics row is malformed")
        coordinates = row["pair"]
        if any(
            type(coordinates.get(key)) is not int
            for key in ("window_size", "window_step")
        ):
            raise ValueError("candidate_metrics pair coordinates must be integers")
        pair = ParameterPair(
            coordinates["window_size"], coordinates["window_step"]
        )
        if (
            row.get("window_size") != pair.window_size
            or row.get("window_step") != pair.window_step
        ):
            raise ValueError("candidate_metrics pair coordinates differ")
        if any(
            type(row.get(key)) is not int
            for key in ("window_size", "window_step")
        ):
            raise ValueError("candidate_metrics pair coordinates must be integers")
        if any(
            isinstance(row.get(key), bool)
            or not isinstance(row.get(key), (int, float))
            or not math.isfinite(float(row[key]))
            for key in numeric_fields
        ):
            raise ValueError("candidate_metrics numeric values are incomplete")
        found.append(pair)
    if len(set(found)) != len(found) or set(found) != set(evaluated):
        raise ValueError(
            "candidate_metrics pair coverage differs from evaluated pairs"
        )


def _validate_final_outputs(paths: Mapping[str, Path]) -> None:
    for relative, path in paths.items():
        if path.name == "selected_setting.json":
            selected = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(selected, dict):
                raise ValueError(f"final selected setting is not an object: {relative}")
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for _ in csv.reader(handle, strict=True):
                    pass


def finalize_stage(
    plan: StagePlan, context: WorkflowContext,
    *, review_record: Path | None = None,
    selection_source: str = "manual_review",
) -> Path:
    if context.dry_run:
        raise ValueError("dry runs cannot finalize a stage")
    lineage_signature = _completed_stage_lineage(plan, context)

    roots = (plan.output_root,)
    phase = "confirmatory" if plan.stage == "C" else "pilot"
    summaries = []
    rows_by_pair: dict[ParameterPair, list[dict[str, Any]]] = {}
    for pair in plan.pairs:
        rows = _selected_rows(roots, pair, phase=phase)
        if rows:
            rows_by_pair[pair] = rows
            summaries.append(summarize_candidate(pair, rows))
    if not summaries:
        raise WorkflowExecutionError(f"Stage {plan.stage} produced no selected settings")
    if plan.stage == "A":
        summaries = annotate_candidate_uncertainty(summaries, rows_by_pair)
    ordered = sorted(
        summaries,
        key=lambda row: (row["pair"].window_size, row["pair"].window_step),
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "stage": plan.stage,
        "completed_at": _now(),
        "evaluated_pairs": [pair.as_dict() for pair in plan.pairs],
        "lineage_signature": lineage_signature,
        "selection_mode": "manual_required",
        "candidate_metrics": [_json_candidate(row) for row in ordered],
        "final_output_sha256": {
            relative: sha256_file(path)
            for relative, path in _final_output_paths(plan, context).items()
        },
    }
    if review_record is not None:
        payload["selection_source"] = selection_source
        payload["review_record"] = str(
            review_record.relative_to(context.output_root)
        ).replace("\\", "/")
        payload["review_record_sha256"] = sha256_file(review_record)
    if plan.stage == "C":
        payload["status"] = "motif_discovery_complete_review_required"
    target = decision_path(context.output_root, plan.stage)
    atomic_write_json(target, payload)
    return target


def _json_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    pair = result.get("pair")
    if isinstance(pair, ParameterPair):
        result["pair"] = pair.as_dict()
    return result


def _evaluated_pairs(payload: Mapping[str, Any]) -> tuple[ParameterPair, ...]:
    return tuple(
        ParameterPair(int(row["window_size"]), int(row["window_step"]))
        for row in payload["evaluated_pairs"]
    )


def _require_current_decision(
    context: WorkflowContext, prior_stage: str, *,
    allow_unavailable_pilot: bool = False,
) -> dict[str, Any]:
    payload = require_decision(
        decision_path(context.output_root, prior_stage),
        prior_stage=prior_stage,
    )
    try:
        evaluated = _evaluated_pairs(payload)
        if payload["stage"] != prior_stage:
            raise ValueError("stage label does not match")
        if not evaluated or len(evaluated) != len(set(evaluated)):
            raise ValueError("evaluated pairs are missing or duplicated")
        if prior_stage == "A":
            plan = build_stage_a_plan(context.output_root)
            if evaluated != plan.pairs:
                raise ValueError("Stage A evaluated pair grid does not match")
        elif prior_stage == "B":
            plan = build_stage_b_plan(context.output_root, evaluated)
        else:
            plan = build_stage_c_plan(context.output_root, evaluated)
        lineage = _completed_stage_lineage(
            plan, context, handoff=True,
            allow_unavailable_pilot=allow_unavailable_pilot,
        )
        if payload["lineage_signature"] != lineage:
            raise ValueError("completed generation differs from decision")
        paths = _final_output_paths(plan, context)
        version = payload.get("schema_version")
        if version == 2 and not isinstance(version, bool):
            digests = payload.get("final_output_sha256")
            if not isinstance(digests, dict) or set(digests) != set(paths):
                raise ValueError(
                    "final_output_sha256 path set differs from completed stage"
                )
            for relative, path in paths.items():
                digest = digests[relative]
                if (
                    not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    raise ValueError(f"invalid final_output_sha256 for {relative}")
                if not allow_unavailable_pilot and sha256_file(path) != digest:
                    raise ValueError(f"final output hash differs: {relative}")
            _validate_candidate_metrics(payload, evaluated)
        elif version is None and "schema_version" not in payload:
            _validate_final_outputs(
                {key: path for key, path in paths.items() if path.is_file()}
                if allow_unavailable_pilot else paths
            )
        else:
            raise ValueError(f"unsupported decision schema_version: {version}")
        if prior_stage in ("B", "C"):
            selection_source = payload.get("selection_source")
            if selection_source not in ("manual_review", "algorithmic_selection",
                                        "algorithmic_reissue"):
                raise ValueError("Stage B/C selection_source must be manual_review, algorithmic_selection, or algorithmic_reissue")
            relative = Path(str(payload["review_record"]))
            if (
                relative.parent != Path("comparisons")
                or not relative.name.startswith(f"stage_{prior_stage.lower()}_review_")
                or relative.suffix != ".json"
            ):
                raise ValueError("review record path is invalid")
            record_path = context.output_root / relative
            if sha256_file(record_path) != payload["review_record_sha256"]:
                raise ValueError("review record hash differs from stage decision")
            record = load_manifest(record_path)
            if record.get("reviewed_pairs") != payload["evaluated_pairs"]:
                raise ValueError("reviewed pairs differ from evaluated pairs")
            source_stage = "A" if prior_stage == "B" else "B"
            source = decision_path(context.output_root, source_stage)
            source_hash = record.get("source_decision_sha256")
            if source_hash != sha256_file(source):
                raise ValueError("review source stage decision has changed")
            if selection_source in ("algorithmic_selection", "algorithmic_reissue"):
                recommendation = record.get("recommendation")
                if (
                    not isinstance(recommendation, dict)
                    or recommendation.get("source_stage") != source_stage
                    or recommendation.get("source_decision_sha256") != source_hash
                    or not isinstance(recommendation.get("rule_version"), str)
                ):
                    raise ValueError("algorithmic recommendation record is invalid")
                recommended = recommendation.get("selected_pairs")
                if selection_source == "algorithmic_selection":
                    if recommended != payload["evaluated_pairs"]:
                        raise ValueError("algorithmic recommendation pairs differ from executed pairs")
                else:
                    if record.get("selection_source") != "algorithmic_reissue":
                        raise ValueError("reissued selection source is inconsistent")
                    if payload.get("selection_rule_version") != recommendation["rule_version"]:
                        raise ValueError("reissued selection rule version differs")
                    pairs = [ParameterPair(int(item["window_size"]), int(item["window_step"]))
                             for item in recommended]
                    if len(pairs) != len(set(pairs)) or set(pairs) != set(evaluated):
                        raise ValueError("reissued recommendation pairs differ from executed pairs")
    except (
        KeyError, TypeError, ValueError, OverflowError, OSError,
        UnicodeError, csv.Error, WorkflowExecutionError,
    ) as exc:
        raise WorkflowExecutionError(
            f"Stage {prior_stage} decision lineage is invalid: {exc}"
        ) from exc
    return payload


def _archive_decisions_for_rerun(output_root: Path, stage: str) -> None:
    remaining = ("A", "B", "C")[("A", "B", "C").index(stage):]
    # Archive downstream first: an interrupted archive cannot leave a usable
    # downstream decision after the current stage decision has been moved.
    for candidate in reversed(remaining):
        target = decision_path(output_root, candidate)
        if not target.is_file():
            continue
        archive = target.with_name(
            f"{target.stem}.superseded_{uuid.uuid4().hex}.json"
        )
        target.replace(archive)


def _write_selection_review(
    context: WorkflowContext,
    stage: str,
    pairs: tuple[ParameterPair, ...],
    *, recommendation: Mapping[str, Any] | None = None,
) -> Path:
    source_stage = "A" if stage == "B" else "B"
    source_hash = sha256_file(decision_path(context.output_root, source_stage))
    fields = {
        "stage": stage,
        "source_decision_sha256": source_hash,
        "reviewed_pairs": [pair.as_dict() for pair in pairs],
    }
    if recommendation is not None:
        fields["selection_source"] = "algorithmic_selection"
        fields["recommendation"] = dict(recommendation)
    digest = unit_signature(fields, {})[:12]
    directory = context.output_root / "comparisons"
    prefix = f"stage_{stage.lower()}_review_{digest}_"
    matches = list(directory.glob(f"{prefix}*.json"))
    if len(matches) > 1:
        raise WorkflowExecutionError("multiple review records match this choice")
    if matches:
        path = matches[0]
        existing = load_manifest(path)
        expected_digest = path.stem[len(prefix):]
        if (
            any(existing.get(key) != value for key, value in fields.items())
            or not isinstance(existing.get("created_at"), str)
            or unit_signature(existing, {})[:16] != expected_digest
        ):
            raise WorkflowExecutionError(f"selection review record conflicts: {path}")
        return path
    payload = {**fields, "created_at": _now()}
    content_digest = unit_signature(payload, {})[:16]
    path = directory / f"{prefix}{content_digest}.json"
    atomic_write_json(path, payload)
    return path


def _validate_selection_report(
    context: WorkflowContext,
    target_stage: str,
    pairs: tuple[ParameterPair, ...],
    report: Mapping[str, Any],
) -> None:
    source_stage = {"B": "A", "C": "B"}[target_stage]
    source = decision_path(context.output_root, source_stage)
    if (
        report.get("source_stage") != source_stage
        or report.get("source_decision_sha256") != sha256_file(source)
    ):
        raise WorkflowExecutionError("algorithmic selection source decision has changed")
    if report.get("selected_pairs") != [pair.as_dict() for pair in pairs]:
        raise WorkflowExecutionError("algorithmic selection differs from requested pairs")
    if not isinstance(report.get("rule_version"), str):
        raise WorkflowExecutionError("algorithmic selection has no rule version")
    from .advancement import recommend_decision_file

    try:
        expected = recommend_decision_file(source)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise WorkflowExecutionError(
            f"computed recommendation is unavailable: {exc}"
        ) from exc
    if dict(report) != expected:
        raise WorkflowExecutionError(
            "algorithmic selection differs from computed recommendation"
        )


def run_workflow(
    stage: str,
    context: WorkflowContext,
    *,
    runner: Runner | None = None,
    reviewed_pairs: tuple[ParameterPair, ...] | None = None,
    stage_c_pairs: tuple[ParameterPair, ...] | None = None,
    selection_report: Mapping[str, Any] | None = None,
) -> Path | None:
    stage = stage.upper()
    if stage not in ("A", "B", "C"):
        raise ValueError("stage must be A, B, or C")
    if reviewed_pairs is not None and stage != "B":
        raise ValueError("reviewed pairs are supported only for Stage B")
    if stage_c_pairs is not None and stage != "C":
        raise ValueError("Stage C pairs are supported only for Stage C")
    if selection_report is not None and stage not in ("B", "C"):
        raise ValueError("algorithmic selection applies only to Stage B or C")
    if stage == "A":
        plan = build_stage_a_plan(context.output_root)
    elif stage == "B":
        if reviewed_pairs is None:
            raise ValueError("Stage B requires selected pairs from the automatic selector")
        prior = _require_current_decision(
            context, "A", allow_unavailable_pilot=True,
        )
        pairs = tuple(reviewed_pairs)
        evaluated = set(_evaluated_pairs(prior))
        if (
            len(pairs) not in (2, 3)
            or len(set(pairs)) != len(pairs)
            or ParameterPair(30, 5) not in pairs
            or not set(pairs).issubset(evaluated)
        ):
            raise WorkflowExecutionError(
                "Stage B pairs must be two or three distinct Stage A "
                "evaluated pairs including baseline 30:5"
            )
        plan = build_stage_b_plan(context.output_root, pairs)
    elif stage == "C":
        if stage_c_pairs is None:
            raise ValueError("Stage C requires selected pairs from the automatic selector")
        prior = _require_current_decision(
            context, "B", allow_unavailable_pilot=True,
        )
        pairs = tuple(stage_c_pairs)
        if (
            len(pairs) != 2
            or len(set(pairs)) != 2
            or not set(pairs).issubset(_evaluated_pairs(prior))
        ):
            raise WorkflowExecutionError(
                "Stage C pairs must be two distinct Stage B evaluated pairs"
            )
        plan = build_stage_c_plan(context.output_root, pairs)
    if selection_report is not None:
        _validate_selection_report(context, stage, plan.pairs, selection_report)

    if not context.dry_run:
        existing = decision_path(context.output_root, stage)
        if existing.is_file() and not context.force:
            try:
                completed = _require_current_decision(context, stage)
            except WorkflowExecutionError:
                pass
            else:
                completed_pairs = _evaluated_pairs(completed)
                if (completed_pairs == plan.pairs or
                        (selection_report is not None and
                         set(completed_pairs) == set(plan.pairs))):
                    if selection_report is not None:
                        _write_selection_review(
                            context, stage, plan.pairs,
                            recommendation=selection_report,
                        )
                    return existing
        preflight(context)
        if stage == "B":
            _guard_stage_b_layout(plan)
            _guard_stage_b_force_links(plan, context)
        _archive_decisions_for_rerun(context.output_root, stage)

    review_record = None
    if stage in ("B", "C") and not context.dry_run:
        review_record = _write_selection_review(
            context, stage, plan.pairs, recommendation=selection_report,
        )

    if stage == "A":
        execute_stage_a(plan, context, runner=runner)
    elif stage == "B":
        execute_stage_b(plan, context, runner=runner)
    else:
        execute_stage_c(plan, context, runner=runner)
    if context.dry_run:
        return None
    return finalize_stage(
        plan, context, review_record=review_record,
        selection_source=("algorithmic_selection" if selection_report is not None
                          else "manual_review"),
    )
