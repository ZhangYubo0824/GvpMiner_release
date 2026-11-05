from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .contracts import DEFAULT_MCS, DEFAULT_MIN_SAMPLES, WorkUnit


@dataclass(frozen=True)
class CommandContext:
    project_root: Path
    dataset_csv: Path
    residue_cache_dirs: tuple[Path, ...] = ()
    residue_cache_fallback_dir: Path | None = None
    python_executable: Path = Path(sys.executable)

    @property
    def src_root(self) -> Path:
        return (Path(self.project_root).resolve() / "src").resolve()


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    log_path: Path
    expected_outputs: tuple[Path, ...] = ()


def _python_command(context: CommandContext, script: Path, *args: str) -> tuple[str, ...]:
    return (str(Path(context.python_executable).resolve()), str(script.resolve()), *args)


def raw_exp3_dir(unit: WorkUnit) -> Path:
    return unit.output_dir / "raw_exp3"


def stability_output_root(unit: WorkUnit, *, name: str = "pilot") -> Path:
    return unit.output_dir / name


def _append_residue_cache_args(argv: list[str], context: CommandContext) -> None:
    for cache_dir in context.residue_cache_dirs:
        argv.extend(("--residue-cache-dir", str(Path(cache_dir).resolve())))
    if context.residue_cache_fallback_dir is not None:
        argv.extend((
            "--residue-cache-fallback-dir",
            str(Path(context.residue_cache_fallback_dir).resolve()),
        ))




def build_grouped_exp3_command(
    context: CommandContext,
    unit: WorkUnit,
    *,
    families: tuple[str, ...],
    methods: tuple[str, ...],
) -> CommandSpec:
    output_dir = raw_exp3_dir(unit)
    argv = list(_python_command(
        context,
        context.src_root / "exp3_domain_discovery.py",
        "--dataset-csv", str(Path(context.dataset_csv).resolve()),
        "--output-dir", str(output_dir.resolve()),
        "--window-size", str(unit.pair.window_size),
        "--window-step", str(unit.pair.window_step),
        "--families", *families,
        "--methods", *methods,
    ))
    _append_residue_cache_args(argv, context)
    expected_outputs = tuple(
        path
        for family in families
        for method in methods
        for path in (
            output_dir / family / "windows_metadata.csv",
            output_dir / family / f"embeddings_{method}_optimized.npy",
        )
    )
    return CommandSpec(
        name=f"exp3-{unit.pair.slug}",
        argv=tuple(argv),
        cwd=Path(context.project_root).resolve(),
        log_path=unit.output_dir / "logs" / "exp3_pair.log",
        expected_outputs=expected_outputs,
    )


def _stability_command(
    context: CommandContext,
    unit: WorkUnit,
    *,
    mcs: tuple[int, ...],
    min_samples: int,
    repeats: int,
    seeds: tuple[int, ...],
    null_repeats: int,
    output_name: str,
) -> CommandSpec:
    output_root = stability_output_root(unit, name=output_name)
    argv = list(_python_command(
        context,
        context.src_root / "run_motif_stability.py",
        "--source-exp3-dir", str(raw_exp3_dir(unit).resolve()),
        "--output-root", str(output_root.resolve()),
        "--families", unit.family,
        "--methods", unit.method,
        "--profile", unit.profile,
        "--mcs", *(str(value) for value in mcs),
        "--min-samples", str(min_samples),
        "--repeats", str(repeats),
        "--seeds", *(str(seed) for seed in seeds),
        "--null-repeats", str(null_repeats),
    ))
    return CommandSpec(
        name=f"stability-{unit.slug}-{output_name}",
        argv=tuple(argv),
        cwd=Path(context.project_root).resolve(),
        log_path=unit.output_dir / "logs" / f"stability_{unit.family}_{unit.method}_{output_name}.log",
        expected_outputs=(
            output_root / "motif_stability" / unit.family / unit.method /
            "selected_setting.json",
            output_root / "exp3_domain_discovery" / unit.family /
            f"motif_catalog_{unit.method}.csv",
            output_root / "exp3_domain_discovery" / unit.family /
            f"motif_occurrences_{unit.method}.csv",
        ),
    )


def build_stability_selection_command(
    context: CommandContext,
    unit: WorkUnit,
) -> CommandSpec:
    return _stability_command(
        context,
        unit,
        mcs=DEFAULT_MCS,
        min_samples=DEFAULT_MIN_SAMPLES[0],
        repeats=10,
        seeds=(0,),
        null_repeats=0,
        output_name="pilot",
    )


def build_confirmatory_command(
    context: CommandContext,
    unit: WorkUnit,
    selected_mcs: int,
) -> CommandSpec:
    if selected_mcs not in DEFAULT_MCS:
        raise ValueError(f"selected MCS must be in {DEFAULT_MCS}")
    return _stability_command(
        context,
        unit,
        mcs=(selected_mcs,),
        min_samples=DEFAULT_MIN_SAMPLES[0],
        repeats=50,
        seeds=(0, 1, 2, 3, 4),
        null_repeats=1000,
        output_name="confirmatory",
    )
