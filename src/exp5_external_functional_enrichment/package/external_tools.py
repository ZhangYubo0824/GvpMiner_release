from __future__ import annotations

from pathlib import Path


def build_interproscan_command(
    *,
    input_fasta: Path,
    output_dir: Path,
    cpu: int = 32,
    interproscan_executable: str = "interproscan.sh",
) -> str:
    return (
        f'{interproscan_executable} '
        f'--input "{input_fasta}" '
        f'--output-dir "{output_dir}" '
        f'--formats TSV '
        f'--seqtype p '
        f'--goterms '
        f'--pathways '
        f'--cpu {cpu}'
    )


def build_eggnog_mapper_command(
    *,
    input_fasta: Path,
    output_dir: Path,
    output_prefix: str = "gvp_eggnog",
    cpu: int = 32,
    mapper_executable: str = "emapper.py",
    search_mode: str = "diamond",
) -> str:
    return (
        f'{mapper_executable} '
        f'-i "{input_fasta}" '
        f'--itype proteins '
        f'-m {search_mode} '
        f'--cpu {cpu} '
        f'--output "{output_prefix}" '
        f'--output_dir "{output_dir}"'
    )


def build_kofamscan_command(
    *,
    input_fasta: Path,
    output_path: Path,
    profile_dir: Path,
    ko_list: Path,
    cpu: int = 32,
    kofamscan_executable: str = "exec_annotation",
) -> str:
    return (
        f'{kofamscan_executable} '
        f'-o "{output_path}" '
        f'-p "{profile_dir}" '
        f'-k "{ko_list}" '
        f'--cpu {cpu} '
        f'"{input_fasta}"'
    )
