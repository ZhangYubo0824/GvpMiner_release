from __future__ import annotations

from typing import Any


__all__ = ["run_exp4_motif_structure_analysis"]


def __getattr__(name: str) -> Any:
    if name == "run_exp4_motif_structure_analysis":
        from .pipeline import run_exp4_motif_structure_analysis

        return run_exp4_motif_structure_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
