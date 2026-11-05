from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MCS = (10, 20, 30, 40, 60, 80)
DEFAULT_MIN_SAMPLES = (10,)
REPRESENTATIVE_FAMILIES = ("GvpK", "GvpN", "GvpF", "GvpM", "GvpL", "GvpH")
ALL_FAMILIES = (
    "GvpA", "GvpC", "GvpD", "GvpF", "GvpG", "GvpH", "GvpJ", "GvpK",
    "GvpL", "GvpM", "GvpN", "GvpO", "GvpQ", "GvpT", "GvpU", "GvpV",
)


@dataclass(frozen=True, order=True)
class ParameterPair:
    window_size: int
    window_step: int

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.window_step <= 0:
            raise ValueError("window size and step must be positive")

    @property
    def slug(self) -> str:
        return f"w{self.window_size}_s{self.window_step}"

    def as_dict(self) -> dict[str, int]:
        return {"window_size": self.window_size, "window_step": self.window_step}


@dataclass(frozen=True)
class WorkUnit:
    stage: str
    pair: ParameterPair
    family: str
    method: str
    profile: str
    output_root: Path

    @property
    def slug(self) -> str:
        return f"{self.pair.slug}_{self.family}_{self.method}"

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.pair.slug


@dataclass(frozen=True)
class StagePlan:
    stage: str
    pairs: tuple[ParameterPair, ...]
    units: tuple[WorkUnit, ...]
    output_root: Path

