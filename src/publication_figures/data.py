"""Validated readers and reusable summaries for publication figures."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def resolve_results_dir(explicit: Path | None, *, project_root: Path) -> Path:
    """Resolve an explicit result directory or the repository's known defaults."""
    candidates = (
        [Path(explicit)]
        if explicit is not None
        else [Path(project_root) / "results"]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "No results directory found. Pass --results-dir explicitly."
    )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    source: Path,
) -> None:
    """Raise a useful error when an input table lacks required columns."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def read_csv(path: Path, required: Iterable[str] = ()) -> pd.DataFrame:
    """Read a CSV and validate its required columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(frame, required, path)
    return frame


def load_exp2_summary(
    path: Path,
    supported_families: set[str],
) -> pd.DataFrame:
    """Flatten Exp2 JSON and distinguish unsupported labels from failures."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for item in payload:
        family = str(item["target_family"])
        status = (
            "supported"
            if family in supported_families
            else "outside_classifier_label_space"
        )
        for method, summary in item.get("method_summaries", {}).items():
            rows.append(
                {
                    "family": family,
                    "method": str(method),
                    "num_generated": int(item.get("num_generated", 0)),
                    "predicted_target_ratio": float(
                        summary.get("predicted_target_ratio", np.nan)
                    ),
                    "mean_target_probability": float(
                        summary.get("mean_target_probability", np.nan)
                    ),
                    "evaluation_status": status,
                }
            )
    return pd.DataFrame(rows)


def standardize_structural_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Robustly standardize effects within each structural endpoint."""
    result = frame.copy()

    def scale(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        center = numeric.median()
        deviations = (numeric - center).abs()
        mad = deviations.median()
        if not np.isfinite(mad) or mad == 0:
            max_abs = deviations.max()
            denominator = max_abs if np.isfinite(max_abs) and max_abs > 0 else 1.0
        else:
            denominator = 1.4826 * mad
        return ((numeric - center) / denominator).clip(-3, 3) / 3

    result["standardized_effect"] = result.groupby(
        "feature",
        group_keys=False,
    )["effect_median_difference"].transform(scale)
    return result


def normalize_term(value: object) -> str:
    """Normalize whitespace and separators for duplicate term detection."""
    return re.sub(r"[_\s]+", " ", str(value).strip()).casefold()


def collapse_function_terms(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate motif-term rows into family-method support summaries."""
    work = frame.copy()
    require_columns(
        work,
        ["family", "method", "motif_cluster", "term", "adjusted_p"],
        Path("<dataframe>"),
    )
    work["normalized_term"] = work["term"].map(normalize_term)
    work["adjusted_p"] = pd.to_numeric(work["adjusted_p"], errors="coerce")
    work = work.dropna(subset=["adjusted_p"])
    work = work.sort_values("adjusted_p").drop_duplicates(
        ["family", "method", "motif_cluster", "normalized_term"]
    )
    grouped = (
        work.groupby(["family", "method", "normalized_term"], as_index=False)
        .agg(
            term=("term", "first"),
            n_motif_clusters=("motif_cluster", "nunique"),
            min_adjusted_p=("adjusted_p", "min"),
        )
    )
    grouped["minus_log10_q"] = (
        -np.log10(grouped["min_adjusted_p"].clip(lower=1e-300))
    ).clip(upper=50)
    return grouped
