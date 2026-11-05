from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .selection import select_parameter_setting


_INTEGER_FIELDS = {
    "min_cluster_size",
    "min_samples",
    "raw_cluster_number",
    "stable_cluster_number",
}

_FLOAT_FIELDS = {
    "median_window_jaccard",
    "median_protein_jaccard",
    "median_occurrence_jaccard",
    "noise_ratio",
    "stable_window_coverage",
    "stable_protein_coverage",
    "median_ami_all",
    "median_noise_agreement",
}


def _read_parameter_screen(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row_number, row in enumerate(rows, start=2):
        converted_row: dict[str, object] = {}
        for key, value in row.items():
            try:
                if key in _INTEGER_FIELDS:
                    converted_row[key] = int(value)
                elif key in _FLOAT_FIELDS and value not in (None, ""):
                    number = float(value)
                    converted_row[key] = number if math.isfinite(number) else None
                else:
                    converted_row[key] = value
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}: invalid {key!r} value on CSV row {row_number}: {value!r}"
                ) from error
        converted.append(converted_row)
    return converted


def collect_parameter_selections(
    output_root: Path,
    *,
    families: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Select settings from completed parameter screens without clustering."""
    stability_root = Path(output_root) / "motif_stability"
    family_filter = None if families is None else set(families)
    method_filter = None if methods is None else set(methods)
    selections: list[dict[str, object]] = []
    if not stability_root.is_dir():
        return selections

    for family_dir in sorted(path for path in stability_root.iterdir() if path.is_dir()):
        if family_filter is not None and family_dir.name not in family_filter:
            continue
        for method_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
            if method_filter is not None and method_dir.name not in method_filter:
                continue
            screen_path = method_dir / "parameter_screen.csv"
            if not screen_path.is_file():
                continue
            rows = _read_parameter_screen(screen_path)
            if not rows:
                continue
            selected = select_parameter_setting(rows)
            selections.append(
                {
                    **selected,
                    "family": family_dir.name,
                    "method": method_dir.name,
                }
            )
    return selections


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    fields = ["family", "method"]
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def write_selection_summary(
    output_root: Path,
    selections: Sequence[Mapping[str, object]],
) -> tuple[Path, Path]:
    """Atomically write the selection-only CSV and JSON summaries."""
    if not selections:
        raise ValueError("no completed parameter screens were found")
    stability_root = Path(output_root) / "motif_stability"
    stability_root.mkdir(parents=True, exist_ok=True)
    csv_path = stability_root / "final_parameter_selection.csv"
    json_path = stability_root / "final_parameter_selection.json"
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")

    fields = _fieldnames(selections)
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selections)
    json_temporary.write_text(
        json.dumps(list(selections), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    csv_temporary.replace(csv_path)
    json_temporary.replace(json_path)
    return csv_path, json_path
