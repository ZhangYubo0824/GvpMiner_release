"""Export helpers for publication figures and their plotted data."""

from __future__ import annotations

import json
from pathlib import Path


def export_figure_bundle(fig, data, output_dir, stem: str):
    """Write a figure as editable vector, high-DPI preview, and source data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "png": output_dir / f"{stem}.png",
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "csv": output_dir / f"{stem}_data.csv",
    }
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], format="svg", bbox_inches="tight", facecolor="white")
    data.to_csv(paths["csv"], index=False)
    return paths


def write_manifest(output_dir, payload):
    """Write a UTF-8 JSON manifest and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "publication_figure_manifest.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
