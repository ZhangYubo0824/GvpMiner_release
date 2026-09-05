from __future__ import annotations

import json
import sys
from pathlib import Path

from .figures import generate_publication_figures, generate_publication_figures_from_summaries


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit("Usage: python -m exp4_motif_structure_analysis.figure_worker <payload.json> <result.json>")
    payload_path = Path(argv[0])
    result_path = Path(argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if "control_type_counts" in payload or "control_contact_density" in payload:
        paths, warnings = generate_publication_figures_from_summaries(
            output_dir=Path(payload["output_dir"]),
            feature_rows=payload.get("feature_rows", []),
            control_type_counts=payload.get("control_type_counts", {}),
            control_contact_density=payload.get("control_contact_density", []),
            stat_rows=payload.get("stat_rows", []),
            formats=payload.get("formats", ["png", "pdf"]),
            isolate_process=False,
        )
    else:
        paths, warnings = generate_publication_figures(
            output_dir=Path(payload["output_dir"]),
            feature_rows=payload.get("feature_rows", []),
            control_rows=payload.get("control_rows", []),
            stat_rows=payload.get("stat_rows", []),
            formats=payload.get("formats", ["png", "pdf"]),
            isolate_process=False,
        )
    result_path.write_text(
        json.dumps({"paths": [str(path) for path in paths], "warnings": warnings}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
