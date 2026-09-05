#!/usr/bin/env python3
"""
从已有的 exp4_motif_structure_analysis 输出 CSV 重新生成论文图。

用法:
    python regenerate_figures.py <output_dir> [--formats png pdf]

    <output_dir> 是 exp4_motif_structure_analysis 的单次输出目录，
    需包含:
      motif_structure_features.csv
      matched_control_features.csv
      structural_enrichment_results.csv

示例:
    python regenerate_figures.py results6/exp4_motif_structure_analysis/GvpA/M2
    python regenerate_figures.py results6/exp4_motif_structure_analysis/GvpA/M2 --formats pdf

Note: 功能富集 (functional enrichment) 由 Exp5 外部工具流程生成，
    对应的输入、富集表和 motif-window 文件请通过 exp5_external_functional_enrichment/run.py 生成。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp4_motif_structure_analysis.figures import _generate_publication_figures_direct
from exp4_motif_structure_analysis.io import read_csv_rows


def main():
    parser = argparse.ArgumentParser(description="Regenerate publication figures from existing CSVs")
    parser.add_argument("output_dir", type=Path, help="Directory containing the analysis CSVs")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Output formats (default: png pdf)")
    args = parser.parse_args()

    result_dir = Path(args.output_dir)
    if not result_dir.is_dir():
        raise SystemExit(f"Directory not found: {result_dir}")

    required = {
        "motif_structure_features.csv": "feature_rows",
        "matched_control_features.csv": "control_rows",
        "structural_enrichment_results.csv": "stat_rows",
    }
    kwargs: dict = {}
    for filename, key in required.items():
        path = result_dir / filename
        if not path.exists():
            raise SystemExit(f"Missing required file: {path}")
        kwargs[key] = read_csv_rows(path)

    figures_dir = result_dir / "figures"
    paths, warnings = _generate_publication_figures_direct(
        output_dir=figures_dir,
        formats=tuple(args.formats),
        **kwargs,
    )

    for p in sorted(paths):
        print(f"  OK  {p}")
    for w in warnings:
        print(f"  WARN  {w}")
    print(f"\nDone: {len(paths)} figures → {figures_dir}")


if __name__ == "__main__":
    main()
