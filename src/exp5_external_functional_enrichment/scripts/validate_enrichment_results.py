#!/usr/bin/env python3
"""Validate enrichment CSVs using the unchanged p.adjust decision boundary.

Example:
  python src2/exp5_external_functional_enrichment/scripts/validate_enrichment_results.py \
    --input results7/exp5_corrected/clusterprofiler/clusterprofiler_enrichment.csv \
    --significant-output results7/exp5_corrected/clusterprofiler/significant.csv \
    --summary-output results7/exp5_corrected/clusterprofiler/summary.csv \
    --padj-cutoff 0.05
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize enrichment results."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--significant-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument(
        "--padj-cutoff",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--qvalue-cutoff",
        type=float,
        default=0.05,
        help=(
            "Deprecated diagnostic parameter retained for CLI compatibility; "
            "qvalue does not alter the p.adjust significance decision."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = pd.read_csv(args.input)

    required = {
        "motif_cluster",
        "ID",
        "p.adjust",
    }
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Enrichment result missing columns: {sorted(missing)}"
        )

    frame["p.adjust"] = pd.to_numeric(
        frame["p.adjust"],
        errors="coerce",
    )

    significant_mask = (
        frame["p.adjust"] <= args.padj_cutoff
    )

    if "qvalue" in frame.columns:
        frame["qvalue"] = pd.to_numeric(
            frame["qvalue"],
            errors="coerce",
        )

    significant = frame.loc[significant_mask].copy()

    args.significant_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    significant.to_csv(
        args.significant_output,
        index=False,
    )

    group_columns = [
        column
        for column in [
            "family",
            "method",
            "motif_cluster",
        ]
        if column in significant.columns
    ]

    if group_columns:
        summary = (
            significant
            .groupby(group_columns, dropna=False)
            .agg(
                significant_terms=("ID", "nunique"),
                enrichment_rows=("ID", "size"),
                minimum_p_adjust=("p.adjust", "min"),
            )
            .reset_index()
            .sort_values(
                [
                    "significant_terms",
                    "minimum_p_adjust",
                ],
                ascending=[False, True],
            )
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv(args.summary_output, index=False)

    print(f"Total enrichment rows: {len(frame):,}")
    print(f"Significant rows: {len(significant):,}")
    print(
        "Motif clusters with significant enrichment: "
        f"{significant['motif_cluster'].nunique():,}"
    )
    print(f"Significant output: {args.significant_output}")
    print(f"Summary output: {args.summary_output}")


if __name__ == "__main__":
    main()
