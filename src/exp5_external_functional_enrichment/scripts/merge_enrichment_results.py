#!/usr/bin/env python3

#python \
#  src/exp5_external_functional_enrichment/scripts/merge_enrichment_results.py \
#  --input-dir \
#    results6/exp5_external_functional_enrichment/clusterprofiler/batch_results \
#  --output \
#    results6/exp5_external_functional_enrichment/clusterprofiler/clusterprofiler_enrichment.csv

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge family and cluster-batch enrichment results."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_result(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"EMPTY: {path}")
        return None

    if frame.empty:
        print(f"NO ROWS: {path}")
        return None

    frame["result_source"] = path.stem
    return frame


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result_files: list[Path] = []

    # 直接按 family-method 运行的结果
    result_files.extend(
        sorted(args.input_dir.glob("Gvp*_M[23].csv"))
    )

    # 按 cluster batch 运行的结果
    result_files.extend(
        sorted(args.input_dir.glob("*_batches/*.csv"))
    )

    # 避免把已合并的输出重新读入
    output_resolved = args.output.resolve()
    result_files = [
        path
        for path in result_files
        if path.resolve() != output_resolved
    ]

    frames = []

    for path in result_files:
        frame = load_result(path)

        if frame is None:
            continue

        frames.append(frame)
        print(f"{path}: {len(frame):,} rows")

    if not frames:
        raise RuntimeError("No non-empty result files found")

    merged = pd.concat(frames, ignore_index=True)

    duplicate_columns = [
        column
        for column in [
            "family",
            "method",
            "motif_cluster",
            "ID",
        ]
        if column in merged.columns
    ]

    if duplicate_columns:
        before = len(merged)
        merged = merged.drop_duplicates(
            subset=duplicate_columns,
            keep="first",
        )
        print(
            f"Removed {before - len(merged):,} duplicate rows"
        )

    merged.to_csv(args.output, index=False)

    print(
        f"Wrote {len(merged):,} rows to {args.output}"
    )


if __name__ == "__main__":
    main()