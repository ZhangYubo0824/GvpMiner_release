#!/usr/bin/env python3
#python \
#  src/exp5_external_functional_enrichment/scripts/split_motif_sets_by_family_method.py \
#  --input \
#    work/results/exp5_external_functional_enrichment/clusterprofiler/motif_sets_long.csv \
#  --output-dir \
#    work/results/exp5_external_functional_enrichment/clusterprofiler/family_method_sets
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split motif sets by family and method."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    handles = {}
    writers = {}
    row_counts = {}
    clusters = {}

    try:
        with args.input.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames

            required = {
                "family",
                "method",
                "motif_cluster",
                "set_type",
                "protein_id",
            }
            missing = required - set(fieldnames or [])

            if missing:
                raise ValueError(
                    f"Input missing columns: {sorted(missing)}"
                )

            for row_number, row in enumerate(reader, start=1):
                family = row["family"]
                method = row["method"]
                key = (family, method)

                if key not in writers:
                    output = (
                        args.output_dir
                        / f"motif_sets_{family}_{method}.csv"
                    )
                    output_handle = output.open("w", newline="")
                    writer = csv.DictWriter(
                        output_handle,
                        fieldnames=fieldnames,
                    )
                    writer.writeheader()

                    handles[key] = output_handle
                    writers[key] = writer
                    row_counts[key] = 0
                    clusters[key] = set()

                writers[key].writerow(row)
                row_counts[key] += 1
                clusters[key].add(row["motif_cluster"])

                if row_number % 10_000_000 == 0:
                    print(f"Processed {row_number:,} rows")

    finally:
        for handle in handles.values():
            handle.close()

    for key in sorted(row_counts):
        family, method = key
        print(
            f"{family}\t{method}\t"
            f"clusters={len(clusters[key]):,}\t"
            f"rows={row_counts[key]:,}"
        )


if __name__ == "__main__":
    main()
