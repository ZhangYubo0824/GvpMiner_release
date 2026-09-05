#!/usr/bin/env python3
#python \
#  src/exp5_external_functional_enrichment/scripts/split_motif_sets_by_cluster.py \
#  --input \
#    work/results/exp5_external_functional_enrichment/clusterprofiler/family_method_sets/motif_sets_GvpK_M2.csv \
#  --output-dir \
#    work/results/exp5_external_functional_enrichment/clusterprofiler/cluster_batches/GvpK \
#  --family GvpK \
#  --method M2 \
#  --clusters-per-batch 10

#python \
#  src/exp5_external_functional_enrichment/scripts/split_motif_sets_by_cluster.py \
#  --input \
#    work/results/exp5_external_functional_enrichment/clusterprofiler/family_method_sets/motif_sets_GvpK_M3.csv \
#  --output-dir \
#    work/results/exp5_external_functional_enrichment/clusterprofiler/cluster_batches/GvpK \
#  --family GvpK \
#  --method M3 \
#  --clusters-per-batch 10
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one family-method motif set by cluster batches."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--clusters-per-batch",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clusters_per_batch < 1:
        raise ValueError("--clusters-per-batch must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cluster_order: list[str] = []
    seen: set[str] = set()

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

        for row in reader:
            cluster = row["motif_cluster"]

            if cluster not in seen:
                seen.add(cluster)
                cluster_order.append(cluster)

    cluster_to_batch = {
        cluster: index // args.clusters_per_batch + 1
        for index, cluster in enumerate(cluster_order)
    }

    handles = {}
    writers = {}
    row_counts = {}

    try:
        with args.input.open(newline="") as handle:
            reader = csv.DictReader(handle)

            for row_number, row in enumerate(reader, start=1):
                cluster = row["motif_cluster"]
                batch_number = cluster_to_batch[cluster]

                if batch_number not in writers:
                    output = args.output_dir / (
                        f"motif_sets_{args.family}_{args.method}_"
                        f"batch_{batch_number:03d}.csv"
                    )

                    output_handle = output.open("w", newline="")
                    writer = csv.DictWriter(
                        output_handle,
                        fieldnames=fieldnames,
                    )
                    writer.writeheader()

                    handles[batch_number] = output_handle
                    writers[batch_number] = writer
                    row_counts[batch_number] = 0

                writers[batch_number].writerow(row)
                row_counts[batch_number] += 1

                if row_number % 5_000_000 == 0:
                    print(f"Processed {row_number:,} rows")

    finally:
        for handle in handles.values():
            handle.close()

    print(
        f"{args.family} {args.method}: "
        f"{len(cluster_order):,} clusters, "
        f"{len(row_counts):,} batches"
    )

    for batch_number in sorted(row_counts):
        print(
            f"batch {batch_number:03d}: "
            f"{row_counts[batch_number]:,} rows"
        )


if __name__ == "__main__":
    main()
