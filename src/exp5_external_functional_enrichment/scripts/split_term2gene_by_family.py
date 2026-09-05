#!/usr/bin/env python3
# python src/exp5_external_functional_enrichment/scripts/split_term2gene_by_family.py \
#  --sequence-index \
#    work/results/exp5_external_functional_enrichment/input/sequence_index.csv \
#  --term2gene \
#    work/results/exp5_external_functional_enrichment/annotations/term2gene.csv \
#  --output-dir \
#    work/results/exp5_external_functional_enrichment/clusterprofiler/family_term2gene
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split TERM2GENE links by Gvp family."
    )
    parser.add_argument("--sequence-index", required=True, type=Path)
    parser.add_argument("--term2gene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protein_to_family: dict[str, str] = {}

    with args.sequence_index.open(newline="") as handle:
        reader = csv.DictReader(handle)

        required = {"protein_id", "family"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"sequence_index missing columns: {sorted(missing)}"
            )

        for row in reader:
            protein_id = row["protein_id"].strip()
            family = row["family"].strip()

            if protein_id and family:
                protein_to_family[protein_id] = family

    families = sorted(set(protein_to_family.values()))

    print(
        f"Loaded {len(protein_to_family):,} proteins "
        f"from {len(families)} families"
    )

    handles = {}
    writers = {}
    counts = {family: 0 for family in families}

    try:
        for family in families:
            output = args.output_dir / f"term2gene_{family}.csv"
            handle = output.open("w", newline="")
            writer = csv.DictWriter(
                handle,
                fieldnames=["term_id", "protein_id"],
            )
            writer.writeheader()

            handles[family] = handle
            writers[family] = writer

        with args.term2gene.open(newline="") as handle:
            reader = csv.DictReader(handle)

            required = {"term_id", "protein_id"}
            missing = required - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    f"term2gene missing columns: {sorted(missing)}"
                )

            for row_number, row in enumerate(reader, start=1):
                protein_id = row["protein_id"].strip()
                family = protein_to_family.get(protein_id)

                if family is not None:
                    writers[family].writerow(
                        {
                            "term_id": row["term_id"],
                            "protein_id": protein_id,
                        }
                    )
                    counts[family] += 1

                if row_number % 1_000_000 == 0:
                    print(
                        f"Processed {row_number:,} TERM2GENE rows"
                    )

    finally:
        for handle in handles.values():
            handle.close()

    for family in families:
        output = args.output_dir / f"term2gene_{family}.csv"
        print(
            f"{family}: {counts[family]:,} links -> {output}"
        )


if __name__ == "__main__":
    main()
    
    
