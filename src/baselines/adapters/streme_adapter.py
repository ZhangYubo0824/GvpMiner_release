#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from adapter_common import (
    build_parser,
    catalog_from_occurrences,
    find_first,
    first_value,
    make_motif_uid,
    one_based_to_zero_based,
    read_delimited,
    write_standard_outputs,
)


def adapt(*, family: str, method: str, input_dir: Path, output_dir: Path) -> None:
    sites_path = find_first(input_dir, ["sites.tsv", "streme_sites.tsv"])
    if sites_path is None:
        raise FileNotFoundError("STREME adapter requires sites.tsv under the raw output directory.")

    occurrences = []
    for index, row in enumerate(read_delimited(sites_path), start=1):
        motif_label = first_value(row, ["motif_ID", "motif_id", "motif", "motif_ALT_ID"], str(index))
        sequence_id = first_value(row, ["seq_ID", "sequence_id", "sequence_name", "parent_id"])
        motif_sequence = first_value(row, ["site", "site_Sequence", "matched_sequence", "motif_sequence", "match"])
        start_text = first_value(row, ["site_Start", "start", "motif_start"])
        end_text = first_value(row, ["site_End", "stop", "end", "motif_end"])
        start, end = one_based_to_zero_based(start_text, end_text, motif_sequence)
        motif_uid = make_motif_uid(family, method, motif_label)
        occurrences.append(
            {
                "family": family,
                "method": method,
                "motif_uid": motif_uid,
                "sequence_id": sequence_id,
                "parent_id": sequence_id,
                "occ_start": start,
                "occ_end": end,
                "score": first_value(row, ["score", "site_score", "site_Score"]),
                "significance": first_value(row, ["site_Pvalue", "site_p-value", "p-value", "pvalue"]),
                "motif_sequence": motif_sequence,
                "consensus_or_label": motif_sequence,
                "source_model": method,
            }
        )

    write_standard_outputs(
        output_dir=output_dir,
        method=method,
        occurrences=occurrences,
        catalog=catalog_from_occurrences(occurrences),
    )


def main() -> None:
    args = build_parser("Convert STREME sites.tsv to baseline motif CSV outputs.").parse_args()
    adapt(family=args.family, method=args.method, input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
