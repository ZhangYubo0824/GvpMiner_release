#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

from adapter_common import (
    build_parser,
    find_first,
    read_fasta_records,
    resolve_family_fasta,
    resolve_sequence_id,
    write_standard_outputs,
)


SCORE_RE = re.compile(r"^Score:\s+(?P<score>\S+)")
SITE_RE = re.compile(
    r"^\s*(?P<name>\S+)\s+(?P<start>\d+)\s+(?P<aligned>[A-Za-z.]+)\s+"
    r"(?P<end>\d+)\s+(?P<strand>[+-])\s+(?P<score>[0-9.eE+-]+)"
)


def _catalog_from_blocks(
    *,
    family: str,
    method: str,
    block_scores: dict[str, str],
    occurrences: list[dict[str, object]],
) -> list[dict[str, str]]:
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for row in occurrences:
        motif_uid = str(row["motif_uid"])
        counts[motif_uid] = counts.get(motif_uid, 0) + 1
        labels.setdefault(motif_uid, str(row.get("consensus_or_label", "")))
    return [
        {
            "family": family,
            "method": method,
            "motif_uid": motif_uid,
            "consensus_or_label": labels.get(motif_uid, motif_uid),
            "n_occurrences": str(count),
            "score": block_scores.get(motif_uid, ""),
            "significance": "",
            "source_model": method,
        }
        for motif_uid, count in sorted(counts.items())
    ]


def adapt(*, family: str, method: str, input_dir: Path, output_dir: Path) -> None:
    glam2_path = find_first(input_dir, ["glam2.txt"])
    if glam2_path is None:
        from standard_csv_adapter import adapt as copy_standard

        copy_standard(family=family, method=method, input_dir=input_dir, output_dir=output_dir)
        return

    sequence_ids = [sequence_id for sequence_id, _sequence in read_fasta_records(resolve_family_fasta(family))]
    occurrences: list[dict[str, object]] = []
    block_scores: dict[str, str] = {}
    block_index = 0
    current_motif_uid = ""

    with glam2_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            score_match = SCORE_RE.match(line)
            if score_match:
                block_index += 1
                current_motif_uid = f"{family}_{method}_{block_index:03d}"
                block_scores[current_motif_uid] = score_match.group("score")
                continue

            if not current_motif_uid:
                continue
            site_match = SITE_RE.match(line)
            if not site_match:
                continue

            aligned = site_match.group("aligned")
            motif_sequence = aligned.replace(".", "")
            start = int(site_match.group("start")) - 1
            end = int(site_match.group("end"))
            sequence_id = resolve_sequence_id(site_match.group("name"), sequence_ids)
            occurrences.append(
                {
                    "family": family,
                    "method": method,
                    "motif_uid": current_motif_uid,
                    "sequence_id": sequence_id,
                    "parent_id": sequence_id,
                    "occ_start": start,
                    "occ_end": end,
                    "score": site_match.group("score"),
                    "significance": "",
                    "motif_sequence": motif_sequence,
                    "consensus_or_label": f"GLAM2-{block_index:03d}",
                    "source_model": method,
                }
            )

    write_standard_outputs(
        output_dir=output_dir,
        method=method,
        occurrences=occurrences,
        catalog=_catalog_from_blocks(
            family=family,
            method=method,
            block_scores=block_scores,
            occurrences=occurrences,
        ),
    )


def main() -> None:
    args = build_parser("Convert GLAM2 glam2.txt alignments to baseline motif CSV outputs.").parse_args()
    adapt(family=args.family, method=args.method, input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
