#!/usr/bin/env python3
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from adapter_common import (
    build_parser,
    catalog_from_occurrences,
    canonical_sequence_id,
    find_first,
    first_value,
    make_motif_uid,
    one_based_to_zero_based,
    read_fasta_records,
    read_delimited,
    resolve_family_fasta,
    write_standard_outputs,
)


def adapt(*, family: str, method: str, input_dir: Path, output_dir: Path) -> None:
    fimo_path = find_first(input_dir, ["fimo.tsv", "meme_fimo.tsv", "sites.tsv"])
    if fimo_path is None:
        xml_path = find_first(input_dir, ["meme.xml"])
        if xml_path is None:
            raise FileNotFoundError(
                "MEME adapter requires FIMO-style fimo.tsv/sites.tsv or MEME meme.xml."
            )
        _adapt_from_meme_xml(family=family, method=method, input_dir=input_dir, output_dir=output_dir, xml_path=xml_path)
        return

    occurrences = []
    for index, row in enumerate(read_delimited(fimo_path), start=1):
        motif_label = first_value(row, ["motif_id", "motif_ID", "pattern_name"], str(index))
        sequence_id = first_value(row, ["sequence_name", "seq_ID", "sequence_id", "parent_id"])
        motif_sequence = first_value(row, ["matched_sequence", "site", "motif_sequence", "match"])
        start_text = first_value(row, ["start", "site_Start"])
        end_text = first_value(row, ["stop", "site_End", "end"])
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
                "score": first_value(row, ["score"]),
                "significance": first_value(row, ["p-value", "pvalue", "q-value", "qvalue"]),
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


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _tag(child) == name]


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    matches = _children(element, name)
    return matches[0] if matches else None




def _int_attr(element: ET.Element, name: str, default: int = 0) -> int:
    value = element.attrib.get(name)
    if value in (None, ""):
        return default
    return int(float(value))


def _letter_symbols(root: ET.Element) -> dict[str, str]:
    result = {}
    for alphabet in root.iter():
        if _tag(alphabet) != "alphabet":
            continue
        for letter in _children(alphabet, "letter"):
            letter_id = letter.attrib.get("id", "")
            symbol = letter.attrib.get("symbol", letter_id)
            if letter_id and symbol:
                result[letter_id] = symbol
        if result:
            return result
    return {}


def _background_frequencies(root: ET.Element, symbols: dict[str, str]) -> dict[str, float]:
    for frequencies in root.iter():
        if _tag(frequencies) not in {"background_frequencies", "letter_frequencies"}:
            continue
        array = _first_child(frequencies, "alphabet_array")
        if array is None:
            continue
        result = {}
        for value in _children(array, "value"):
            letter = symbols.get(value.attrib.get("letter_id", ""))
            if letter and value.text:
                result[letter] = float(value.text)
        if result:
            return result
    letters = sorted(symbols.values())
    if not letters:
        return {}
    uniform = 1.0 / len(letters)
    return {letter: uniform for letter in letters}


def _probability_matrix(motif: ET.Element, symbols: dict[str, str]) -> list[dict[str, float]]:
    probabilities = _first_child(motif, "probabilities")
    if probabilities is None:
        return []
    matrix = _first_child(probabilities, "alphabet_matrix")
    if matrix is None:
        return []
    rows = []
    for array in _children(matrix, "alphabet_array"):
        row = {}
        for value in _children(array, "value"):
            letter = symbols.get(value.attrib.get("letter_id", ""))
            if letter and value.text:
                row[letter] = float(value.text)
        if row:
            rows.append(row)
    return rows


def _score_window(window: str, matrix: list[dict[str, float]], background: dict[str, float]) -> float:
    score = 0.0
    pseudocount = 1e-12
    fallback_bg = 1.0 / max(1, len(background))
    for letter, row in zip(window, matrix):
        probability = max(row.get(letter, pseudocount), pseudocount)
        bg = max(background.get(letter, fallback_bg), pseudocount)
        score += math.log(probability / bg)
    return score


def _best_window(
    sequence: str,
    matrix: list[dict[str, float]],
    background: dict[str, float],
) -> tuple[int, int, float] | None:
    width = len(matrix)
    if width <= 0 or len(sequence) < width:
        return None
    best: tuple[int, int, float] | None = None
    for start in range(0, len(sequence) - width + 1):
        end = start + width
        score = _score_window(sequence[start:end], matrix, background)
        if best is None or score > best[2]:
            best = (start, end, score)
    return best


def _adapt_from_meme_xml(
    *,
    family: str,
    method: str,
    input_dir: Path,
    output_dir: Path,
    xml_path: Path,
) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    symbols = _letter_symbols(root)
    background = _background_frequencies(root, symbols)
    records = read_fasta_records(resolve_family_fasta(family))
    occurrences = []

    for motif_index, motif in enumerate((node for node in root.iter() if _tag(node) == "motif"), start=1):
        matrix = _probability_matrix(motif, symbols)
        if not matrix:
            continue
        motif_label = motif.attrib.get("alt") or motif.attrib.get("id") or str(motif_index)
        motif_uid = make_motif_uid(family, method, motif_label)
        site_limit = _int_attr(motif, "sites", default=len(records))
        e_value = motif.attrib.get("e_value", "")
        consensus = motif.attrib.get("name") or motif_label
        candidates = []
        for record_index, (sequence_id, sequence) in enumerate(records):
            best = _best_window(sequence, matrix, background)
            if best is None:
                continue
            start, end, score = best
            candidates.append((score, record_index, canonical_sequence_id(sequence_id), start, end, sequence[start:end]))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for score, _record_index, sequence_id, start, end, motif_sequence in candidates[:site_limit]:
            occurrences.append(
                {
                    "family": family,
                    "method": method,
                    "motif_uid": motif_uid,
                    "sequence_id": sequence_id,
                    "parent_id": sequence_id,
                    "occ_start": start,
                    "occ_end": end,
                    "score": f"{score:.6g}",
                    "significance": e_value,
                    "motif_sequence": motif_sequence,
                    "consensus_or_label": consensus,
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
    args = build_parser("Convert MEME/FIMO site TSV to baseline motif CSV outputs.").parse_args()
    adapt(family=args.family, method=args.method, input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
