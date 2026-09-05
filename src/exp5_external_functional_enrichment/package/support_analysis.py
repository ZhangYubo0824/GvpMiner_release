from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence


Row = Mapping[str, object]


def _text(value: object) -> str:
    return str(value or "").strip()


def _float_or_none(value: object) -> float | None:
    text = _text(value)
    if not text or text.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _integer_or_none(value: object) -> int | None:
    number = _float_or_none(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _motif_key(row: Row) -> tuple[str, str, str]:
    return (
        _text(row.get("family")),
        _text(row.get("method")),
        _text(row.get("motif_cluster") or row.get("motif_uid") or row.get("motif_id")),
    )


def _catalog_ranks(catalog_rows: Iterable[Row]) -> dict[tuple[str, str, str], int]:
    ranks: dict[tuple[str, str, str], int] = {}
    for row in catalog_rows:
        key = _motif_key(row)
        rank = _integer_or_none(row.get("rank_within_method"))
        if all(key) and rank is not None and rank > 0:
            ranks[key] = rank
    return ranks


def _native_rank(motif_cluster: str) -> int | None:
    for pattern in (r"_motif_(\d+)$", r"_(\d+)(?:-|$)"):
        matches = re.findall(pattern, motif_cluster)
        if matches:
            return int(matches[-1])
    return None


def _summarise_family_terms(
    rows: Iterable[Row],
    *,
    min_gene_set_size: int,
    max_gene_set_size: int,
) -> tuple[bool, int, set[str]]:
    """Summarise a term-sorted TERM2GENE stream with bounded working memory."""
    has_annotations = False
    eligible_term_count = 0
    eligible_proteins: set[str] = set()
    current_term: str | None = None
    current_proteins: set[str] = set()
    current_oversized = False

    def finish_term() -> None:
        nonlocal eligible_term_count
        if (
            current_term is not None
            and not current_oversized
            and min_gene_set_size <= len(current_proteins) <= max_gene_set_size
        ):
            eligible_term_count += 1
            eligible_proteins.update(current_proteins)

    for row in rows:
        term_id = _text(row.get("term_id") or row.get("term"))
        protein_id = _text(row.get("protein_id") or row.get("gene"))
        if not term_id or not protein_id:
            continue
        has_annotations = True
        if current_term is not None and term_id < current_term:
            raise ValueError(
                "family TERM2GENE rows must be sorted by term_id for bounded-memory analysis"
            )
        if term_id != current_term:
            finish_term()
            current_term = term_id
            current_proteins = set()
            current_oversized = False
        if not current_oversized:
            current_proteins.add(protein_id)
            if len(current_proteins) > max_gene_set_size:
                current_proteins.clear()
                current_oversized = True
    finish_term()
    return has_annotations, eligible_term_count, eligible_proteins


def _rank_group(
    rows: list[dict[str, object]],
    catalog_ranks: Mapping[tuple[str, str, str], int],
) -> None:
    method = _text(rows[0]["method"])

    def rank_key(row: dict[str, object]) -> tuple[object, ...]:
        key = _motif_key(row)
        if key in catalog_ranks:
            row["rank_source"] = "catalog_rank"
            return (0, catalog_ranks[key], _text(row["motif_cluster"]))
        if method.casefold() == "motifae":
            row["rank_source"] = "foreground_protein_support"
            return (1, -int(row["protein_count"]), _text(row["motif_cluster"]))
        native = _native_rank(_text(row["motif_cluster"]))
        if native is not None:
            row["rank_source"] = "native_motif_number"
            return (1, native, _text(row["motif_cluster"]))
        row["rank_source"] = "foreground_protein_support_fallback"
        return (2, -int(row["protein_count"]), _text(row["motif_cluster"]))

    rows.sort(key=rank_key)
    for rank, row in enumerate(rows, start=1):
        row["motif_rank"] = rank


def analyse_motifs(
    motif_set_rows: Iterable[Row],
    enrichment_rows: Iterable[Row],
    family_term_rows: Mapping[str, Iterable[Row]],
    *,
    catalog_rows: Iterable[Row] = (),
    alpha: float = 0.05,
    min_annotated_foreground: int = 3,
    min_gene_set_size: int = 10,
    max_gene_set_size: int = 500,
) -> list[dict[str, object]]:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if min_annotated_foreground < 1:
        raise ValueError("min_annotated_foreground must be at least 1")
    if min_gene_set_size < 1 or max_gene_set_size < min_gene_set_size:
        raise ValueError("invalid gene-set size range")

    foregrounds: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in motif_set_rows:
        if _text(row.get("set_type")).casefold() != "foreground":
            continue
        key = _motif_key(row)
        protein_id = _text(row.get("protein_id"))
        if all(key) and protein_id:
            foregrounds[key].add(protein_id)

    minimum_adjusted_pvalues: dict[tuple[str, str, str], float] = {}
    for row in enrichment_rows:
        key = _motif_key(row)
        adjusted = _float_or_none(row.get("p.adjust"))
        if all(key) and adjusted is not None:
            previous = minimum_adjusted_pvalues.get(key)
            if previous is None or adjusted < previous:
                minimum_adjusted_pvalues[key] = adjusted

    family_annotations: dict[str, tuple[bool, int, set[str]]] = {}
    for family, rows in family_term_rows.items():
        family_annotations[family] = _summarise_family_terms(
            rows,
            min_gene_set_size=min_gene_set_size,
            max_gene_set_size=max_gene_set_size,
        )

    analysed: list[dict[str, object]] = []
    for (family, method, motif_cluster), foreground in sorted(foregrounds.items()):
        family_summary = family_annotations.get(family)
        has_annotations, eligible_term_count, eligible_proteins = (
            family_summary if family_summary is not None else (False, 0, set())
        )
        eligible_foreground = foreground & eligible_proteins
        protein_count = len(foreground)
        annotation_coverage = len(eligible_foreground) / protein_count if protein_count else 0.0

        if not has_annotations:
            status = "unevaluable_no_family_annotations"
            is_testable = False
        elif eligible_term_count == 0:
            status = "unevaluable_no_eligible_gene_sets"
            is_testable = False
        elif len(eligible_foreground) < min_annotated_foreground:
            status = "unevaluable_insufficient_annotated_foreground"
            is_testable = False
        else:
            is_testable = True
            minimum_adjusted = minimum_adjusted_pvalues.get(
                (family, method, motif_cluster)
            )
            is_supported = minimum_adjusted is not None and minimum_adjusted <= alpha
            status = "tested_supported" if is_supported else "tested_not_significant"

        is_supported = status == "tested_supported"
        minimum_adjusted = minimum_adjusted_pvalues.get((family, method, motif_cluster))
        analysed.append(
            {
                "family": family,
                "method": method,
                "motif_cluster": motif_cluster,
                "protein_count": protein_count,
                "eligible_term_count": eligible_term_count,
                "family_eligible_annotated_protein_count": len(eligible_proteins),
                "eligible_annotated_foreground_count": len(eligible_foreground),
                "annotation_coverage": annotation_coverage,
                "protein_size_bin": int(math.floor(math.log2(max(1, protein_count)))),
                "annotation_coverage_bin": min(4, int(math.floor(annotation_coverage * 5))),
                "is_testable": is_testable,
                "is_supported": is_supported,
                "status": status,
                "minimum_p_adjust": minimum_adjusted if minimum_adjusted is not None else "",
                "alpha": alpha,
                "significance_rule": "at least one clusterProfiler row with p.adjust <= alpha",
            }
        )

    ranks = _catalog_ranks(catalog_rows)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in analysed:
        grouped[(_text(row["family"]), _text(row["method"]))].append(row)
    for rows in grouped.values():
        _rank_group(rows, ranks)
    return sorted(
        analysed,
        key=lambda row: (
            _text(row["family"]),
            _text(row["method"]),
            int(row["motif_rank"]),
        ),
    )


def select_top_k(rows: Iterable[Row], top_k: int) -> list[dict[str, object]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(_text(row.get("family")), _text(row.get("method")))].append(row)
    selected: list[dict[str, object]] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda row: (
                int(row.get("motif_rank") or 10**12),
                _text(row.get("motif_cluster")),
            ),
        )
        selected.extend(dict(row) for row in ranked[:top_k])
    return selected


def derived_seed(
    global_seed: int,
    family: str,
    method: str,
    iteration: int,
) -> int:
    material = f"{global_seed}|{family}|{method}|{iteration}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def draw_matched_random_selection(
    candidates: Iterable[Row],
    observed_rows: Iterable[Row],
    *,
    global_seed: int,
    family: str,
    method: str,
    iteration: int,
) -> list[dict[str, object]]:
    strata_needed = Counter(
        (int(row["protein_size_bin"]), int(row["annotation_coverage_bin"]))
        for row in observed_rows
    )
    pools: dict[tuple[int, int], list[Row]] = defaultdict(list)
    for row in candidates:
        if _text(row.get("family")) != family or _text(row.get("method")) != method:
            continue
        stratum = (int(row["protein_size_bin"]), int(row["annotation_coverage_bin"]))
        pools[stratum].append(row)

    for population in pools.values():
        population.sort(key=lambda row: _text(row.get("motif_cluster")))
    return _draw_from_pools(
        pools,
        strata_needed,
        seed=derived_seed(global_seed, family, method, iteration),
    )


def _draw_from_pools(
    pools: Mapping[tuple[int, int], Sequence[Row]],
    strata_needed: Mapping[tuple[int, int], int],
    *,
    seed: int,
) -> list[dict[str, object]]:
    generator = random.Random(seed)
    selected: list[dict[str, object]] = []
    for stratum, count in sorted(strata_needed.items()):
        population = pools.get(stratum, [])
        if len(population) < count:
            raise ValueError(
                f"stratum {stratum} has {len(population)} candidates but {count} are required"
            )
        selected.extend(dict(row) for row in generator.sample(population, count))
    return sorted(selected, key=lambda row: _text(row.get("motif_cluster")))


def summarise_top_k(rows: Iterable[Row], top_k: int) -> list[dict[str, object]]:
    all_rows = [dict(row) for row in rows]
    selected = select_top_k(all_rows, top_k)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[(_text(row["family"]), _text(row["method"]))].append(row)
    summaries: list[dict[str, object]] = []
    for (family, method), group in sorted(grouped.items()):
        testable = sum(bool(row.get("is_testable")) for row in group)
        supported = sum(bool(row.get("is_supported")) for row in group)
        summaries.append(
            {
                "family": family,
                "method": method,
                "top_k_requested": top_k,
                "selected_motif_count": len(group),
                "testable_motif_count": testable,
                "supported_motif_count": supported,
                "supported_fraction_of_selected": supported / len(group) if group else "",
                "supported_fraction_of_testable": supported / testable if testable else "",
            }
        )
    return summaries


def summarise_methods(
    family_method_summaries: Iterable[Row],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in family_method_summaries:
        grouped[_text(row.get("method"))].append(row)
    output: list[dict[str, object]] = []
    for method, rows in sorted(grouped.items()):
        selected = sum(int(row["selected_motif_count"]) for row in rows)
        testable = sum(int(row["testable_motif_count"]) for row in rows)
        supported = sum(int(row["supported_motif_count"]) for row in rows)
        output.append(
            {
                "method": method,
                "family_count": len(rows),
                "selected_motif_count": selected,
                "testable_motif_count": testable,
                "supported_motif_count": supported,
                "supported_fraction_of_selected": supported / selected if selected else "",
                "supported_fraction_of_testable": supported / testable if testable else "",
            }
        )
    return output


def _quantile(values: Sequence[int], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def random_background_summaries(
    rows: Iterable[Row],
    *,
    top_k: int,
    iterations: int,
    global_seed: int,
) -> list[dict[str, object]]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    all_rows = [dict(row) for row in rows]
    observed = select_top_k(all_rows, top_k)
    all_by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    observed_by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        all_by_group[(_text(row["family"]), _text(row["method"]))].append(row)
    for row in observed:
        observed_by_group[(_text(row["family"]), _text(row["method"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (family, method), observed_rows in sorted(observed_by_group.items()):
        observed_supported = sum(bool(row.get("is_supported")) for row in observed_rows)
        observed_testable = sum(bool(row.get("is_testable")) for row in observed_rows)
        random_supported_counts: list[int] = []
        random_testable_counts: list[int] = []
        strata_needed = Counter(
            (
                int(row["protein_size_bin"]),
                int(row["annotation_coverage_bin"]),
            )
            for row in observed_rows
        )
        pools: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
        for row in all_by_group[(family, method)]:
            pools[
                (
                    int(row["protein_size_bin"]),
                    int(row["annotation_coverage_bin"]),
                )
            ].append(row)
        for population in pools.values():
            population.sort(key=lambda row: _text(row.get("motif_cluster")))
        for iteration in range(iterations):
            random_rows = _draw_from_pools(
                pools,
                strata_needed,
                seed=derived_seed(global_seed, family, method, iteration),
            )
            random_supported_counts.append(
                sum(bool(row.get("is_supported")) for row in random_rows)
            )
            random_testable_counts.append(
                sum(bool(row.get("is_testable")) for row in random_rows)
            )
        summaries.append(
            {
                "family": family,
                "method": method,
                "top_k_requested": top_k,
                "observed_selected_count": len(observed_rows),
                "observed_testable_count": observed_testable,
                "observed_supported_count": observed_supported,
                "random_iterations": iterations,
                "global_seed": global_seed,
                "random_testable_mean": sum(random_testable_counts) / iterations,
                "random_supported_mean": sum(random_supported_counts) / iterations,
                "random_supported_q025": _quantile(random_supported_counts, 0.025),
                "random_supported_q975": _quantile(random_supported_counts, 0.975),
                "empirical_p_ge_observed": (
                    1
                    + sum(count >= observed_supported for count in random_supported_counts)
                )
                / (iterations + 1),
                "sampling_contract": (
                    "without replacement; matched on floor(log2(protein_count)) "
                    "and five annotation-coverage bins"
                ),
            }
        )
    return summaries
