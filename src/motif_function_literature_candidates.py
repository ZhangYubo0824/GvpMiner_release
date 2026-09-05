from __future__ import annotations

from typing import Iterable, Mapping


LITERATURE_CANDIDATE_STATUSES = {
    "external_enrichment_not_significant",
    "external_enrichment_no_terms",
}


def select_literature_candidates(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("batch_coverage_status", "")) == "has_clusterprofiler_csv"
        and str(row.get("external_enrichment_status", ""))
        in LITERATURE_CANDIDATE_STATUSES
    ]
