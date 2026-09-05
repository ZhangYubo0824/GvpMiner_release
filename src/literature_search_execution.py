from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen


EUROPE_PMC_SEARCH_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
)
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
USER_AGENT = "GvpMotifLiteratureEnrichment/1.0"


def _request_json(url: str, timeout: float = 30.0) -> dict[str, object]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def europe_pmc_hit_count(query: str, timeout: float = 30.0) -> int:
    parameters = urlencode(
        {
            "query": query,
            "format": "json",
            "pageSize": 1,
        }
    )
    url = f"{EUROPE_PMC_SEARCH_URL}?{parameters}"
    payload = _request_json(url, timeout=timeout)
    return int(payload["hitCount"])


def crossref_hit_count(query: str, timeout: float = 30.0) -> int:
    parameters = urlencode(
        {
            "query.bibliographic": query,
            "filter": "until-pub-date:2026-06-14",
            "rows": 0,
        }
    )
    url = f"{CROSSREF_WORKS_URL}?{parameters}"
    payload = _request_json(url, timeout=timeout)
    message = payload.get("message", {})
    if not isinstance(message, Mapping):
        raise ValueError("Crossref response is missing message metadata")
    return int(message["total-results"])


def execute_public_searches(
    rows: Iterable[Mapping[str, object]],
    *,
    europe_pmc_fetcher: Callable[[str], int] = europe_pmc_hit_count,
    crossref_fetcher: Callable[[str], int] = crossref_hit_count,
    executed_at_utc: str | None = None,
) -> list[dict[str, object]]:
    timestamp = executed_at_utc or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    output: list[dict[str, object]] = []
    for raw in rows:
        row = dict(raw)
        database = str(row.get("database", "")).lower()
        if str(row.get("status", "")).lower() == "manual_required":
            output.append(row)
            continue
        if "crossref" in database:
            fetcher = crossref_fetcher
            executed_via = "Crossref REST API"
        elif "europe pmc" in database:
            fetcher = europe_pmc_fetcher
            executed_via = "Europe PMC REST API"
        else:
            output.append(row)
            continue
        try:
            row["hit_count"] = fetcher(str(row.get("query", "")))
            row["status"] = "executed_public_api"
            row["executed_via"] = executed_via
            row["executed_at_utc"] = timestamp
            row["execution_error"] = ""
        except Exception as exc:
            row["hit_count"] = ""
            row["status"] = "execution_failed"
            row["executed_via"] = executed_via
            row["executed_at_utc"] = timestamp
            row["execution_error"] = f"{type(exc).__name__}: {exc}"
        output.append(row)
    return output
