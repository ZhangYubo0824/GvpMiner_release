from __future__ import annotations

import csv
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GO_RE = re.compile(r"GO:\d+")
KO_RE = re.compile(r"K\d{5}")
PFAM_RE = re.compile(r"PF\d{5}")
IPR_RE = re.compile(r"IPR\d{6}")
EC_RE = re.compile(r"\d+\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)")


@dataclass(frozen=True)
class AnnotationTerm:
    source: str
    namespace: str
    term_id: str
    protein_id: str
    term_name: str = ""


@dataclass
class AnnotationFilterAudit:
    parsed: Counter[tuple[str, str]]
    retained: Counter[tuple[str, str]]

    def __init__(self) -> None:
        self.parsed = Counter()
        self.retained = Counter()

    def record(self, term: AnnotationTerm, *, retained: bool) -> None:
        key = (term.source, term.namespace)
        self.parsed[key] += 1
        if retained:
            self.retained[key] += 1


def _clean(value: object) -> str:
    return str(value or "").strip()


def _split_multi(value: object) -> list[str]:
    text = _clean(value)
    if not text or text == "-":
        return []
    for separator in ["|", ";"]:
        text = text.replace(separator, ",")
    return [part.strip() for part in text.split(",") if part.strip() and part.strip() != "-"]


def _term(source: str, namespace: str, term_id: str, protein_id: str, term_name: str = "") -> AnnotationTerm | None:
    term_id = _clean(term_id)
    protein_id = _clean(protein_id)
    if not term_id or not protein_id or term_id == "-":
        return None
    return AnnotationTerm(source, namespace, term_id, protein_id, _clean(term_name))


def iter_interproscan_tsv(
    path: Path,
    *,
    include_pathways: bool = False,
    audit: AnnotationFilterAudit | None = None,
) -> Iterable[AnnotationTerm]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            raw = raw.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            fields = raw.split("\t")
            if len(fields) < 5:
                continue
            protein_id = fields[0]
            analysis = fields[3] if len(fields) > 3 else ""
            signature_accession = fields[4] if len(fields) > 4 else ""
            signature_name = fields[5] if len(fields) > 5 else ""
            interpro_accession = fields[11] if len(fields) > 11 else ""
            interpro_name = fields[12] if len(fields) > 12 else ""
            go_text = fields[13] if len(fields) > 13 else ""
            pathway_text = fields[14] if len(fields) > 14 else ""

            if analysis == "Pfam" or PFAM_RE.fullmatch(signature_accession):
                term = _term("InterProScan", "Pfam", signature_accession, protein_id, signature_name)
                if term:
                    if audit:
                        audit.record(term, retained=True)
                    yield term
            if IPR_RE.fullmatch(interpro_accession):
                term = _term("InterProScan", "InterPro", interpro_accession, protein_id, interpro_name)
                if term:
                    if audit:
                        audit.record(term, retained=True)
                    yield term
            for go_id in GO_RE.findall(go_text):
                term = _term("InterProScan", "GO", go_id, protein_id)
                if term:
                    if audit:
                        audit.record(term, retained=True)
                    yield term
            if include_pathways or audit is not None:
                for pathway in _split_multi(pathway_text):
                    term = _term("InterProScan", "Pathway", pathway, protein_id)
                    if term:
                        if audit:
                            audit.record(term, retained=include_pathways)
                        if include_pathways:
                            yield term




def _iter_eggnog_rows(path: Path) -> Iterable[dict[str, str]]:
    header: list[str] | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("#query"):
                header = [field.lstrip("#") for field in line.split("\t")]
                continue
            if line.startswith("#") or header is None:
                continue
            values = line.split("\t")
            values.extend([""] * max(0, len(header) - len(values)))
            yield dict(zip(header, values[:len(header)]))


def iter_eggnog_annotations(path: Path) -> Iterable[AnnotationTerm]:
    for row in _iter_eggnog_rows(path):
        protein_id = row.get("query", "")
        for go_id in GO_RE.findall(row.get("GOs", "")):
            term = _term("eggNOG", "GO", go_id, protein_id)
            if term:
                yield term
        for ko in KO_RE.findall(row.get("KEGG_ko", "")):
            term = _term("eggNOG", "KEGG_KO", ko, protein_id)
            if term:
                yield term
        for pathway in _split_multi(row.get("KEGG_Pathway", "")):
            term = _term("eggNOG", "KEGG_Pathway", pathway, protein_id)
            if term:
                yield term
        for ec in EC_RE.findall(row.get("EC", "")):
            term = _term("eggNOG", "EC", ec, protein_id)
            if term:
                yield term
        for pfam in PFAM_RE.findall(row.get("PFAMs", "")):
            term = _term("eggNOG", "Pfam", pfam, protein_id)
            if term:
                yield term
        cog = _clean(row.get("COG_category", ""))
        if cog and cog != "-":
            for category in cog:
                term = _term("eggNOG", "COG", f"COG:{category}", protein_id)
                if term:
                    yield term




def iter_kofamscan_tsv(path: Path) -> Iterable[AnnotationTerm]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith("*"):
                continue
            fields = line.split(maxsplit=6)
            if len(fields) < 6 or fields[0] != "*":
                continue
            protein_id = fields[1]
            ko_id = fields[2]
            ko_match = KO_RE.fullmatch(ko_id)
            if not ko_match:
                continue
            name = fields[6] if len(fields) > 6 else ""
            term = _term("KofamScan", "KEGG_KO", ko_id, protein_id, name)
            if term:
                yield term






def write_filter_audit(audit: AnnotationFilterAudit, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "namespace",
                "parsed_links",
                "retained_links",
                "filtered_links",
                "filter_reason",
            ]
        )
        for source, namespace in sorted(audit.parsed):
            parsed = audit.parsed[(source, namespace)]
            retained = audit.retained[(source, namespace)]
            reason = (
                "excluded_by_default_unless_include_interpro_pathways"
                if source == "InterProScan" and namespace == "Pathway" and parsed > retained
                else ""
            )
            writer.writerow(
                [source, namespace, parsed, retained, parsed - retained, reason]
            )
    temporary.replace(path)
    return path


def write_term_tables(
    terms: Iterable[AnnotationTerm],
    output_dir: Path,
    *,
    batch_size: int = 10_000,
) -> dict[str, Path]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "functional_annotations_long.csv"
    term2gene_path = output_dir / "term2gene.csv"
    term2name_path = output_dir / "term2name.csv"
    namespace_counts_path = output_dir / "annotation_namespace_counts.csv"
    database_path = output_dir / ".annotation_writer.sqlite3"
    temporary_paths = {
        path: path.with_name(path.name + ".tmp")
        for path in [long_path, term2gene_path, term2name_path, namespace_counts_path]
    }
    database_artifacts = [
        database_path,
        database_path.with_name(database_path.name + "-wal"),
        database_path.with_name(database_path.name + "-shm"),
    ]
    for path in [*database_artifacts, *temporary_paths.values()]:
        path.unlink(missing_ok=True)

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute(
        """
        CREATE TABLE annotation (
            source TEXT NOT NULL,
            namespace TEXT NOT NULL,
            term_id TEXT NOT NULL,
            protein_id TEXT NOT NULL,
            term_name TEXT NOT NULL,
            PRIMARY KEY (source, namespace, term_id, protein_id)
        ) WITHOUT ROWID
        """
    )
    batch: list[tuple[str, str, str, str, str]] = []

    def flush() -> None:
        connection.executemany(
            """
            INSERT INTO annotation(source, namespace, term_id, protein_id, term_name)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source, namespace, term_id, protein_id) DO UPDATE
            SET term_name = excluded.term_name
            WHERE annotation.term_name = '' AND excluded.term_name <> ''
            """,
            batch,
        )
        connection.commit()
        batch.clear()

    completed = False
    try:
        for term in terms:
            batch.append(
                (
                    term.source,
                    term.namespace,
                    term.term_id,
                    term.protein_id,
                    term.term_name,
                )
            )
            if len(batch) >= batch_size:
                flush()
        if batch:
            flush()
        connection.execute(
            "CREATE INDEX annotation_term_gene ON annotation(term_id, protein_id)"
        )
        connection.commit()

        with temporary_paths[long_path].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source", "namespace", "term_id", "term_name", "protein_id"])
            for source, namespace, term_id, term_name, protein_id in connection.execute(
                """
                SELECT source, namespace, term_id, term_name, protein_id
                FROM annotation
                ORDER BY source, namespace, term_id, protein_id
                """
            ):
                writer.writerow([source, namespace, term_id, term_name, protein_id])

        with temporary_paths[term2gene_path].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["term_id", "protein_id"])
            writer.writerows(
                connection.execute(
                    """
                    SELECT DISTINCT term_id, protein_id
                    FROM annotation
                    ORDER BY term_id, protein_id
                    """
                )
            )

        with temporary_paths[term2name_path].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["term_id", "term_name"])
            for term_id, term_name in connection.execute(
                """
                SELECT term_id, MAX(NULLIF(term_name, ''))
                FROM annotation
                GROUP BY term_id
                ORDER BY term_id
                """
            ):
                writer.writerow([term_id, term_name or term_id])

        with temporary_paths[namespace_counts_path].open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["source", "namespace", "annotation_links", "unique_terms", "unique_proteins"]
            )
            writer.writerows(
                connection.execute(
                    """
                    SELECT source, namespace, COUNT(*),
                           COUNT(DISTINCT term_id), COUNT(DISTINCT protein_id)
                    FROM annotation
                    GROUP BY source, namespace
                    ORDER BY source, namespace
                    """
                )
            )
        completed = True
    finally:
        connection.close()
        if not completed:
            for path in [*database_artifacts, *temporary_paths.values()]:
                path.unlink(missing_ok=True)

    for final_path, temporary_path in temporary_paths.items():
        temporary_path.replace(final_path)
    for path in database_artifacts:
        path.unlink(missing_ok=True)

    return {
        "long": long_path,
        "term2gene": term2gene_path,
        "term2name": term2name_path,
        "namespace_counts": namespace_counts_path,
    }
