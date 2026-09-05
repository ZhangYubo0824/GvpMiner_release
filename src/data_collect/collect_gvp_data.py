#!/usr/bin/env python3
"""
Comprehensive Gvp (Gas Vesicle Protein) Multi-Database Data Collector (v2)
==========================================================================
Collects data for all 23 Gvp families (A–W) from:
  1. NCBI GenBank / RefSeq  (via Entrez API)
  2. UniProtKB / Swiss-Prot  (via REST API, full pagination)
  3. InterPro / Pfam          (via REST API, all families, no limits)
  4. AlphaFold DB             (via REST API, no limits)
  5. Literature-curated reference table

Changes from v1:
  - Removed all data quantity limits (NCBI, UniProt, InterPro, AlphaFold)
  - Expanded InterPro/Pfam entries to cover ALL Gvp families (A-W)
  - Added text-based InterPro search for families without dedicated accessions
  - Full pagination support for UniProt REST API
  - Output to data2/ directory

Usage:
    python collect_gvp_data.py [--email YOUR_EMAIL] [--skip-alphafold] [--families A,C,N]
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
import re
from pathlib import Path
from io import StringIO
from datetime import datetime

import requests
from Bio import Entrez, SeqIO

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()

GVP_FAMILIES = list("ABCDEFGHIJKLMNOPQRSTUVW")

# NCBI search queries per family - broadened for comprehensive collection
NCBI_QUERIES = {
    f: f'(gvp{f}[Gene Name] OR gvp{f}[Protein Name] OR "gas vesicle protein {f}"[Protein Name]) AND "gas vesicle"'
    for f in GVP_FAMILIES
}

# Broadened fallback queries for families with few results
NCBI_BROAD_QUERIES = {
    f: f'gvp{f}[Gene Name] OR "gas vesicle protein {f}"[Title]'
    for f in GVP_FAMILIES
}

# Pfam / InterPro accessions relevant to ALL Gvp families
# Expanded to cover as many families as possible
INTERPRO_ENTRIES = {
    # GvpA / GvpJ / GvpM related (structural shell proteins share this domain)
    "PF00741":  "Gas_vesicle (GvpA/GvpJ/GvpM domain)",
    "IPR008654": "Gas vesicle protein GvpA",
    "IPR047870": "Gas vesicle protein GvpA superfamily",
    "IPR050530": "Gas vesicle protein GvpA-like",
    "IPR018493": "Gas vesicle protein GvpA conserved site",
    # GvpC related (outer reinforcement protein)
    "PF01304":  "GVPc repeated domain (GvpC)",
    "IPR000638": "Gas vesicle protein GvpC repeat",
    # GvpD related (negative regulator, P-loop NTPase)
    "PF07088":  "GvpD gas vesicle protein, P-loop domain",
    "IPR010645": "Gas vesicle protein GvpD",
    # GvpF / GvpL related (accessory / scaffold proteins)
    "PF06386":  "Gas vesicle synthesis protein GvpL/GvpF",
    "IPR009430": "Gas vesicle synthesis protein GvpL/GvpF",
    # GvpG related (accessory multimer)
    "PF05120":  "Gas vesicle protein GvpG",
    # GvpO related (essential for formation)
    "PF05800":  "Gas vesicle protein GvpO",
}

# Gvp families that DO NOT have a dedicated Pfam/InterPro accession
# These will be searched using InterPro text search API
GVP_TEXT_SEARCH_FAMILIES = {
    "GvpB": ["gas vesicle protein B", "GvpB"],
    "GvpE": ["gas vesicle protein E", "GvpE transcriptional activator"],
    "GvpH": ["gas vesicle protein H", "GvpH"],
    "GvpI": ["gas vesicle protein I", "GvpI"],
    "GvpK": ["gas vesicle protein K", "GvpK"],
    "GvpN": ["gas vesicle protein N", "GvpN ATPase", "gas vesicle AAA-ATPase"],
    "GvpP": ["gas vesicle protein P", "GvpP"],
    "GvpQ": ["gas vesicle protein Q", "GvpQ"],
    "GvpR": ["gas vesicle protein R", "GvpR"],
    "GvpS": ["gas vesicle protein S", "GvpS"],
    "GvpT": ["gas vesicle protein T", "GvpT"],
    "GvpU": ["gas vesicle protein U", "GvpU"],
    "GvpV": ["gas vesicle protein V", "GvpV"],
    "GvpW": ["gas vesicle protein W", "GvpW"],
}

# UniProt REST API base
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

# AlphaFold API base
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"

# Rate limiting
NCBI_DELAY = 0.4  # seconds between NCBI requests
API_DELAY = 0.3   # seconds between other API calls

# Batch sizes
NCBI_BATCH_SIZE = 500
UNIPROT_PAGE_SIZE = 500

# Literature reference data (curated from key publications)
LITERATURE_DATA = [
    # GvpA
    {"family": "GvpA", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPD4", "ncbi": "AAG19706.1",
     "function": "Major structural shell protein of gas vesicles, forms ribs via alpha-helix/beta-sheet",
     "key_reference": "Walsby 1994; Pfeifer 2012", "doi": "10.1128/mr.58.1.94-144.1994"},
    {"family": "GvpA", "organism": "Anabaena flos-aquae", "uniprot": "P10397", "ncbi": "P10397.3",
     "function": "GvpA structural monomer, highly conserved in cyanobacteria",
     "key_reference": "Hayes et al. 1992", "doi": "10.1016/0022-2836(92)90533-P"},
    {"family": "GvpA", "organism": "Bacillus megaterium", "uniprot": "O68680", "ncbi": "O68680.1",
     "function": "GvpA1 structural component in Firmicutes gas vesicles",
     "key_reference": "Li & Cannon 1998", "doi": "10.1128/JB.180.9.2450-2458.1998"},
    {"family": "GvpA", "organism": "Microcystis aeruginosa", "uniprot": "P08412", "ncbi": "P08412.3",
     "function": "Major gas vesicle shell protein in bloom-forming cyanobacterium",
     "key_reference": "Damerval et al. 1987", "doi": "10.1099/00221287-133-10-2695"},
    {"family": "GvpA", "organism": "Serratia sp. ATCC 39006", "uniprot": "Q8GIK6", "ncbi": "AAM47041.1",
     "function": "Structural protein enabling buoyancy in Enterobacteria",
     "key_reference": "Ramsay et al. 2011", "doi": "10.1111/j.1365-2958.2011.07901.x"},
    # GvpC
    {"family": "GvpC", "organism": "Anabaena flos-aquae", "uniprot": "P09413", "ncbi": "P09413.1",
     "function": "Outer reinforcement protein with 5x33aa tandem repeats, stabilizes shell",
     "key_reference": "Hayes et al. 1988; Buchholz et al. 1993", "doi": "10.1016/S0022-2836(88)80029-3"},
    {"family": "GvpC", "organism": "Halobacterium salinarum", "uniprot": "P19975", "ncbi": "AAA72325.1",
     "function": "GvpC stabilizer, fewer repeats than cyanobacterial homologs",
     "key_reference": "Halladay et al. 1993", "doi": "10.1128/JB.175.3.684-692.1993"},
    {"family": "GvpC", "organism": "Microcystis aeruginosa PCC 7806", "uniprot": "P81001", "ncbi": "ARI81316.1",
     "function": "GvpC with multiple GvpC repeat domains stiffening the vesicle",
     "key_reference": "Mlouka et al. 2004", "doi": "10.1099/mic.0.26771-0"},
    # GvpD
    {"family": "GvpD", "organism": "Halobacterium salinarum", "uniprot": "P19030", "ncbi": "AAA72316.1",
     "function": "Negative regulator; p-loop NTPase that represses GvpA expression",
     "key_reference": "Pfeifer 2012", "doi": "10.1038/nrmicro2834"},
    # GvpE
    {"family": "GvpE", "organism": "Halobacterium salinarum", "uniprot": "Q48331", "ncbi": "AAA72317.1",
     "function": "Transcriptional activator of gvpA promoter",
     "key_reference": "Zimmermann & Pfeifer 2003", "doi": "10.1046/j.1365-2958.2003.03437.x"},
    # GvpF
    {"family": "GvpF", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPD0", "ncbi": "AAG19710.1",
     "function": "Accessory protein, interacts with GvpA; coiled-coil structure",
     "key_reference": "Völkner et al. 2020", "doi": "10.3389/fmicb.2020.01651"},
    # GvpG
    {"family": "GvpG", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPC9", "ncbi": "AAG19711.1",
     "function": "Accessory protein, forms multimers, essential for early assembly",
     "key_reference": "Völkner et al. 2020", "doi": "10.3389/fmicb.2020.01651"},
    # GvpH
    {"family": "GvpH", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPC8", "ncbi": "AAG19712.1",
     "function": "Accessory protein, its absence produces altered gas vesicle shapes",
     "key_reference": "Offner et al. 2000", "doi": "10.1046/j.1365-2958.2000.01990.x"},
    # GvpI
    {"family": "GvpI", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPC7", "ncbi": "AAG19713.1",
     "function": "Accessory protein essential for vesicle formation; haloarchaea-specific",
     "key_reference": "Pfeifer 2012", "doi": "10.1038/nrmicro2834"},
    # GvpJ
    {"family": "GvpJ", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPC6", "ncbi": "AAG19714.1",
     "function": "GvpA homolog, minor structural component, multimer formation",
     "key_reference": "Völkner et al. 2020", "doi": "10.3389/fmicb.2020.01651"},
    # GvpK
    {"family": "GvpK", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPC5", "ncbi": "AAG19715.1",
     "function": "Accessory protein that forms multimers, interacts with GvpL",
     "key_reference": "Völkner et al. 2020", "doi": "10.3389/fmicb.2020.01651"},
    # GvpL
    {"family": "GvpL", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPC4", "ncbi": "AAG19716.1",
     "function": "Platform/scaffold protein bringing accessory Gvps together",
     "key_reference": "Völkner et al. 2020", "doi": "10.3389/fmicb.2020.01651"},
    # GvpM
    {"family": "GvpM", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPC3", "ncbi": "AAG19717.1",
     "function": "GvpA-like nucleation seed; initiates vesicle assembly",
     "key_reference": "Tavlaridou et al. 2014", "doi": "10.1111/mmi.12766"},
    # GvpN
    {"family": "GvpN", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPD3", "ncbi": "AAG19707.1",
     "function": "AAA-type ATPase required for vesicle enlargement",
     "key_reference": "Pfeifer 2012", "doi": "10.1038/nrmicro2834"},
    {"family": "GvpN", "organism": "Anabaena flos-aquae", "uniprot": "P30118", "ncbi": "P30118.1",
     "function": "ATPase for gas vesicle growth in cyanobacteria",
     "key_reference": "DasSarma & DasSarma 2015", "doi": "10.1007/978-3-319-22834-0_6"},
    # GvpO
    {"family": "GvpO", "organism": "Halobacterium salinarum NRC-1", "uniprot": "Q9HPD2", "ncbi": "AAG19708.1",
     "function": "Essential for vesicle formation; function not fully characterized",
     "key_reference": "Pfeifer 2012", "doi": "10.1038/nrmicro2834"},
]

# ============================================================
# Logging Setup
# ============================================================
def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


# ============================================================
# 1. NCBI GenBank / RefSeq (NO quantity limits)
# ============================================================
def collect_ncbi(output_dir: Path, email: str, families: list):
    """Download protein sequences and metadata from NCBI for each Gvp family.
    No quantity limits - all results are downloaded in batches."""
    Entrez.email = email
    ncbi_dir = output_dir / "ncbi"
    summary_rows = []

    # Prepare search targets (pure families + hybrid families)
    search_targets = []
    for f in families:
        search_targets.append((f"Gvp{f}", NCBI_QUERIES[f], NCBI_BROAD_QUERIES.get(f, "")))
    
    # Add specific hybrid queries to isolate multi-domain or cross-annotated sequences
    search_targets.append(("GvpL_F_hybrid", 'gvpL[All Fields] AND gvpF[All Fields] AND "gas vesicle"', ''))
    search_targets.append(("GvpA_J_hybrid", 'gvpA[All Fields] AND gvpJ[All Fields] AND "gas vesicle"', ''))

    for folder_name, query, query_broad in search_targets:
        family_dir = ncbi_dir / folder_name
        family_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[NCBI] Searching {folder_name}: {query}")

        try:
            # Search
            handle = Entrez.esearch(db="protein", term=query, retmax=100000, usehistory="y")
            result = Entrez.read(handle)
            handle.close()
            count = int(result["Count"])
            logging.info(f"[NCBI] {folder_name}: {count} proteins found")

            if count == 0 and query_broad:
                # Try broader search
                logging.info(f"[NCBI] Trying broader query for {folder_name}: {query_broad}")
                handle = Entrez.esearch(db="protein", term=query_broad, retmax=100000, usehistory="y")
                result = Entrez.read(handle)
                handle.close()
                count = int(result["Count"])
                logging.info(f"[NCBI] {folder_name} (broad): {count} proteins found")

            if count == 0:
                summary_rows.append({"family": folder_name, "ncbi_count": 0, "status": "no_results"})
                continue

            webenv = result["WebEnv"]
            query_key = result["QueryKey"]

            # Download FASTA - ALL results, no limit
            fasta_path = family_dir / f"{folder_name}_proteins.fasta"
            batch_size = NCBI_BATCH_SIZE
            with open(fasta_path, "w") as fasta_out:
                for start in range(0, count, batch_size):
                    time.sleep(NCBI_DELAY)
                    logging.info(f"[NCBI] {folder_name}: downloading FASTA batch {start}-{min(start+batch_size, count)}/{count}")
                    fetch = Entrez.efetch(
                        db="protein", rettype="fasta", retmode="text",
                        retstart=start, retmax=batch_size,
                        webenv=webenv, query_key=query_key,
                    )
                    fasta_out.write(fetch.read())
                    fetch.close()
            logging.info(f"[NCBI] {folder_name}: saved FASTA -> {fasta_path}")

            # Download GenPept (metadata) - ALL results, NO 2000 limit
            metadata_rows = []
            for start in range(0, count, batch_size):
                time.sleep(NCBI_DELAY)
                logging.info(f"[NCBI] {folder_name}: downloading metadata batch {start}-{min(start+batch_size, count)}/{count}")
                try:
                    fetch = Entrez.efetch(
                        db="protein", rettype="gp", retmode="text",
                        retstart=start, retmax=batch_size,
                        webenv=webenv, query_key=query_key,
                    )
                    text = fetch.read()
                    fetch.close()
                    # Parse GenPept records
                    for record in SeqIO.parse(StringIO(text), "genbank"):
                        row = {
                            "protein_id": record.id,
                            "protein_name": record.name,
                            "description": record.description,
                            "organism": record.annotations.get("organism", ""),
                            "taxonomy": "; ".join(record.annotations.get("taxonomy", [])),
                            "sequence_length": len(record.seq),
                            "sequence": str(record.seq),
                            "accessions": "; ".join(record.annotations.get("accessions", [])),
                            "db_source": record.annotations.get("db_source", ""),
                            "date": record.annotations.get("date", ""),
                            "family": folder_name,
                        }
                        # Extract CDS/features
                        domains = []
                        for feat in record.features:
                            if feat.type == "Region":
                                note = feat.qualifiers.get("region_name", [""])[0]
                                db_xref = feat.qualifiers.get("db_xref", [""])[0]
                                domains.append(f"{note}|{db_xref}")
                        row["domains"] = "; ".join(domains)
                        metadata_rows.append(row)
                except Exception as batch_err:
                    logging.warning(f"[NCBI] {folder_name}: batch {start} metadata error: {batch_err}, continuing...")

            # Save metadata CSV
            meta_path = family_dir / f"{folder_name}_metadata.csv"
            if metadata_rows:
                keys = metadata_rows[0].keys()
                with open(meta_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(metadata_rows)
                logging.info(f"[NCBI] {folder_name}: saved {len(metadata_rows)} metadata rows -> {meta_path}")

            summary_rows.append({
                "family": folder_name,
                "ncbi_count": count,
                "fasta_records": len(metadata_rows),
                "status": "ok",
            })

        except Exception as e:
            logging.error(f"[NCBI] {folder_name} error: {e}")
            summary_rows.append({"family": folder_name, "ncbi_count": 0, "status": f"error: {e}"})

    # Write summary
    summary_path = ncbi_dir / "all_gvp_ncbi_summary.csv"
    if summary_rows:
        all_keys = ["family", "ncbi_count", "fasta_records", "status"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})
    logging.info(f"[NCBI] Summary saved -> {summary_path}")
    return summary_rows


# ============================================================
# 2. UniProtKB / Swiss-Prot (full pagination, NO size limit)
# ============================================================
def _get_next_link(response):
    """Extract 'next' page URL from UniProt Link header."""
    link_header = response.headers.get("Link", "")
    re_next = re.compile(r'<(.+)>; rel="next"')
    match = re_next.search(link_header)
    if match:
        return match.group(1)
    return None


def collect_uniprot(output_dir: Path, families: list):
    """Download protein data from UniProt REST API for each Gvp family.
    Uses full pagination - NO size limit."""
    uniprot_dir = output_dir / "uniprot"
    uniprot_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for family in families:
        logging.info(f"[UniProt] Searching Gvp{family}...")
        # Query: search for gas vesicle proteins with this gene name
        query = f'(gene:gvp{family} OR protein_name:"gas vesicle protein {family}") AND keyword:"gas vesicle"'

        try:
            # --- Download TSV with full pagination ---
            all_tsv_lines = []
            header_line = None
            page_url = f"{UNIPROT_API}/search"
            params = {
                "query": query,
                "format": "tsv",
                "fields": "accession,id,gene_names,protein_name,organism_name,organism_id,lineage,length,mass,sequence,go_id,go_p,go_c,go_f,xref_pfam,xref_interpro,xref_pdb,xref_alphafolddb,reviewed,annotation_score",
                "size": UNIPROT_PAGE_SIZE,
            }
            is_first_request = True

            while page_url:
                time.sleep(API_DELAY)
                if is_first_request:
                    resp = requests.get(page_url, params=params, timeout=60)
                    is_first_request = False
                else:
                    resp = requests.get(page_url, timeout=60)

                if resp.status_code != 200:
                    logging.warning(f"[UniProt] Gvp{family}: HTTP {resp.status_code} at {page_url}")
                    break

                lines = resp.text.strip().split("\n")
                if header_line is None and lines:
                    header_line = lines[0]
                    all_tsv_lines.extend(lines[1:])
                else:
                    # Skip header on subsequent pages
                    if lines and lines[0] == header_line:
                        all_tsv_lines.extend(lines[1:])
                    else:
                        all_tsv_lines.extend(lines)

                page_url = _get_next_link(resp)
                if page_url:
                    logging.info(f"[UniProt] Gvp{family}: fetched {len(all_tsv_lines)} entries so far, getting next page...")

            # If strict query yields nothing, try without keyword filter
            if not all_tsv_lines:
                query_broad = f'(gene:gvp{family} OR protein_name:"gas vesicle protein {family}")'
                logging.info(f"[UniProt] Gvp{family}: no results with keyword filter, trying broader query")
                page_url = f"{UNIPROT_API}/search"
                params["query"] = query_broad
                is_first_request = True
                header_line = None

                while page_url:
                    time.sleep(API_DELAY)
                    if is_first_request:
                        resp = requests.get(page_url, params=params, timeout=60)
                        is_first_request = False
                    else:
                        resp = requests.get(page_url, timeout=60)

                    if resp.status_code != 200:
                        break

                    lines = resp.text.strip().split("\n")
                    if header_line is None and lines:
                        header_line = lines[0]
                        all_tsv_lines.extend(lines[1:])
                    else:
                        if lines and lines[0] == header_line:
                            all_tsv_lines.extend(lines[1:])
                        else:
                            all_tsv_lines.extend(lines)

                    page_url = _get_next_link(resp)

            record_count = len(all_tsv_lines)

            if record_count > 0 and header_line:
                # Save TSV
                tsv_path = uniprot_dir / f"Gvp{family}_uniprot.tsv"
                with open(tsv_path, "w", encoding="utf-8") as f:
                    f.write(header_line + "\n")
                    f.write("\n".join(all_tsv_lines) + "\n")
                logging.info(f"[UniProt] Gvp{family}: {record_count} entries -> {tsv_path}")

                # Also download FASTA with full pagination
                fasta_parts = []
                fasta_url = f"{UNIPROT_API}/search"
                fasta_params = {
                    "query": params["query"],
                    "format": "fasta",
                    "size": UNIPROT_PAGE_SIZE,
                }
                is_first_fasta = True

                while fasta_url:
                    time.sleep(API_DELAY)
                    if is_first_fasta:
                        fasta_resp = requests.get(fasta_url, params=fasta_params, timeout=60)
                        is_first_fasta = False
                    else:
                        fasta_resp = requests.get(fasta_url, timeout=60)

                    if fasta_resp.status_code == 200:
                        fasta_parts.append(fasta_resp.text)
                    else:
                        break
                    fasta_url = _get_next_link(fasta_resp)

                if fasta_parts:
                    fasta_path = uniprot_dir / f"Gvp{family}_uniprot.fasta"
                    with open(fasta_path, "w", encoding="utf-8") as f:
                        f.write("".join(fasta_parts))
                    logging.info(f"[UniProt] Gvp{family}: FASTA saved -> {fasta_path}")

                summary_rows.append({"family": f"Gvp{family}", "uniprot_count": record_count, "status": "ok"})
            else:
                logging.warning(f"[UniProt] Gvp{family}: no results found")
                summary_rows.append({"family": f"Gvp{family}", "uniprot_count": 0, "status": "no_results"})

        except Exception as e:
            logging.error(f"[UniProt] Gvp{family} error: {e}")
            summary_rows.append({"family": f"Gvp{family}", "uniprot_count": 0, "status": f"error: {e}"})

        time.sleep(API_DELAY)

    # Write summary
    summary_path = uniprot_dir / "all_gvp_uniprot_summary.csv"
    if summary_rows:
        keys = summary_rows[0].keys()
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
    logging.info(f"[UniProt] Summary saved -> {summary_path}")
    return summary_rows


# ============================================================
# 3. InterPro / Pfam (ALL families, NO quantity limits)
# ============================================================
def collect_interpro(output_dir: Path):
    """Download domain-annotated protein lists from InterPro/Pfam.
    Covers all Gvp families. No quantity limits - full pagination."""
    interpro_dir = output_dir / "interpro"
    interpro_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    # --- Part A: Accession-based queries (known Pfam/InterPro entries) ---
    for accession, desc in INTERPRO_ENTRIES.items():
        logging.info(f"[InterPro] Fetching {accession}: {desc}")

        # Determine DB type
        if accession.startswith("PF"):
            db_type = "pfam"
        elif accession.startswith("IPR"):
            db_type = "interpro"
        else:
            continue

        try:
            # Get protein list from InterPro API
            if db_type == "pfam":
                url = f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/entry/pfam/{accession}/?page_size=200"
            else:
                url = f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/entry/interpro/{accession}/?page_size=200"

            all_proteins = []
            page_url = url
            page_num = 0

            # NO LIMIT - download ALL pages
            while page_url:
                time.sleep(API_DELAY)
                page_num += 1
                resp = requests.get(page_url, headers={"Accept": "application/json"}, timeout=120)
                if resp.status_code != 200:
                    logging.warning(f"[InterPro] {accession}: HTTP {resp.status_code} on page {page_num}")
                    break
                data = resp.json()
                results = data.get("results", [])
                for entry in results:
                    meta = entry.get("metadata", {})
                    all_proteins.append({
                        "interpro_accession": accession,
                        "interpro_description": desc,
                        "uniprot_accession": meta.get("accession", ""),
                        "protein_name": meta.get("name", ""),
                        "source_organism": meta.get("source_organism", {}).get("scientificName", ""),
                        "taxon_id": meta.get("source_organism", {}).get("taxId", ""),
                        "length": meta.get("length", ""),
                    })
                page_url = data.get("next")
                if page_url and page_num % 5 == 0:
                    logging.info(f"[InterPro] {accession}: {len(all_proteins)} proteins collected so far (page {page_num})...")

            # Save
            out_path = interpro_dir / f"{db_type}_{accession}_proteins.tsv"
            if all_proteins:
                keys = all_proteins[0].keys()
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
                    writer.writeheader()
                    writer.writerows(all_proteins)
            logging.info(f"[InterPro] {accession}: {len(all_proteins)} proteins -> {out_path}")
            summary_rows.append({
                "accession": accession,
                "description": desc,
                "protein_count": len(all_proteins),
                "status": "ok",
            })

        except Exception as e:
            logging.error(f"[InterPro] {accession} error: {e}")
            summary_rows.append({"accession": accession, "description": desc, "protein_count": 0, "status": f"error: {e}"})

    # --- Part B: Text-based search for families without dedicated accessions ---
    logging.info("[InterPro] Starting text-based search for additional Gvp families...")
    for family_name, search_terms in GVP_TEXT_SEARCH_FAMILIES.items():
        logging.info(f"[InterPro] Text search for {family_name}...")
        all_text_proteins = []

        for term in search_terms:
            try:
                # Search UniProt via InterPro text search
                search_url = f"https://www.ebi.ac.uk/interpro/api/protein/UniProt/?search={term}&page_size=200"
                page_url = search_url

                while page_url:
                    time.sleep(API_DELAY)
                    resp = requests.get(page_url, headers={"Accept": "application/json"}, timeout=120)
                    if resp.status_code != 200:
                        logging.warning(f"[InterPro] Text search '{term}': HTTP {resp.status_code}")
                        break
                    data = resp.json()
                    for entry in data.get("results", []):
                        meta = entry.get("metadata", {})
                        protein_entry = {
                            "interpro_accession": f"text_search:{family_name}",
                            "interpro_description": f"Text search: {term}",
                            "uniprot_accession": meta.get("accession", ""),
                            "protein_name": meta.get("name", ""),
                            "source_organism": meta.get("source_organism", {}).get("scientificName", ""),
                            "taxon_id": meta.get("source_organism", {}).get("taxId", ""),
                            "length": meta.get("length", ""),
                        }
                        # Deduplicate by UniProt accession
                        if protein_entry["uniprot_accession"] not in {p["uniprot_accession"] for p in all_text_proteins}:
                            all_text_proteins.append(protein_entry)
                    page_url = data.get("next")

            except Exception as e:
                logging.warning(f"[InterPro] Text search '{term}' error: {e}")

        # Save text-search results
        if all_text_proteins:
            out_path = interpro_dir / f"text_search_{family_name}_proteins.tsv"
            keys = all_text_proteins[0].keys()
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
                writer.writeheader()
                writer.writerows(all_text_proteins)
            logging.info(f"[InterPro] {family_name} text search: {len(all_text_proteins)} proteins -> {out_path}")
        else:
            logging.info(f"[InterPro] {family_name} text search: no results")

        summary_rows.append({
            "accession": f"text:{family_name}",
            "description": f"Text search for {family_name}",
            "protein_count": len(all_text_proteins),
            "status": "ok" if all_text_proteins else "no_results",
        })

    # Write summary
    summary_path = interpro_dir / "interpro_domain_summary.csv"
    if summary_rows:
        keys = summary_rows[0].keys()
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
    logging.info(f"[InterPro] Summary saved -> {summary_path}")
    return summary_rows


# ============================================================
# 4. AlphaFold DB (NO structure count limit)
# ============================================================
def collect_alphafold(output_dir: Path):
    """Download predicted structures from AlphaFold DB for ALL UniProt accessions.
    No quantity limit - downloads all available structures."""
    af_dir = output_dir / "alphafold"
    af_dir.mkdir(parents=True, exist_ok=True)
    download_log = []

    # Collect UniProt accessions from the UniProt TSV files already downloaded
    uniprot_dir = output_dir / "uniprot"
    accessions = set()

    if uniprot_dir.exists():
        for tsv_file in uniprot_dir.glob("*.tsv"):
            try:
                with open(tsv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        acc = row.get("Entry", "") or row.get("accession", "")
                        if acc and "." not in acc:  # valid UniProt accession format
                            accessions.add(acc)
            except Exception:
                pass

    # Also collect from InterPro results
    interpro_dir = output_dir / "interpro"
    if interpro_dir.exists():
        for tsv_file in interpro_dir.glob("*.tsv"):
            try:
                with open(tsv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        acc = row.get("uniprot_accession", "")
                        if acc and "." not in acc:
                            accessions.add(acc)
            except Exception:
                pass

    # Also add known Swiss-Prot accessions from literature
    for entry in LITERATURE_DATA:
        up = entry.get("uniprot", "")
        if up and "." not in up:
            accessions.add(up)

    logging.info(f"[AlphaFold] {len(accessions)} unique UniProt accessions to query")

    downloaded = 0
    failed = 0
    for i, acc in enumerate(sorted(accessions), 1):
        try:
            time.sleep(API_DELAY)
            # Get prediction metadata
            resp = requests.get(f"{ALPHAFOLD_API}/prediction/{acc}", timeout=30)
            if resp.status_code != 200:
                download_log.append({"uniprot": acc, "status": f"not_found (HTTP {resp.status_code})"})
                failed += 1
                continue

            pred_data = resp.json()
            if isinstance(pred_data, list) and len(pred_data) > 0:
                pred_data = pred_data[0]

            # Download PDB file
            pdb_url = pred_data.get("pdbUrl", "")

            if pdb_url:
                time.sleep(API_DELAY)
                pdb_resp = requests.get(pdb_url, timeout=60)
                if pdb_resp.status_code == 200:
                    pdb_path = af_dir / f"{acc}.pdb"
                    with open(pdb_path, "w") as f:
                        f.write(pdb_resp.text)
                    downloaded += 1
                    download_log.append({"uniprot": acc, "pdb_file": str(pdb_path), "status": "ok"})
                else:
                    download_log.append({"uniprot": acc, "status": f"pdb_download_failed ({pdb_resp.status_code})"})
                    failed += 1
            else:
                download_log.append({"uniprot": acc, "status": "no_pdb_url"})
                failed += 1

            # Progress logging every 20 entries
            if i % 20 == 0:
                logging.info(f"[AlphaFold] Progress: {i}/{len(accessions)} queried, {downloaded} downloaded, {failed} failed/not_found")

        except Exception as e:
            logging.error(f"[AlphaFold] {acc} error: {e}")
            download_log.append({"uniprot": acc, "status": f"error: {e}"})
            failed += 1

    logging.info(f"[AlphaFold] Complete: {downloaded} structures downloaded out of {len(accessions)} accessions")

    # Save log
    log_path = af_dir / "alphafold_download_log.csv"
    if download_log:
        keys = ["uniprot", "pdb_file", "status"]
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(download_log)
    logging.info(f"[AlphaFold] Download log saved -> {log_path}")
    return download_log


# ============================================================
# 5. Literature-Curated Reference Table
# ============================================================
def create_literature_reference(output_dir: Path):
    """Write the literature-curated Gvp reference table."""
    lit_dir = output_dir / "literature"
    lit_dir.mkdir(parents=True, exist_ok=True)

    ref_path = lit_dir / "gvp_literature_reference.csv"
    keys = ["family", "organism", "uniprot", "ncbi", "function", "key_reference", "doi"]
    with open(ref_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(LITERATURE_DATA)
    logging.info(f"[Literature] {len(LITERATURE_DATA)} entries saved -> {ref_path}")

    # Also create a family function summary
    func_path = lit_dir / "gvp_family_functions.csv"
    family_funcs = {
        "GvpA": "Major structural shell protein; forms ribs via alpha-helix/beta-sheet arrangement; highly conserved",
        "GvpB": "GvpA paralog in some organisms; possible minor structural component",
        "GvpC": "Outer reinforcement protein with tandem repeats; stabilizes shell against collapse",
        "GvpD": "Negative regulator; p-loop NTPase repressing GvpA transcription (haloarchaea)",
        "GvpE": "Transcriptional activator of gvpA promoter (haloarchaea)",
        "GvpF": "Accessory protein; coiled-coil interactions with GvpA; essential for assembly",
        "GvpG": "Accessory protein; forms multimers; essential for early assembly stages",
        "GvpH": "Accessory protein; absence leads to altered vesicle morphology",
        "GvpI": "Essential accessory protein; haloarchaea-specific in some species",
        "GvpJ": "GvpA homolog; minor structural component; multimer formation",
        "GvpK": "Accessory protein; forms multimers; interacts with GvpL",
        "GvpL": "Scaffold/platform protein; brings other accessory Gvps together",
        "GvpM": "GvpA-like nucleation seed protein; initiates vesicle assembly",
        "GvpN": "AAA-type ATPase; required for vesicle enlargement (tiny vesicles in its absence)",
        "GvpO": "Essential for vesicle formation; function not fully characterized",
    }
    with open(func_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "function_summary", "role_category"])
        writer.writeheader()
        for fam, func in family_funcs.items():
            category = "structural" if fam in ("GvpA", "GvpB", "GvpC", "GvpJ", "GvpM") else \
                       "regulatory" if fam in ("GvpD", "GvpE") else \
                       "assembly" if fam in ("GvpF", "GvpG", "GvpH", "GvpI", "GvpK", "GvpL", "GvpN", "GvpO") else "unknown"
            writer.writerow({"family": fam, "function_summary": func, "role_category": category})
    logging.info(f"[Literature] Family function summary saved -> {func_path}")


# ============================================================
# 6. Generate README
# ============================================================
def generate_readme(output_dir: Path, ncbi_summary, uniprot_summary, interpro_summary, af_log):
    """Generate a README.md summarizing the collected data."""
    readme_path = output_dir / "README.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Gvp Multi-Database Data Collection (v2 - Comprehensive, No Limits)",
        f"",
        f"Generated: {now}",
        "",
        "## Changes from v1",
        "",
        "- **No data quantity limits**: All results are downloaded via batch/pagination",
        "- **Expanded InterPro/Pfam**: Covers all Gvp families (A-O), not just GvpA/GvpC",
        "- **Text-based search**: Supplementary InterPro search for families without dedicated accessions",
        "- **Full UniProt pagination**: Uses Link header pagination for complete results",
        "",
        "## Data Sources",
        "",
        "| Source | Directory | Description |",
        "|--------|-----------|-------------|",
        "| NCBI GenBank/RefSeq | `ncbi/` | Protein sequences + metadata for each Gvp family |",
        "| UniProtKB | `uniprot/` | Curated annotations, GO terms, cross-references |",
        "| InterPro/Pfam | `interpro/` | Domain-annotated protein lists (all families) |",
        "| AlphaFold DB | `alphafold/` | Predicted 3D structures (PDB format) |",
        "| Literature | `literature/` | Curated reference table from key publications |",
        "",
        "## NCBI Summary",
        "",
        "| Family | Proteins Found | Status |",
        "|--------|---------------|--------|",
    ]
    for row in (ncbi_summary or []):
        lines.append(f"| {row.get('family', '')} | {row.get('ncbi_count', 0)} | {row.get('status', '')} |")

    lines += [
        "",
        "## UniProt Summary",
        "",
        "| Family | Entries | Status |",
        "|--------|---------|--------|",
    ]
    for row in (uniprot_summary or []):
        lines.append(f"| {row.get('family', '')} | {row.get('uniprot_count', 0)} | {row.get('status', '')} |")

    lines += [
        "",
        "## InterPro/Pfam Summary",
        "",
        "| Accession | Description | Proteins | Status |",
        "|-----------|-------------|----------|--------|",
    ]
    for row in (interpro_summary or []):
        lines.append(f"| {row.get('accession', '')} | {row.get('description', '')} | {row.get('protein_count', 0)} | {row.get('status', '')} |")

    lines += [
        "",
        f"## AlphaFold Structures",
        f"",
        f"Downloaded {sum(1 for r in (af_log or []) if r.get('status') == 'ok')} structures",
        "",
        "## Gvp Family Overview",
        "",
        "| Family | Role | Category |",
        "|--------|------|----------|",
        "| GvpA | Major structural shell protein | Structural |",
        "| GvpB | GvpA paralog | Structural |",
        "| GvpC | Outer reinforcement with tandem repeats | Structural |",
        "| GvpD | Transcriptional repressor | Regulatory |",
        "| GvpE | Transcriptional activator | Regulatory |",
        "| GvpF | Assembly accessory (coiled-coil) | Assembly |",
        "| GvpG | Assembly accessory (multimer) | Assembly |",
        "| GvpH | Assembly accessory (morphology) | Assembly |",
        "| GvpI | Essential accessory (haloarchaea) | Assembly |",
        "| GvpJ | Minor structural (GvpA homolog) | Structural |",
        "| GvpK | Assembly accessory (multimer) | Assembly |",
        "| GvpL | Scaffold/platform protein | Assembly |",
        "| GvpM | Nucleation seed (GvpA-like) | Structural |",
        "| GvpN | AAA-ATPase for enlargement | Assembly |",
        "| GvpO | Essential, unknown function | Assembly |",
        "| GvpP | Gas vesicle associated protein | Unknown |",
        "| GvpQ | Gas vesicle associated protein | Unknown |",
        "| GvpR | Gas vesicle associated protein | Unknown |",
        "| GvpS | Gas vesicle associated protein | Unknown |",
        "| GvpT | Gas vesicle associated protein | Unknown |",
        "| GvpU | Gas vesicle associated protein | Unknown |",
        "| GvpV | Gas vesicle associated protein | Unknown |",
        "| GvpW | Gas vesicle associated protein | Unknown |",
        "",
        "## InterPro/Pfam Coverage by Family",
        "",
        "| Family | Known Accessions | Search Method |",
        "|--------|-----------------|---------------|",
        "| GvpA/J/M | PF00741, IPR008654, IPR047870, IPR050530, IPR018493 | Accession-based |",
        "| GvpC | PF01304, IPR000638 | Accession-based |",
        "| GvpD | PF07088, IPR010645 | Accession-based |",
        "| GvpF/L | PF06386, IPR009430 | Accession-based |",
        "| GvpG | PF05120 | Accession-based |",
        "| GvpO | PF05800 | Accession-based |",
        "| GvpB | - | Text search |",
        "| GvpE | - | Text search |",
        "| GvpH | - | Text search |",
        "| GvpI | - | Text search |",
        "| GvpK | - | Text search |",
        "| GvpN | - | Text search |",
        "| GvpP | - | Text search |",
        "| GvpQ | - | Text search |",
        "| GvpR | - | Text search |",
        "| GvpS | - | Text search |",
        "| GvpT | - | Text search |",
        "| GvpU | - | Text search |",
        "| GvpV | - | Text search |",
        "| GvpW | - | Text search |",
    ]

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"[README] Generated -> {readme_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Gvp multi-database data collector (v2 - no limits)")
    parser.add_argument("--email", default="gvp_collector@example.com",
                        help="Email for NCBI Entrez API (required by NCBI policy)")
    parser.add_argument("--skip-alphafold", action="store_true",
                        help="Skip AlphaFold DB structure downloads")
    parser.add_argument("--families", default=",".join(GVP_FAMILIES),
                        help=f"Comma-separated family letters (default: all {','.join(GVP_FAMILIES)})")
    parser.add_argument("--skip-ncbi", action="store_true", help="Skip NCBI collection")
    parser.add_argument("--skip-uniprot", action="store_true", help="Skip UniProt collection")
    parser.add_argument("--skip-interpro", action="store_true", help="Skip InterPro collection")
    args = parser.parse_args()

    families = [f.strip().upper() for f in args.families.split(",")]
    log_file = setup_logging(BASE_DIR / "logs")

    logging.info("=" * 60)
    logging.info("Gvp Multi-Database Data Collection v2 (No Limits) Started")
    logging.info(f"Families: {', '.join(f'Gvp{f}' for f in families)}")
    logging.info(f"Output directory: {BASE_DIR}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"InterPro entries: {len(INTERPRO_ENTRIES)} accession-based + {len(GVP_TEXT_SEARCH_FAMILIES)} text-search families")
    logging.info("=" * 60)

    ncbi_summary = []
    uniprot_summary = []
    interpro_summary = []
    af_log = []

    # 1. NCBI
    if not args.skip_ncbi:
        logging.info("\n>>> Phase 1: NCBI GenBank / RefSeq (no limits) <<<")
        ncbi_summary = collect_ncbi(BASE_DIR, args.email, families)
    else:
        logging.info(">>> Skipping NCBI <<<")

    # 2. UniProt
    if not args.skip_uniprot:
        logging.info("\n>>> Phase 2: UniProtKB / Swiss-Prot (full pagination) <<<")
        uniprot_summary = collect_uniprot(BASE_DIR, families)
    else:
        logging.info(">>> Skipping UniProt <<<")

    # 3. InterPro / Pfam
    if not args.skip_interpro:
        logging.info("\n>>> Phase 3: InterPro / Pfam (all families, no limits) <<<")
        interpro_summary = collect_interpro(BASE_DIR)
    else:
        logging.info(">>> Skipping InterPro <<<")

    # 4. AlphaFold
    if not args.skip_alphafold:
        logging.info("\n>>> Phase 4: AlphaFold DB (no limits) <<<")
        af_log = collect_alphafold(BASE_DIR)
    else:
        logging.info(">>> Skipping AlphaFold <<<")

    # 5. Literature
    logging.info("\n>>> Phase 5: Literature-Curated Reference <<<")
    create_literature_reference(BASE_DIR)

    # 6. README
    logging.info("\n>>> Generating README <<<")
    generate_readme(BASE_DIR, ncbi_summary, uniprot_summary, interpro_summary, af_log)

    logging.info("\n" + "=" * 60)
    logging.info("Collection Complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
