# GVP motif literature enrichment search protocol

## Search boundary

- Search coverage: database inception through 2026-06-14.
- Primary bibliographic sources: PubMed/MEDLINE, Europe PMC and Crossref.
- Citation tracking: Web of Science, Scopus and Google Scholar.
- Database cross-checks: UniProt, InterPro, Pfam, PROSITE, NCBI CDD and RCSB PDB where available; the automated local run uses InterProScan coordinates.
- Chinese-language supplementation: CNKI and Wanfang may identify leads but do not replace an original primary paper.
- Exact public-API hit counts and execution timestamps are recorded in
  `literature_search_log.csv`; subscription or login-dependent platforms remain
  `manual_required` until a researcher records their execution.

## Evidence policy

- Reviews are discovery aids only.
- Region evidence requires a primary mutation, deletion, truncation, interaction, complementation, structure, pressure, localization or engineering experiment.
- Natural and engineering evidence tracks are reported separately.
- Engineering papers remain in the source table for discovery and context, but
  are not promoted to localized motif evidence until an experimentally tested
  residue or interval has been verified from the primary paper.
- Match grades are `A_direct_exact`, `B_homologous_residue`, `C_homologous_region`, `D_context_only` and `no_localized_support`.
- B-grade transfer requires motif identity >= 0.70, mapping fraction >= 0.80, complete coverage of the tested region and exact conservation of tested residues.
- C-grade transfer requires a valid homologous mapping and either motif overlap >= 0.20 or experimental-region overlap >= 0.50.
- Exact 30-aa text search is an audit aid, not the primary evidence method.

## Strict novel-candidate rule

A strict candidate must have A/B evidence, at least 10 unique sequences, verified DOI/PMID metadata, no overlapping local database annotation and no unresolved conflict. C-grade rows remain exploratory. D-grade rows never establish motif function.

## Run summary

- Motifs audited: 18854
- Verified/deduplicated sources: 42
- Experimental regions: 63
- Motifs with A-C localized support: 4270
- Strict database-unannotated literature candidates: 0

## Interpretation limitation

`no_localized_support` means that no verified local experiment was mapped by this evidence set. It must not be interpreted as evidence that a motif lacks function.
