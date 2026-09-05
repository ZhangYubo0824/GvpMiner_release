# Motif literature and database validation protocol

## Search date and scope

- Search date: 2026-06-08 (Asia/Shanghai).
- Structural shortlist: 12 motifs, one high-priority representative motif per family with usable Exp4 structural evidence.
- Functional shortlist: 17 motifs selected from external enrichment states requiring follow-up.
- Search representation: only Exp3 representative-window accession, start, end and sequence were used. Exp4/Exp5 statistics were read but not modified.

## Databases and sources

- Bibliographic verification: PubMed, Europe PMC, publisher/PMC full-text landing pages and DOI records.
- Structure: RCSB PDB and primary structural papers; AlphaFold DB was considered, but no exact stable mapping from most RefSeq representative accessions to an AlphaFold entry was asserted.
- Protein/domain annotation: local InterProScan 5.77-108.0 and eggNOG-mapper 2.1.13 raw outputs, with official InterPro/Pfam/NCBIfam identifiers retained in the evidence text.
- Functional context: primary mutation, deletion/reconstitution, interaction and engineering papers were preferred over reviews.

## Query design

Queries combined:

1. `family + representative_sequence + representative_accession + representative_start-end`
2. `family + gas vesicle assembly OR shell stability`
3. `family + acoustic reporter OR ultrasound imaging`
4. `family + delivery OR display OR transport OR localization`
5. `family + mutation OR deletion OR fusion OR engineering`
6. Exact paper-title, DOI, PMID and PMCID verification queries.

Exact 30-aa sequence searches generally returned no direct experimental study. Family/protein papers were therefore retained only as homologous or contextual evidence, never as proof that the exact motif is functional.

## Inclusion rules

- Include peer-reviewed primary studies, stable PDB/database records and local raw annotation outputs tied to the exact representative accession.
- Prefer direct mutation, deletion, reconstitution, interaction or structural evidence.
- Preserve clusterProfiler status exactly. `not_tested_external`, `external_enrichment_unavailable` and `external_filter_failed` are not treated as nonsignificant.
- Assign evidence grades as: A exact protein/region directly tested; B homologous protein or neighboring-region experimental support; C database/review/structural inference; D weak background.
- Retain both `evidence_scope` and `limitations`.

## Exclusion rules

- Exclude unverified titles, DOI/PMID pairs, database pages or conclusions.
- Exclude generic family function as motif-level proof.
- Exclude internal Fisher enrichment as a selection criterion; it may only be used later as cross-reference.
- Exclude exact AlphaFold/UniProt claims when the RefSeq representative accession could not be mapped reliably.

## Selection

Structural motifs were chosen by significance (`q <= 0.05`), strong/moderate structural evidence, mean motif pLDDT >= 70 where available, and family-level instance support. Functional motifs were chosen to represent external nonsignificant, not-tested, unavailable and filter-failed states and to span all available Gvp families. GvpN contributes two rows so `not_tested_external` and `external_enrichment_unavailable` remain separately visible.

## Current status summary

- Functional status counts: {'external_filter_failed': 4, 'external_enrichment_not_significant': 11, 'not_tested_external': 1, 'external_enrichment_unavailable': 1}
- Functional evidence-grade counts: {'B': 16, 'C': 1}
- No A-grade row was assigned because no paper directly tested an exact selected representative accession and 30-aa interval.

## Limitations

- Representative windows identify searchable examples of a cluster; they are not inferred minimal functional domains.
- Most experimental studies use homologs from different organisms, so B-grade evidence supports family/region plausibility rather than exact motif causality.
- Predicted structural enrichment and pLDDT support structural plausibility but cannot establish biochemical function.
- GvpA-like paralogy affects some GvpJ/GvpM representatives, and AAA+ homology can confound GvpN annotation.
- GvpQ and GvpV remain comparatively weakly characterized; their candidate functions should be treated as exploratory.
