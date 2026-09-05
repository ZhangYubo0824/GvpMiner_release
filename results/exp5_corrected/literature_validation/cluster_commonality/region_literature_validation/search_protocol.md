# Region-level literature validation protocol

## Scope

- Search and verification date: 2026-06-08.
- Structural input: cluster-wide ESMFold/PyDSSP commonality results.
- Functional input: motifs with current clusterProfiler status
  `external_enrichment_not_significant` or `external_enrichment_no_terms`.
- Representative-window fields are used only as search/mapping coordinates.
- Exp4/Exp5 statistics remain unchanged and continue to use all cluster windows.
- `core_start`, `core_end`, and `core_length` are not used.

## Structural interpretation

Representative full proteins were globally aligned to experimentally studied
homologs. Motif intervals were projected through the alignment and intersected
with PDB-resolved secondary-structure elements or experimentally tested local
regions. A biological interpretation was retained only when the mapped interval
overlapped a curated region. Generic secondary-structure possibilities were not
promoted to literature-supported mechanisms.

## Functional search

Exact 30-aa text searching is retained only as a negative audit. The primary
test is region-level homology mapping to verified mutation, deletion, truncation,
or fragment-interaction experiments. Family-level function alone does not count
as motif support. GvpK remains in the audit but is not called supported unless a
GvpK-localized experimental region is found. No evidence grades are assigned in
the functional tables.

## Thresholds

- Global alignment identity >= 0.25.
- Global query coverage >= 0.60.
- Motif mapping fraction >= 0.80.
- At least one mapped residue must overlap the curated experimental region.

## Outputs

- Structural rows: 14782; region-supported: 3368.
- Functional candidate rows: 2584; region-supported: 465.

## Limitations

- All positive mappings in this dataset are homologous-region transfers, not
  experiments on the exact representative accession.
- Broad fragment assays localize function to a region, not every residue.
- Repeated GvpC positions and the GvpU five-site mutant were perturbed jointly;
  individual occurrences/residues cannot be separated.
- Absence of a mapped experiment means `no_localized_support`, not evidence that
  the motif lacks function.
