# Final results and interpretation

`results/` is a read-only final snapshot. It contains final scientific outputs and the source tables needed to check or redraw them. New analyses, transient caches, trained models, logs and historical trials belong under `work/` or an external run directory.

| Location | Purpose |
| --- | --- |
| `catalogues/GvpMiner_candidate_evidence_latest.csv` | Primary GvpMiner catalogue: 522 candidates and their evidence fields |
| `catalogues/GvpMiner_M2_M3/` | Primary and cross-representation comparison catalogues with actual occurrence intervals |
| `structure/`, `annotation/` | Final structural endpoints, annotation results and output-coverage records |
| `stability/final_setting/` | Selected parameters and final stability evidence |
| `sequence_representation/` | Final per-candidate profiles, aligned fragments and consensus records |
| `evidence/` | Detailed coordinate, conservation, cross-model and evidence tables |
| `paper/source_data/`, `paper/figure_inputs/` | Final numerical source data and scientific figure inputs |
| `figures/` | Fourteen final PNGs, captions and available vector artwork |

## Current comparison counts

| Method | Candidates | Strict structure support | Annotation output | Annotation support |
| --- | ---: | ---: | ---: | ---: |
| GvpMiner | 522 | 348 | 283 | 224 |
| GLAM2 | 160 | 102 | 120 | 89 |
| MEME | 320 | 149 | 174 | 123 |
| STREME | 529 | 262 | 323 | 212 |
| MotifAE | 8378 | 2028 | 5685 | 2542 |
| InterPLM | 875 | 93 | 382 | 188 |

The primary GvpMiner catalogue uses ESM-2 protein representations and contains 522 candidates. A comparison catalogue using ProtBERT-BFD contains 464 candidates for cross-representation analysis. Only the primary GvpMiner catalogue enters the six-method comparison. MotifAE counts refer to the completed `guarded_first_20261101_all` experiment across sixteen families.

Strict structural support requires the original significant primary structural endpoint, mean pLDDT at least 70 and composition-matched control coverage at least 0.9. Annotation support in the current comparison means any original within-candidate BH-adjusted `p.adjust <= 0.05`, with the presence of an enrichment output recorded separately. Conditional annotation support uses output-bearing candidates as denominator; overall support uses all input candidates. No-output and tested-without-support states are retained separately.

**Inherited fields need care.** In `Table_S14_candidate_status.csv`, current annotation support is `annotation_supported_raw` and the current output flag is `annotation_has_output`. The older `annotation_testable`/`annotation_supported` columns retain the supplementary minimum-three-foreground-member rule. They are not interchangeable. The current Figure 5 tables and `annotation_output_status` encode the main comparison. Additional eligibility and source fields preserve the statistical definitions and provenance of the numerical data.

Occurrence intervals use 0-based half-open `[start, end)` coordinates unless the field explicitly says 1-based. Native MotifAE aligned cores can be noncontinuous; keep the explicit core residue lists rather than replacing them with a bounding interval. An occurrence is not a unique protein, and an annotation term is not a unique candidate.

`cross_species_count` in the inherited catalogue equals protein count and must not be used as biological species diversity. Use current `tax_` fields. `msa_` fields are descriptive conservation measures with no added null comparison or significance claim. Curated literature mapping is member/site-level evidence, not whole-cluster causal validation. Current literature statuses are in `literature_status_current` and the final source tables; older reviewed mapping files are retained as provenance.

The final figures are in `results/figures/`, with available vectors in its `vector/` subdirectory. Their file identities, together with the numerical results and source code, are recorded in `manifests/SHA256SUMS.txt`. Run `python tools/verify_results.py --hashes` to check the complete release.
