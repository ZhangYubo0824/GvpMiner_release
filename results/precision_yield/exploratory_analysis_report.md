# Exp5 precision–yield exploratory analysis report

## Scope and immutable inputs

- Motif-level source SHA-256: `165f3cb7a173343cb51ad7c7e835ccffebdde2931d38b27912d9ad5ec77ea358`
- Existing random-summary SHA-256: `d9fdb1ae5603e96fd4cef160abeff10a2468e843fb0144ca2a80327c1da1a18e`
- This downstream analysis did not rerun clusterProfiler and did not change its multiplicity handling.
- Protein family is the uncertainty cluster for bootstrap intervals and robust model standard errors.

## Method-level candidate conversion

| Method | All | Testable | Supported | Supported/all | Supported/testable |
|---|---:|---:|---:|---:|---:|
| GLAM2 | 160 | 120 | 93 | 0.581 | 0.775 |
| M2 | 9,525 | 2,236 | 1,844 | 0.194 | 0.825 |
| M3 | 9,329 | 2,144 | 1,727 | 0.185 | 0.806 |
| MEME | 320 | 164 | 123 | 0.384 | 0.750 |
| MotifAE | 800 | 499 | 316 | 0.395 | 0.633 |
| STREME | 529 | 291 | 212 | 0.401 | 0.729 |

These quantities describe scale, annotation testability and support yield. They do not establish that the method with the largest count has the best enrichment performance.

## Top-K sensitivity and rank semantics

Top-K values: 1, 3, 5, 10, 20, 50. Family-cluster bootstrap iterations: 2000; seed: 20260812.
- GLAM2: `native_motif_number`.
- M2: `native_motif_number`.
- M3: `native_motif_number`.
- MEME: `foreground_protein_support_fallback`.
- MotifAE: `foreground_protein_support`.
- STREME: `native_motif_number`.

Rank measures are normalized only within family–method groups; normalized ranks are not treated as commensurable native scores across methods.

## Rank models

- GLAM2: not estimable (non-convergence or separation).
- M2: not estimable (non-convergence or separation).
- M3: not estimable (non-convergence or separation).
- MEME: not estimable (non-convergence or separation).
- MotifAE: not estimable (non-convergence or separation).
- STREME: not estimable (non-convergence or separation).

## Method-level matched-background test

Joint matched-background iterations: 10000; seed: 20260813. The observed Top-10 motifs remained eligible for random sampling, matching the corrected original workflow.

| Method | Observed supported | Null mean | Difference | p | BH q |
|---|---:|---:|---:|---:|---:|
| GLAM2 | 93 | 93.000 | +0.000 pp | 1 | 1 |
| M2 | 71 | 69.795 | +0.753 pp | 0.1868 | 1 |
| M3 | 67 | 67.260 | -0.162 pp | 1 | 1 |
| MEME | 72 | 72.225 | -0.141 pp | 1 | 1 |
| MotifAE | 70 | 75.659 | -3.537 pp | 0.9936 | 1 |
| STREME | 61 | 61.690 | -0.445 pp | 0.8882 | 1 |

Existing family–method audit: 0/96 comparisons had empirical p < 0.05. This is retained as a separate 1,000-draw family-level result.

## Outcome-independent inclusion gate

| Analysis | Provisional destination | Validity basis |
|---|---|---|
| candidate_yield | main_candidate | complete status decomposition; six-method completeness disclosed |
| precision_yield_map | main_candidate | finite count-derived precision and yield measures |
| evidence_chain | main_narrative_candidate | evidence units remain separate with explicit provenance |
| topk_sensitivity | supplement_candidate | all declared K values computed; rank-source heterogeneity retained |
| rank_trend | report_only | destination depends only on numerical identifiability and stability |
| global_matched_background | supplement_candidate | matched draws conserved and null distribution reproducible |

Destinations were assigned from identifiability, comparability, stability and redundancy. Effect direction and p/q thresholds were not used to decide inclusion.

## Permitted interpretation

- GvpMiner may be described as expanding the candidate search space while retaining an annotation-supported subset for prioritization.
- Equal-size Top-K and matched-background results may be described as prioritization diagnostics with method-specific uncertainty.
- Null or negative findings must be reported when the corresponding valid analysis is presented.

## Prohibited interpretation

- Do not claim that GvpMiner is the functionally best enrichment method.
- Do not interpret annotation enrichment as direct proof that an exact 30-aa candidate has a causal function.
- Do not treat heterogeneous native motif ranks as a common continuous score across methods.
- Do not merge the historical 2,584/465 literature audit with motif-level counts without its machine-readable linkage source and hash.
