# Experimental verification

Validation performed with Python 3.12 on Windows:

- Main regression suite: 124 passed.
- MotifAE suite: 167 passed, 7 skipped, 32 subtests passed. Skips cover unavailable Torch and Linux-only process/signal behavior.
- Candidate-coordinate validation, cross-model matching, taxonomy, 522 sequence profiles, conservation analysis and conservation verification executed on the reference inputs.
- All 1,569 regenerated sequence-representation files matched the released results byte-for-byte after decompression.
- Figure 2–7 and supplementary S5/S6 plotting commands completed using the supplied numerical inputs.
- The result checker verifies 10,784 unique six-method candidates, M2=522, M3=464, 84,012 occurrence sequences/coordinates, summary consistency and the presence of fourteen final figures. With `--hashes`, it also checks the released files against their SHA256 values.

## Run checks

```bash
python tools/verify_results.py --hashes
python -m pytest -q
cd baselines/motifae
python -m pytest -p no:cacheprovider tests src/exp5_external_functional_enrichment/tests -q
```

`manifests/SHA256SUMS.txt` covers code, inputs, results and figures. It excludes itself and ignored runtime outputs/caches.

## Coverage

GPU embeddings, model training/inference, bulk structure prediction, complete external-database annotation and licensed native Gibbs sampling were not rerun as part of these validation checks. The bundled results are completed experimental outputs. MotifAE requires Linux installation qualification matching the code, inputs and native executable before sampling. Changed software, hardware, inputs or models require corresponding run records.
