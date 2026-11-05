# MotifAE experiment baseline

This directory contains the current experiment implementation: published
MotifAE k3/80000 model inference, declared preprint-v1 peptide extraction,
qualified guarded-first Gibbs alignment, strict occurrence handoff, and the
Exp4/Exp5 consumers that preserve noncontinuous aligned-core coordinates.
The upstream model dictionary and MIT license are in
`scripts/baseline_adapters/vendor/motifae/`.

Before a new alignment run, qualify the native installation with the supplied
fixtures. Qualification and resume records must match the exact code, inputs
and installed executable.

All default experiment data/output paths resolve under the enclosing package
root: `data/reference/preprocessed`, `work/results/structures`, and
`work/results/baselines/motifae`. External checkpoint, ESM weights, annotation data,
and a licensed GibbsCluster installation are supplied separately. No trained
weights or licensed sampler executable are bundled.

From the package root, discover one family using the published k3 checkpoint:

```bash
python baselines/motifae/scripts/baseline_adapters/run_motifae_pipeline.py \
  --family GvpA --fasta data/reference/preprocessed/by_family/GvpA_cleaned.fasta \
  --checkpoint /path/to/published_k3.pt --device cuda \
  --activation-protocol preprint-v1 \
  --output-dir work/results/baselines/motifae/discovery/GvpA/run_001
```

Full-protein inference is the default and does not crop or stitch windows.
Long inputs can require substantial device memory. Cached full-length ESM
embeddings can be supplied with `--input-dir` instead of `--fasta`.
Custom and incomplete training checkpoints are not supported. Discovery
outputs are peptide candidates; they are not completed PSSM motifs.

After discovering all selected families, run qualification and alignment on
Linux x86_64 with the installation version checked by the code:

```bash
export MOTIFAE_GIBBS=/path/to/licensed/gibbscluster-2.0
bash baselines/motifae/run_all_families.sh
```

The wrapper resolves the current package root from its own location and can
be launched from any working directory. `MOTIFAE_RESULTS`, `MOTIFAE_FASTA`,
`MOTIFAE_QUAL`, `MOTIFAE_RUN`, and `PYTHON` can override defaults. Resume only
the same source/input/qualification identity with `--resume`; changed code
requires fresh qualification and a new run name.

The wrapper publishes `alignment/_runs/<run>/family_runs.json` only after all
selected families pass strict validation. Use
`scripts/baseline_adapters/motifae_handoff.py --help` for strict handoff export,
verification, annotation binding, and Exp5 preparation. The current consumers
are `src/exp4_motif_structure_analysis.py` and
`src/exp5_external_functional_enrichment/run.py`; both support `--help`.
They retain the full protein background and exact aligned-core residue lists.

Portable checks (no native sampler execution):

```bash
python -m unittest discover -s baselines/motifae/tests -v
python -m unittest discover -s baselines/motifae/src/exp5_external_functional_enrichment/tests -v
```

The qualification fixtures include an unchanged reference native output and
fixed fault/singleton inputs. Portable tests mock native execution and do not
substitute for fresh Linux installation qualification or biological validation.
