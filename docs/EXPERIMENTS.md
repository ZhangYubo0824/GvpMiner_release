# Current experimental entrypoints

Run commands from the repository root. Input data are under `data/reference/`; frozen numerical results are under `results/`; new work is under `work/`. Set `GVP_OUTPUT_DIR` to change the main experimental output root and `GVP_SOURCE_RESULTS_DIR` to change the preprocessed reference-input root. Changing inputs requires new run identities and new scientific validation.

## CPU analyses of the final input snapshot

```bash
python -m scripts.candidate_evidence.evidence
python -m scripts.candidate_evidence.cross_model --source-root . --output-root work/cross_model
python -m scripts.candidate_evidence.taxonomy --source-root . --output-root work/taxonomy
python -m scripts.motif_representation.build
python -m scripts.candidate_evidence.conservation
python -m scripts.candidate_evidence.verify_conservation --source-root . --output-root work/conservation
python -m scripts.literature.run --families GvpM --event-ids M_G22A
```

The evidence exporter validates every parent sequence and 0-based half-open occurrence interval. Cross-model mapping uses actual residues on shared proteins. Taxonomy uses accession metadata. Conservation/profile construction can reuse the bundled verified MSAs after input/hash checks. To realign changed inputs, install MAFFT and supply its executable through the alignment CLI options. Curated site mapping is experimental sequence-coordinate analysis; it does not search or draft literature text, and a compatible mapping still needs scientific review.

## Upstream experiments

Inspect the supported options before launching a GPU or external-tool job:

```bash
python src/exp1_classification.py --help
python src/exp2_generation_validation.py --help
python src/exp3_domain_discovery.py --help
python src/run_motif_stability.py --help
python src/run_stable_parameter_workflow.py --help
python src/exp4_motif_structure_analysis.py --help
python src/predict_family_structures.py --help
python src/exp5_external_functional_enrichment/run.py --help
```

Exp3 defaults to window size 20 and stride 15, using UMAP plus HDBSCAN. The current family-specific HDBSCAN selections are recorded in `configs/paper_snapshot.json` and `results/stability/final_setting/`. A normal discovery run alone does not repeat the complete A/B/C selection and stability workflow. Use the workflow entrypoint with explicit source CSV, configuration and a fresh output root to repeat that process.

ESM2 uses `esm2_t33_650M_UR50D`, layer 33; ProtBERT uses `Rostlab/prot_bert_bfd`. Downloaded model caches belong under `work/models/`. Structure prediction and structural analysis share `work/results/structures/<backend>/<family>/` by default. Both use the newly preprocessed table when present, otherwise the bundled input table. Annotation experiments require the relevant InterProScan, eggNOG/Kofam and R/clusterProfiler resources; these databases are not included. Sequence collection is optional and can change the dataset; use the frozen inputs for snapshot checks.

Current comparison adapters are in `baselines/adapters/`. For example, run `python baselines/adapters/meme_adapter.py --help` or `python baselines/adapters/interplm_adapter.py --help` from the repository root. MotifAE has its own [instructions](../baselines/motifae/README.md), published-checkpoint checks and fresh native installation qualification. No model weights or licensed GibbsCluster executable are bundled.

## Scientific figures

```bash
python scripts/figures/plot_figure2.py
python scripts/figures/plot_figure3.py
python scripts/figures/plot_figure4.py
python scripts/figures/plot_figure5.py
python scripts/figures/plot_figure6.py
python scripts/figures/plot_figure7.py
python scripts/figures/plot_supplement56.py
```

These render retained final measurements to `work/figures/` and QA records under `work/figure_qa/`. They do not replace the exact final figures in `results/figures/`. Chinese labels require a suitable font; the verified local rendering used Microsoft YaHei. Figure 1 and the other supplementary schematics are supplied as final artwork, with available vectors, rather than reconstructed by an invented generator.
