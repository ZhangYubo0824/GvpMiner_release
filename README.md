# GvpMiner experimental code

GvpMiner provides code for gas vesicle protein experiments, motif discovery and comparative evaluation, together with reference inputs, final results and scientific figures.

## Layout

| Directory | Contents |
| --- | --- |
| `src/` | Data collection/preprocessing, classification/generation validation, motif discovery, parameter stability, structural and annotation experiments |
| `scripts/` | Candidate validation, cross-model matching, taxonomy, conservation, sequence profiles, curated site mapping and scientific plotting |
| `baselines/` | Comparison-method adapters in `adapters/`, plus MotifAE inference, guarded alignment and exact-core consumers in `motifae/` |
| `data/` | Reference sequences, accession records, verified MSAs and curated site-mapping inputs |
| `results/` | Final catalogues, statistical outputs, sequence representations, figure source data and figures |
| `configs/` | Experimental settings and selected family parameters |
| `tests/` | Regression tests for experimental code, analyses and paths |
| `tools/` | Read-only validation of the released results |
| `docs/` | Experiment commands, result definitions and verification coverage |
| `manifests/` | File SHA256 checksums and a brief usage note |
| `work/` | New outputs and caches; excluded by `.gitignore` |

## Dependencies

Python 3.12 is the tested version. The three dependency files serve different tasks:

| File | Purpose | Includes |
| --- | --- | --- |
| `requirements.txt` | Core experiments, numerical analysis and plotting | NumPy, pandas, SciPy, scikit-learn, UMAP, HDBSCAN and related libraries |
| `requirements-models.txt` | Protein-language-model inference and structure-related code | Core dependencies plus PyTorch, ESM, Transformers and PyDSSP |
| `requirements-dev.txt` | Running regression tests | Core dependencies plus pytest |

The model and development files both include `requirements.txt` through `-r`; core packages are listed once. Choose the file for the task:

```bash
# Analysis and plotting
python -m pip install -r requirements.txt

# Model inference, including core dependencies
python -m pip install -r requirements-models.txt

# Tests, including core dependencies
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Select the PyTorch/CUDA build for the target machine. Model weights, native alignment programs and annotation databases are obtained separately. See [experiment commands](docs/EXPERIMENTS.md) and [MotifAE instructions](baselines/motifae/README.md) for additional requirements.

## Inspect the results

These commands require only the Python standard library:

```bash
python scripts/summarize_results.py
python tools/verify_results.py --hashes
```

The results include GvpMiner motif candidates, sequence representations, structural analyses and annotation-enrichment results. See [the results guide](docs/RESULTS_GUIDE.md) for file organization, field definitions and interpretation.

## Run analyses

```bash
python -m scripts.candidate_evidence.evidence
python -m scripts.motif_representation.build
python -m scripts.candidate_evidence.conservation
```

Reference inputs are under `data/`; new outputs default to `work/`. See [input data](data/README.md), [experiment commands](docs/EXPERIMENTS.md) and [verification](docs/VERIFICATION.md). MotifAE native sampling requires installation qualification matching the code, inputs and native executable.

## Licensing

See [licensing status and third-party notices](LICENSES.md). A project license has not yet been selected.
