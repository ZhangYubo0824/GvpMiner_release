# Licensing and external resources

This repository is an author-review snapshot. No license for the original GvpMiner code or bundled
research data has been granted yet. Keep the repository private until the author chooses a release
license and confirms the redistribution terms of every input and result artifact.

Third-party source trees, pretrained model weights, external annotation databases, bulk predicted
structures, and tool workspaces are not redistributed. Obtain them from their upstream projects and
follow the licenses and citation requirements published there:

- [PyTorch](https://github.com/pytorch/pytorch), [fair-esm / ESM-2 and ESMFold](https://github.com/facebookresearch/esm), and [Hugging Face Transformers](https://github.com/huggingface/transformers).
- [MEME Suite](https://meme-suite.org/), [SLiMSuite / SLiMFinder](https://github.com/slimsuite/SLiMSuite), [DiMotif](https://github.com/ehsanasgari/dimotif), and [MotifAE](https://github.com/CHAOHOU-97/MotifAE).
- [InterProScan](https://www.ebi.ac.uk/interpro/about/interproscan/), [eggNOG-mapper](https://github.com/eggnogdb/eggnog-mapper), [KofamScan](https://www.genome.jp/tools/kofamkoala/), and [clusterProfiler](https://bioconductor.org/packages/clusterProfiler/).
- [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) and the upstream ESMFold/OpenFold stack used for local predictions.

The Python packages listed in `requirements.txt` retain their own upstream licenses. The links above
are for dependency acquisition and review; they do not add any third-party license to this repository.
