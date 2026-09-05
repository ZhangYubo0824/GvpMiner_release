"""Central paths and experiment parameters for the GvpMiner workflow."""

import os
from pathlib import Path


# Project paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.resolve()
DATA_DIR = (PROJECT_ROOT / "data").resolve()
NCBI_DIR = (DATA_DIR / "ncbi").resolve()
INTERPRO_DIR = (DATA_DIR / "interpro").resolve()

# Read-only final-result snapshot used by publication and audit commands.
SNAPSHOT_RESULTS_DIR = (PROJECT_ROOT / "results").resolve()

# New runs are isolated from the bundled snapshot. Override this location on a
# server with GVPMINER_OUTPUT_DIR.
OUTPUT_DIR = Path(
    os.environ.get("GVPMINER_OUTPUT_DIR", PROJECT_ROOT / "work" / "results")
).resolve()

PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
EXP1_DIR = OUTPUT_DIR / "exp1_classification"
EXP2_DIR = OUTPUT_DIR / "exp2_generation_validation"
EXP3_DIR = OUTPUT_DIR / "exp3_domain_discovery"
GLOBAL_RESIDUE_CACHE_DIR = OUTPUT_DIR / "residue_embeddings"
STRUCTURES_DIR = OUTPUT_DIR / "structures"

# Exp4 consumes the current run's preprocessing, motif, and structure outputs.
EXP4_SEQUENCE_TABLE = PREPROCESSED_DIR / "cleaned_gvp_dataset.csv"
EXP4_EXP3_INPUT_DIR = EXP3_DIR
EXP4_STRUCTURE_INPUT_DIR = STRUCTURES_DIR


# Gvp family list (hybrid families are intentionally absent).
GVP_FAMILIES = [
    "GvpA", "GvpB", "GvpC", "GvpD", "GvpE", "GvpF", "GvpG",
    "GvpH", "GvpI", "GvpJ", "GvpK", "GvpL", "GvpM", "GvpN",
    "GvpO", "GvpP", "GvpQ", "GvpR", "GvpS", "GvpT", "GvpU",
    "GvpV", "GvpW",
]


# Data preprocessing
CDHIT_THRESHOLD = 0.90
MIN_SEQ_LENGTH = 30
MAX_SEQ_LENGTH = 2000
MIN_FAMILY_SIZE = 100
INVALID_AA_CHARS = set("XBZJOU*")
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


# Embedding models
MODELS_DIR = PROJECT_ROOT / "work" / "models"

# M2: ESM-2 650M, 1280-dimensional final-layer representation.
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"

# M3: ProtBERT-BFD
PROTBERT_MODEL_NAME = "Rostlab/prot_bert_bfd"
PROTBERT_EMBEDDING_DIM = 1024
PROTBERT_USE_HALF = True


# Experiment 1: family classification
EXP1_CV_FOLDS = 5
EXP1_RF_ESTIMATORS = 200
EXP1_RANDOM_STATE = 42


# Experiment 3: domain discovery
EXP3_WINDOW_SIZE = 30
EXP3_WINDOW_STEP = 5
EXP3_DBSCAN_MIN_SAMPLES = 5


# Experiment 4: motif-structure analysis
EXP4_ESMFOLD_CHUNK_SIZE = 128
