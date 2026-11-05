"""Current GvpMiner experiment paths and numerical defaults."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'
NCBI_DIR = DATA_DIR / 'ncbi'
SOURCE_RESULTS_DIR = Path(os.environ.get('GVP_SOURCE_RESULTS_DIR', PROJECT_ROOT / 'data' / 'reference')).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get('GVP_OUTPUT_DIR', PROJECT_ROOT / 'work' / 'results')).expanduser().resolve()
PREPROCESSED_DIR = OUTPUT_DIR / 'preprocessed'
EXP1_DIR = OUTPUT_DIR / 'exp1_classification'
EXP2_DIR = OUTPUT_DIR / 'exp2_generation_validation'
EXP3_DIR = OUTPUT_DIR / 'exp3_domain_discovery'
EXP4_EXP3_INPUT_DIR = EXP3_DIR
EXP4_STRUCTURE_INPUT_DIR = OUTPUT_DIR / 'structures'
GLOBAL_RESIDUE_CACHE_DIR = OUTPUT_DIR / 'residue_embeddings'
STRUCTURES_DIR = OUTPUT_DIR / 'structures'
GVP_FAMILIES = ['GvpA', 'GvpB', 'GvpC', 'GvpD', 'GvpE', 'GvpF', 'GvpG', 'GvpH', 'GvpI', 'GvpJ', 'GvpK', 'GvpL', 'GvpM', 'GvpN', 'GvpO', 'GvpP', 'GvpQ', 'GvpR', 'GvpS', 'GvpT', 'GvpU', 'GvpV', 'GvpW']
CDHIT_THRESHOLD = 0.9
MIN_SEQ_LENGTH = 30
MAX_SEQ_LENGTH = 2000
MIN_FAMILY_SIZE = 100
INVALID_AA_CHARS = set('XBZJOU*')
STANDARD_AA = 'ACDEFGHIKLMNPQRSTVWY'
MODELS_DIR = PROJECT_ROOT / 'work' / 'models'
ESM2_MODEL_NAME = 'esm2_t33_650M_UR50D'
ESM2_REPR_LAYER = 33
PROTBERT_MODEL_NAME = 'Rostlab/prot_bert_bfd'
EXP1_CV_FOLDS = 5
EXP1_RF_ESTIMATORS = 200
EXP1_RANDOM_STATE = 42
EXP3_WINDOW_SIZE = 20
EXP3_WINDOW_STEP = 15
EXP4_ESMFOLD_CHUNK_SIZE = 128
EXP3_MIN_WINDOWS = 15
SEQUENCE_TABLE = (PREPROCESSED_DIR / "cleaned_gvp_dataset.csv" if (PREPROCESSED_DIR / "cleaned_gvp_dataset.csv").exists() else SOURCE_RESULTS_DIR / "preprocessed" / "cleaned_gvp_dataset.csv")

EXP4_SEQUENCE_TABLE = SEQUENCE_TABLE
