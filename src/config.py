"""
GvpMiner 鍏ㄥ眬閰嶇疆
==================
鎵€鏈夎矾寰勩€佹ā鍨嬪弬鏁般€佸疄楠岃秴鍙傛暟闆嗕腑绠＄悊銆?
"""

from pathlib import Path

# ============================================================
# 椤圭洰璺緞
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # Task1/
SRC_DIR = PROJECT_ROOT / "src2"
DATA3_DIR = PROJECT_ROOT / "data3"
NCBI_DIR = DATA3_DIR / "ncbi"
UNIPROT_DIR = DATA3_DIR / "uniprot"
INTERPRO_DIR = DATA3_DIR / "interpro"
ALPHAFOLD_DIR = DATA3_DIR / "alphafold"

# Structural-analysis inputs are reused read-only from the previous result set.
PRIOR_RESULTS_DIR = PROJECT_ROOT / "result6"
EXP4_SEQUENCE_TABLE = PRIOR_RESULTS_DIR / "preprocessed" / "cleaned_gvp_dataset.csv"
EXP4_EXP3_INPUT_DIR = PRIOR_RESULTS_DIR / "exp3_domain_discovery"
EXP4_STRUCTURE_INPUT_DIR = PRIOR_RESULTS_DIR / "structures"

# New outputs from src2 are isolated under results7.
OUTPUT_DIR = PROJECT_ROOT / "results7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EXP1_DIR = OUTPUT_DIR / "exp1_classification"
EXP1_DIR.mkdir(parents=True, exist_ok=True)

EXP2_DIR = OUTPUT_DIR / "exp2_generation_validation"
EXP2_DIR.mkdir(parents=True, exist_ok=True)

EXP3_DIR = OUTPUT_DIR / "exp3_domain_discovery"
EXP3_DIR.mkdir(parents=True, exist_ok=True)

EXP4_DIR = OUTPUT_DIR / "exp4_motif_structure_analysis"
EXP4_DIR.mkdir(parents=True, exist_ok=True)

EXP5_EXTERNAL_DIR = OUTPUT_DIR / "exp5_external_functional_enrichment"
EXP5_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

RESULT_BASELINE_DIR = OUTPUT_DIR / "result_baseline"
RESULT_BASELINE_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_RESIDUE_CACHE_DIR = OUTPUT_DIR / "residue_embeddings"
GLOBAL_RESIDUE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURES_DIR = OUTPUT_DIR / "structures"
STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

EXP_CONTROL_DIR = OUTPUT_DIR / "exp3_control"
EXP_CONTROL_DIR.mkdir(parents=True, exist_ok=True)

MOTIF_STRUCTURE_DIR = OUTPUT_DIR / "exp4_motif_structure_analysis"
MOTIF_STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Gvp 瀹舵棌鍒楄〃
# ============================================================
GVP_FAMILIES = [
    "GvpA", "GvpB", "GvpC", "GvpD", "GvpE", "GvpF", "GvpG",
    "GvpH", "GvpI", "GvpJ", "GvpK", "GvpL", "GvpM", "GvpN",
    "GvpO", "GvpP", "GvpQ", "GvpR", "GvpS", "GvpT", "GvpU",
    "GvpV", "GvpW",
]

# 鎺掗櫎hybrid瀹舵棌锛堝彲閫夌撼鍏ワ級
GVP_HYBRID_FAMILIES = ["GvpA_J_hybrid", "GvpL_F_hybrid"]

# 鏁版嵁閲忚緝灏戠殑瀹舵棌锛堝彲鑳介渶瑕佺壒娈婂鐞嗭級
GVP_SMALL_FAMILIES = ["GvpB", "GvpE", "GvpI", "GvpR", "GvpS"]

# ============================================================
# 鏁版嵁棰勫鐞嗗弬鏁?
# ============================================================
CDHIT_THRESHOLD = 0.90          # CD-HIT 鍘诲啑浣欓槇鍊?
MIN_SEQ_LENGTH = 30             # 鏈€鐭簭鍒楅暱搴?
MAX_SEQ_LENGTH = 2000           # 鏈€闀垮簭鍒楅暱搴?
MIN_FAMILY_SIZE = 100            # 瀹舵棌鑷冲皯瑕佹湁N鏉″簭鍒楁墠鍙備笌瀹為獙

# 闈炴爣鍑嗘皑鍩洪吀瀛楃
INVALID_AA_CHARS = set("XBZJOU*")
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# ============================================================
# 宓屽叆妯″瀷鍙傛暟
# ============================================================
# 鏈湴妯″瀷鐩綍
MODELS_DIR = PROJECT_ROOT / "models"

# M2: ESM-2 (650M 鐗堟湰)
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"  # 650M鍙傛暟, 1280D
ESM2_REPR_LAYER = 33                     # 650M 妯″瀷鐨勬渶鍚庝竴灞傛槸绗?33 灞?
ESM2_EMBEDDING_DIM = 1280                # 650M 妯″瀷鐨勮緭鍑虹淮搴︽槸 1280

# M3: ProtBERT-BFD
PROTBERT_MODEL_NAME = "Rostlab/prot_bert_bfd"
PROTBERT_EMBEDDING_DIM = 1024
PROTBERT_USE_HALF = True  # 鍗婄簿搴﹁妭鐪佹樉瀛?

# 鎵瑰鐞嗗ぇ灏?(鏍规嵁4GB鏄惧瓨璋冩暣)
EMBEDDING_BATCH_SIZE = 32

# ============================================================
# 瀹為獙1: 鍒嗙被鍙傛暟
# ============================================================
EXP1_CV_FOLDS = 5               # 浜ゅ弶楠岃瘉鎶樻暟
EXP1_RF_ESTIMATORS = 200        # 闅忔満妫灄鏍戞暟閲?
EXP1_RANDOM_STATE = 42

# ============================================================
# Experiment 3: domain-discovery parameters
# ============================================================
EXP3_WINDOW_SIZE = 30
EXP3_WINDOW_STEP = 5
EXP3_KMEANS_K_RANGE = (3, 20)
EXP3_DBSCAN_MIN_SAMPLES = 5
EXP3_DBSCAN_EPS_PERCENTILE = 90

# ============================================================
# Experiment 4: motif-structure analysis parameters
# ============================================================
EXP4_DEFAULT_BACKENDS = ("esmfold",)
EXP4_DEFAULT_METHODS = ("M2", "M3")
EXP4_NUM_RANDOM_CONTROLS = 10
EXP4_CONTACT_THRESHOLD = 8.0
EXP4_LOW_PLDDT_CUTOFF = 70.0
EXP4_ESMFOLD_CHUNK_SIZE = 128

# ============================================================
# 鍙鍖栧弬鏁?
# ============================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
COLOR_PALETTE = {
    "M2_ESM2": "#FF9800",        # 姗欒壊
    "M3_ProtBERT": "#4CAF50",    # 缁胯壊
}
