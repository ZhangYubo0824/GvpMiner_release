"""
GvpMiner 鍏ㄥ眬閰嶇疆
==================
鎵€鏈夎矾寰勩€佹ā鍨嬪弬鏁般€佸疄楠岃秴鍙傛暟闆嗕腑绠＄悊銆?
"""

import os
from pathlib import Path

# ============================================================
# 椤圭洰璺緞
# ============================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.resolve()
DATA_DIR = (PROJECT_ROOT / "data").resolve()
DATA3_DIR = DATA_DIR
NCBI_DIR = (DATA_DIR / "ncbi").resolve()
UNIPROT_DIR = (DATA_DIR / "uniprot").resolve()
INTERPRO_DIR = (DATA_DIR / "interpro").resolve()
ALPHAFOLD_DIR = (DATA_DIR / "alphafold").resolve()

# The bundled results are a read-only snapshot used to reproduce paper figures.
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
EXP4_DIR = OUTPUT_DIR / "exp4_motif_structure_analysis"
EXP5_EXTERNAL_DIR = OUTPUT_DIR / "exp5_external_functional_enrichment"
RESULT_BASELINE_DIR = OUTPUT_DIR / "result_baseline"
GLOBAL_RESIDUE_CACHE_DIR = OUTPUT_DIR / "residue_embeddings"
STRUCTURES_DIR = OUTPUT_DIR / "structures"
EXP_CONTROL_DIR = OUTPUT_DIR / "exp3_control"
MOTIF_STRUCTURE_DIR = EXP4_DIR

# Exp4 consumes the current run's preprocessing, motif, and structure outputs.
# The snapshot can be selected explicitly through the Exp4 CLI when figures or
# summaries must be regenerated without rerunning the upstream experiments.
PRIOR_RESULTS_DIR = OUTPUT_DIR
EXP4_SEQUENCE_TABLE = PREPROCESSED_DIR / "cleaned_gvp_dataset.csv"
EXP4_EXP3_INPUT_DIR = EXP3_DIR
EXP4_STRUCTURE_INPUT_DIR = STRUCTURES_DIR

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
MODELS_DIR = PROJECT_ROOT / "work" / "models"

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
