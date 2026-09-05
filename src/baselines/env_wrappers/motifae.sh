#!/usr/bin/env bash
set -Eeuo pipefail

: "${FAMILY:?}"
: "${FAMILY_FASTA:?}"
: "${MODEL_RAW_DIR:?}"

MOTIFAE_REPO="${MOTIFAE_REPO:-/data/coding/MotifAE}"
PYTHON_BIN="${MOTIFAE_PYTHON:-python}"

rm -rf "$MODEL_RAW_DIR/input" "$MODEL_RAW_DIR/embeddings" "$MODEL_RAW_DIR/sae" "$MODEL_RAW_DIR/tmp_config"
mkdir -p "$MODEL_RAW_DIR"/{input,embeddings,sae,tmp_config}

"$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

family = os.environ["FAMILY"]
fasta = Path(os.environ["FAMILY_FASTA"])
raw = Path(os.environ["MODEL_RAW_DIR"])
out = raw / "input" / "motifae_input.csv"
mapping = raw / "input" / "id_map.csv"

records = []
name = None
seq = []
for line in fasta.read_text().splitlines():
    if line.startswith(">"):
        if name:
            records.append((name, "".join(seq)))
        name = line[1:].split()[0]
        seq = []
    else:
        seq.append(line.strip())
if name:
    records.append((name, "".join(seq)))

limit = int(os.environ.get("MOTIFAE_LIMIT", "0") or 0)
if limit > 0:
    records = records[:limit]

with out.open("w", newline="") as handle, mapping.open("w", newline="") as map_handle:
    writer = csv.writer(handle)
    map_writer = csv.writer(map_handle)
    writer.writerow(["uniprot", "sequence"])
    map_writer.writerow(["safe_id", "original_id"])
    for i, (original_id, sequence) in enumerate(records, 1):
        safe = f"{family}_{i:06d}"
        writer.writerow([safe, sequence])
        map_writer.writerow([safe, original_id])

print(f"Wrote {out} records: {len(records)}")
print(f"Wrote {mapping}")
PY

cat > "$MODEL_RAW_DIR/tmp_config/config.py" <<PY
my_config = {
    "df_path": "$MODEL_RAW_DIR/input/motifae_input.csv",
    "embed_path": "$MODEL_RAW_DIR/embeddings",
    "save_dir": "$MODEL_RAW_DIR/sae",
    "df_name_col": "uniprot",
    "df_seq_col": "random_subregion",
    "device": "cuda:0",
    "batch_size": 1,
    "dataloader_num_workers": 0,
    "activation_dim": 1280,
    "dict_size": int(1280*32),
    "resample_steps": None,
    "tied": False,
    "lr": 1e-3,
    "warmup_steps": 50,
    "l1_penalty": 0.04,
    "smoothness_relative_penalty": 1,
    "smoothness_window": 3,
    "l1_annealing_steps": 500,
    "seed": 42,
    "n_epoch": 1,
    "save_steps": 100,
    "log_steps": 20,
}
PY

"$PYTHON_BIN" - "$MOTIFAE_REPO/model/train_esm_sae.py" "$MODEL_RAW_DIR/tmp_config/train_esm_sae.py" <<'PY'
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:])
text = src.read_text()
text = text.replace(
    "df = pd.read_csv(my_config['df_path'], nrows=10)",
    "df = pd.read_csv(my_config['df_path'])",
)
dst.write_text(text)
PY

PYTHONPATH="$MODEL_RAW_DIR/tmp_config:$MOTIFAE_REPO/model" "$PYTHON_BIN" -c "import get_esm2_embedding"
PYTHONPATH="$MODEL_RAW_DIR/tmp_config:$MOTIFAE_REPO/model" "$PYTHON_BIN" -c "import train_esm_sae"
