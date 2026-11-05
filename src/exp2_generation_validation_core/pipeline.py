from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.validation import (
    AA_STANDARD,
    SequenceRecord,
    compute_embeddings,
)
from config import NCBI_DIR, EXP2_DIR, EXP1_DIR


T02_FAMILY_PATTERN_PREFIX = "Gvp"

# ---------------------------------------------------------------------------
# 全局绘图样式
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

METHOD_COLORS = {"M2": "#FF9800", "M3": "#4CAF50"}
METHOD_LABELS = {"M2": "M2 ESM-2", "M3": "M3 ProtBERT"}


def is_t02_standard_family(name: str) -> bool:
    return len(name) == 4 and name.startswith(T02_FAMILY_PATTERN_PREFIX) and name[-1].isalpha()


def clean_t02_sequence(sequence: str) -> str:
    seq = "".join(ch for ch in str(sequence).upper() if not ch.isspace() and ch != "*")
    if not seq or not set(seq).issubset(set(AA_STANDARD)):
        return ""
    return seq


def format_t02_training_line(record: SequenceRecord) -> str:
    seq = clean_t02_sequence(record.sequence)
    spaced = " ".join(seq)
    return f"<|{record.family}|> {spaced} <|endoftext|>"


def compute_t02_manifest_signature(records: Sequence[SequenceRecord], families: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for family in families:
        digest.update(family.encode("utf-8"))
    for record in sorted(records, key=lambda row: (row.family, row.seq_id, row.sequence)):
        digest.update(record.seq_id.encode("utf-8"))
        digest.update(record.family.encode("utf-8"))
        digest.update(clean_t02_sequence(record.sequence).encode("utf-8"))
    return digest.hexdigest()


def resolve_t02_family_fasta(data_dir: Path, family: str) -> Optional[Path]:
    family_dir = Path(data_dir) / family
    candidates = [
        family_dir / f"{family}_FINAL_TRAINING_SET.fasta",
        family_dir / f"{family}_proteins.fasta",
        family_dir / f"{family}_filtered.fasta",
        family_dir / f"{family}.fasta",
    ]
    for path in candidates:
        if path.exists():
            return path
    fasta_files = sorted(family_dir.glob("*.fasta")) if family_dir.is_dir() else []
    return fasta_files[0] if fasta_files else None


def load_t02_reference_records(data_dir: Path = NCBI_DIR) -> List[SequenceRecord]:
    from Bio import SeqIO

    records: List[SequenceRecord] = []
    data_dir = Path(data_dir)
    for family_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        family = family_dir.name
        if not is_t02_standard_family(family):
            continue
        fasta_path = resolve_t02_family_fasta(data_dir, family)
        if fasta_path is None:
            continue
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq = clean_t02_sequence(str(record.seq))
            if not seq:
                continue
            records.append(SequenceRecord(seq_id=str(record.id), family=family, sequence=seq))
    if not records:
        raise FileNotFoundError(f"No T02-compatible Gvp FASTA records found in {data_dir}")
    return records


def prepare_t02_corpus(
    *,
    records: Sequence[SequenceRecord],
    target_families: Sequence[str],
    output_dir: Path = EXP2_DIR,
    val_ratio: float = 0.1,
    seed: int = 2025,
) -> Dict[str, Any]:
    requested = list(dict.fromkeys(target_families))
    available = {record.family for record in records}
    missing = [family for family in requested if family not in available]
    if missing:
        raise ValueError(f"Unknown target families: {', '.join(missing)}")

    selected: List[SequenceRecord] = []
    for record in records:
        if record.family not in requested:
            continue
        seq = clean_t02_sequence(record.sequence)
        if not seq:
            continue
        selected.append(SequenceRecord(seq_id=record.seq_id, family=record.family, sequence=seq))

    shuffled = list(selected)
    random.Random(seed).shuffle(shuffled)
    val_count = int(len(shuffled) * val_ratio)
    if val_ratio > 0 and shuffled:
        val_count = max(1, min(len(shuffled), val_count))

    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "dataset_train.txt"
    val_path = output_dir / "dataset_val.txt"
    manifest_path = output_dir / "dataset_manifest.json"

    train_path.write_text(
        "\n".join(format_t02_training_line(record) for record in train_records),
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(format_t02_training_line(record) for record in val_records),
        encoding="utf-8",
    )

    manifest = {
        "experiment": "exp2_generation_validation",
        "generation_logic": "T02_BioGPT",
        "families": requested,
        "family_tokens": [f"<|{family}|>" for family in requested],
        "num_total_records": len(selected),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "val_ratio": val_ratio,
        "seed": seed,
        "dataset_signature": compute_t02_manifest_signature(selected, requested),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "train_path": train_path,
        "val_path": val_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def build_t02_tokenizer(families: Sequence[str]):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    special_tokens = [f"<|{family}|>" for family in families]
    vocab = ["<|endoftext|>", "<|pad|>", "<|unk|>"] + special_tokens + list(AA_STANDARD)
    vocab_map = {token: idx for idx, token in enumerate(vocab)}

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab_map, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.add_special_tokens(
        {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "bos_token": "<|endoftext|>",
            "additional_special_tokens": special_tokens,
        }
    )
    return fast_tokenizer


class T02ProteinDataset:
    def __init__(self, tokenizer, file_path: Path, block_size: int):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self):
                self.examples = []
                with open(file_path, "r", encoding="utf-8") as handle:
                    lines = [line.strip() for line in handle if line.strip()]
                for line in lines:
                    tokenized = tokenizer(
                        line,
                        truncation=True,
                        max_length=block_size,
                        padding="max_length",
                    )
                    self.examples.append(tokenized)

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, i):
                item = {key: value for key, value in self.examples[i].items()}
                item["input_ids"] = torch.tensor(item["input_ids"], dtype=torch.long)
                item["labels"] = torch.tensor(item["input_ids"], dtype=torch.long)
                return item

        self.dataset = _Dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


def is_complete_t02_model_dir(model_dir: Path) -> bool:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").exists()
    has_tokenizer = any(
        (model_dir / name).exists()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    )
    has_weights = any(
        (model_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack")
    )
    return has_config and has_tokenizer and has_weights


def model_dir_has_t02_family_tokens(model_dir: Path, families: Sequence[str]) -> bool:
    tokenizer_config_path = Path(model_dir) / "tokenizer_config.json"
    special_tokens_path = Path(model_dir) / "special_tokens_map.json"
    token_payloads = []
    for path in (tokenizer_config_path, special_tokens_path):
        if not path.exists():
            continue
        try:
            token_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue

    found = set()
    for payload in token_payloads:
        tokens = payload.get("additional_special_tokens", [])
        for token in tokens:
            if isinstance(token, dict):
                content = token.get("content")
            else:
                content = token
            if isinstance(content, str):
                found.add(content)

    if not found:
        return False
    expected = {f"<|{family}|>" for family in families}
    return expected.issubset(found)


def train_t02_generator(
    *,
    corpus_info: Dict[str, Any],
    target_families: Sequence[str],
    output_dir: Path = EXP2_DIR,
    model_dir: Optional[Path] = None,
    force_retrain: bool = False,
    block_size: int = 256,
    batch_size: int = 8,
    epochs: int = 15,
    learning_rate: float = 5e-4,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 8,
) -> Dict[str, Path]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    model_dir = Path(model_dir) if model_dir else Path(output_dir) / "gvp_biogpt_model"
    expected_families = list(dict.fromkeys(target_families))
    if is_complete_t02_model_dir(model_dir) and not force_retrain:
        if model_dir_has_t02_family_tokens(model_dir, expected_families):
            return {"model_dir": model_dir}
        raise ValueError(
            "Existing T02 generator model does not contain the requested family-token set. "
            f"Run with --force-retrain or use a different --biogpt-model-dir. Directory: {model_dir}"
        )

    import torch
    from transformers import (
        DataCollatorForLanguageModeling,
        GPT2Config,
        GPT2LMHeadModel,
        Trainer,
        TrainingArguments,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_t02_tokenizer(target_families)
    train_ds = T02ProteinDataset(tokenizer, Path(corpus_info["train_path"]), block_size).dataset
    val_ds = T02ProteinDataset(tokenizer, Path(corpus_info["val_path"]), block_size).dataset

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=learning_rate,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=[],
        run_name="exp2_generation_validation",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return {"model_dir": model_dir}


def calculate_t02_properties(sequence: str) -> Tuple[float, float, float]:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        analyser = ProteinAnalysis(sequence)
        return (
            float(analyser.molecular_weight()),
            float(analyser.isoelectric_point()),
            float(analyser.instability_index()),
        )
    except Exception:
        return 0.0, 0.0, 100.0


def clean_generated_text(text: str, prompt: str) -> str:
    seq = (
        text.replace(prompt, "")
        .replace("<|endoftext|>", "")
        .replace("<|pad|>", "")
        .replace("<|unk|>", "")
        .replace(" ", "")
        .strip()
        .upper()
    )
    return "".join(ch for ch in seq if ch in set(AA_STANDARD))


def generate_t02_sequences(
    *,
    target_family: str,
    model_info: Dict[str, Any],
    output_dir: Path = EXP2_DIR,
    num_generate: int = 500,
    min_length: int = 30,
    max_length: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> List[str]:
    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    model_dir = Path(model_info["model_dir"])
    if not is_complete_t02_model_dir(model_dir):
        raise FileNotFoundError(
            "T02 generator model directory is incomplete. "
            f"Run the train stage first or pass --force-retrain. Directory: {model_dir}"
        )

    output_dir = Path(output_dir)
    family_dir = output_dir / "generated" / target_family
    family_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    prompt = f"<|{target_family}|> "
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = set()
    rows: List[Dict[str, Any]] = []

    for _ in range(num_generate):
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        sequence = clean_generated_text(full_text, prompt)
        if sequence in generated or len(sequence) < min_length:
            continue
        generated.add(sequence)
        _, _, instability = calculate_t02_properties(sequence)
        score = max(0.0, 100.0 - instability)
        rows.append(
            {
                "ID": f"AI_{len(rows) + 1:03d}",
                "Sequence": sequence,
                "Length": len(sequence),
                "Instability": round(instability, 2),
                "Score": round(score, 2),
            }
        )

    if not rows:
        raise RuntimeError(f"No valid T02 generated sequences for {target_family}.")

    rows = sorted(rows, key=lambda row: row["Score"], reverse=True)
    output_csv = family_dir / f"AI_Designed_{target_family}.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Sequence", "Length", "Instability", "Score"])
        writer.writeheader()
        writer.writerows(rows)
    return [row["Sequence"] for row in rows]


def classifier_bundle_path(base_dir: Path, method: str) -> Path:
    import sklearn

    skver = sklearn.__version__.replace(".", "_")
    return Path(base_dir) / f"classifier_bundle_{method}_sk{skver}.pkl"


def resolve_classifier_bundle_path(base_dir: Path, method: str) -> Path:
    exact = classifier_bundle_path(base_dir, method)
    if exact.exists():
        return exact
    matches = sorted(Path(base_dir).glob(f"classifier_bundle_{method}_sk*.pkl"))
    if matches:
        return matches[-1]
    return exact


def patch_sklearn_forest_compatibility(classifier) -> None:
    for estimator in getattr(classifier, "estimators_", []):
        if not hasattr(estimator, "monotonic_cst"):
            estimator.monotonic_cst = None


def load_exp1_classifier_bundle(
    *,
    method: str,
    exp1_dir: Path = EXP1_DIR,
):
    path = resolve_classifier_bundle_path(exp1_dir, method)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Exp1 classifier bundle for method={method}: {path}. "
            "Run exp1_classification.py for the requested method first."
        )
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    classifier = bundle["classifier"]
    patch_sklearn_forest_compatibility(classifier)
    return classifier, bundle["label_encoder"], path


def predict_with_exp1_classifier(
    *,
    sequences: Sequence[str],
    method: str,
    target_family: str,
    output_dir: Path,
    exp1_dir: Path = EXP1_DIR,
) -> Tuple[List[Dict[str, Any]], Path]:
    classifier, label_encoder, source_path = load_exp1_classifier_bundle(
        method=method,
        exp1_dir=exp1_dir,
    )
    cache_path = Path(output_dir) / "embedding_cache" / f"generated_{target_family}_{method}.npy"
    X = compute_embeddings(method, list(sequences), cache_path=cache_path)
    y_pred = classifier.predict(X)
    pred_labels = label_encoder.inverse_transform(y_pred)
    probabilities = classifier.predict_proba(X)

    rows: List[Dict[str, Any]] = []
    for idx, sequence in enumerate(sequences):
        class_probs = {
            str(label_encoder.classes_[col]): float(probabilities[idx, col])
            for col in range(len(label_encoder.classes_))
        }
        target_probability = float(class_probs.get(target_family, 0.0))
        rows.append(
            {
                "generated_id": f"gen_{idx:04d}",
                "sequence": sequence,
                "length": len(sequence),
                "predicted_family": str(pred_labels[idx]),
                "predicted_confidence": float(np.max(probabilities[idx])),
                "target_family": target_family,
                "target_probability": target_probability,
                "is_target_family": str(pred_labels[idx]) == target_family,
                "probabilities": class_probs,
            }
        )
    return rows, source_path


def write_prediction_csv(path: Path, rows: Sequence[Dict[str, Any]], target_family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "generated_id",
            "sequence",
            "length",
            "predicted_family",
            "predicted_confidence",
            "target_family",
            "target_probability",
            "is_target_family",
            f"prob_{target_family}",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "generated_id": row["generated_id"],
                    "sequence": row["sequence"],
                    "length": row["length"],
                    "predicted_family": row["predicted_family"],
                    "predicted_confidence": row["predicted_confidence"],
                    "target_family": row["target_family"],
                    "target_probability": row["target_probability"],
                    "is_target_family": row["is_target_family"],
                    f"prob_{target_family}": row["target_probability"],
                }
            )


def write_classification_report(
    *,
    sequences: Sequence[str],
    target_family: str,
    methods: Sequence[str],
    output_dir: Path = EXP2_DIR,
    exp1_dir: Path = EXP1_DIR,
) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    output_dir = Path(output_dir)
    family_dir = output_dir / "generated" / target_family
    family_dir.mkdir(parents=True, exist_ok=True)

    method_summaries: Dict[str, Dict[str, Any]] = {}
    classifier_sources: Dict[str, str] = {}
    method_rows_map: Dict[str, List[Dict[str, Any]]] = {}
    for method in methods:
        rows, source_path = predict_with_exp1_classifier(
            sequences=sequences,
            method=method,
            target_family=target_family,
            output_dir=output_dir,
            exp1_dir=exp1_dir,
        )
        method_rows_map[method] = rows
        write_prediction_csv(family_dir / f"classification_{method}.csv", rows, target_family)
        predicted_target = sum(1 for row in rows if row["is_target_family"])
        method_summaries[method] = {
            "method": method,
            "target_family": target_family,
            "num_generated": len(rows),
            "predicted_target_ratio": predicted_target / max(1, len(rows)),
            "mean_target_probability": float(np.mean([row["target_probability"] for row in rows])) if rows else 0.0,
        }
        classifier_sources[method] = str(source_path)

    report = {
        "experiment": "exp2_generation_validation",
        "generation_logic": "T02_BioGPT",
        "classifier_source": "exp1_saved_bundle",
        "target_family": target_family,
        "num_generated": len(sequences),
        "method_summaries": method_summaries,
        "classifier_sources": classifier_sources,
    }
    (family_dir / "classification_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report, method_rows_map


# ---------------------------------------------------------------------------
# 优化输出: 可视化
# ---------------------------------------------------------------------------
def plot_probability_distribution(
    method_rows_map: Dict[str, List[Dict[str, Any]]],
    method_summaries: Dict[str, Dict[str, Any]],
    target_family: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, rows in method_rows_map.items():
        if not rows:
            continue
        probs = [r["target_probability"] for r in rows]
        color = METHOD_COLORS.get(method, "#888")
        label = METHOD_LABELS.get(method, method)
        ax.hist(probs, bins=30, alpha=0.45, color=color, label=label, edgecolor="white", linewidth=0.5)
        summary = method_summaries.get(method, {})
        mean_p = summary.get("mean_target_probability", 0.0)
        ax.axvline(mean_p, color=color, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel(f"P( {target_family} )")
    ax.set_ylabel("Count")
    ax.set_title(f"Generated Sequence Classification — {target_family}", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cross_family_heatmap(
    all_reports: List[Dict[str, Any]],
    methods: Sequence[str],
    metric: str,
    title: str,
    save_path: Path,
) -> None:
    import pandas as pd

    rows = []
    for report in all_reports:
        fam = report["target_family"]
        summaries = report.get("method_summaries", {})
        for method in methods:
            if method in summaries:
                rows.append({
                    "Family": fam,
                    "Method": METHOD_LABELS.get(method, method),
                    metric: summaries[method].get(metric, 0.0),
                })

    if not rows:
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Family", columns="Method", values=metric)

    annot = pivot.copy().astype(str)
    for fam in pivot.index:
        for method in pivot.columns:
            val = pivot.loc[fam, method]
            if pd.notna(val):
                annot.loc[fam, method] = f"{val:.3f}"

    fig, ax = plt.subplots(
        figsize=(max(6, 2 * len(pivot.columns)), max(4, 0.7 * len(pivot.index) + 2)),
    )
    sns.heatmap(
        pivot.astype(float),
        annot=annot.values,
        fmt="",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": metric.replace("_", " ").title()},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    ax.set_title(title, fontweight="bold", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# 优化输出: CSV 导出
# ---------------------------------------------------------------------------
def export_passed_sequences(
    method_rows_map: Dict[str, List[Dict[str, Any]]],
    target_family: str,
    family_dir: Path,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    passed: List[Dict[str, Any]] = []
    for method, rows in method_rows_map.items():
        for row in rows:
            if row["is_target_family"] and row["target_probability"] >= threshold:
                passed.append({
                    "family": target_family,
                    "method": method,
                    "generated_id": row["generated_id"],
                    "sequence": row["sequence"],
                    "length": row["length"],
                    "predicted_family": row["predicted_family"],
                    "target_probability": row["target_probability"],
                })

    if passed:
        passed_csv = family_dir / f"AI_Designed_{target_family}_passed.csv"
        with open(passed_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "family", "method", "generated_id", "sequence", "length",
                "predicted_family", "target_probability",
            ])
            writer.writeheader()
            writer.writerows(passed)
    return passed


def write_merged_summary_csv(
    all_reports: List[Dict[str, Any]],
    methods: Sequence[str],
    output_dir: Path,
) -> None:
    rows = []
    for report in all_reports:
        fam = report["target_family"]
        summaries = report.get("method_summaries", {})
        for method in methods:
            s = summaries.get(method, {})
            if s:
                rows.append({
                    "family": fam,
                    "method": method,
                    "num_generated": s.get("num_generated", 0),
                    "predicted_target_ratio": f"{s.get('predicted_target_ratio', 0.0):.4f}",
                    "mean_target_probability": f"{s.get('mean_target_probability', 0.0):.4f}",
                })

    if rows:
        csv_path = output_dir / "cross_family_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "family", "method", "num_generated",
                "predicted_target_ratio", "mean_target_probability",
            ])
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# 优化输出: Markdown 报告
# ---------------------------------------------------------------------------
def write_markdown_report(
    all_reports: List[Dict[str, Any]],
    all_passed: List[Dict[str, Any]],
    methods: Sequence[str],
    output_dir: Path,
) -> None:
    lines = [
        "# Exp2 - Generation Validation Report",
        "",
        "## Overview",
        "",
        "| Family | Method | Generated | Target Ratio | Mean Prob | Passed |",
        "|--------|--------|-----------|-------------|-----------|--------|",
    ]

    passed_by_fam_method: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in all_passed:
        passed_by_fam_method[p["family"]][p["method"]] += 1

    for report in all_reports:
        fam = report["target_family"]
        summaries = report.get("method_summaries", {})
        for method in methods:
            s = summaries.get(method, {})
            if not s:
                continue
            n_pass = passed_by_fam_method.get(fam, {}).get(method, 0)
            lines.append(
                f"| {fam} | {METHOD_LABELS.get(method, method)} "
                f"| {s.get('num_generated', 0)} "
                f"| {s.get('predicted_target_ratio', 0.0):.3f} "
                f"| {s.get('mean_target_probability', 0.0):.3f} "
                f"| {n_pass} |"
            )

    lines.append("")
    lines.append("## Top-5 Sequences per Family")
    lines.append("")

    for report in all_reports:
        fam = report["target_family"]
        passed_fam = [p for p in all_passed if p["family"] == fam]
        if not passed_fam:
            continue
        top5 = sorted(passed_fam, key=lambda r: -r["target_probability"])[:5]
        lines.append(f"### {fam}")
        lines.append("")
        for i, row in enumerate(top5, 1):
            seq = row["sequence"]
            seq_display = seq[:60] + "..." if len(seq) > 60 else seq
            lines.append(
                f"**#{i}** [{row['method']}] P={row['target_probability']:.4f} "
                f"len={row['length']}"
            )
            lines.append(f"```\n{seq_display}\n```")
        lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Summary JSON: `{output_dir}/summaries/exp2_generation_validation_summary.json`")
    lines.append(f"- Merged CSV: `{output_dir}/cross_family_summary.csv`")
    lines.append(f"- Passed sequences: `{output_dir}/all_passed_sequences.csv`")
    lines.append(f"- Probability heatmap: `{output_dir}/heatmap_target_probability.png`")
    lines.append(f"- Ratio heatmap: `{output_dir}/heatmap_target_ratio.png`")
    for report in all_reports:
        fam = report["target_family"]
        lines.append(f"- `{fam}` → `generated/{fam}/`")
    lines.append("")

    report_path = output_dir / "exp2_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def group_records_by_family(records: Sequence[SequenceRecord]) -> Dict[str, List[SequenceRecord]]:
    groups: Dict[str, List[SequenceRecord]] = defaultdict(list)
    for record in records:
        groups[record.family].append(record)
    return groups


def resolve_target_families(args, available_families: Sequence[str]) -> List[str]:
    if args.families:
        requested = args.families
    elif args.target_family:
        requested = [args.target_family]
    else:
        requested = sorted(available_families)
    deduped = list(dict.fromkeys(requested))
    available = set(available_families)
    missing = [family for family in deduped if family not in available]
    if missing:
        raise ValueError(f"Families not found in filtered dataset: {', '.join(missing)}")
    return deduped


def run_pipeline(args) -> List[Dict[str, Any]]:
    output_dir = Path(getattr(args, "output_dir", "") or EXP2_DIR)
    t02_records = load_t02_reference_records()
    family_groups = group_records_by_family(t02_records)
    t02_families = sorted(family_groups.keys())
    target_families = resolve_target_families(args, family_groups.keys())

    corpus_info = None
    if "prepare" in args.stages or "train" in args.stages:
        corpus_info = prepare_t02_corpus(
            records=t02_records,
            target_families=t02_families,
            output_dir=output_dir,
            val_ratio=args.train_val_split,
            seed=args.seed,
        )

    model_info = None
    if "train" in args.stages:
        model_info = train_t02_generator(
            corpus_info=corpus_info,
            target_families=t02_families,
            output_dir=output_dir,
            model_dir=Path(args.biogpt_model_dir) if args.biogpt_model_dir else None,
            force_retrain=args.force_retrain,
            block_size=args.block_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            n_embd=args.biogpt_n_embd,
            n_layer=args.biogpt_n_layer,
            n_head=args.biogpt_n_head,
        )
    elif "generate" in args.stages:
        model_dir = Path(args.biogpt_model_dir) if args.biogpt_model_dir else output_dir / "gvp_biogpt_model"
        if not model_dir_has_t02_family_tokens(model_dir, t02_families):
            raise ValueError(
                "Existing T02 generator model does not contain the requested family-token set. "
                f"Run the train stage with --force-retrain or pass a matching --biogpt-model-dir. Directory: {model_dir}"
            )
        model_info = {"model_dir": model_dir}

    reports: List[Dict[str, Any]] = []
    all_method_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    all_passed: List[Dict[str, Any]] = []

    for family in target_families:
        sequences = None
        if "generate" in args.stages:
            sequences = generate_t02_sequences(
                target_family=family,
                model_info=model_info,
                output_dir=output_dir,
                num_generate=args.num_generate,
                min_length=args.min_length,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        elif "classify" in args.stages:
            designed_csv = output_dir / "generated" / family / f"AI_Designed_{family}.csv"
            if not designed_csv.exists():
                raise FileNotFoundError(f"Missing generated sequence CSV for classify stage: {designed_csv}")
            with open(designed_csv, newline="", encoding="utf-8") as handle:
                sequences = [row["Sequence"] for row in csv.DictReader(handle) if row.get("Sequence")]

        method_rows_map: Dict[str, List[Dict[str, Any]]] = {}
        if "classify" in args.stages:
            report, method_rows_map = write_classification_report(
                sequences=sequences or [],
                target_family=family,
                methods=args.methods,
                output_dir=output_dir,
                exp1_dir=Path(args.exp1_dir),
            )
            all_method_rows[family] = method_rows_map

            family_dir = output_dir / "generated" / family
            figures_dir = family_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            plot_probability_distribution(
                method_rows_map=method_rows_map,
                method_summaries=report.get("method_summaries", {}),
                target_family=family,
                save_path=figures_dir / "probability_distribution.png",
            )

            passed = export_passed_sequences(
                method_rows_map=method_rows_map,
                target_family=family,
                family_dir=family_dir,
            )
            all_passed.extend(passed)
        else:
            report = {"target_family": family}
        reports.append(report)

    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "exp2_generation_validation_summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- 跨家族输出 ---
    if "classify" in args.stages and len(reports) >= 1:
        write_merged_summary_csv(reports, args.methods, output_dir)

        if all_passed:
            all_passed_csv = output_dir / "all_passed_sequences.csv"
            with open(all_passed_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "family", "method", "generated_id", "sequence", "length",
                    "predicted_family", "target_probability",
                ])
                writer.writeheader()
                writer.writerows(all_passed)

        if len(reports) >= 2:
            plot_cross_family_heatmap(
                all_reports=reports,
                methods=args.methods,
                metric="mean_target_probability",
                title="Mean Target Probability — Cross-Family Summary",
                save_path=output_dir / "heatmap_target_probability.png",
            )
            plot_cross_family_heatmap(
                all_reports=reports,
                methods=args.methods,
                metric="predicted_target_ratio",
                title="Predicted Target Ratio — Cross-Family Summary",
                save_path=output_dir / "heatmap_target_ratio.png",
            )

        write_markdown_report(reports, all_passed, args.methods, output_dir)

    return reports


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 2: family-conditioned generation validated with saved Exp1 classifiers."
    )
    parser.add_argument("--target-family", default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["prepare", "train", "generate", "classify"],
        default=["prepare", "train", "generate", "classify"],
    )
    parser.add_argument("--methods", nargs="+", choices=["M2", "M3"], default=["M2", "M3"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--exp1-dir", default=str(EXP1_DIR))
    parser.add_argument("--biogpt-model-dir", default="")
    parser.add_argument("--train-val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--force-retrain", action="store_true")

    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--biogpt-n-embd", type=int, default=256)
    parser.add_argument("--biogpt-n-layer", type=int, default=6)
    parser.add_argument("--biogpt-n-head", type=int, default=8)

    parser.add_argument("--num-generate", type=int, default=500)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    reports = run_pipeline(args)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
