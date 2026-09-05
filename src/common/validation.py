from __future__ import annotations

import csv
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    PREPROCESSED_DIR,
    EXP1_DIR,
    EXP3_DIR,
    MIN_FAMILY_SIZE,
    EXP1_RF_ESTIMATORS,
    EXP1_RANDOM_STATE,
)


AA_STANDARD = "ACDEFGHIKLMNPQRSTVWY"
AA_NORMALIZATION = {
    "B": "D",
    "J": "L",
    "O": "K",
    "U": "C",
    "X": "A",
    "Z": "E",
}


@dataclass
class SequenceRecord:
    seq_id: str
    family: str
    sequence: str


def similarity_from_distance(distance: int, max_length: int) -> float:
    if max_length <= 0:
        return 0.0
    return 1.0 - (distance / max_length)


def normalize_aa_sequence(sequence: str) -> str:
    chars = []
    for ch in sequence.strip().upper():
        if ch in AA_STANDARD:
            chars.append(ch)
        elif ch in AA_NORMALIZATION:
            chars.append(AA_NORMALIZATION[ch])
    return "".join(chars)


def levenshtein_distance(a: str, b: str, max_distance: Optional[int] = None) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            sub_cost = previous[j - 1] + (0 if ca == cb else 1)
            value = min(insert_cost, delete_cost, sub_cost)
            current.append(value)
            if value < row_min:
                row_min = value
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def insert_subsequence(sequence: str, subseq: str, start: int) -> str:
    if not sequence:
        return ""
    start = max(0, min(start, len(sequence)))
    end = min(start + len(subseq), len(sequence))
    trimmed_subseq = subseq[: end - start]
    return sequence[:start] + trimmed_subseq + sequence[end:]


def randomize_subsequence(
    sequence: str,
    start: int,
    end: int,
    rng: np.random.Generator,
    alphabet: str = AA_STANDARD,
) -> str:
    if not sequence:
        return ""
    start = max(0, min(start, len(sequence)))
    end = max(start, min(end, len(sequence)))
    if end == start:
        return sequence
    randomized = "".join(rng.choice(list(alphabet), size=end - start))
    return sequence[:start] + randomized + sequence[end:]


def permutation_test_p_value(
    group_a: Sequence[float],
    group_b: Sequence[float],
    num_permutations: int = 5000,
    seed: int = 42,
) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    observed = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)

    count = 0
    for _ in range(num_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        diff = float(np.mean(perm_a) - np.mean(perm_b))
        if diff >= observed:
            count += 1
    return (count + 1) / (num_permutations + 1)


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_cleaned_dataset() -> List[SequenceRecord]:
    csv_path = PREPROCESSED_DIR / "cleaned_gvp_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}. Run data_preprocessing.py first.")

    records: List[SequenceRecord] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            seq = normalize_aa_sequence(row.get("sequence", ""))
            family = row.get("family", "").strip()
            if not seq or not family:
                continue
            seq_id = row.get("protein_id", "").strip() or f"seq_{i}"
            records.append(SequenceRecord(seq_id=seq_id, family=family, sequence=seq))
    return records


def filter_min_family_size(records: List[SequenceRecord], min_size: int = MIN_FAMILY_SIZE) -> List[SequenceRecord]:
    counts = Counter(r.family for r in records)
    valid = {f for f, c in counts.items() if c >= min_size}
    return [r for r in records if r.family in valid]


def group_by_family(records: List[SequenceRecord]) -> Dict[str, List[SequenceRecord]]:
    groups: Dict[str, List[SequenceRecord]] = defaultdict(list)
    for record in records:
        groups[record.family].append(record)
    return groups


def get_embedder(method: str):
    if method == "M2":
        from embeddings.esm2 import ESM2Embedder

        return ESM2Embedder()
    if method == "M3":
        from embeddings.protbert import ProtBERTEmbedder

        return ProtBERTEmbedder()
    raise ValueError(f"Unknown method: {method}")


def compute_embeddings(method: str, sequences: Sequence[str], cache_path: Optional[Path] = None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        X = np.load(cache_path)
        if X.shape[0] == len(sequences):
            return X

    embedder = get_embedder(method)
    X = embedder.embed_batch(list(sequences))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
    return X


def get_classifier_bundle_paths(method: str, artifact_root: Path, exp1_root: Path = EXP1_DIR) -> Tuple[Path, Path]:
    import sklearn

    skver = sklearn.__version__.replace(".", "_")
    bundle_name = f"classifier_bundle_{method}_sk{skver}.pkl"
    return exp1_root / bundle_name, artifact_root / bundle_name


def load_saved_classifier_bundle(
    method: str,
    artifact_root: Path,
    exp1_root: Path = EXP1_DIR,
):
    exp1_model_path, _ = get_classifier_bundle_paths(method, artifact_root, exp1_root=exp1_root)
    if not exp1_model_path.exists():
        raise FileNotFoundError(f"Missing saved classifier bundle for method={method}: {exp1_model_path}")

    with open(exp1_model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["classifier"], bundle["label_encoder"]


def train_or_load_classifier(
    records: Sequence[SequenceRecord],
    method: str,
    artifact_root: Path,
    exp1_root: Path = EXP1_DIR,
    force_retrain: bool = False,
    rf_n_jobs: int = 1,
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    exp1_model_path, local_model_path = get_classifier_bundle_paths(method, artifact_root, exp1_root=exp1_root)
    if not force_retrain:
        for model_path in (exp1_model_path, local_model_path):
            if model_path.exists():
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                return bundle["classifier"], bundle["label_encoder"]

    sequences = [r.sequence for r in records]
    labels = [r.family for r in records]

    exp1_cache = exp1_root / f"embeddings_{method}.npy"
    local_cache = artifact_root / f"embeddings_train_{method}.npy"
    cache = exp1_cache if exp1_cache.exists() else local_cache
    X = compute_embeddings(method, sequences, cache_path=cache)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = RandomForestClassifier(
        n_estimators=EXP1_RF_ESTIMATORS,
        random_state=EXP1_RANDOM_STATE,
        n_jobs=rf_n_jobs,
    )
    clf.fit(X, y)

    local_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_model_path, "wb") as f:
        pickle.dump({"classifier": clf, "label_encoder": le, "method": method}, f)
    return clf, le


def classify_generated_sequences(
    method: str,
    sequences: Sequence[str],
    target_family: str,
    cache_name: str,
    cache_root: Path,
    artifact_root: Path,
    exp1_root: Path = EXP1_DIR,
) -> List[Dict]:
    classifier, label_encoder = load_saved_classifier_bundle(
        method=method,
        artifact_root=artifact_root,
        exp1_root=exp1_root,
    )
    rows = predict_with_classifier(
        classifier=classifier,
        label_encoder=label_encoder,
        method=method,
        sequences=sequences,
        cache_name=cache_name,
        cache_root=cache_root,
    )
    for row in rows:
        target_probability = float(row["probabilities"].get(target_family, 0.0))
        row["target_family"] = target_family
        row["target_probability"] = target_probability
        row["is_target_family"] = row["predicted_family"] == target_family
    return rows


def predict_with_classifier(
    classifier,
    label_encoder,
    method: str,
    sequences: Sequence[str],
    cache_name: str,
    cache_root: Path,
) -> List[Dict]:
    cache = cache_root / f"{cache_name}_{method}.npy"
    X = compute_embeddings(method, sequences, cache_path=cache)
    y_pred = classifier.predict(X)
    pred_labels = label_encoder.inverse_transform(y_pred)
    proba = classifier.predict_proba(X)

    rows: List[Dict] = []
    for i, seq in enumerate(sequences):
        class_probs = {label_encoder.classes_[k]: float(proba[i, k]) for k in range(len(label_encoder.classes_))}
        rows.append(
            {
                "generated_id": f"gen_{i:04d}",
                "sequence": seq,
                "length": len(seq),
                "predicted_family": pred_labels[i],
                "predicted_confidence": float(np.max(proba[i])),
                "probabilities": class_probs,
            }
        )
    return rows


def parse_motif_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def motif_catalog_path(target_family: str, method: str) -> Path:
    return EXP3_DIR / target_family / f"motif_catalog_{method}.csv"


def motif_catalog_detail_path(target_family: str, method: str) -> Path:
    return EXP3_DIR / target_family / f"motif_catalog_{method}.json"


def load_motif_catalog(target_family: str, method: str) -> List[Dict]:
    path = motif_catalog_path(target_family, method)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing motif catalog for family={target_family}, method={method}: {path}"
        )
    return parse_motif_rows(path)


def list_motif_catalog_rows(target_family: str, method: str) -> List[Dict]:
    return load_motif_catalog(target_family, method)


def get_motif_row(target_family: str, method: str, motif_uid: str) -> Dict:
    for row in load_motif_catalog(target_family, method):
        if row.get("motif_uid") == motif_uid:
            return row
    raise KeyError(f"Motif not found for family={target_family}, method={method}, motif_uid={motif_uid}")


def load_motif_detail(target_family: str, method: str, motif_uid: str | None = None):
    path = motif_catalog_detail_path(target_family, method)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing motif detail catalog for family={target_family}, method={method}: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if motif_uid is None:
        return payload
    for row in payload:
        if row.get("motif_uid") == motif_uid:
            return row
    raise KeyError(f"Motif detail not found for family={target_family}, method={method}, motif_uid={motif_uid}")


def choose_motif_file(target_family: str, preferred_method: str = "M2") -> Tuple[Optional[Path], str]:
    family_dir = EXP3_DIR / target_family
    preferred = family_dir / f"motif_positions_{preferred_method}.csv"
    if preferred.exists():
        return preferred, preferred_method

    for method in ("M2", "M3"):
        p = family_dir / f"motif_positions_{method}.csv"
        if p.exists():
            return p, method
    return None, ""


def consensus_motif(sequences: Sequence[str], start: int, end: int) -> str:
    if end <= start:
        return ""
    frags = [seq[start:end] for seq in sequences if len(seq) >= end]
    if not frags:
        return ""
    motif_chars = []
    for i in range(end - start):
        cnt = Counter(fragment[i] for fragment in frags)
        motif_chars.append(cnt.most_common(1)[0][0])
    return "".join(motif_chars)


def aa_distribution(sequences: Sequence[str]) -> np.ndarray:
    counts = Counter("".join(sequences))
    vec = np.array([counts.get(a, 0) for a in AA_STANDARD], dtype=float)
    if vec.sum() <= 0:
        vec = np.ones(len(AA_STANDARD), dtype=float)
    vec = vec / vec.sum()
    return vec


def sample_random_sequence(length: int, probs: np.ndarray, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list(AA_STANDARD), p=probs, size=length))


def build_motif_essentiality_sequences(
    natural_target_sequences: Sequence[str],
    motif_consensus: str,
    motif_start: int,
    n_samples: int,
    seed: int,
) -> Tuple[List[str], List[str], str]:
    rng = np.random.default_rng(seed)
    motif = str(motif_consensus or "")
    if not motif:
        raise ValueError("Could not derive a motif consensus for essentiality validation.")

    lengths = [len(s) for s in natural_target_sequences]
    probs = aa_distribution(natural_target_sequences)
    motif_len = len(motif)

    group_a = []
    group_b = []
    for _ in range(n_samples):
        length = int(rng.choice(lengths))
        length = max(length, motif_start + motif_len + 1)
        base = sample_random_sequence(length, probs, rng)
        seq_a = insert_subsequence(base, motif, motif_start)
        seq_b = randomize_subsequence(seq_a, motif_start, motif_start + motif_len, rng=rng)
        group_a.append(seq_a)
        group_b.append(seq_b)

    return group_a, group_b, motif


def scan_motif_in_sequence(sequence: str, motif_consensus: str) -> Tuple[int, float]:
    if not motif_consensus:
        return 0, 0.0
    motif_len = len(motif_consensus)
    if len(sequence) < motif_len:
        return 0, 0.0
    best_start = 0
    best_score = 0.0
    for start in range(len(sequence) - motif_len + 1):
        window = sequence[start:start + motif_len]
        matches = sum(1 for a, b in zip(window, motif_consensus) if a == b)
        score = matches / motif_len
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_score


def build_motif_essentiality_with_natural_fragments(
    natural_sequences: Sequence[str],
    motif_consensus: str,
    n_samples: int,
    seed: int,
    min_identity: float = 0.2,
) -> Tuple[List[str], List[str], List[Dict]]:
    if not motif_consensus:
        raise ValueError("motif_consensus is required for essentiality scanning.")
    motif_len = len(motif_consensus)
    if motif_len == 0:
        raise ValueError("motif_consensus must be non-empty.")

    rng = np.random.default_rng(seed)
    probs = aa_distribution(natural_sequences)

    scored = []
    for idx, seq in enumerate(natural_sequences):
        if len(seq) < motif_len:
            continue
        best_start, best_score = scan_motif_in_sequence(seq, motif_consensus)
        if best_score < min_identity:
            continue
        fragment = seq[best_start:best_start + motif_len]
        scored.append({
            "idx": idx,
            "start": best_start,
            "score": best_score,
            "fragment": fragment,
            "length": len(seq),
        })

    scored.sort(key=lambda r: (-r["score"], -len(r["fragment"]), r["idx"]))
    selected = scored[:n_samples]

    group_a: List[str] = []
    group_b: List[str] = []
    fragment_infos: List[Dict] = []

    for info in selected:
        length = info["length"]
        start = info["start"]
        fragment = info["fragment"]
        pair_rng = np.random.default_rng(seed + info["idx"] * 7919)
        base = sample_random_sequence(length, probs, pair_rng)
        seq_a = insert_subsequence(base, fragment, start)
        seq_b = randomize_subsequence(seq_a, start, start + motif_len, rng=pair_rng)
        group_a.append(seq_a)
        group_b.append(seq_b)
        fragment_infos.append({
            "start": start,
            "end": start + motif_len,
            "fragment": fragment,
            "scan_score": info["score"],
        })

    return group_a, group_b, fragment_infos


def extract_target_probs(pred_rows: Sequence[Dict], target_family: str) -> np.ndarray:
    vals = []
    for row in pred_rows:
        probs = row.get("probabilities", {})
        vals.append(float(probs.get(target_family, 0.0)))
    return np.asarray(vals, dtype=float)


def resolve_motif_interval(row: Dict) -> Tuple[int, int]:
    start = None
    end = None

    for key in ("min_start", "core_start", "start", "motif_start"):
        if row.get(key) not in (None, ""):
            start = int(float(row[key]))
            break

    for key in ("core_end", "max_end", "end", "motif_end"):
        if row.get(key) not in (None, ""):
            end = int(float(row[key]))
            break

    if start is None or end is None or end <= start:
        raise ValueError(f"Invalid motif interval row: {row}")
    return start, end


def select_top_motif(motif_rows: Sequence[Dict]) -> Dict:
    if not motif_rows:
        return {}

    def _score(row: Dict) -> float:
        try:
            n_unique = int(float(row.get("n_unique_sequences", 0)))
        except Exception:
            n_unique = 0
        try:
            n_windows = int(float(row.get("n_windows", 0)))
        except Exception:
            n_windows = 0
        try:
            coverage_peak = float(row.get("coverage_peak", 0) or 0)
        except Exception:
            coverage_peak = 0.0
        try:
            conservation_score = float(row.get("conservation_score", 0) or 0)
        except Exception:
            conservation_score = 0.0
        try:
            position_std = float(row.get("position_std", 0) or 0)
        except Exception:
            position_std = 0.0
        try:
            core_length = int(float(row.get("core_length", 0) or 0))
        except Exception:
            core_length = 0

        if core_length <= 0:
            try:
                start, end = resolve_motif_interval(row)
                core_length = max(0, end - start)
            except Exception:
                core_length = 0

        return (
            3.0 * n_unique
            + 2.0 * coverage_peak
            + 20.0 * conservation_score
            - 0.5 * core_length
            - 2.0 * position_std
            + 0.05 * n_windows
        )

    return max(motif_rows, key=_score)


def resolve_target_families(args, available_families: Sequence[str]) -> List[str]:
    requested = args.families if args.families else [args.target_family]
    deduped = []
    seen = set()
    for fam in requested:
        if fam not in seen:
            deduped.append(fam)
            seen.add(fam)

    missing = [fam for fam in deduped if fam not in available_families]
    if missing:
        raise ValueError(f"Families not found in filtered dataset: {missing}")
    return deduped
