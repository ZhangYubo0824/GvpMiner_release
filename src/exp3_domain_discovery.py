#!/usr/bin/env python3
"""Local ESM-2/ProtBERT window discovery with UMAP and HDBSCAN.

The paper profile uses a 20-aa window and 15-aa stride. New outputs are
written under work/results; use the A/B/C workflow for final stable selection.
"""

import sys
import csv
import hashlib
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    EXP3_DIR, PREPROCESSED_DIR, SEQUENCE_TABLE, GLOBAL_RESIDUE_CACHE_DIR,
    EXP3_WINDOW_SIZE, EXP3_WINDOW_STEP,
    EXP3_MIN_WINDOWS,
    MIN_FAMILY_SIZE, ESM2_MODEL_NAME, ESM2_REPR_LAYER,
    PROTBERT_MODEL_NAME,
)
from embeddings import residue_cache
from motif_representative_windows import build_family_method_representatives


# ============================================================
# Step 1: 滑动窗口切片
# ============================================================
def generate_sliding_windows(sequences, seq_ids, families,
                              window_size=None, step=None):
    """
    对所有序列生成滑动窗口。
    
    Returns:
        windows: list of dict, 每个窗口的元数据
        window_seqs: list of str, 窗口序列
    """
    if window_size is None:
        window_size = EXP3_WINDOW_SIZE
    if step is None:
        step = EXP3_WINDOW_STEP

    windows = []
    window_seqs = []

    for seq, seq_id, family in tqdm(zip(sequences, seq_ids, families),
                                     total=len(sequences),
                                     desc="    滑动窗口切片", leave=False):
        if len(seq) < window_size:
            continue

        for start in range(0, len(seq) - window_size + 1, step):
            end = start + window_size
            window_seq = seq[start:end]

            windows.append({
                "window_id": f"{seq_id}_w{start}_{end}",
                "parent_id": seq_id,
                "family": family,
                "start": start,
                "end": end,
                "window_seq": window_seq,
                "window_seq_full": seq,  # 新增：保存原始长序列引用
            })
            window_seqs.append(window_seq)

    return windows, window_seqs


# ============================================================
# Step 2: protein language model embeddings
# ============================================================
# ============================================================
# Step 2: M2/M3 窗口级嵌入
# ============================================================
def get_window_embedder(method_name):
    """获取窗口级嵌入器 (仅 M2/M3)"""
    if method_name == "M2":
        from embeddings.esm2 import ESM2Embedder
        return ESM2Embedder()
    elif method_name == "M3":
        from embeddings.protbert import ProtBERTEmbedder
        return ProtBERTEmbedder()
    else:
        raise ValueError(f"未知嵌入方法: {method_name}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_window_ids(windows):
    digest = hashlib.sha256()
    for window in windows:
        digest.update(f"{window['window_id']}\n".encode("utf-8"))
    return digest.hexdigest()


def _window_cache_paths(cache_dir, method_name):
    cache_dir = Path(cache_dir)
    stem = f"embeddings_{method_name}_optimized"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.manifest.json"


def _embedding_model_signature(method_name):
    if method_name == "M2":
        return {"model_name": ESM2_MODEL_NAME, "repr_layer": ESM2_REPR_LAYER}
    if method_name == "M3":
        return {"model_name": PROTBERT_MODEL_NAME, "repr_layer": None}
    raise ValueError(f"未知嵌入方法: {method_name}")


def _expected_window_cache_manifest(
    method_name,
    windows,
    *,
    window_size,
    window_step,
    dataset_fingerprint,
    array_shape=None,
    dtype=None,
):
    manifest = {
        "schema_version": 1,
        "method": method_name,
        "window_size": int(window_size),
        "window_step": int(window_step),
        "dataset_fingerprint": str(dataset_fingerprint),
        "window_count": len(windows),
        "window_id_fingerprint": fingerprint_window_ids(windows),
        **_embedding_model_signature(method_name),
    }
    if array_shape is not None:
        manifest["array_shape"] = list(array_shape)
    if dtype is not None:
        manifest["dtype"] = str(np.dtype(dtype))
    return manifest


def save_window_embedding_cache(
    cache_dir,
    method_name,
    array,
    windows,
    *,
    window_size,
    window_step,
    dataset_fingerprint,
):
    cache_path, manifest_path = _window_cache_paths(cache_dir, method_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array)
    manifest = _expected_window_cache_manifest(
        method_name,
        windows,
        window_size=window_size,
        window_step=window_step,
        dataset_fingerprint=dataset_fingerprint,
        array_shape=array.shape,
        dtype=array.dtype,
    )

    temporary_cache = cache_path.with_suffix(".npy.tmp")
    with temporary_cache.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary_cache.replace(cache_path)

    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_manifest.replace(manifest_path)


def load_window_embedding_cache(
    cache_dir,
    method_name,
    windows,
    *,
    window_size,
    window_step,
    dataset_fingerprint,
):
    cache_path, manifest_path = _window_cache_paths(cache_dir, method_name)
    if not cache_path.exists() and not manifest_path.exists():
        return None
    if not cache_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            f"incomplete window cache: expected {cache_path.name} and "
            f"{manifest_path.name}"
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        actual = json.load(handle)
    expected = _expected_window_cache_manifest(
        method_name,
        windows,
        window_size=window_size,
        window_step=window_step,
        dataset_fingerprint=dataset_fingerprint,
    )
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"window cache manifest mismatch: {key}: "
                f"expected {value!r}, got {actual.get(key)!r}"
            )

    array = np.load(cache_path, allow_pickle=False)
    for key, value in {
        "array_shape": list(array.shape),
        "dtype": str(array.dtype),
    }.items():
        if actual.get(key) != value:
            raise ValueError(
                f"window cache manifest mismatch: {key}: "
                f"expected {value!r}, got {actual.get(key)!r}"
            )
    expected_shape = (len(windows), residue_cache.EXPECTED_DIMS[method_name])
    if array.shape != expected_shape:
        raise ValueError(
            f"window cache manifest mismatch: array_shape: "
            f"expected {list(expected_shape)!r}, got {list(array.shape)!r}"
        )
    return array


def embed_windows(
    method_name,
    windows,
    cache_dir,
    *,
    window_size,
    window_step,
    dataset_fingerprint,
    residue_cache_dirs,
    residue_cache_fallback_dir,
    audit_callback=None,
    force_recompute=False,
):
    """Pool validated per-residue embeddings without retaining all proteins."""
    cached = None
    if not force_recompute:
        cached = load_window_embedding_cache(
            cache_dir,
            method_name,
            windows,
            window_size=window_size,
            window_step=window_step,
            dataset_fingerprint=dataset_fingerprint,
        )
    if cached is not None:
        return cached

    print(f"    提取 {method_name} 残基嵌入并计算窗口向量...")
    embedder_holder = {}

    def compute_residue_embedding(sequence):
        if "embedder" not in embedder_holder:
            embedder_holder["embedder"] = get_window_embedder(method_name)
        return embedder_holder["embedder"].embed_per_residue(sequence)

    dimension = residue_cache.EXPECTED_DIMS[method_name]
    vectors = np.empty((len(windows), dimension), dtype=np.float32)

    indexes_by_sequence = defaultdict(list)
    for index, window in enumerate(windows):
        indexes_by_sequence[window["window_seq_full"]].append(index)

    for sequence, indexes in tqdm(
        indexes_by_sequence.items(),
        desc="    [pLM/cache] 序列处理",
    ):
        result = residue_cache.load_or_compute(
            method=method_name,
            sequence=sequence,
            primary_dirs=residue_cache_dirs,
            supplement_dir=residue_cache_fallback_dir,
            compute=lambda seq=sequence: compute_residue_embedding(seq),
            audit_callback=audit_callback,
        )
        residue_array = result.array
        for index in indexes:
            window = windows[index]
            start = int(window["start"])
            end = int(window["end"])
            if start < 0 or end > len(sequence) or end <= start:
                raise ValueError(
                    f"Invalid window bounds for {window['window_id']}: "
                    f"{start}:{end} on sequence length {len(sequence)}"
                )
            vectors[index] = residue_array[start:end].mean(axis=0)
        del residue_array

    save_window_embedding_cache(
        cache_dir,
        method_name,
        vectors,
        windows,
        window_size=window_size,
        window_step=window_step,
        dataset_fingerprint=dataset_fingerprint,
    )
    return vectors


# ============================================================
# Step 3: 双层聚类
# ============================================================
def _cache_clustering_results(labels, metrics, method_name, cache_dir):
    """缓存聚类结果"""
    labels_path = cache_dir / f"clustering_{method_name}_labels.npy"
    metrics_path = cache_dir / f"clustering_{method_name}_metrics.json"
    np.save(labels_path, labels)
    metrics_serializable = {k: v for k, v in metrics.items()
                            if not isinstance(v, np.ndarray)}
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2, default=str)


def _load_cached_clustering(method_name, cache_dir, expected_size=None):
    """尝试加载缓存的聚类结果"""
    labels_path = cache_dir / f"clustering_{method_name}_labels.npy"
    metrics_path = cache_dir / f"clustering_{method_name}_metrics.json"
    if labels_path.exists() and metrics_path.exists():
        labels = np.load(labels_path)
        if expected_size is not None and len(labels) != expected_size:
            print(
                f"    跳过缓存: {labels_path.name} 长度 {len(labels)} "
                f"!= 当前输入长度 {expected_size}"
            )
            return None, None
        print(f"    从缓存加载聚类结果: {labels_path.name}")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return labels, metrics
    return None, None


def run_clustering(X, method_name, cache_dir, *, force_recompute=False):
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    """执行聚类 (UMAP + HDBSCAN)"""

    labels, metrics = (None, None)
    if not force_recompute:
        labels, metrics = _load_cached_clustering(
            method_name, cache_dir, expected_size=len(X)
        )
    if labels is not None:
        return labels, metrics

    print(f"    运行聚类 — UMAP+HDBSCAN ({method_name})...")

    import umap
    import hdbscan

    if len(X) < 10:
        print("      - [警告] 样本太少")
        metrics = {
            "k_optimal": 0, "n_motifs": 0, "n_core_samples": 0,
            "noise_ratio": 1.0, "silhouette_final": -1.0,
            "davies_bouldin": -1.0, "calinski_harabasz": -1.0
        }
        return np.full(len(X), -1), metrics

    # 1️⃣ UMAP
    n_neighbors = min(15, max(2, len(X) - 1))
    n_components = min(20, max(2, len(X) - 1))

    print(f"      - UMAP降维 (n_neighbors={n_neighbors}, n_components={n_components})...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=n_components,
        metric="cosine",
        random_state=42
    )

    Xr = reducer.fit_transform(X)

    # 2️⃣ HDBSCAN
    min_cluster_size = min(20, max(2, len(X) // 5))
    min_samples = min(10, max(1, min_cluster_size // 2))

    print(f"      - HDBSCAN聚类 (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    labels = clusterer.fit_predict(Xr)

    # 3️⃣ silhouette
    mask = labels != -1
    n_clusters = len(set(labels[mask]))

    # 修改 run_clustering 函数中的计算逻辑
    mask = labels != -1
    n_clusters = len(set(labels[mask]))

    if mask.sum() > 1 and n_clusters > 1:
        # 轮廓系数（代码中已有）
        score = silhouette_score(Xr[mask], labels[mask], metric="cosine")
        
        # 新增：计算 Davies-Bouldin Index (越小越好)
        db_score = davies_bouldin_score(Xr[mask], labels[mask])
        
        # 新增：计算 Calinski-Harabasz Index (越大越好)
        ch_score = calinski_harabasz_score(Xr[mask], labels[mask])
    else:
        score = -1.0
        db_score = -1.0
        ch_score = -1.0

    # 更新 metrics 字典
    metrics = {
        "k_optimal": n_clusters,
        "n_motifs": n_clusters,
        "n_core_samples": int(mask.sum()),
        "noise_ratio": float(1.0 - (mask.sum() / len(X))),
        "silhouette_final": float(score),
        "davies_bouldin": float(db_score),    # 替换 -1.0
        "calinski_harabasz": float(ch_score)  # 替换 -1.0
    }

    _cache_clustering_results(labels, metrics, method_name, cache_dir)

    return labels, metrics


# ============================================================
# Step 5: InterPro 交叉验证
# ============================================================




# ============================================================
# 主流程
# ============================================================
def load_dataset(csv_path=None):
    """加载预处理后的数据集"""
    csv_path = Path(csv_path or (PREPROCESSED_DIR / "cleaned_gvp_dataset.csv"))
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到预处理数据集: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    return records


def make_cache_audit_callback(path):
    path = Path(path)
    fields = [
        "method", "sequence_hash", "path", "source", "shape", "dtype", "validation",
    ]

    def append_record(record):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        row = dict(record)
        if isinstance(row.get("shape"), (list, tuple)):
            row["shape"] = "x".join(map(str, row["shape"]))
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    return append_record


def run_exp3(
    *,
    dataset_csv,
    output_dir,
    window_size,
    window_step,
    methods,
    families=None,
    top_n=None,
    residue_cache_dirs=(),
    residue_cache_fallback_dir=GLOBAL_RESIDUE_CACHE_DIR,
    audit_callback=None,
    force_recompute=False,
):
    dataset_csv = Path(dataset_csv)
    output_dir = Path(output_dir)
    residue_cache_dirs = [Path(path) for path in residue_cache_dirs]
    residue_cache_fallback_dir = Path(residue_cache_fallback_dir)
    if int(window_size) <= 0 or int(window_step) <= 0:
        raise ValueError("window_size and window_step must be positive")
    unknown_methods = sorted(set(methods) - set(residue_cache.EXPECTED_DIMS))
    if unknown_methods:
        raise ValueError(f"Unsupported embedding methods: {unknown_methods}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_fingerprint = sha256_file(dataset_csv)
    print("=" * 60)
    print("实验3: 微观功能域无监督发现 (核心实验)")
    print("=" * 60)
    records = load_dataset(dataset_csv)

    # 按家族分组
    family_groups = defaultdict(list)
    for rec in records:
        family_groups[rec["family"]].append(rec)

    # 筛选家族
    if families:
        target_families = list(families)
    else:
        # 默认: 按数据量排序, 排除太小的
        sorted_families = sorted(family_groups.items(), key=lambda x: -len(x[1]))
        target_families = [f for f, recs in sorted_families if len(recs) >= MIN_FAMILY_SIZE]

    if top_n:
        target_families = target_families[:top_n]

    print(f"\n将处理 {len(target_families)} 个家族, 方法: {list(methods)}")
    for f in target_families:
        print(f"  {f}: {len(family_groups[f])} 条序列")

    # 对每个家族执行完整流程
    all_family_metrics = []

    for fi, family_name in enumerate(tqdm(target_families, desc="家族总进度"), 1):
        print(f"\n{'='*60}")
        print(f"[{fi}/{len(target_families)}] 家族: {family_name} ({len(family_groups[family_name])} 条序列)")
        print(f"{'='*60}")

        family_records = family_groups[family_name]
        metrics = process_family(
            family_name,
            family_records,
            methods,
            output_dir,
            window_size=window_size,
            window_step=window_step,
            residue_cache_dirs=residue_cache_dirs,
            residue_cache_fallback_dir=residue_cache_fallback_dir,
            dataset_fingerprint=dataset_fingerprint,
            audit_callback=audit_callback,
            force_recompute=force_recompute,
        )

        if metrics:
            all_family_metrics.extend(metrics)

    # 保存汇总对比
    if all_family_metrics:
        summary_path = output_dir / "clustering_comparison.csv"
        fieldnames = ["family", "method", "window_size", "window_step",
                       "n_windows_raw", "n_windows_deduplicated",
                       "k_optimal", "n_motifs",
                       "n_core_samples", "noise_ratio", "silhouette_final",
                       "davies_bouldin", "calinski_harabasz"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for m in all_family_metrics:
                writer.writerow({k: m.get(k, "") for k in fieldnames})
        print(f"\n汇总对比: {summary_path}")

        # 绘制汇总图
        try:
            from utils.visualization import plot_clustering_comparison
            plot_clustering_comparison(
                all_family_metrics,
                output_dir / "clustering_comparison.png",
            )
        except Exception as e:
            print(f"[警告] 汇总图绘制失败: {e}")

    print(f"\n{'='*60}")
    print("实验3完成! ✓")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*60}")
    return all_family_metrics


def main():
    parser = argparse.ArgumentParser(description="实验3: 功能域无监督发现")
    parser.add_argument("--families", nargs="+", default=None,
                        help="指定家族, 默认处理所有大家族")
    parser.add_argument("--methods", nargs="+", default=["M2", "M3"],
                        choices=["M2", "M3"],
                        help="选择嵌入方法 (默认全部)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="仅处理数据量最大的前N个家族")
    parser.add_argument("--window-size", type=int, default=EXP3_WINDOW_SIZE)
    parser.add_argument("--window-step", type=int, default=EXP3_WINDOW_STEP)
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=SEQUENCE_TABLE,
    )
    parser.add_argument("--output-dir", type=Path, default=EXP3_DIR)
    parser.add_argument(
        "--residue-cache-dir",
        action="append",
        type=Path,
        default=[],
        help="只读逐残基 embedding 缓存目录（可重复）",
    )
    parser.add_argument(
        "--residue-cache-fallback-dir",
        type=Path,
        default=GLOBAL_RESIDUE_CACHE_DIR,
        help="缓存缺失时唯一允许写入的补充目录",
    )
    parser.add_argument(
        "--cache-audit-csv",
        type=Path,
        default=None,
        help="可选的脱敏逐残基缓存审计 CSV",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Rebuild window embeddings and clustering while still reusing residue caches",
    )
    args = parser.parse_args()

    run_exp3(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        window_size=args.window_size,
        window_step=args.window_step,
        methods=args.methods,
        families=args.families,
        top_n=args.top_n,
        residue_cache_dirs=args.residue_cache_dir,
        residue_cache_fallback_dir=args.residue_cache_fallback_dir,
        audit_callback=(
            make_cache_audit_callback(args.cache_audit_csv)
            if args.cache_audit_csv is not None
            else None
        ),
        force_recompute=args.force_recompute,
    )


def deduplicate_exact_windows_within_parent(windows):
    """
    保守去冗余：仅移除同一原始序列内完全重复的窗口序列。

    保留不同窗口序列及其位置轨迹，不按嵌入相似度裁剪重叠窗口。
    """
    seq_groups = defaultdict(list)
    for i, w in enumerate(windows):
        seq_groups[w["parent_id"]].append(i)

    keep_global = []
    for _, indices in seq_groups.items():
        seen_window_seqs = set()
        for idx in indices:
            window_seq = windows[idx].get("window_seq", "")
            if window_seq and window_seq in seen_window_seqs:
                continue
            if window_seq:
                seen_window_seqs.add(window_seq)
            keep_global.append(idx)

    keep_global = np.array(sorted(keep_global))
    total_windows = len(windows)
    print(f"    exact-window deduplication retained {len(keep_global)} / {total_windows} "
          f"({len(keep_global)/total_windows:.4f})")
    return keep_global


def summarize_motif_clusters(windows, labels, method_name):
    """Summarize stable window clusters without inferring motif-core boundaries."""
    motif_clusters = defaultdict(list)
    for window, label in zip(windows, labels):
        if int(label) != -1:
            motif_clusters[int(label)].append(window)

    summaries = []
    for cluster_label, cluster_windows in motif_clusters.items():
        starts = [int(window["start"]) for window in cluster_windows]
        ends = [int(window["end"]) for window in cluster_windows]
        families = Counter(str(window["family"]) for window in cluster_windows)
        parent_ids = {str(window["parent_id"]) for window in cluster_windows}
        summaries.append({
            "motif_id": int(cluster_label),
            "cluster_label": int(cluster_label),
            "method": method_name,
            "n_windows": len(cluster_windows),
            "n_unique_sequences": len(parent_ids),
            "position_std": float(np.std(starts)) if starts else 0.0,
            "mean_start": float(np.mean(starts)),
            "mean_end": float(np.mean(ends)),
            "min_start": min(starts),
            "max_end": max(ends),
            "families": dict(families),
            "dominant_family": families.most_common(1)[0][0] if families else "",
            "cross_species_count": len(parent_ids),
            "support_protein_ids": sorted(parent_ids),
        })
    return summaries


def finalize_motif_catalog(family_name, method_name, motifs):
    def _sort_key(row):
        stability_floor = min(
            float(row.get("median_window_jaccard", 0.0) or 0.0),
            float(row.get("median_protein_jaccard", 0.0) or 0.0),
            float(row.get("median_occurrence_jaccard", 0.0) or 0.0),
        )
        return (
            -int(row.get("n_unique_sequences", 0) or 0),
            -int(row.get("n_windows", 0) or 0),
            -stability_floor,
            int(row.get("cluster_label", row.get("motif_id", 0)) or 0),
        )

    finalized = []
    for rank, row in enumerate(sorted(motifs, key=_sort_key), start=1):
        motif = dict(row)
        motif["family"] = family_name
        motif["method"] = method_name
        motif["cluster_label"] = int(motif.get("cluster_label", motif.get("motif_id", 0)))
        motif["rank_within_method"] = rank
        motif["motif_uid"] = f"{family_name}_{method_name}_motif_{rank:03d}"
        finalized.append(motif)
    return finalized


def _split_windows_into_occurrence_segments(cluster_windows):
    if not cluster_windows:
        return []

    ordered = sorted(cluster_windows, key=lambda row: (int(row["start"]), int(row["end"])))
    segments = [[ordered[0]]]
    current_end = int(ordered[0]["end"])

    for window in ordered[1:]:
        start = int(window["start"])
        end = int(window["end"])
        if start <= current_end:
            segments[-1].append(window)
            current_end = max(current_end, end)
            continue
        segments.append([window])
        current_end = end

    return segments


def build_sequence_motif_occurrences(windows, labels, motifs, method_name):
    motif_by_cluster = {
        int(motif.get("cluster_label", motif.get("motif_id", -1))): motif
        for motif in motifs
    }

    grouped = defaultdict(list)
    for window, label in zip(windows, labels):
        if int(label) == -1:
            continue
        cluster_label = int(label)
        if cluster_label not in motif_by_cluster:
            continue
        grouped[(window["parent_id"], cluster_label)].append(window)

    occurrences = []
    for (parent_id, cluster_label), cluster_windows in sorted(grouped.items()):
        motif = motif_by_cluster[cluster_label]
        for occ_idx, segment_windows in enumerate(_split_windows_into_occurrence_segments(cluster_windows), start=1):
            starts = [int(window["start"]) for window in segment_windows]
            ends = [int(window["end"]) for window in segment_windows]

            occurrence = {
                "occurrence_id": f"{motif['motif_uid']}_occ_{occ_idx:03d}",
                "motif_uid": motif["motif_uid"],
                "family": motif.get("family", segment_windows[0].get("family", "")),
                "method": method_name,
                "cluster_label": cluster_label,
                "parent_id": parent_id,
                "occ_start": min(starts),
                "occ_end": max(ends),
                "occ_length": max(ends) - min(starts),
                "n_support_windows": len(segment_windows),
                "support_window_ids": [window.get("window_id", "") for window in segment_windows],
            }
            occurrences.append(occurrence)

    occurrences.sort(key=lambda row: (row["parent_id"], row["occ_start"], row["motif_uid"]))
    return occurrences


def process_family(
    family_name,
    family_records,
    methods,
    output_dir,
    *,
    window_size=EXP3_WINDOW_SIZE,
    window_step=EXP3_WINDOW_STEP,
    residue_cache_dirs=(),
    residue_cache_fallback_dir=GLOBAL_RESIDUE_CACHE_DIR,
    dataset_fingerprint="unspecified-dataset",
    audit_callback=None,
    force_recompute=False,
):
    family_dir = Path(output_dir) / family_name
    family_dir.mkdir(parents=True, exist_ok=True)

    sequences = [r["sequence"] for r in family_records]
    seq_ids = [r.get("protein_id", f"seq_{i}") for i, r in enumerate(family_records)]
    families = [r["family"] for r in family_records]

    print(f"  Step 1: sliding windows (size={window_size}, step={window_step})")
    windows, window_seqs = generate_sliding_windows(
        sequences, seq_ids, families,
        window_size, window_step,
    )
    print(f"    Generated {len(windows)} windows from {len(sequences)} sequences")

    if len(windows) < EXP3_MIN_WINDOWS:
        print(f"    [skip] Too few windows ({len(windows)})")
        return None

    windows_csv_path = family_dir / "windows_metadata.csv"
    with open(windows_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "window_id", "parent_id", "family", "start", "end", "window_seq", "window_seq_full"
        ])
        writer.writeheader()
        writer.writerows(windows)

    all_motif_positions = {}
    all_metrics = []

    for method in methods:
        representative_embeddings = None
        print(f"\n  Method: {method}")
        print(f"  {'-'*40}")

        X = embed_windows(
            method,
            windows,
            family_dir,
            window_size=window_size,
            window_step=window_step,
            dataset_fingerprint=dataset_fingerprint,
            residue_cache_dirs=residue_cache_dirs,
            residue_cache_fallback_dir=residue_cache_fallback_dir,
            audit_callback=audit_callback,
            force_recompute=force_recompute,
        ).astype(np.float32)
        if np.isnan(X).any():
            print("    [warning] Embeddings contain NaN; replacing with zero")
            X = np.nan_to_num(X)

        windows_full = windows
        X_full = X
        keep_idx = deduplicate_exact_windows_within_parent(windows_full)
        X_dedup = X_full[keep_idx]
        labels_dedup, metrics = run_clustering(
            X_dedup, method, family_dir, force_recompute=force_recompute
        )
        labels = np.full(len(X_full), -1)
        labels[keep_idx] = labels_dedup
        use_windows = windows_full
        representative_embeddings = X_full

        metrics["method"] = method
        metrics["family"] = family_name
        metrics["window_size"] = int(window_size)
        metrics["window_step"] = int(window_step)
        metrics["n_windows_raw"] = len(X_full)
        metrics["n_windows_deduplicated"] = len(X_dedup)
        all_metrics.append(metrics)

        motifs = summarize_motif_clusters(use_windows, labels, method)
        motifs = finalize_motif_catalog(family_name, method, motifs)
        motif_occurrences = build_sequence_motif_occurrences(use_windows, labels, motifs, method)
        total_sequences = max(1, len(set(seq_ids)))
        for motif in motifs:
            motif["support_ratio"] = float(motif["n_unique_sequences"]) / float(total_sequences)
        all_motif_positions[method] = motifs

        if representative_embeddings is not None and motifs:
            representative_path, representative_rows = build_family_method_representatives(
                family_dir=family_dir,
                method=method,
                windows=use_windows,
                labels=labels,
                embeddings=representative_embeddings,
                catalog_rows=motifs,
            )
            print(
                f"    Representative windows: {len(representative_rows)} -> "
                f"{representative_path.name}"
            )

        motif_csv = family_dir / f"motif_positions_{method}.csv"
        with open(motif_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "motif_id", "method", "n_windows", "n_unique_sequences",
                "position_std",
                "mean_start", "mean_end", "min_start", "max_end",
                "dominant_family", "cross_species_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(motifs)

        catalog_csv = family_dir / f"motif_catalog_{method}.csv"
        with open(catalog_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                    "motif_uid", "family", "method", "cluster_label", "rank_within_method",
                    "min_start", "max_end", "mean_start", "mean_end", "position_std",
                    "n_windows", "n_unique_sequences", "support_ratio",
                    "dominant_family", "cross_species_count",
                ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(motifs)

        catalog_json = family_dir / f"motif_catalog_{method}.json"
        with open(catalog_json, "w", encoding="utf-8") as f:
            json.dump(motifs, f, indent=2, ensure_ascii=False)

        occurrence_csv = family_dir / f"motif_occurrences_{method}.csv"
        with open(occurrence_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "occurrence_id", "motif_uid", "family", "method", "cluster_label",
                "parent_id", "occ_start", "occ_end", "occ_length",
                "n_support_windows", "support_window_ids",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(motif_occurrences)

        occurrence_json = family_dir / f"motif_occurrences_{method}.json"
        with open(occurrence_json, "w", encoding="utf-8") as f:
            json.dump(motif_occurrences, f, indent=2, ensure_ascii=False)

        print(f"    Discovered {len(motifs)} candidate motif regions")

    try:
        from utils.visualization import plot_motif_heatmap

        heatmap_data = {}
        for method, motifs in all_motif_positions.items():
            positions = [
                (
                    int(m["min_start"]),
                    int(m["max_end"]),
                    m["motif_id"],
                )
                for m in motifs
            ]
            heatmap_data[method] = positions

        plot_motif_heatmap(
            heatmap_data, family_name, methods,
            family_dir / f"motif_heatmap_{family_name}.png",
        )
    except Exception as e:
        print(f"  [warning] Motif heatmap generation failed: {e}")

    return all_metrics


if __name__ == "__main__":
    main()
