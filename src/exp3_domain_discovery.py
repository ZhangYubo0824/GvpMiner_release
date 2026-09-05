#!/usr/bin/env python3
"""
实验3: 微观功能域无监督发现 (⭐核心实验)
=========================================
滑动窗口 → 两种蛋白语言模型方法 (ESM-2/ProtBERT) → 双层聚类 → InterPro交叉验证 → Motif发现

方法:
    M2: ESM-2 语言模型嵌入 → KMeans + DBSCAN
    M3: ProtBERT 语言模型嵌入 → KMeans + DBSCAN

用法:
    python exp3_domain_discovery.py                         # 全部家族+全部方法
    python exp3_domain_discovery.py --families GvpA GvpN    # 指定家族
    python exp3_domain_discovery.py --methods M2 M3         # 指定方法
    python exp3_domain_discovery.py --families GvpA --methods M2  # 组合

输出:
    work/results/exp3_domain_discovery/
        {family}/
            windows_metadata.csv              — 滑动窗口元数据
            embeddings_{M2|M3}.npy            — 窗口嵌入缓存
            clustering_{method}_labels.npy    — 聚类标签
            clustering_{method}_metrics.json  — 聚类质量指标
            motif_positions_{method}.csv      — 发现的Motif位置
            motif_heatmap_{family}.png        — 两方法Motif位置热图
        clustering_comparison.csv             — 两方法聚类质量对比
        clustering_comparison.png             — 聚类质量柱状图
"""

import sys
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    EXP3_DIR, PREPROCESSED_DIR, INTERPRO_DIR, GLOBAL_RESIDUE_CACHE_DIR,
    EXP3_WINDOW_SIZE, EXP3_WINDOW_STEP,
    EXP3_DBSCAN_MIN_SAMPLES,
    MIN_FAMILY_SIZE,
)
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


def embed_windows(method_name, sequences, windows, cache_dir):
    """
    逻辑优化：先提取全序列残基嵌入，再通过切片计算窗口均值
    """
    cache_path = cache_dir / f"embeddings_{method_name}_optimized.npy"
    if cache_path.exists():
        return np.load(cache_path)

    import hashlib

    print(f"    提取 {method_name} 残基嵌入并计算窗口向量...")
    embedder = get_window_embedder(method_name)  # 获取 ESM2 或 ProtBERT 实例

    # 1. 预计算所有唯一序列的残基嵌入
    unique_seqs = list(set(sequences))
    seq_to_embedding = {}

    for seq in tqdm(unique_seqs, desc="    [pLM] 序列推理"):
        # 计算序列哈希作为缓存键
        seq_hash = hashlib.md5(seq.encode('utf-8')).hexdigest()
        seq_cache_path = GLOBAL_RESIDUE_CACHE_DIR / f"{method_name}_{seq_hash}.npy"

        if seq_cache_path.exists():
            seq_to_embedding[seq] = np.load(seq_cache_path)
        else:
            emb = embedder.embed_per_residue(seq)
            np.save(seq_cache_path, emb)
            seq_to_embedding[seq] = emb

    # 2. 根据窗口元数据进行内存切片和池化
    window_vectors = []
    for w in tqdm(windows, desc="    [Memory] 窗口切片"):
        parent_seq = w["window_seq_full"]  # 需要在 generate_sliding_windows 中保留原序列参考
        start, end = w["start"], w["end"]

        # 从全序列矩阵中截取对应段落并取均值
        res_embeds = seq_to_embedding[parent_seq][start:end, :]
        if res_embeds.shape[0] == 0:
            window_vec = np.zeros(seq_to_embedding[parent_seq].shape[1])
        else:
            window_vec = res_embeds.mean(axis=0)
        window_vectors.append(window_vec)

    X = np.array(window_vectors)
    np.save(cache_path, X)
    return X


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


def run_clustering(X, method_name, cache_dir):
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    """执行聚类 (UMAP + HDBSCAN)"""

    labels, metrics = _load_cached_clustering(method_name, cache_dir, expected_size=len(X))
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
def interpro_cross_validation(motif_positions, family_name):
    """
    将发现的Motif与InterPro已知domain做位置重叠分析。
    TODO: 需要根据具体InterPro数据格式实现。
    """
    # InterPro 已知domain位置 (需要从data3/interpro/加载)
    # 这里提供框架，具体实现取决于InterPro数据格式
    interpro_domains = load_interpro_domains(family_name)

    if not interpro_domains:
        return {"status": "no_interpro_data", "overlap_rate": None}

    # 计算重叠
    overlap_count = 0
    for motif in motif_positions:
        m_start = motif.get("core_start", motif["min_start"])
        m_end = motif.get("core_end", motif["max_end"])
        for domain in interpro_domains:
            d_start = domain.get("start", 0)
            d_end = domain.get("end", 0)
            overlap = max(0, min(m_end, d_end) - max(m_start, d_start))
            if overlap > 0:
                overlap_count += 1
                break

    overlap_rate = overlap_count / len(motif_positions) if motif_positions else 0

    return {
        "status": "ok",
        "n_motifs": len(motif_positions),
        "n_interpro_domains": len(interpro_domains),
        "n_overlapping": overlap_count,
        "overlap_rate": overlap_rate,
    }


def load_interpro_domains(family_name):
    """加载InterPro已知domain信息"""
    # 尝试加载对应文件
    if not INTERPRO_DIR.exists():
        return []

    # 映射家族到Pfam
    family_pfam_map = {
        "GvpA": ["PF00741"],
        "GvpJ": ["PF00741"],
        "GvpC": ["PF01304"],
        "GvpD": ["PF07088"],
        "GvpF": ["PF06386"],
        "GvpL": ["PF06386"],
        "GvpG": ["PF05120"],
        "GvpO": ["PF05800"],
    }

    pfam_ids = family_pfam_map.get(family_name, [])
    domains = []

    for pfam_id in pfam_ids:
        tsv_path = INTERPRO_DIR / f"pfam_{pfam_id}_proteins.tsv"
        if tsv_path.exists():
            try:
                with open(tsv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        domains.append({
                            "pfam_id": pfam_id,
                            "uniprot": row.get("uniprot_accession", ""),
                            "length": int(row.get("length", 0)) if row.get("length") else 0,
                        })
            except Exception:
                pass

    return domains


# ============================================================
# 主流程
# ============================================================
def load_dataset():
    """加载预处理后的数据集"""
    csv_path = PREPROCESSED_DIR / "cleaned_gvp_dataset.csv"
    if not csv_path.exists():
        print(f"错误: 找不到 {csv_path}")
        print("请先运行 data_preprocessing.py")
        sys.exit(1)

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    return records


def main():
    parser = argparse.ArgumentParser(description="实验3: 功能域无监督发现")
    parser.add_argument("--families", nargs="+", default=None,
                        help="指定家族, 默认处理所有大家族")
    parser.add_argument("--methods", nargs="+", default=["M2", "M3"],
                        choices=["M2", "M3"],
                        help="选择嵌入方法 (默认全部)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="仅处理数据量最大的前N个家族")
    args = parser.parse_args()

    print("=" * 60)
    print("实验3: 微观功能域无监督发现 (核心实验)")
    print("=" * 60)

    # 加载数据
    records = load_dataset()

    # 按家族分组
    family_groups = defaultdict(list)
    for rec in records:
        family_groups[rec["family"]].append(rec)

    # 筛选家族
    if args.families:
        target_families = args.families
    else:
        # 默认: 按数据量排序, 排除太小的
        sorted_families = sorted(family_groups.items(), key=lambda x: -len(x[1]))
        target_families = [f for f, recs in sorted_families if len(recs) >= MIN_FAMILY_SIZE]

    if args.top_n:
        target_families = target_families[:args.top_n]

    print(f"\n将处理 {len(target_families)} 个家族, 方法: {args.methods}")
    for f in target_families:
        print(f"  {f}: {len(family_groups[f])} 条序列")

    # 对每个家族执行完整流程
    all_family_metrics = []

    for fi, family_name in enumerate(tqdm(target_families, desc="家族总进度"), 1):
        print(f"\n{'='*60}")
        print(f"[{fi}/{len(target_families)}] 家族: {family_name} ({len(family_groups[family_name])} 条序列)")
        print(f"{'='*60}")

        family_records = family_groups[family_name]
        metrics = process_family(family_name, family_records, args.methods, EXP3_DIR)

        if metrics:
            all_family_metrics.extend(metrics)

    # 保存汇总对比
    if all_family_metrics:
        summary_path = EXP3_DIR / "clustering_comparison.csv"
        fieldnames = ["family", "method", "k_optimal", "n_motifs",
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
                EXP3_DIR / "clustering_comparison.png",
            )
        except Exception as e:
            print(f"[警告] 汇总图绘制失败: {e}")

    print(f"\n{'='*60}")
    print("实验3完成! ✓")
    print(f"结果保存在: {EXP3_DIR}")
    print(f"{'='*60}")


def deduplicate_exact_windows_within_parent(windows):
    """
    保守去冗余：仅移除同一原始序列内完全重复的窗口序列。

    旧逻辑按 embedding 余弦相似度裁剪重叠窗口，会把连续 motif 轨迹压缩到极少数点。
    这里保留所有不同窗口序列，只清理明显重复项。
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


def build_cluster_position_profile(cluster_windows):
    profile = defaultdict(int)
    for window in cluster_windows:
        start = int(window["start"])
        end = int(window["end"])
        for pos in range(start, end):
            profile[pos] += 1
    return dict(profile)


def _segments_from_positions(positions):
    if not positions:
        return []

    segments = []
    start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        segments.append((start, prev + 1))
        start = pos
        prev = pos
    segments.append((start, prev + 1))
    return segments


def _segment_metrics(profile, start, end):
    values = [int(profile[pos]) for pos in range(start, end)]
    return {
        "core_start": start,
        "core_end": end,
        "core_length": end - start,
        "coverage_peak": max(values) if values else 0,
        "coverage_mass": int(sum(values)),
    }


def extract_core_interval(profile, min_core_length=5, threshold_ratio=0.7):
    if not profile:
        return {
            "core_start": 0,
            "core_end": 0,
            "core_length": 0,
            "coverage_peak": 0,
            "coverage_mass": 0,
        }

    peak = max(int(value) for value in profile.values())
    cutoff = max(1, int(np.floor(peak * threshold_ratio)))
    dense_positions = sorted(pos for pos, count in profile.items() if count >= cutoff)
    if not dense_positions:
        dense_positions = sorted(pos for pos, count in profile.items() if count == peak)

    peak_positions = sorted(pos for pos, count in profile.items() if count == peak)
    peak_segments = _segments_from_positions(peak_positions)
    eligible_peak_segments = [
        _segment_metrics(profile, start, end)
        for start, end in peak_segments
        if (end - start) >= min_core_length
    ]
    if eligible_peak_segments:
        return max(
            eligible_peak_segments,
            key=lambda seg: (seg["coverage_mass"], seg["core_length"], -seg["core_start"]),
        )

    dense_segments = [
        _segment_metrics(profile, start, end)
        for start, end in _segments_from_positions(dense_positions)
    ]
    return max(
        dense_segments,
        key=lambda seg: (seg["coverage_mass"], seg["coverage_peak"], seg["core_length"], -seg["core_start"]),
    )


def compute_conservation_score(fragments):
    usable = [fragment for fragment in fragments if fragment]
    if not usable:
        return 0.0

    width = min(len(fragment) for fragment in usable)
    if width <= 0:
        return 0.0

    scores = []
    for idx in range(width):
        counts = Counter(fragment[idx] for fragment in usable)
        scores.append(counts.most_common(1)[0][1] / len(usable))
    return float(np.mean(scores)) if scores else 0.0


def collect_unique_core_fragments(cluster_windows, core_start, core_end):
    unique_fragments = []
    seen_parents = set()

    for window in cluster_windows:
        parent_id = window.get("parent_id")
        if not parent_id or parent_id in seen_parents:
            continue

        full_seq = window.get("window_seq_full", "")
        if not full_seq or len(full_seq) < core_end:
            continue

        seen_parents.add(parent_id)
        unique_fragments.append({
            "parent_id": parent_id,
            "fragment": full_seq[core_start:core_end],
        })

    return unique_fragments


def build_motif_consensus(fragments):
    usable = [fragment for fragment in fragments if fragment]
    if not usable:
        return ""

    width = min(len(fragment) for fragment in usable)
    if width <= 0:
        return ""

    consensus = []
    for idx in range(width):
        counts = Counter(fragment[idx] for fragment in usable)
        consensus.append(counts.most_common(1)[0][0])
    return "".join(consensus)


def build_position_frequency_matrix(fragments):
    usable = [fragment for fragment in fragments if fragment]
    if not usable:
        return []

    width = min(len(fragment) for fragment in usable)
    if width <= 0:
        return []

    matrix = []
    for idx in range(width):
        counts = Counter(fragment[idx] for fragment in usable)
        total = float(sum(counts.values()))
        matrix.append({
            aa: counts[aa] / total for aa in sorted(counts)
        })
    return matrix


def finalize_motif_catalog(family_name, method_name, motifs):
    def _sort_key(row):
        return (
            -int(row.get("n_unique_sequences", 0) or 0),
            -float(row.get("coverage_peak", 0) or 0.0),
            -float(row.get("conservation_score", 0) or 0.0),
            int(row.get("core_start", 0) or 0),
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
            profile = build_cluster_position_profile(segment_windows)
            core = extract_core_interval(profile)

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
                "occ_core_start": int(core["core_start"]),
                "occ_core_end": int(core["core_end"]),
                "occ_core_length": int(core["core_length"]),
                "n_support_windows": len(segment_windows),
                "support_window_ids": [window.get("window_id", "") for window in segment_windows],
            }
            occurrences.append(occurrence)

    occurrences.sort(key=lambda row: (row["parent_id"], row["occ_start"], row["motif_uid"]))
    return occurrences


def extract_motif_positions(windows, labels, method_name):
    motif_positions = []
    motif_clusters = defaultdict(list)

    for w, label in zip(windows, labels):
        if label == -1:
            continue
        motif_clusters[label].append(w)

    for motif_id, cluster_windows in motif_clusters.items():
        starts = [w["start"] for w in cluster_windows]
        ends = [w["end"] for w in cluster_windows]
        families = Counter(w["family"] for w in cluster_windows)
        parent_ids = set(w["parent_id"] for w in cluster_windows)
        profile = build_cluster_position_profile(cluster_windows)
        core = extract_core_interval(profile)

        fragment_records = collect_unique_core_fragments(
            cluster_windows,
            core["core_start"],
            core["core_end"],
        )
        core_fragments = [item["fragment"] for item in fragment_records]
        motif_consensus = build_motif_consensus(core_fragments)
        position_frequency_matrix = build_position_frequency_matrix(core_fragments)

        motif_positions.append({
            "motif_id": int(motif_id),
            "cluster_label": int(motif_id),
            "method": method_name,
            "n_windows": len(cluster_windows),
            "n_unique_sequences": len(parent_ids),
            "core_start": core["core_start"],
            "core_end": core["core_end"],
            "core_length": core["core_length"],
            "coverage_peak": core["coverage_peak"],
            "coverage_mass": core["coverage_mass"],
            "position_std": float(np.std(starts)) if starts else 0.0,
            "conservation_score": compute_conservation_score(core_fragments),
            "is_primary_core": True,
            "mean_start": np.mean(starts),
            "mean_end": np.mean(ends),
            "min_start": min(starts),
            "max_end": max(ends),
            "families": dict(families),
            "dominant_family": families.most_common(1)[0][0] if families else "",
            "cross_species_count": len(parent_ids),
            "motif_consensus": motif_consensus,
            "support_protein_ids": [item["parent_id"] for item in fragment_records],
            "support_fragments": core_fragments,
            "position_frequency_matrix": position_frequency_matrix,
        })

    return motif_positions


def process_family(family_name, family_records, methods, output_dir):
    family_dir = output_dir / family_name
    family_dir.mkdir(parents=True, exist_ok=True)

    sequences = [r["sequence"] for r in family_records]
    seq_ids = [r.get("protein_id", f"seq_{i}") for i, r in enumerate(family_records)]
    families = [r["family"] for r in family_records]

    print(f"  Step 1: 婊戝姩绐楀彛 (size={EXP3_WINDOW_SIZE}, step={EXP3_WINDOW_STEP})")
    windows, window_seqs = generate_sliding_windows(
        sequences, seq_ids, families,
        EXP3_WINDOW_SIZE, EXP3_WINDOW_STEP,
    )
    print(f"    鐢熸垚 {len(windows)} 涓獥鍙?(鏉ヨ嚜 {len(sequences)} 鏉″簭鍒?)")

    if len(windows) < EXP3_DBSCAN_MIN_SAMPLES * 3:
        print(f"    [璺宠繃] 绐楀彛鏁板お灏?({len(windows)})")
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
        print(f"\n  鏂规硶: {method}")
        print(f"  {'鈹€'*40}")

        X = embed_windows(method, sequences, windows, family_dir).astype(np.float32)
        if np.isnan(X).any():
            print("    [璀﹀憡] 宓屽叆鍚玁aN, 鏇挎崲涓?0")
            X = np.nan_to_num(X)

        windows_full = windows
        X_full = X
        keep_idx = deduplicate_exact_windows_within_parent(windows_full)
        X_dedup = X_full[keep_idx]
        labels_dedup, metrics = run_clustering(X_dedup, method, family_dir)
        labels = np.full(len(X_full), -1)
        labels[keep_idx] = labels_dedup
        use_windows = windows_full
        representative_embeddings = X_full

        metrics["method"] = method
        metrics["family"] = family_name
        all_metrics.append(metrics)

        motifs = extract_motif_positions(use_windows, labels, method)
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
        if motifs:
            with open(motif_csv, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "motif_id", "method", "n_windows", "n_unique_sequences",
                    "core_start", "core_end", "core_length", "coverage_peak",
                    "coverage_mass", "position_std", "conservation_score", "is_primary_core",
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
                    "motif_consensus", "core_start", "core_end", "core_length",
                    "min_start", "max_end", "mean_start", "mean_end", "position_std",
                    "n_windows", "n_unique_sequences", "support_ratio",
                    "coverage_peak", "coverage_mass", "conservation_score",
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
                    "occ_core_start", "occ_core_end", "occ_core_length",
                    "n_support_windows", "support_window_ids",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(motif_occurrences)

            occurrence_json = family_dir / f"motif_occurrences_{method}.json"
            with open(occurrence_json, "w", encoding="utf-8") as f:
                json.dump(motif_occurrences, f, indent=2, ensure_ascii=False)

        print(f"    鍙戠幇 {len(motifs)} 涓€欓€塎otif鍖哄煙")

    try:
        from utils.visualization import plot_motif_heatmap

        heatmap_data = {}
        for method, motifs in all_motif_positions.items():
            positions = [
                (
                    int(m.get("core_start", m["min_start"])),
                    int(m.get("core_end", m["max_end"])),
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
        print(f"  [璀﹀憡] 鐑浘缁樺埗澶辫触: {e}")

    return all_metrics


if __name__ == "__main__":
    main()
