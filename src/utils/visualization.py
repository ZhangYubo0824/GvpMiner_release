"""
可视化工具
==========
论文级别的 Figure 生成 (DPI≥300)。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 无头模式

import seaborn as sns
from pathlib import Path

# 统一样式
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

METHOD_COLORS = {
    "M2_ESM2": "#FF9800",
    "M3_ProtBERT": "#4CAF50",
}


def plot_confusion_matrices(cms, method_names, class_names, save_path):
    """绘制多个混淆矩阵并排 (三方法对比)"""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, cm, name in zip(axes, cms, method_names):
        color = METHOD_COLORS.get(name, "#666666")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(name.replace("_", " "), fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  保存: {save_path}")


def plot_classification_comparison(results_dict, save_path):
    """
    绘制三种方法分类性能柱状图对比。
    results_dict: {"M2_ESM2": {"acc": 0.95, "f1": 0.93}, ...}
    """
    methods = list(results_dict.keys())
    metrics = ["acc", "f1"]
    metric_labels = ["Accuracy", "Macro F1"]

    x = np.arange(len(methods))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results_dict[m].get(metric, 0) for m in methods]
        colors = [METHOD_COLORS.get(m, "#999") for m in methods]
        bars = ax.bar(x + i * width, values, width, label=label, alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Classification Performance Comparison", fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in methods])
    ax.legend()
    ax.set_ylim(0.8, 1.02)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  保存: {save_path}")


def plot_elbow_silhouette(silhouette_scores, k_range, k_optimal, method_name, save_path):
    """绘制 Silhouette Score vs K 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(range(k_range[0], k_range[0] + len(silhouette_scores)))
    ax.plot(ks, silhouette_scores, "o-", color=METHOD_COLORS.get(method_name, "#333"),
            linewidth=2, markersize=6)
    ax.axvline(x=k_optimal, color="red", linestyle="--", alpha=0.7, label=f"Optimal k={k_optimal}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"K-Means Elbow Analysis — {method_name}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  保存: {save_path}")


def plot_motif_heatmap(motif_positions, family_name, method_names, save_path):
    """
    绘制 Motif 位置热图 (每个方法一行, 序列位置为列)。
    motif_positions: {method_name: [(start, end, motif_id), ...]}
    """
    if not motif_positions:
        return

    # 确定最大序列长度
    max_pos = 0
    for method, positions in motif_positions.items():
        for start, end, _ in positions:
            max_pos = max(max_pos, end)

    if max_pos == 0:
        return

    # 构建热图矩阵
    matrix = np.zeros((len(method_names), max_pos))
    for i, method in enumerate(method_names):
        if method in motif_positions:
            for start, end, motif_id in motif_positions[method]:
                matrix[i, start:end] += 1

    fig, ax = plt.subplots(figsize=(14, 3))
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd", yticklabels=method_names,
                xticklabels=False, cbar_kws={"label": "Motif Density"})
    ax.set_xlabel("Sequence Position (aa)")
    ax.set_title(f"Motif Position Heatmap — {family_name}", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  保存: {save_path}")


def plot_clustering_comparison(metrics_list, save_path):
    """
    绘制三种方法的聚类质量指标对比柱状图。
    metrics_list: [{"method": "M2", "silhouette_final": .., ...}, ...]
    """
    methods = [m["method"] for m in metrics_list]
    metric_keys = ["silhouette_final", "n_motifs", "noise_ratio"]
    metric_labels = ["Silhouette Score", "# Motifs Found", "Noise Ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        values = [m.get(key, 0) for m in metrics_list]
        colors = [METHOD_COLORS.get(m_name, "#999") for m_name in methods]
        ax.bar(methods, values, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  保存: {save_path}")


def plot_dimensionality_reduction(X, y_labels, method_name, class_names, save_path):
    """
    绘制降维可视化图 (PCA, t-SNE, UMAP).
    对于高维嵌入矩阵 X 进行降维，并在 2D 平面散点图上按家庭大类着色。
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    has_umap = False
    try:
        import umap
        has_umap = True
    except ImportError:
        print("  [警告] 未安装 umap-learn，将跳过 UMAP 绘图。")

    n_plots = 3 if has_umap else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
        
    print(f"  进行 PCA 降维...")
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X)
    
    print(f"  进行 t-SNE 降维...")
    X_tsne = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(X)
    
    points_list = [(X_pca, "PCA"), (X_tsne, "t-SNE")]
    
    if has_umap:
        print(f"  进行 UMAP 降维...")
        # 针对一些新版本umap，如果不加n_jobs可能会有warning，这里通常默认
        X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        points_list.append((X_umap, "UMAP"))
        
    # 为保证颜色一致性，获取 unique labels
    unique_labels = sorted(list(set(y_labels)))
    palette = sns.color_palette("husl", len(unique_labels))
    
    for ax, (X_proj, title) in zip(axes, points_list):
        sns.scatterplot(
            x=X_proj[:, 0], y=X_proj[:, 1],
            hue=y_labels,
            hue_order=unique_labels,
            palette=palette,
            s=15, alpha=0.8, edgecolor=None, ax=ax
        )
        ax.set_title(f"{title} - {method_name}", fontweight="bold")
        ax.set_xlabel(f"{title} 1")
        ax.set_ylabel(f"{title} 2")
        
        # 只在最后一个图显示 legend，以防遮挡图表
        if title == points_list[-1][1]:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=1, title="Family")
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  降维可视化已保存: {save_path}")
