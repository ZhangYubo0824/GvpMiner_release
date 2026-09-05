"""
双层聚类: K-Means 粗定位 + DBSCAN 精筛
========================================
Stage 1: K-Means 确定粗类别 (k 由 Silhouette 自动选择)
Stage 2: 在每个粗簇内用 DBSCAN 筛选高密度核心区域
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances


class TwoStageClustering:
    """双层聚类器"""

    def __init__(self, k_range=(3, 20), dbscan_min_samples=5, dbscan_eps_percentile=90,
                 random_state=42, use_pca=True, pca_variance=0.95, use_cosine=False):
        """
        Args:
            k_range: K-Means 搜索的 k 范围 (min, max)
            dbscan_min_samples: DBSCAN 最小样本数
            dbscan_eps_percentile: DBSCAN eps 自适应计算的百分位 (建议ESM2用75)
            random_state: 随机种子
            use_pca: 是否对高维数据进行PCA降维 (默认开启)
            pca_variance: PCA保留的方差比例 (默认0.95)
            use_cosine: 是否使用Cosine相似度 (对高维嵌入更鲁棒)
        """
        self.k_range = k_range
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_eps_percentile = dbscan_eps_percentile
        self.random_state = random_state
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.use_cosine = use_cosine
        self.scaler = StandardScaler()
        self.pca = None

    def fit_predict(self, X, verbose=True):
        """
        执行双层聚类。
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            verbose: 是否打印过程信息
            
        Returns:
            final_labels: 聚类标签 (-1 = 噪声)
            metrics: 聚类质量指标字典
            stage_info: 各阶段详细信息
        """
        n_samples, n_features = X.shape
        if verbose:
            print(f"  输入: {n_samples} 个样本, {n_features} 维特征")

        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # ---- PCA 降维（可选，针对高维数据）----
        if self.use_pca and n_features > 50:
            if verbose:
                print(f"  [PCA] 开始降维 (当前 {n_features}D, 目标保留 {self.pca_variance*100:.0f}% 方差)...")
            self.pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
            X_pca = self.pca.fit_transform(X_scaled)
            n_components = X_pca.shape[1]
            if verbose:
                print(f"  [PCA] 降维完成: {n_features}D → {n_components}D "
                      f"(解释方差={self.pca.explained_variance_ratio_.sum():.4f})")
            X_scaled = X_pca

        # ---- Stage 1: K-Means 粗定位 ----
        k_optimal, silhouette_scores = self._find_optimal_k(X_scaled, verbose)
        kmeans = KMeans(n_clusters=k_optimal, random_state=self.random_state, n_init=10)
        coarse_labels = kmeans.fit_predict(X_scaled)

        if verbose:
            best_idx = int(np.argmax(silhouette_scores)) if silhouette_scores else 0
            best_sil = silhouette_scores[best_idx] if silhouette_scores else float('nan')
            print(f"  Stage 1: K-Means k={k_optimal}, Silhouette={best_sil:.4f}")

        # ---- Stage 2: DBSCAN 精细筛选 ----
        final_labels = np.full(len(X), -1)
        motif_id = 0
        stage2_info = []

        for cluster_id in range(k_optimal):
            mask = coarse_labels == cluster_id
            n_in_cluster = mask.sum()

            if n_in_cluster < self.dbscan_min_samples * 2:
                if verbose:
                    print(f"    粗簇 {cluster_id}: {n_in_cluster} 样本 -> 跳过 (太小)")
                continue

            X_sub = X_scaled[mask]
            eps_auto = self._auto_eps(X_sub)

            db = DBSCAN(eps=eps_auto, min_samples=self.dbscan_min_samples)
            fine_labels = db.fit_predict(X_sub)

            n_fine_clusters = len(set(fine_labels) - {-1})
            n_noise = (fine_labels == -1).sum()

            for fl in set(fine_labels):
                if fl == -1:
                    continue
                indices = np.where(mask)[0][fine_labels == fl]
                final_labels[indices] = motif_id
                motif_id += 1

            stage2_info.append({
                "coarse_cluster": cluster_id,
                "n_samples": n_in_cluster,
                "eps": eps_auto,
                "n_fine_clusters": n_fine_clusters,
                "n_noise": n_noise,
            })

            if verbose:
                print(f"    粗簇 {cluster_id}: {n_in_cluster} 样本, eps={eps_auto:.3f}, "
                      f"{n_fine_clusters} 个精细簇, {n_noise} 噪声")

        # ---- 计算评估指标 ----
        metrics = self._compute_metrics(X_scaled, final_labels, k_optimal)
        metrics["silhouette_scores_by_k"] = silhouette_scores

        if verbose:
            print(f"  最终: {motif_id} 个Motif簇, "
                  f"{(final_labels != -1).sum()} 核心样本, "
                  f"{(final_labels == -1).sum()} 噪声")
            if "silhouette_final" in metrics:
                print(f"  最终 Silhouette={metrics['silhouette_final']:.4f}")

        return final_labels, metrics, stage2_info

    def _find_optimal_k(self, X_scaled, verbose=True):
        """通过 Silhouette Score 自动选择最优 k"""
        n_samples = X_scaled.shape[0]
        k_min, k_max = self.k_range
        # 保证 k 的合理上下界
        k_min = max(k_min, 2)
        k_max = min(k_max, max(k_min, n_samples // 2))
        k_max = min(k_max, max(k_min, n_samples))

        k_values = [k for k in range(k_min, k_max + 1) if k <= n_samples]
        if not k_values:
            k_values = [max(1, min(n_samples, k_min))]

        scores = []
        for k in k_values:
            # 跳过不合法的 k
            if k < 2 or k > n_samples:
                scores.append(-1.0)
                continue
            try:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = km.fit_predict(X_scaled)
                if len(set(labels)) < 2:
                    scores.append(-1.0)
                    continue
                if self.use_cosine:
                    # 使用Cosine距离计算Silhouette (预计算距离)
                    dist_matrix = pairwise_distances(X_scaled, metric='cosine')
                    score = silhouette_score(dist_matrix, labels, metric='precomputed')
                else:
                    score = silhouette_score(X_scaled, labels)
            except Exception:
                score = -1.0
            scores.append(score)

        # 诊断输出
        if verbose:
            print("  [诊断] K-Means Silhouette分数:")
            for k, score in zip(k_values, scores):
                marker = " ← 最优" if k == k_values[int(np.argmax(scores))] else ""
                status = "✓" if score > 0 else "✗"
                print(f"    k={k:2d}: {score:7.4f} {status}{marker}")
            
            min_score = min(scores)
            if min_score < 0:
                print("  [警告] 所有k值的Silhouette都为负! 建议:")
                print("    • 降低k_range (如改为 (2,8))")
                print("    • 检查数据是否有明显聚类结构")
                print("    • 调整PCA参数或特征预处理")

        k_optimal = k_values[int(np.argmax(scores))]
        return k_optimal, scores

    def _auto_eps(self, X_sub):
        """基于 k-距离图自适应计算 DBSCAN eps (改进版)"""
        k = min(self.dbscan_min_samples, len(X_sub) - 1)
        if k < 1:
            return 0.5

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_sub)
        distances, _ = nn.kneighbors(X_sub)
        
        # 使用自动百分位调整
        # 如果eps_percentile过高，自动降低以避免所有点都成为噪声
        eps_percentile = self.dbscan_eps_percentile
        eps = np.percentile(distances[:, -1], eps_percentile)
        
        # 检查是否会产生过多噪声
        noise_ratio = np.mean(distances[:, -1] > eps)
        if noise_ratio > 0.5:  # 如果>50%的点会成为噪声，降低eps_percentile
            eps = np.percentile(distances[:, -1], max(50, eps_percentile - 20))
        
        return max(eps, 0.01)  # 防止eps为0

    def _compute_metrics(self, X_scaled, labels, k_optimal):
        """计算聚类质量指标"""
        valid_mask = labels != -1
        n_valid = valid_mask.sum()
        n_clusters = len(set(labels[valid_mask])) if n_valid > 0 else 0

        metrics = {
            "k_optimal": k_optimal,
            "n_motifs": n_clusters,
            "n_core_samples": int(n_valid),
            "n_noise": int((~valid_mask).sum()),
            "noise_ratio": float(1 - valid_mask.mean()) if len(labels) > 0 else 1.0,
        }

        if n_valid > 1 and n_clusters > 1:
            try:
                if getattr(self, 'use_cosine', False):
                    dist_matrix = pairwise_distances(X_scaled[valid_mask], metric='cosine')
                    metrics["silhouette_final"] = float(
                        silhouette_score(dist_matrix, labels[valid_mask], metric='precomputed'))
                else:
                    metrics["silhouette_final"] = float(
                        silhouette_score(X_scaled[valid_mask], labels[valid_mask]))
                metrics["davies_bouldin"] = float(
                    davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask]))
                metrics["calinski_harabasz"] = float(
                    calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask]))
            except Exception:
                pass

        return metrics
