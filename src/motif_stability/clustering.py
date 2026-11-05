from __future__ import annotations

from itertools import product
from typing import Callable, Iterable

import numpy as np
from joblib import Memory


RepresentationBuilder = Callable[[np.ndarray], np.ndarray]
ClusterRunner = Callable[..., np.ndarray]


def cluster_parameter_grid(
    embeddings: np.ndarray,
    *,
    mcs_values: Iterable[int],
    min_samples_values: Iterable[int],
    representation_builder: RepresentationBuilder,
    cluster_runner: ClusterRunner,
) -> dict[tuple[int, int], np.ndarray]:
    """Build one representation and reuse it for every clustering setting."""
    values = np.asarray(embeddings)
    if values.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    representation = np.asarray(representation_builder(values))
    if representation.ndim != 2 or len(representation) != len(values):
        raise ValueError("representation_builder must preserve the number of rows")

    labels_by_parameter: dict[tuple[int, int], np.ndarray] = {}
    for min_cluster_size, min_samples in product(mcs_values, min_samples_values):
        key = (int(min_cluster_size), int(min_samples))
        labels = np.asarray(
            cluster_runner(
                representation,
                min_cluster_size=key[0],
                min_samples=key[1],
            ),
            dtype=int,
        )
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError(f"cluster_runner returned invalid labels for {key}")
        labels_by_parameter[key] = labels
    return labels_by_parameter


def make_umap_builder(
    *,
    n_neighbors: int = 30,
    n_components: int = 10,
    seed: int = 42,
    min_dist: float = 0.0,
    metric: str = "cosine",
) -> RepresentationBuilder:
    """Return a server-friendly UMAP builder with deterministic defaults."""

    def build(values: np.ndarray) -> np.ndarray:
        if len(values) <= 2:
            return np.asarray(values, dtype=float)
        import umap

        reducer = umap.UMAP(
            n_neighbors=max(2, min(int(n_neighbors), len(values) - 1)),
            n_components=max(1, min(int(n_components), len(values) - 1)),
            min_dist=float(min_dist),
            metric=metric,
            random_state=int(seed),
            transform_seed=int(seed),
        )
        return np.asarray(reducer.fit_transform(values), dtype=float)

    return build


def make_pca_builder(*, n_components: int = 10, seed: int = 42) -> RepresentationBuilder:
    """Return a PCA builder used for representation-robustness checks."""

    def build(values: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        components = max(1, min(int(n_components), len(values), values.shape[1]))
        return np.asarray(
            PCA(n_components=components, random_state=int(seed)).fit_transform(values),
            dtype=float,
        )

    return build




def make_cached_hdbscan_clusterer(memory: Memory) -> ClusterRunner:
    """Reuse HDBSCAN's tree computation across MCS values for one task."""

    def cluster(
        values: np.ndarray,
        *,
        min_cluster_size: int,
        min_samples: int,
    ) -> np.ndarray:
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            cluster_selection_method="eom",
            prediction_data=False,
            memory=memory,
        )
        return np.asarray(model.fit_predict(values), dtype=int)

    return cluster
