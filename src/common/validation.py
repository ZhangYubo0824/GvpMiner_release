"""Shared sequence records and embeddings used by the generation experiment."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import numpy as np

AA_STANDARD = "ACDEFGHIKLMNPQRSTVWY"

@dataclass
class SequenceRecord:
    seq_id: str
    family: str
    sequence: str


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

