"""Embedding module exports with optional heavy dependencies."""

try:
    from .esm2 import ESM2Embedder
except Exception:  # pragma: no cover
    ESM2Embedder = None

try:
    from .protbert import ProtBERTEmbedder
except Exception:  # pragma: no cover
    ProtBERTEmbedder = None

__all__ = [
    "ESM2Embedder",
    "ProtBERTEmbedder",
]
