"""Compatibility wrapper; prefer importing from the helpit package."""

from helpit import EmbeddingBackend, HFEmbeddingBackend, aihelp, object_header

__all__ = ["aihelp", "EmbeddingBackend", "HFEmbeddingBackend", "object_header"]
