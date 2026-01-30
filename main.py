"""Compatibility wrapper; prefer importing from the aihelp package."""

from aihelp import EmbeddingBackend, HFEmbeddingBackend, aihelp, object_header

__all__ = ["aihelp", "EmbeddingBackend", "HFEmbeddingBackend", "object_header"]

