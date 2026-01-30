"""Compatibility wrapper; prefer importing from the helpit package."""

from helpit import EmbeddingBackend, HFEmbeddingBackend, helpit, object_header, set_default_client

__all__ = ["helpit", "set_default_client", "EmbeddingBackend", "HFEmbeddingBackend", "object_header"]
