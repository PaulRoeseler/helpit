from __future__ import annotations

import pydoc
import textwrap
from typing import List


def capture_help_text(obj) -> str:
    """Return plain-text help() output for an object using pydoc."""
    try:
        return pydoc.render_doc(obj, renderer=pydoc.plaintext)
    except Exception:
        return ""


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    """
    Character-based chunking with overlap to preserve context.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap >= max_chars:
        overlap = max_chars // 4

    clean = textwrap.dedent(text).strip()
    chunks: List[str] = []
    start = 0
    n = len(clean)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(clean[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


__all__ = ["capture_help_text", "chunk_text"]

