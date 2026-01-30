from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from .docs import capture_help_text, chunk_text
from .embeddings import EmbeddingBackend, _get_default_embedder, rank_chunks
from .object_info import _short_repr, object_header


def aihelp(
    fn_or_value: Any,
    question: str,
    *,
    model: str = "gpt-5-mini",
    verbosity: str = "low",
    reasoning_effort: str = "minimal",
    max_output_tokens: int = 250,
    add_documentation: bool = False,
    top_k_docs: int = 3,
    chunk_chars: int = 900,
    overlap_chars: int = 150,
    embedder: Optional[EmbeddingBackend] = None,
    openai_client: Any = None,
) -> str:
    """
    Ask OpenAI about a function/value with optional auto-doc retrieval via help().
    """
    try:
        header = object_header(fn_or_value)
    except Exception:
        header = {"schema": "objhdr.error", "repr": _short_repr(fn_or_value, 160)}
    payload: Dict[str, Any] = {"question": question, "object_header": header}

    if callable(fn_or_value):
        fn = fn_or_value
        payload["callable"] = {
            "name": getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))),
            "module": getattr(fn, "__module__", None),
        }
        bound_self = getattr(fn, "__self__", None)
        if bound_self is not None:
            payload["bound_self_header"] = object_header(bound_self)
    else:
        payload["value_repr"] = _short_repr(fn_or_value, 240)

    if add_documentation:
        try:
            help_text = capture_help_text(fn_or_value).strip()
        except Exception:
            help_text = ""
        chunks: List[str] = []
        if help_text:
            try:
                chunks = chunk_text(help_text, max_chars=chunk_chars, overlap=overlap_chars)
            except Exception:
                chunks = []
        if chunks:
            embedder = embedder or _get_default_embedder()
            ranked = rank_chunks(question, chunks, embedder, top_k=top_k_docs)
            doc_chunks = [c for c, _ in ranked]
            if doc_chunks:
                payload["documentation_chunks"] = doc_chunks

    instructions = (
        "You are a concise Python assistant. Use the provided structured info to answer. "
        "If key runtime info is missing, ask for exactly what you need (e.g., shape). "
        "Output: (1) short explanation (2) minimal code snippet. "
        "If documentation_chunks are present, treat them as authoritative documentation for the object."
    )

    client = openai_client
    if client is None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed; install openai>=2.15.0 to call helpit.")
        client = OpenAI()

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
        text={"verbosity": verbosity},
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
    )

    return getattr(response, "output_text", response)


__all__ = ["aihelp"]
