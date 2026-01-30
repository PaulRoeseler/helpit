from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .docs import capture_help_text, chunk_text
from .embeddings import EmbeddingBackend, _get_default_embedder, rank_chunks
from .object_info import _short_repr, object_header

_DEFAULT_CLIENT: Optional[Any] = None


def set_default_client(client: Optional[Any]) -> None:
    """
    Set a process-wide default OpenAI-compatible client used when helpit() is
    called without an explicit openai_client. Pass None to clear.
    """
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = client


def helpit(
    fn_or_value: Any,
    question: str,
    *,
    model: str = "gpt-5-mini",
    verbosity: str = "low",
    reasoning_effort: str = "minimal",
    max_output_tokens: int = 500,
    add_documentation: bool = False,
    top_k_docs: int = 3,
    chunk_chars: int = 900,
    overlap_chars: int = 150,
    embedder: Optional[EmbeddingBackend] = None,
    openai_client: Optional[Any] = None,
    echo: bool = False,
) -> Optional[str]:
    """
    Ask OpenAI about a function/value with optional auto-doc retrieval via help().
    Requires an OpenAI-compatible client passed via `openai_client` or configured
    once via set_default_client().
    Always prints the answer (like built-in help()).
    Returns the string only when `echo=True`; otherwise returns None to avoid duplicate REPL output.
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
        "Output: (1) short explanation (2) minimal code snippet. "
        "If documentation_chunks are present, treat them as authoritative documentation for the object."
    )

    client = openai_client if openai_client is not None else _DEFAULT_CLIENT
    if client is None:
        raise ValueError(
            "openai_client is required. Pass an OpenAI() client or compatible Responses API client, "
            "or call set_default_client(OpenAI(...)) once per process."
        )

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
        text={"verbosity": verbosity},
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
    )

    output_text = getattr(response, "output_text", response)
    print(output_text)
    return output_text if echo else None


__all__ = ["helpit", "set_default_client"]
