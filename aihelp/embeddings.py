from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    import math

    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return [0.0 for _ in vec]
    return [x / norm for x in vec]


class EmbeddingBackend:
    """Minimal embedding interface so real and stub backends match."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


class HFEmbeddingBackend(EmbeddingBackend):
    """
    Hugging Face backend using intfloat/multilingual-e5-small.
    Lazily loads to avoid cost when add_documentation=False.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._normalize = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            import torch.nn.functional as F
            from transformers.utils import logging as hf_logging
            from huggingface_hub import constants as hf_constants
            from huggingface_hub import logging as hub_logging
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for add_documentation=True. "
                "Install with `pip install transformers torch sentencepiece`."
            ) from exc

        hub_logging.set_verbosity_error()
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        hf_constants.HF_HUB_CACHE
        os.environ.setdefault("HF_HUB_OFFLINE", "0")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._torch = torch
        self._normalize = F.normalize

        target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(target)
        self.device = target

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        self._ensure_loaded()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        normalize = self._normalize

        batch = tokenizer(
            list(texts),
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**batch)
            last_hidden = outputs.last_hidden_state
            attn = batch["attention_mask"]
            masked = last_hidden.masked_fill(~attn[..., None].bool(), 0.0)
            pooled = masked.sum(dim=1) / attn.sum(dim=1)[..., None]
            pooled = normalize(pooled, p=2, dim=1)
        return pooled.cpu().tolist()


_DEFAULT_EMBEDDER: Optional[EmbeddingBackend] = None


def _get_default_embedder() -> EmbeddingBackend:
    """
    Module-level singleton so the HF model is loaded only once per process.
    """
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = HFEmbeddingBackend()
    return _DEFAULT_EMBEDDER


def rank_chunks(question: str, chunks: Sequence[str], embedder: EmbeddingBackend, *, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Embed query + chunks and return top_k (chunk, score) pairs sorted by score desc.
    """
    if not chunks:
        return []

    texts = [f"query: {question}"] + [f"passage: {c}" for c in chunks]
    try:
        vectors = embedder.embed(texts)
    except Exception:
        return []
    if not vectors:
        return []

    query_vec = _l2_normalize(vectors[0])
    chunk_vecs = [_l2_normalize(v) for v in vectors[1:]]

    def _sim(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    scored = [(chunk, _sim(query_vec, v)) for chunk, v in zip(chunks, chunk_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, top_k)]


__all__ = [
    "EmbeddingBackend",
    "HFEmbeddingBackend",
    "_get_default_embedder",
    "rank_chunks",
]
