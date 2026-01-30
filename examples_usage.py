"""Runnable examples for calling helpit with and without documentation retrieval."""

import json
from typing import Sequence

from helpit import EmbeddingBackend, helpit


class _StaticResponse:
    def __init__(self, text: str):
        self.output_text = text


class StubResponses:
    def __init__(self, parent):
        self.parent = parent

    def create(self, **kwargs):
        self.parent.calls.append(kwargs)
        return _StaticResponse(self.parent.reply_text)


class StubOpenAIClient:
    def __init__(self, reply_text: str = "ok"):
        self.reply_text = reply_text
        self.calls = []
        self.responses = StubResponses(self)


class TinyEmbedder(EmbeddingBackend):
    def embed(self, texts: Sequence[str]):
        vecs = []
        for t in texts:
            score = 1.0 if "range" in t.lower() else 0.2
            vecs.append([score, 1 - score])
        return vecs


def demo_basic():
    """Call helpit without documentation retrieval."""
    client = StubOpenAIClient("Stubbed answer about len.")
    helpit(len, "What does len return for a list?", openai_client=client, echo=False)
    print("Returned value when echo=False:", None)
    print("Payload sent:", client.calls[-1]["input"])


def demo_with_documentation():
    """Call helpit with help()-based documentation retrieval."""
    client = StubOpenAIClient("Doc-aware stubbed answer.")
    result = helpit(
        range,
        "How do I use start/stop/step?",
        add_documentation=True,
        embedder=TinyEmbedder(),
        chunk_chars=200,
        overlap_chars=40,
        openai_client=client,
        echo=True,
    )
    print("Result:", result)
    print("Top doc chunks attached:", json.loads(client.calls[-1]["input"])["documentation_chunks"])


if __name__ == "__main__":
    demo_basic()
    print("-" * 60)
    demo_with_documentation()
