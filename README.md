# helpit

`help()` is great… until it isn’t.

It often dumps a wall of documentation that:
- isn’t tailored to *your* object (its current attributes, fields, shapes, dtypes, etc.)
- isn’t tailored to *your* question
- is verbose when you just want the “do this” answer

**helpit** flips that: it inspects the **runtime object you already have**, packages up **safe, useful metadata** (type/module/signature/fields + lightweight hints for things like pandas/torch/pathlib), and asks a small OpenAI model to answer **your specific question** quickly.

If the object is complex or the answer likely lives in docs, you can turn on:

- `add_documentation=True`

…and `helpit` will grab the object’s `help()` output, chunk it, and attach only the **most relevant snippets** (picked via embeddings using `intfloat/multilingual-e5-small`) so the model can ground its answer.

---

## Install

### Option A: venv + editable install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
````

### Option B: uv

```bash
uv sync
```

---

## Quickstart 

```python
from helpit import aihelp

def scale(x, factor=2):
    return x * factor

print(aihelp(scale, "How do I use this to double a list?"))
````

---

## When things get tricky: add doc snippets

Use this when you suspect the model might need extra *documentation* to be correct

```python
from openai import OpenAI
from helpit import aihelp

client = OpenAI()  # expects OPENAI_API_KEY

# Example: a built-in function where you want details grounded in docs
answer = aihelp(
    len,
    "How does len behave on nested lists, and what errors should I expect?",
    model="gpt-5-mini",
    verbosity="low",
    reasoning_effort="minimal",
    max_output_tokens=250,
    add_documentation=True,
    top_k_docs=2,
    openai_client=client,
)

print(answer)
```

---

## Using local OpenAI-compatible servers (Ollama, vLLM, etc.)

`helpit` talks to the OpenAI **Responses** API. Any local server that exposes a compatible `/v1/responses` endpoint can be used by passing a custom `OpenAI` client:

```python
from openai import OpenAI
from helpit import aihelp

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # Ollama example

print(
    aihelp(
        len,
        "How does len behave on a list?",
        model="llama3.2",            # whatever your server calls the model
        max_output_tokens=200,
        openai_client=client,
    )
)
```

- **Ollama**: `/v1/responses` is supported in v0.13.3+ (stateless only). Use any token as `api_key`, set `base_url` to your Ollama host, and pick a local model name (e.g., `llama3.2`, `qwen2.5`).
- **vLLM**: run the OpenAI-compatible server and point `base_url` to it; use the served model name.
- If your backend lacks `/v1/responses`, upgrade or run a thin proxy that maps Responses requests to chat/completions, or fork `helpit` to call chat/completions directly.

---

## Parameters

* **fn_or_value**: object or callable to describe
* **question**: text passed to the model
* **model**: OpenAI Responses model name
* **verbosity**: value for Responses API `text.verbosity`
* **reasoning_effort**: value for Responses API `reasoning.effort`
* **max_output_tokens**: cap for model output tokens
* **add_documentation**: when `True`, include ranked `help()` passages
* **top_k_docs**: maximum doc chunks to attach
* **chunk_chars**: maximum characters per `help()` chunk
* **overlap_chars**: overlap size between chunks
* **embedder**: `EmbeddingBackend` implementation; defaults to HF `multilingual-e5-small`
* **openai_client**: OpenAI client or stub; defaults to `OpenAI()`

---


## Offline demo

```bash
python examples_usage.py
```

Runs two stubbed calls (no network): a basic query and a doc-enriched query using a tiny deterministic embedder.

---

## Tests

```bash
python -m unittest -v
```
