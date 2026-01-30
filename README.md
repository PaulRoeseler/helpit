# helpit

[![PyPI](https://img.shields.io/pypi/v/helpit.svg)](https://pypi.org/project/helpit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Python's `help()` is great… until it isn’t.**

`helpit` lets you ask questions about `help()` output and get answers tailored to the **exact object in front of you**.

<video src="assets/helpit_gif.mp4" width="720" controls muted loop></video>

---

## Install

```bash
pip install helpit
```

---

## Quickstart

```python
from openai import OpenAI
from helpit import helpit, set_default_client

set_default_client(OpenAI())  

import torch

x = torch.randn(1, 32, 1)
helpit(
    x.squeeze,
    "How do I remove the last dimension and keep the leading dim?",
)
```

---

## Examples

```python
import pandas as pd
from helpit import helpit

df = pd.DataFrame({"city": [None, "ZRH", None], "sales": [3, 10, 2]})

helpit(df, "Group by city, keep NaNs, and sum sales—what's the right dropna setting?")
```


```python
from helpit import helpit

def stream():
    for i in range(1_000_000):
        yield i

it = stream()
helpit(it, "How do I safely consume only the first 10 items without exhausting the iterator?")
```

```python
from sklearn.ensemble import RandomForestClassifier
from helpit import helpit

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=0,
)

helpit(rf, "Which hyperparameters matter most for overfitting here? Show how to adjust them.")
```


---

## When docs matter 📚 (grounded answers)

`helpit` can attach the most relevant `help()` snippets when `add_documentation=True`, keeping answers grounded without dumping full docs.

```python
from transformers import pipeline
from helpit import helpit

pipe = pipeline("sentiment-analysis")

helpit(pipe, "How does this pipeline work?", add_documentation=True)
```

---

## Run locally 🏠 (fast + private)

`helpit` talks to the OpenAI **Responses API**. Any compatible local server works.

```python
from openai import OpenAI
from helpit import helpit

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",   # any token works
)

helpit(
    len,
    "How does len behave on nested lists?",
    model="llama3.2",
    openai_client=client,
)
```

---

## Tests

```bash
python3 -m unittest discover -s tests -v
```
