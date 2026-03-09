# Data Description

## Overview

The competition data consists of **binary response matrices** recording whether
AI models (subjects) correctly answered benchmark items. The matrices are drawn
from real AI evaluation benchmarks processed through the
[torch_measure](https://github.com/sttruong/torch_measure) library.

---

## Data Format

### Response Matrix

The core data structure is a CSV file where:
- **Rows** represent AI models (subjects)
- **Columns** represent benchmark items (test questions / coding tasks / etc.)
- **Values** are binary: `1` (correct), `0` (incorrect), or empty (unobserved)

```
,item_0001,item_0002,item_0003,...
model_001,1,0,,1,...
model_002,0,1,1,,...
model_003,,1,0,1,...
```

### Metadata Files

Additional metadata is provided as supplementary CSVs:

**model_metadata.csv**
```
model_id,org,model_name,param_count,is_instruct,release_date
model_001,OpenAI,gpt-4,1760000000000,True,2023-03-14
model_002,Meta,llama-2-70b,70000000000,False,2023-07-18
...
```

**item_metadata.csv**
```
item_id,benchmark,category,difficulty_estimate
item_0001,mmlu,abstract_algebra,0.72
item_0002,humaneval,coding,0.45
...
```

**item_embeddings.npy** (NumPy array)
- Shape: `(n_items, embedding_dim)`
- Pre-computed text embeddings of item content (for use with Amortized IRT or
  other embedding-based approaches).

---

## Track-Specific Data

### Track 1 — Response Prediction

**Training data:**
- `train_responses.csv` — Partially observed response matrix.
- `model_metadata.csv` — Model attributes.
- `item_metadata.csv` — Item attributes.
- `item_embeddings.npy` — Item embeddings.

**Test data:**
- `test_pairs.csv` — List of (model_id, item_id) pairs to predict.

**Ground truth (hidden):**
- Binary labels for each test pair.

### Track 2 — Robust Scoring

**Training data:**
- `contaminated_responses.csv` — Response matrix with both genuine and
  artificially inflated entries. Some models have suspiciously high accuracy on
  certain benchmarks (simulating data leakage / benchmark contamination).
- `model_metadata.csv` — Model attributes.
- `item_metadata.csv` — Item attributes.

**Ground truth (hidden):**
- Ability scores computed from clean (uncontaminated) data only using a
  reference IRT model.

---

## Data Statistics (Development Set)

| Property | Approximate Value |
|----------|-------------------|
| Number of models | 100–200 |
| Number of items | 5,000–20,000 |
| Matrix density | 30–60% |
| Number of benchmarks | 5–10 |
| Contamination rate (Track 2) | 5–15% of entries |

---

## Downloading the Data

- **Development data** is available for download from the competition page
  (click "Files" in the phase panel).
- **Final phase data** is released at the start of the Final Phase.

---

## Notes

- NaN / empty cells indicate unobserved entries, **not** incorrect responses.
- The contaminated entries in Track 2 are realistic: they inflate the accuracy
  of specific models on specific benchmarks, mimicking what happens when training
  data leaks into evaluation sets.
- Item embeddings are provided as a convenience. You are free to compute your
  own features or ignore them entirely.
