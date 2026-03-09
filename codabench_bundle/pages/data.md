# Data Description

## Overview

The task is **amortized prediction**: predict how AI models (test-takers) will
perform on *new, unseen items*. During training you see item content. During
evaluation, your code processes hidden test items — you never see them directly.

Data is drawn from real AI evaluation benchmarks processed through the
[torch_measure](https://github.com/sttruong/torch_measure) library.

---

## Training Data

Available in `train_dir/`:

### train_responses.csv

Long-form response matrix: each row is one observed (model, item) response.

```
model_id,item_id,response
model_001,item_0001,1
model_001,item_0002,0
model_002,item_0001,0
...
```

- `response`: binary — `1` (correct), `0` (incorrect).
- The matrix is sparse: not every (model, item) pair is observed.

### train_items.csv

Full item content for all training items.

```
item_id,item_text,benchmark,category
item_0001,"What is the capital of France?",mmlu,geography
item_0002,"def fibonacci(n): ...",humaneval,coding
...
```

Participants can use item text to build custom featurizers.

### model_metadata.csv

```
model_id,org,param_count,is_instruct,release_date
model_001,OpenAI,175000000000,True,2023-03-14
model_002,Meta,70000000000,False,2023-07-18
...
```

### item_embeddings.npy

Pre-computed text embeddings of training items.

- Shape: `(n_train_items, embedding_dim)`
- Provided as a convenience for participants who want to use embedding-based
  methods (e.g., Amortized IRT) without building their own featurizer.

---

## Hidden Test Data

Available only to your code inside the air-gapped container, in `test_dir/`:

### test_items.csv

Same format as `train_items.csv` but for new, unseen items. Your featurizer
code can read this to extract features from test item content.

### test_pairs.csv

```
model_id,item_id
model_001,item_0042
model_001,item_0099
model_002,item_0042
...
```

The list of (model, item) pairs you must predict. All models appeared in
training; all items are new (not in training).

### test_item_embeddings.npy

Pre-computed embeddings for test items, same embedding model as training.

- Shape: `(n_test_items, embedding_dim)`

---

## Data Statistics (Development Set)

| Property | Approximate Value |
|----------|-------------------|
| Number of models | 100–200 |
| Number of training items | 5,000–20,000 |
| Number of test items | 500–2,000 |
| Response matrix density | 30–60% |
| Number of benchmarks | 5–10 |
| Embedding dimension | 384–768 |

---

## Key Design Principle

**Training items** — you see everything (text, category, embeddings, responses).
Use this to train your model and featurizer.

**Test items** — your *code* sees everything (text, embeddings), but *you* never
see the test item content. The air-gapped container prevents data exfiltration.
Your featurizer must generalize to process unseen items.

This design tests **amortized prediction**: can your learned featurizer +
prediction model generalize to new items without you hand-crafting features
for them?

---

## Downloading the Data

- **Development data** is available from the competition page ("Files" tab).
- **Final phase data** is released at the start of the Final Phase.
- Training data can also be loaded via `torch_measure.datasets`.
