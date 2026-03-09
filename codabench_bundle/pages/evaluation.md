# Evaluation

## Submission Format

All submissions are **code submissions**. Participants submit a ZIP file
containing their prediction pipeline, which is executed inside an air-gapped
Docker container against hidden test items.

### Required Interface

Your submission ZIP must contain a `model.py` that implements:

```python
def train(train_dir: str) -> None:
    """Train your model. You can see all training item content."""

def predict(train_dir: str, test_dir: str, output_dir: str) -> None:
    """Predict on hidden test items. Your code sees test item content,
    but you (the participant) do not."""
```

### Data Available to Your Code

**During `train(train_dir)`:**
```
train_dir/
    train_responses.csv       # model_id, item_id, response (0 or 1)
    train_items.csv           # item_id, item_text, benchmark, category, ...
    model_metadata.csv        # model_id, param_count, release_date, org, ...
    item_embeddings.npy       # (n_train_items, embedding_dim)
```

**During `predict(train_dir, test_dir, output_dir)`:**
```
test_dir/
    test_items.csv            # item_id, item_text, benchmark, category, ...
    test_pairs.csv            # (model_id, item_id) pairs to predict
    test_item_embeddings.npy  # (n_test_items, embedding_dim)
```

Your code can read `test_items.csv` and process test item content (e.g., run
your featurizer on item text). But because the container is air-gapped, you
cannot exfiltrate this data.

### Required Output

Write to `output_dir/track1_predictions.csv`:

```
model_id,item_id,predicted_probability
model_001,item_0042,0.8731
model_001,item_0099,0.2145
...
```

- One row per (model_id, item_id) pair from `test_pairs.csv`.
- `predicted_probability` must be in [0, 1].

### Including a Custom Featurizer

If your pipeline uses a custom featurizer (fine-tuned encoder, embedding
model, etc.), include it in your submission ZIP:

```
submission.zip/
    model.py              # implements train() and predict()
    featurizer.py         # your custom featurizer code
    featurizer_weights/   # saved model weights (safetensors format)
    requirements.txt      # additional pip dependencies (optional)
```

For LLM-based featurizers, upload the model to HuggingFace and load it
in your `model.py` via the pre-downloaded weights mounted in the container.

---

## Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **AUC-ROC** (primary) | Area Under the ROC Curve. Measures ranking quality of predicted probabilities against binary ground truth. | Higher is better |
| **Log-Loss** (secondary) | Binary cross-entropy. Measures calibration of predicted probabilities. | Lower is better |

---

## Scoring Details

1. The ingestion program runs your `train()` then `predict()` inside an
   air-gapped Docker container.
2. Your predictions (`track1_predictions.csv`) are passed to the scoring
   program, which compares them against hidden ground-truth labels.
3. Predictions are matched to ground truth by (model_id, item_id).
4. Missing pairs result in an error. Duplicate pairs result in an error.
5. Predicted probabilities outside [0, 1] are clipped with a warning.

---

## Execution Environment

- **Docker image:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- **GPU:** Available via `torch.cuda`
- **Time limit:** 30 min (Development), 60 min (Final) for train + predict combined
- **Network:** Disabled (air-gapped) — no internet access during execution
- **Pre-installed:** PyTorch, NumPy, Pandas, scikit-learn, scipy, transformers, safetensors
- **Custom packages:** Include a `requirements.txt` in your ZIP for additional pip packages

See `starting_kit/sample_code_submission/` for a template.
