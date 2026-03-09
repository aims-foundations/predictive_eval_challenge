# Evaluation

## Submission Format

Participants submit a **ZIP file** containing one or both of the following CSVs.
If only one track is targeted, include only the corresponding file.

### Track 1 — Response Prediction

File: `track1_predictions.csv`

```
model_id,item_id,predicted_probability
model_001,item_0042,0.8731
model_001,item_0099,0.2145
model_002,item_0042,0.6500
...
```

- **model_id** and **item_id** must match the IDs provided in the test data.
- **predicted_probability** must be in [0, 1].
- Every (model_id, item_id) pair in the test set must appear exactly once.

### Track 2 — Robust Scoring

File: `track2_scores.csv`

```
model_id,ability_score
model_001,1.234
model_002,-0.567
model_003,0.891
...
```

- **model_id** must match the IDs in the test data.
- **ability_score** is a real-valued scalar (no range restriction).
- Every model_id in the test set must appear exactly once.

---

## Metrics

### Track 1 Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **AUC-ROC** (primary) | Area Under the ROC Curve. Measures ranking quality of predicted probabilities against binary ground truth. | Higher is better |
| **Log-Loss** (secondary) | Negative log-likelihood of the binary labels under the predicted probabilities. Measures calibration. | Lower is better |

### Track 2 Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **Kendall's Tau** (primary) | Rank correlation between submitted ability scores and ground-truth abilities computed from clean data. | Higher is better |
| **Spearman's Rho** (secondary) | Spearman rank correlation. Another measure of monotonic agreement. | Higher is better |

---

## Scoring Details

1. Submissions are unzipped on the server.
2. The scoring program reads ground-truth labels from the reference data and
   participant predictions from the submission directory.
3. For Track 1, predictions are matched to ground truth by (model_id, item_id).
   Missing pairs result in an error.
4. For Track 2, scores are matched to ground truth by model_id.
5. Results are written to `scores.json` and displayed on the leaderboard.

### Edge Cases

- Duplicate (model_id, item_id) pairs → error.
- Predicted probabilities outside [0, 1] → clipped with a warning.
- Missing model_id entries → error with a message listing missing IDs.
- Non-numeric values → error.

---

## Code Submissions

For participants who wish to submit **code** instead of precomputed predictions:

1. Package your code in a ZIP containing at minimum `model.py`.
2. Your `model.py` must implement `train(data_dir)` and
   `predict(test_data_dir, output_dir)`.
3. The ingestion program will call these functions, and the outputs will be
   passed to the scoring program.
4. Time limit: 30 minutes (Development), 60 minutes (Final).
5. GPU access is available (CUDA 12.1 via the competition Docker image).

See `starting_kit/sample_code_submission/` for a template.
