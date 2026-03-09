# Reference Data

This directory contains the ground-truth data used by the scoring program to
evaluate submissions.

## Files

### track1_ground_truth.csv

Ground-truth binary labels for Track 1 (Response Prediction).

Format:
```
model_id,item_id,label
model_001,item_0042,1
model_001,item_0099,0
model_002,item_0042,1
...
```

- `model_id` — Unique identifier for the AI model.
- `item_id` — Unique identifier for the benchmark item.
- `label` — Binary ground truth: 1 (correct) or 0 (incorrect).

### track2_ground_truth.csv

Ground-truth ability scores for Track 2 (Robust Scoring).

Format:
```
model_id,ability_score
model_001,1.234
model_002,-0.567
...
```

- `model_id` — Unique identifier for the AI model.
- `ability_score` — True ability score computed from clean (uncontaminated) data
  using a reference IRT model.

## Note

The example files in this directory contain small synthetic data for testing.
The actual competition data will be larger and will be substituted when the
competition is deployed.
