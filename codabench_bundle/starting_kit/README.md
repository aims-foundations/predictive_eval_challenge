# Starting Kit — Predictive AI Evaluation Challenge

This starting kit provides everything you need to get started with the
competition.

## Contents

```
starting_kit/
  README.md                         # This file
  track1_baseline_submission.csv    # Example Track 1 submission
  track2_baseline_submission.csv    # Example Track 2 submission
  example_code/
    rasch_baseline.py               # Rasch (1PL) IRT baseline
    twoPL_baseline.py               # 2PL IRT baseline
    amortized_irt_baseline.py       # Amortized IRT baseline
    requirements.txt                # Python dependencies
  sample_code_submission/
    model.py                        # Template for code submissions
    metadata.yaml                   # Codabench metadata
```

## Quick Start

### Option 1: Submit Pre-computed Predictions (Recommended to start)

1. Download the competition data from the "Files" tab on the competition page.
2. Run one of the baseline scripts to generate predictions:
   ```bash
   pip install -r example_code/requirements.txt
   python example_code/rasch_baseline.py --data-dir /path/to/data --output-dir ./submission
   ```
3. Zip the output CSV files and upload to Codabench:
   ```bash
   cd submission
   zip ../my_submission.zip track1_predictions.csv track2_scores.csv
   ```

### Option 2: Submit Code

1. Copy `sample_code_submission/model.py` and implement the `train()` and
   `predict()` functions.
2. Zip your code directory:
   ```bash
   cd my_code_submission
   zip -r ../my_code_submission.zip model.py metadata.yaml *.py
   ```
3. Upload the ZIP file to Codabench. The ingestion program will call your
   `train()` and `predict()` functions automatically.

## Submission Format

### Track 1 — Response Prediction

CSV file named `track1_predictions.csv`:
```
model_id,item_id,predicted_probability
model_001,item_0042,0.8731
model_001,item_0099,0.2145
...
```

### Track 2 — Robust Scoring

CSV file named `track2_scores.csv`:
```
model_id,ability_score
model_001,1.234
model_002,-0.567
...
```

## Baselines

| Baseline | Track 1 AUC-ROC (approx.) | Track 2 Kendall Tau (approx.) | Description |
|----------|---------------------------|-------------------------------|-------------|
| Random | 0.50 | 0.00 | Random predictions |
| Rasch (1PL) | 0.70–0.75 | 0.60–0.70 | Simple IRT model |
| 2PL | 0.72–0.78 | 0.65–0.75 | Adds item discrimination |
| Amortized IRT | 0.75–0.82 | 0.68–0.78 | Uses item embeddings |

(Exact numbers depend on the dataset and split.)

## Dependencies

The baseline scripts require the `torch_measure` package:

```bash
pip install torch-measure
```

Or install from source:
```bash
git clone https://github.com/sttruong/torch_measure.git
cd torch_measure
pip install -e .
```

## Tips

- **Start simple.** The Rasch model is a strong baseline and trains in seconds.
- **Use metadata.** Model size, release date, and item embeddings carry
  predictive signal.
- **Handle sparsity.** The response matrix is only 30–60% observed. Models that
  exploit the pattern of missingness can gain an edge.
- **For Track 2,** consider robust estimation techniques. Standard IRT will fit
  the contaminated entries. Think about outlier detection, trimmed estimation,
  or multi-facet models that separate "genuine" from "inflated" responses.
