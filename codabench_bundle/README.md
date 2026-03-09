# Predictive AI Evaluation Challenge — Codabench Competition Bundle

This directory contains the complete competition bundle for the **Predictive AI
Evaluation Challenge**, ready to be zipped and uploaded to
[Codabench](https://www.codabench.org/).

## Directory Structure

```
codabench_bundle/
  competition.yaml              # Main Codabench competition configuration (v2)
  logo.png                      # Competition logo (TO BE ADDED)
  README.md                     # This file
  pages/
    description.md              # Competition overview (shown on Codabench)
    evaluation.md               # Evaluation criteria and submission format
    terms.md                    # Terms and conditions
    data.md                     # Dataset description
  scoring_program/
    score.py                    # Scoring script (computes AUC-ROC, log-loss, tau, rho)
    metadata.yaml               # Codabench execution metadata
    requirements.txt            # Python dependencies for scoring
  ingestion_program/
    ingestion.py                # Runs participant code submissions
    metadata.yaml               # Codabench execution metadata
    requirements.txt            # Python dependencies for ingestion
  reference_data/
    README.md                   # Ground truth format documentation
    track1_ground_truth_example.csv   # Example ground truth (Track 1)
    track2_ground_truth_example.csv   # Example ground truth (Track 2)
  starting_kit/
    README.md                   # Getting started guide for participants
    track1_baseline_submission.csv    # Example Track 1 submission
    track2_baseline_submission.csv    # Example Track 2 submission
    example_code/
      rasch_baseline.py         # Rasch (1PL) IRT baseline
      twoPL_baseline.py         # 2PL IRT baseline
      amortized_irt_baseline.py # Amortized IRT baseline
      requirements.txt          # Dependencies (includes torch_measure)
    sample_code_submission/
      model.py                  # Template for code submissions
      metadata.yaml             # Codabench metadata
```

## How to Deploy

### 1. Prepare Reference Data

Replace the example CSV files in `reference_data/` with your actual ground
truth data:

- `track1_ground_truth.csv` — Binary labels for Track 1 test pairs
- `track2_ground_truth.csv` — True ability scores for Track 2

### 2. Add a Logo

Place a `logo.png` (recommended: 200x200 px) in the bundle root directory.

### 3. Create the Bundle ZIP

```bash
cd codabench_bundle
zip -r ../predictive_eval_challenge_bundle.zip \
    competition.yaml \
    pages/ \
    scoring_program/ \
    ingestion_program/ \
    reference_data/ \
    starting_kit/
```

Note: The `logo.png` should be included in the ZIP if present. Add it to the
zip command above when ready.

### 4. Upload to Codabench

1. Go to https://www.codabench.org/
2. Log in and navigate to "Management" > "Benchmarks" > "Upload"
3. Upload the ZIP file.
4. Review and publish.

## Configuration Details

### Phases

| Phase | Dates | Submissions/Day | Time Limit |
|-------|-------|-----------------|------------|
| Development | Apr 1 – Jul 1, 2026 | 5 | 30 min |
| Final | Jul 1 – Aug 1, 2026 | 2 | 60 min |

### Leaderboards

**Track 1 — Response Prediction**
- AUC-ROC (descending, primary)
- Log-Loss (ascending)

**Track 2 — Robust Scoring**
- Kendall's Tau (descending, primary)
- Spearman's Rho (descending)

### Docker Image

`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

Includes PyTorch 2.1, CUDA 12.1, and Python 3.10. The scoring program
additionally requires scikit-learn, scipy, pandas, and numpy (installed via
the scoring program's requirements.txt).

## Testing Locally

You can test the scoring program locally:

```bash
# Create mock directories
mkdir -p /tmp/test_scoring/input/res /tmp/test_scoring/input/ref /tmp/test_scoring/output

# Copy example files
cp starting_kit/track1_baseline_submission.csv /tmp/test_scoring/input/res/track1_predictions.csv
cp starting_kit/track2_baseline_submission.csv /tmp/test_scoring/input/res/track2_scores.csv
cp reference_data/track1_ground_truth_example.csv /tmp/test_scoring/input/ref/track1_ground_truth.csv
cp reference_data/track2_ground_truth_example.csv /tmp/test_scoring/input/ref/track2_ground_truth.csv

# Run scorer
pip install -r scoring_program/requirements.txt
python scoring_program/score.py /tmp/test_scoring/input /tmp/test_scoring/output

# Check results
cat /tmp/test_scoring/output/scores.json
```
