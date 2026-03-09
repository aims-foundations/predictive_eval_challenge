#!/usr/bin/env python3
"""Scoring program for the Predictive AI Evaluation Challenge.

This script is executed inside the Codabench Docker container. It reads
participant predictions and reference ground-truth data, computes metrics
for Track 1 (Response Prediction) and Track 2 (Robust Scoring), and writes
results to scores.json.

Usage (called by Codabench via metadata.yaml):
    python3 score.py <input_dir> <output_dir>

Directory layout inside the container:
    <input_dir>/
        res/               # Participant submission (unzipped)
            track1_predictions.csv   (optional)
            track2_scores.csv        (optional)
        ref/               # Reference data
            track1_ground_truth.csv  (if Track 1 task)
            track2_ground_truth.csv  (if Track 2 task)
    <output_dir>/
        scores.json        # Written by this script
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    """Print a timestamped log message to stdout (visible in Codabench logs)."""
    print(f"[SCORING] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Area Under the ROC Curve."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        log("WARNING: Only one class present in ground truth. AUC-ROC is undefined; returning 0.5.")
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Compute binary log-loss (cross-entropy)."""
    from sklearn.metrics import log_loss

    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return float(log_loss(y_true, y_pred_clipped))


def compute_kendall_tau(scores_pred: np.ndarray, scores_true: np.ndarray) -> float:
    """Compute Kendall's tau rank correlation."""
    from scipy.stats import kendalltau

    tau, _ = kendalltau(scores_pred, scores_true)
    if np.isnan(tau):
        log("WARNING: Kendall's tau is NaN (likely constant input). Returning 0.0.")
        return 0.0
    return float(tau)


def compute_spearman_rho(scores_pred: np.ndarray, scores_true: np.ndarray) -> float:
    """Compute Spearman's rank correlation."""
    from scipy.stats import spearmanr

    rho, _ = spearmanr(scores_pred, scores_true)
    if np.isnan(rho):
        log("WARNING: Spearman's rho is NaN (likely constant input). Returning 0.0.")
        return 0.0
    return float(rho)


# ---------------------------------------------------------------------------
# Track 1: Response Prediction
# ---------------------------------------------------------------------------

def score_track1(submission_dir: Path, reference_dir: Path) -> dict[str, float]:
    """Score Track 1 (Response Prediction) submission.

    Returns dict with keys: track1_auc_roc, track1_log_loss
    """
    pred_path = submission_dir / "track1_predictions.csv"
    truth_path = reference_dir / "track1_ground_truth.csv"

    if not pred_path.exists():
        log(f"Track 1 prediction file not found at {pred_path}. Skipping Track 1.")
        return {}

    if not truth_path.exists():
        log(f"Track 1 ground truth not found at {truth_path}. Skipping Track 1.")
        return {}

    log("Scoring Track 1 — Response Prediction")
    log(f"  Reading predictions from {pred_path}")
    log(f"  Reading ground truth from {truth_path}")

    # Load predictions
    try:
        pred_df = pd.read_csv(pred_path)
    except Exception as e:
        raise ValueError(f"Failed to parse track1_predictions.csv: {e}") from e

    required_cols = {"model_id", "item_id", "predicted_probability"}
    if not required_cols.issubset(pred_df.columns):
        missing = required_cols - set(pred_df.columns)
        raise ValueError(
            f"track1_predictions.csv is missing columns: {missing}. "
            f"Required columns: {required_cols}"
        )

    # Load ground truth
    try:
        truth_df = pd.read_csv(truth_path)
    except Exception as e:
        raise ValueError(f"Failed to parse track1_ground_truth.csv: {e}") from e

    truth_required = {"model_id", "item_id", "label"}
    if not truth_required.issubset(truth_df.columns):
        missing = truth_required - set(truth_df.columns)
        raise ValueError(
            f"track1_ground_truth.csv is missing columns: {missing}. "
            f"Required columns: {truth_required}"
        )

    # Ensure model_id and item_id are strings for consistent merging
    for df in [pred_df, truth_df]:
        df["model_id"] = df["model_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)

    # Check for duplicates in predictions
    pred_keys = pred_df[["model_id", "item_id"]]
    if pred_keys.duplicated().any():
        n_dups = pred_keys.duplicated().sum()
        raise ValueError(
            f"track1_predictions.csv contains {n_dups} duplicate (model_id, item_id) pairs. "
            f"Each pair must appear exactly once."
        )

    # Merge predictions with ground truth
    merged = truth_df.merge(
        pred_df[["model_id", "item_id", "predicted_probability"]],
        on=["model_id", "item_id"],
        how="left",
    )

    # Check for missing predictions
    n_missing = merged["predicted_probability"].isna().sum()
    if n_missing > 0:
        missing_examples = merged[merged["predicted_probability"].isna()][["model_id", "item_id"]].head(5)
        raise ValueError(
            f"Missing predictions for {n_missing} (model_id, item_id) pairs. "
            f"First few missing:\n{missing_examples.to_string(index=False)}"
        )

    # Validate predicted_probability values
    preds = merged["predicted_probability"].values.astype(np.float64)
    if np.any(np.isnan(preds)):
        raise ValueError("predicted_probability contains NaN values.")

    # Clip out-of-range values with warning
    out_of_range = np.sum((preds < 0) | (preds > 1))
    if out_of_range > 0:
        log(f"WARNING: {out_of_range} predicted_probability values outside [0, 1]. Clipping.")
        preds = np.clip(preds, 0.0, 1.0)

    labels = merged["label"].values.astype(np.float64)

    log(f"  Number of predictions: {len(preds)}")
    log(f"  Label distribution: {np.mean(labels):.4f} positive rate")

    auc = compute_auc_roc(labels, preds)
    ll = compute_log_loss(labels, preds)

    log(f"  AUC-ROC: {auc:.6f}")
    log(f"  Log-Loss: {ll:.6f}")

    return {"track1_auc_roc": auc, "track1_log_loss": ll}


# ---------------------------------------------------------------------------
# Track 2: Robust Scoring
# ---------------------------------------------------------------------------

def score_track2(submission_dir: Path, reference_dir: Path) -> dict[str, float]:
    """Score Track 2 (Robust Scoring) submission.

    Returns dict with keys: track2_kendall_tau, track2_spearman_rho
    """
    pred_path = submission_dir / "track2_scores.csv"
    truth_path = reference_dir / "track2_ground_truth.csv"

    if not pred_path.exists():
        log(f"Track 2 scores file not found at {pred_path}. Skipping Track 2.")
        return {}

    if not truth_path.exists():
        log(f"Track 2 ground truth not found at {truth_path}. Skipping Track 2.")
        return {}

    log("Scoring Track 2 — Robust Scoring")
    log(f"  Reading predictions from {pred_path}")
    log(f"  Reading ground truth from {truth_path}")

    # Load predictions
    try:
        pred_df = pd.read_csv(pred_path)
    except Exception as e:
        raise ValueError(f"Failed to parse track2_scores.csv: {e}") from e

    required_cols = {"model_id", "ability_score"}
    if not required_cols.issubset(pred_df.columns):
        missing = required_cols - set(pred_df.columns)
        raise ValueError(
            f"track2_scores.csv is missing columns: {missing}. "
            f"Required columns: {required_cols}"
        )

    # Load ground truth
    try:
        truth_df = pd.read_csv(truth_path)
    except Exception as e:
        raise ValueError(f"Failed to parse track2_ground_truth.csv: {e}") from e

    truth_required = {"model_id", "ability_score"}
    if not truth_required.issubset(truth_df.columns):
        missing = truth_required - set(truth_df.columns)
        raise ValueError(
            f"track2_ground_truth.csv is missing columns: {missing}. "
            f"Required columns: {truth_required}"
        )

    # Ensure model_id is string
    pred_df["model_id"] = pred_df["model_id"].astype(str)
    truth_df["model_id"] = truth_df["model_id"].astype(str)

    # Check for duplicate model_ids in predictions
    if pred_df["model_id"].duplicated().any():
        n_dups = pred_df["model_id"].duplicated().sum()
        raise ValueError(
            f"track2_scores.csv contains {n_dups} duplicate model_id entries. "
            f"Each model_id must appear exactly once."
        )

    # Merge on model_id
    merged = truth_df.merge(
        pred_df[["model_id", "ability_score"]],
        on="model_id",
        how="left",
        suffixes=("_true", "_pred"),
    )

    # Check for missing models
    n_missing = merged["ability_score_pred"].isna().sum()
    if n_missing > 0:
        missing_models = merged[merged["ability_score_pred"].isna()]["model_id"].head(10).tolist()
        raise ValueError(
            f"Missing ability scores for {n_missing} models. "
            f"Missing model_ids (first 10): {missing_models}"
        )

    # Validate scores
    scores_pred = merged["ability_score_pred"].values.astype(np.float64)
    scores_true = merged["ability_score_true"].values.astype(np.float64)

    if np.any(np.isnan(scores_pred)):
        raise ValueError("ability_score contains NaN values in submission.")
    if np.any(np.isinf(scores_pred)):
        raise ValueError("ability_score contains Inf values in submission.")

    log(f"  Number of models: {len(scores_pred)}")
    log(f"  Predicted score range: [{scores_pred.min():.4f}, {scores_pred.max():.4f}]")
    log(f"  Ground truth score range: [{scores_true.min():.4f}, {scores_true.max():.4f}]")

    tau = compute_kendall_tau(scores_pred, scores_true)
    rho = compute_spearman_rho(scores_pred, scores_true)

    log(f"  Kendall Tau: {tau:.6f}")
    log(f"  Spearman Rho: {rho:.6f}")

    return {"track2_kendall_tau": tau, "track2_spearman_rho": rho}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 score.py <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Codabench places submission files in <input_dir>/res/
    # and reference data in <input_dir>/ref/
    submission_dir = input_dir / "res"
    reference_dir = input_dir / "ref"

    log(f"Input directory: {input_dir}")
    log(f"Submission directory: {submission_dir}")
    log(f"Reference directory: {reference_dir}")
    log(f"Output directory: {output_dir}")

    # List files for debugging
    if submission_dir.exists():
        log(f"Submission files: {sorted(os.listdir(submission_dir))}")
    else:
        log(f"WARNING: Submission directory does not exist: {submission_dir}")

    if reference_dir.exists():
        log(f"Reference files: {sorted(os.listdir(reference_dir))}")
    else:
        log(f"WARNING: Reference directory does not exist: {reference_dir}")

    scores = {}
    errors = []

    # Score Track 1
    try:
        track1_scores = score_track1(submission_dir, reference_dir)
        scores.update(track1_scores)
    except Exception as e:
        error_msg = f"Track 1 scoring failed: {e}"
        log(f"ERROR: {error_msg}")
        log(traceback.format_exc())
        errors.append(error_msg)

    # Score Track 2
    try:
        track2_scores = score_track2(submission_dir, reference_dir)
        scores.update(track2_scores)
    except Exception as e:
        error_msg = f"Track 2 scoring failed: {e}"
        log(f"ERROR: {error_msg}")
        log(traceback.format_exc())
        errors.append(error_msg)

    # Validate that at least one track was scored
    if not scores:
        error_detail = " | ".join(errors) if errors else "No prediction files found."
        log(f"FATAL: No scores computed. {error_detail}")
        log(
            "Please ensure your submission ZIP contains at least one of:\n"
            "  - track1_predictions.csv (for Track 1)\n"
            "  - track2_scores.csv (for Track 2)"
        )
        # Write a minimal scores.json so Codabench does not crash, but with
        # sentinel values indicating failure.
        scores = {
            "track1_auc_roc": 0.0,
            "track1_log_loss": 999.0,
            "track2_kendall_tau": 0.0,
            "track2_spearman_rho": 0.0,
        }

    # Write scores
    scores_path = output_dir / "scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)

    log(f"Scores written to {scores_path}")
    log(f"Final scores: {json.dumps(scores, indent=2)}")
    log("Scoring complete.")


if __name__ == "__main__":
    main()
