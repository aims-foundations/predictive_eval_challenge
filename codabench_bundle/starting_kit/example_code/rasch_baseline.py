#!/usr/bin/env python3
"""Rasch (1PL) IRT Baseline for the Predictive AI Evaluation Challenge.

This script fits a Rasch model to the training response matrix and produces
predictions for both Track 1 (response prediction) and Track 2 (robust scoring).

Usage:
    python rasch_baseline.py --data-dir /path/to/data --output-dir ./submission

The Rasch model is the simplest IRT model:
    P(correct | model_i, item_j) = sigmoid(theta_i - b_j)

where theta_i is the ability of model i and b_j is the difficulty of item j.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_measure.datasets import load_csv
from torch_measure.models import Rasch


def main():
    parser = argparse.ArgumentParser(description="Rasch (1PL) IRT Baseline")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing competition data")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write submission files")
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading training data...")

    # Track 1: Load the training response matrix
    train_path = data_dir / "train_responses.csv"
    if train_path.exists():
        rm = load_csv(str(train_path))
        print(f"Training response matrix: {rm}")
    else:
        print(f"WARNING: {train_path} not found. Looking for alternative files...")
        # Try loading any CSV that looks like a response matrix
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        rm = load_csv(str(csv_files[0]))
        print(f"Loaded: {csv_files[0]} -> {rm}")

    n_subjects, n_items = rm.shape
    response_matrix = rm.data  # (n_subjects, n_items) tensor with NaN for missing

    # ── Fit Rasch model ───────────────────────────────────────────────────
    print(f"Fitting Rasch model ({n_subjects} subjects, {n_items} items)...")
    model = Rasch(n_subjects=n_subjects, n_items=n_items, device=args.device)
    history = model.fit(
        response_matrix.to(args.device),
        max_epochs=args.max_epochs,
        lr=args.lr,
        verbose=True,
    )
    print(f"Training complete. Final loss: {history['losses'][-1]:.6f}")

    # ── Track 1: Response Prediction ──────────────────────────────────────
    test_pairs_path = data_dir / "test_pairs.csv"
    if test_pairs_path.exists():
        print("Generating Track 1 predictions...")
        test_pairs = pd.read_csv(test_pairs_path)

        # Build lookup from IDs to indices
        subject_to_idx = {sid: i for i, sid in enumerate(rm.subject_ids)} if rm.subject_ids else {}
        item_to_idx = {iid: i for i, iid in enumerate(rm.item_ids)} if rm.item_ids else {}

        # Compute full probability matrix
        with torch.no_grad():
            prob_matrix = model.predict().cpu().numpy()

        predictions = []
        for _, row in test_pairs.iterrows():
            mid = str(row["model_id"])
            iid = str(row["item_id"])
            si = subject_to_idx.get(mid)
            ii = item_to_idx.get(iid)
            if si is not None and ii is not None:
                prob = float(prob_matrix[si, ii])
            else:
                # Fallback: use global mean for unknown models/items
                prob = float(np.nanmean(response_matrix.numpy()))
            predictions.append({"model_id": mid, "item_id": iid, "predicted_probability": prob})

        pred_df = pd.DataFrame(predictions)
        pred_path = output_dir / "track1_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Track 1 predictions written to {pred_path} ({len(pred_df)} rows)")
    else:
        print(f"No test_pairs.csv found at {test_pairs_path}. Skipping Track 1.")

    # ── Track 2: Robust Scoring ───────────────────────────────────────────
    print("Generating Track 2 ability scores...")

    # For the Rasch baseline, ability scores are simply the learned theta parameters.
    # Note: This naive baseline does NOT handle contamination robustly.
    # A better approach would use outlier detection or robust estimation.
    with torch.no_grad():
        abilities = model.ability.cpu().numpy()

    if rm.subject_ids:
        scores_df = pd.DataFrame({
            "model_id": rm.subject_ids,
            "ability_score": abilities,
        })
    else:
        scores_df = pd.DataFrame({
            "model_id": [f"model_{i:03d}" for i in range(n_subjects)],
            "ability_score": abilities,
        })

    scores_path = output_dir / "track2_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Track 2 scores written to {scores_path} ({len(scores_df)} rows)")

    print("Done!")


if __name__ == "__main__":
    main()
