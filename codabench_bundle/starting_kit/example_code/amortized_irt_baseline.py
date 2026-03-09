#!/usr/bin/env python3
"""Amortized IRT Baseline for the Predictive AI Evaluation Challenge.

This script fits an Amortized IRT model that predicts item parameters from
item embeddings. This enables zero-shot generalization to new items that were
not seen during training, making it especially well-suited for Track 1.

Usage:
    python amortized_irt_baseline.py --data-dir /path/to/data --output-dir ./submission

The Amortized IRT model:
    P(correct | model_i, item_j) = c_j + (1-c_j) * sigmoid(a_j * (theta_i - b_j))

where (b_j, a_j, c_j) = f(embedding_j) are predicted by a neural network
from item embeddings, and theta_i are learned ability parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_measure.datasets import load_csv
from torch_measure.models import AmortizedIRT


def main():
    parser = argparse.ArgumentParser(description="Amortized IRT Baseline")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing competition data")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write submission files")
    parser.add_argument("--max-epochs", type=int, default=500, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for embedding network")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers in embedding network")
    parser.add_argument("--pl", type=int, default=2, choices=[1, 2, 3], help="IRT model: 1PL, 2PL, or 3PL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading training data...")

    train_path = data_dir / "train_responses.csv"
    if train_path.exists():
        rm = load_csv(str(train_path))
    else:
        contam_path = data_dir / "contaminated_responses.csv"
        if contam_path.exists():
            rm = load_csv(str(contam_path))
            print("Using contaminated_responses.csv")
        else:
            csv_files = sorted(data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {data_dir}")
            rm = load_csv(str(csv_files[0]))

    print(f"Response matrix: {rm}")
    n_subjects, n_items = rm.shape
    response_matrix = rm.data

    # ── Load item embeddings ──────────────────────────────────────────────
    embeddings_path = data_dir / "item_embeddings.npy"
    if not embeddings_path.exists():
        print(f"WARNING: {embeddings_path} not found.")
        print("Generating random embeddings as placeholder (results will be poor).")
        embedding_dim = 128
        embeddings = torch.randn(n_items, embedding_dim)
    else:
        print(f"Loading item embeddings from {embeddings_path}...")
        embeddings_np = np.load(str(embeddings_path))
        embeddings = torch.from_numpy(embeddings_np).float()
        embedding_dim = embeddings.shape[1]
        print(f"Embeddings shape: {embeddings.shape}")

    if embeddings.shape[0] != n_items:
        raise ValueError(
            f"Number of embeddings ({embeddings.shape[0]}) does not match "
            f"number of items ({n_items})"
        )

    # ── Fit Amortized IRT model ──────────────────────────────────────────
    print(f"Fitting Amortized IRT ({args.pl}PL) on {args.device}...")
    print(f"  {n_subjects} subjects, {n_items} items, {embedding_dim}-dim embeddings")
    print(f"  hidden_dim={args.hidden_dim}, n_layers={args.n_layers}")

    model = AmortizedIRT(
        n_subjects=n_subjects,
        n_items=n_items,
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        pl=args.pl,
        device=args.device,
    )

    history = model.fit(
        response_matrix.to(args.device),
        embeddings=embeddings.to(args.device),
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

        subject_to_idx = {sid: i for i, sid in enumerate(rm.subject_ids)} if rm.subject_ids else {}
        item_to_idx = {iid: i for i, iid in enumerate(rm.item_ids)} if rm.item_ids else {}

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
                # For the amortized model, we could in principle compute
                # predictions for new items if we have their embeddings.
                # Here we fall back to the global mean.
                prob = float(np.nanmean(response_matrix.numpy()))
            predictions.append({"model_id": mid, "item_id": iid, "predicted_probability": prob})

        pred_df = pd.DataFrame(predictions)
        pred_path = output_dir / "track1_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Track 1 predictions written to {pred_path} ({len(pred_df)} rows)")
    else:
        print(f"No test_pairs.csv found. Skipping Track 1.")

    # ── Track 2: Robust Scoring ───────────────────────────────────────────
    print("Generating Track 2 ability scores...")

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
