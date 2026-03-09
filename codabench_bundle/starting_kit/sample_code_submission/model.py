"""Sample code submission for the Predictive AI Evaluation Challenge.

This module serves as a template for code submissions. Participants must
implement the ``train()`` and ``predict()`` functions below.

The ingestion program will:
    1. Import this module.
    2. Call ``train(data_dir)`` with the path to the input data directory.
    3. Call ``predict(test_data_dir, output_dir)`` to generate predictions.

Both functions receive string paths. The output directory is where you must
write your prediction CSV files.

Required output files (write at least one):
    - ``track1_predictions.csv`` — For Track 1 (Response Prediction)
    - ``track2_scores.csv`` — For Track 2 (Robust Scoring)

See the starting kit README and baseline scripts for detailed examples.
"""

from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Global state (populated by train, used by predict)
# ──────────────────────────────────────────────────────────────────────────
MODEL = None


def train(data_dir: str) -> None:
    """Train your model on the competition data.

    This function is called once before ``predict()``. Use it to load data,
    fit your model, and store any state needed for prediction in module-level
    variables.

    Parameters
    ----------
    data_dir : str
        Absolute path to the input data directory. Contains:
            - train_responses.csv           (Track 1 training data)
            - contaminated_responses.csv    (Track 2 training data)
            - model_metadata.csv            (model attributes)
            - item_metadata.csv             (item attributes)
            - item_embeddings.npy           (pre-computed item embeddings)
            - test_pairs.csv                (Track 1: pairs to predict)

    Notes
    -----
    - You can import any libraries available in the Docker image
      (pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime) plus any packages
      listed in your submission's requirements.txt.
    - GPU is available via torch.cuda.
    - Time limit: 30 min (Development) / 60 min (Final) for train+predict combined.
    """
    global MODEL

    data_path = Path(data_dir)

    # ── Example: Load data ────────────────────────────────────────────────
    # import pandas as pd
    # import numpy as np
    # import torch
    # from torch_measure.datasets import load_csv
    # from torch_measure.models import Rasch  # or TwoPL, AmortizedIRT, etc.
    #
    # rm = load_csv(str(data_path / "train_responses.csv"))
    # model = Rasch(n_subjects=rm.n_subjects, n_items=rm.n_items, device="cuda")
    # model.fit(rm.data.cuda(), max_epochs=1000, lr=0.01)
    # MODEL = {"model": model, "rm": rm}

    # TODO: Replace this placeholder with your training code.
    print("[model.py] train() called — implement your training logic here.")
    MODEL = {"data_dir": data_dir}


def predict(test_data_dir: str, output_dir: str) -> None:
    """Generate predictions and write them to the output directory.

    This function is called after ``train()``. Use the trained model to
    produce predictions and write them as CSV files.

    Parameters
    ----------
    test_data_dir : str
        Absolute path to the input data directory (same as train's data_dir).
    output_dir : str
        Absolute path to the output directory. Write your prediction files here:
            - track1_predictions.csv  (columns: model_id, item_id, predicted_probability)
            - track2_scores.csv       (columns: model_id, ability_score)

    Notes
    -----
    You must write at least one of the two files. If you are only competing
    in one track, write only that track's file.
    """
    global MODEL

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = Path(test_data_dir)

    # ── Example: Track 1 predictions ──────────────────────────────────────
    # import pandas as pd
    # import torch
    #
    # model = MODEL["model"]
    # rm = MODEL["rm"]
    # test_pairs = pd.read_csv(data_path / "test_pairs.csv")
    #
    # subject_to_idx = {s: i for i, s in enumerate(rm.subject_ids)}
    # item_to_idx = {s: i for i, s in enumerate(rm.item_ids)}
    #
    # with torch.no_grad():
    #     prob_matrix = model.predict().cpu().numpy()
    #
    # predictions = []
    # for _, row in test_pairs.iterrows():
    #     si = subject_to_idx[str(row["model_id"])]
    #     ii = item_to_idx[str(row["item_id"])]
    #     predictions.append({
    #         "model_id": row["model_id"],
    #         "item_id": row["item_id"],
    #         "predicted_probability": float(prob_matrix[si, ii]),
    #     })
    #
    # pd.DataFrame(predictions).to_csv(output_path / "track1_predictions.csv", index=False)

    # ── Example: Track 2 scores ───────────────────────────────────────────
    # import torch
    #
    # with torch.no_grad():
    #     abilities = model.ability.cpu().numpy()
    #
    # pd.DataFrame({
    #     "model_id": rm.subject_ids,
    #     "ability_score": abilities,
    # }).to_csv(output_path / "track2_scores.csv", index=False)

    # TODO: Replace this placeholder with your prediction logic.
    print("[model.py] predict() called — implement your prediction logic here.")
    print(f"  test_data_dir: {test_data_dir}")
    print(f"  output_dir: {output_dir}")
