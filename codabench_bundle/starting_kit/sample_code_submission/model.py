"""Sample code submission for the Predictive AI Evaluation Challenge.

Participants must implement two functions:
    train(train_dir)                         — fit your model on training data
    predict(train_dir, test_dir, output_dir) — predict on hidden test items

Key concept — amortized prediction:
    During training, you see item content (text, category, embeddings).
    During prediction, your CODE receives hidden test item content so your
    featurizer can process it, but YOU never see the test items directly.
    This is enforced by the air-gapped Docker container.

If you use a custom featurizer (e.g., a fine-tuned LLM encoder), include
it in your submission ZIP alongside model.py.

Directory layout provided to your code:

    train_dir/
        train_responses.csv       # model_id, item_id, response (0 or 1)
        train_items.csv           # item_id, item_text, benchmark, category, ...
        model_metadata.csv        # model_id, param_count, release_date, org, ...
        item_embeddings.npy       # (n_train_items, embedding_dim) pre-computed

    test_dir/
        test_items.csv            # item_id, item_text, benchmark, category, ...
        test_pairs.csv            # model_id, item_id — pairs to predict
        test_item_embeddings.npy  # (n_test_items, embedding_dim) pre-computed

Required output:
    output_dir/track1_predictions.csv
        Columns: model_id, item_id, predicted_probability
        One row per (model_id, item_id) pair from test_pairs.csv.
"""

from __future__ import annotations

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# Global state (populated by train, used by predict)
# ──────────────────────────────────────────────────────────────────────────
MODEL = None


def train(train_dir: str) -> None:
    """Train your model on the competition data.

    Parameters
    ----------
    train_dir : str
        Path to training data directory containing:
            - train_responses.csv     (model_id, item_id, response)
            - train_items.csv         (item_id, item_text, benchmark, ...)
            - model_metadata.csv      (model_id, param_count, ...)
            - item_embeddings.npy     (pre-computed item embeddings)
    """
    global MODEL

    train_path = Path(train_dir)

    # ── Example using torch_measure ────────────────────────────────────
    # import pandas as pd
    # import numpy as np
    # import torch
    # from torch_measure.models import AmortizedIRT
    #
    # responses = pd.read_csv(train_path / "train_responses.csv")
    # items = pd.read_csv(train_path / "train_items.csv")
    # embeddings = np.load(train_path / "item_embeddings.npy")
    #
    # # Build response matrix from long-form CSV
    # ...
    #
    # model = AmortizedIRT(
    #     n_subjects=n_models,
    #     n_items=n_items,
    #     embedding_dim=embeddings.shape[1],
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )
    # model.fit(response_matrix, embeddings=torch.tensor(embeddings, dtype=torch.float32))
    # MODEL = {"model": model, "items": items, "embeddings": embeddings}

    # TODO: Replace with your training code.
    print("[model.py] train() called — implement your training logic here.")
    MODEL = {"train_dir": train_dir}


def predict(train_dir: str, test_dir: str, output_dir: str) -> None:
    """Generate predictions for hidden test items.

    Your featurizer processes the test item content here. You have access
    to item text and pre-computed embeddings for the test items.

    Parameters
    ----------
    train_dir : str
        Path to training data (same as in train()).
    test_dir : str
        Path to hidden test data containing:
            - test_items.csv            (item_id, item_text, benchmark, ...)
            - test_pairs.csv            (model_id, item_id pairs to predict)
            - test_item_embeddings.npy  (pre-computed embeddings for test items)
    output_dir : str
        Write track1_predictions.csv here.
    """
    global MODEL

    test_path = Path(test_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Example ────────────────────────────────────────────────────────
    # import pandas as pd
    # import numpy as np
    # import torch
    #
    # test_pairs = pd.read_csv(test_path / "test_pairs.csv")
    # test_items = pd.read_csv(test_path / "test_items.csv")
    # test_embeddings = np.load(test_path / "test_item_embeddings.npy")
    #
    # model = MODEL["model"]
    #
    # # Use the amortized model to predict on new items via their embeddings:
    # # The model maps embeddings -> item parameters, then computes P(correct).
    # with torch.no_grad():
    #     test_emb_tensor = torch.tensor(test_embeddings, dtype=torch.float32)
    #     # predict_new_items() takes embeddings for unseen items and returns
    #     # a probability matrix of shape (n_models, n_test_items)
    #     prob_matrix = model.predict_new_items(test_emb_tensor).cpu().numpy()
    #
    # # Or, if you have a custom featurizer:
    # # test_emb = my_featurizer.encode(test_items["item_text"].tolist())
    # # prob_matrix = model.predict_new_items(test_emb).cpu().numpy()
    #
    # predictions = []
    # for _, row in test_pairs.iterrows():
    #     si = subject_to_idx[str(row["model_id"])]
    #     ii = test_item_to_idx[str(row["item_id"])]
    #     predictions.append({
    #         "model_id": row["model_id"],
    #         "item_id": row["item_id"],
    #         "predicted_probability": float(prob_matrix[si, ii]),
    #     })
    #
    # pd.DataFrame(predictions).to_csv(
    #     output_path / "track1_predictions.csv", index=False
    # )

    # TODO: Replace with your prediction code.
    print("[model.py] predict() called — implement your prediction logic here.")
    print(f"  train_dir: {train_dir}")
    print(f"  test_dir: {test_dir}")
    print(f"  output_dir: {output_dir}")
