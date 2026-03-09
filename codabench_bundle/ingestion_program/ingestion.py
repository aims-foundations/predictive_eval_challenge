#!/usr/bin/env python3
"""Ingestion program for the Predictive AI Evaluation Challenge.

This script handles code submissions for amortized prediction. It loads the
participant's model.py, calls their train() and predict() functions, and
writes predictions to the output directory for the scoring program.

Key design: participants train on items whose content they can see. At test
time, their *code* receives hidden test item content (text, embeddings) so
their featurizer can process it, but the participant never directly sees
the test items — enforced by the air-gapped Docker container.

Usage (called by Codabench via metadata.yaml):
    python3 ingestion.py <input_data_dir> <output_dir> <program_dir> <submission_dir>

Directory layout:
    <input_data_dir>/
        train/                      # Training data (participant can inspect)
            train_responses.csv     # Partial response matrix (model_id, item_id, response)
            train_items.csv         # Item content: item_id, item_text, benchmark, category, ...
            model_metadata.csv      # model_id, param_count, release_date, org, ...
            item_embeddings.npy     # Pre-computed embeddings for training items (optional)
        test/                       # Hidden test data (only code sees this)
            test_items.csv          # item_id, item_text, benchmark, category, ...
            test_pairs.csv          # (model_id, item_id) pairs to predict
            test_item_embeddings.npy  # Pre-computed embeddings for test items (optional)
    <output_dir>/                   # Where predictions are written (passed to scoring)
        track1_predictions.csv
    <submission_dir>/               # Participant's code (unzipped)
        model.py                    # Must implement train() and predict()
        (featurizer weights, configs, helper modules, etc.)
"""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
import time
import traceback
from pathlib import Path


def log(msg: str) -> None:
    """Print a timestamped log message."""
    elapsed = time.time() - START_TIME
    print(f"[INGESTION {elapsed:7.1f}s] {msg}", flush=True)


START_TIME = time.time()

# Time limit in seconds (can be overridden by environment variable)
TIME_LIMIT = int(os.environ.get("INGESTION_TIME_LIMIT", 1800))


class TimeLimitExceeded(Exception):
    """Raised when participant code exceeds the time limit."""
    pass


def timeout_handler(signum, frame):
    raise TimeLimitExceeded(f"Participant code exceeded the time limit of {TIME_LIMIT} seconds.")


def load_participant_module(submission_dir: Path):
    """Dynamically import the participant's model.py."""
    model_path = submission_dir / "model.py"

    if not model_path.exists():
        raise FileNotFoundError(
            f"model.py not found in submission directory: {submission_dir}. "
            f"Contents: {sorted(os.listdir(submission_dir))}"
        )

    # Add submission directory to sys.path so participant can import their
    # own helper modules / featurizer code.
    sys.path.insert(0, str(submission_dir))

    spec = importlib.util.spec_from_file_location("participant_model", str(model_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec from {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def validate_module(module) -> None:
    """Check that the participant module implements the required interface."""
    if not hasattr(module, "train"):
        raise AttributeError(
            "model.py must define a train(train_dir) function. "
            "See the starting kit for an example."
        )
    if not hasattr(module, "predict"):
        raise AttributeError(
            "model.py must define a predict(train_dir, test_dir, output_dir) function. "
            "See the starting kit for an example."
        )
    if not callable(module.train):
        raise TypeError("model.train must be callable.")
    if not callable(module.predict):
        raise TypeError("model.predict must be callable.")


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python3 ingestion.py <input_data_dir> <output_dir> "
            "<program_dir> <submission_dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    program_dir = Path(sys.argv[3])
    submission_dir = Path(sys.argv[4])

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = input_data_dir / "train"
    test_dir = input_data_dir / "test"

    log(f"Input data directory: {input_data_dir}")
    log(f"  Train directory: {train_dir}")
    log(f"  Test directory: {test_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Submission directory: {submission_dir}")
    log(f"Time limit: {TIME_LIMIT}s")

    # List available files
    for label, d in [("Train", train_dir), ("Test", test_dir), ("Submission", submission_dir)]:
        if d.exists():
            log(f"{label} files: {sorted(os.listdir(d))}")
        else:
            log(f"WARNING: {label} directory does not exist: {d}")

    # Set time limit via signal (Unix only)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIME_LIMIT)
        log(f"Timer set: {TIME_LIMIT}s")
    except (AttributeError, OSError):
        log("WARNING: Could not set SIGALRM timer (non-Unix platform?).")

    try:
        # Step 1: Load participant module
        log("Loading participant model.py ...")
        module = load_participant_module(submission_dir)
        validate_module(module)
        log("model.py loaded and validated successfully.")

        # Step 2: Train — participant sees training items (content visible)
        log("Calling participant's train(train_dir) ...")
        train_start = time.time()
        module.train(str(train_dir))
        train_elapsed = time.time() - train_start
        log(f"train() completed in {train_elapsed:.1f}s")

        # Step 3: Predict — code receives hidden test items
        # The participant's featurizer processes test item content here.
        # The participant never directly sees the test items because the
        # container is air-gapped (no network access).
        log("Calling participant's predict(train_dir, test_dir, output_dir) ...")
        predict_start = time.time()
        module.predict(str(train_dir), str(test_dir), str(output_dir))
        predict_elapsed = time.time() - predict_start
        log(f"predict() completed in {predict_elapsed:.1f}s")

        # Step 4: Verify outputs exist
        log(f"Output directory contents: {sorted(os.listdir(output_dir))}")

        t1_exists = (output_dir / "track1_predictions.csv").exists()
        if not t1_exists:
            log(
                "WARNING: track1_predictions.csv not found in output directory. "
                "The scoring program will report an error."
            )
        else:
            log("track1_predictions.csv found.")

    except TimeLimitExceeded as e:
        log(f"TIMEOUT: {e}")
        sys.exit(2)
    except Exception as e:
        log(f"ERROR: Participant code raised an exception: {e}")
        log(traceback.format_exc())
        sys.exit(1)
    finally:
        # Cancel the alarm
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass

    total_elapsed = time.time() - START_TIME
    log(f"Ingestion complete. Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
