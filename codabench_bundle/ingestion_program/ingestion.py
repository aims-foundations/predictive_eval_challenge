#!/usr/bin/env python3
"""Ingestion program for the Predictive AI Evaluation Challenge.

This script handles **code submissions**. It loads the participant's model.py,
calls their train() and predict() functions, and writes predictions to the
output directory for the scoring program.

Usage (called by Codabench via metadata.yaml):
    python3 ingestion.py <input_data_dir> <output_dir> <program_dir> <submission_dir>

Directory layout:
    <input_data_dir>/   # Competition input data (training + test data)
        train_responses.csv
        contaminated_responses.csv
        model_metadata.csv
        item_metadata.csv
        item_embeddings.npy
        test_pairs.csv
    <output_dir>/       # Where predictions are written (passed to scoring)
        track1_predictions.csv
        track2_scores.csv
    <program_dir>/      # This ingestion program
    <submission_dir>/   # Participant's code (unzipped)
        model.py
        metadata.yaml
        (other participant files)
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


class TimeoutError(Exception):
    """Raised when participant code exceeds the time limit."""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError(f"Participant code exceeded the time limit of {TIME_LIMIT} seconds.")


def load_participant_module(submission_dir: Path):
    """Dynamically import the participant's model.py.

    Parameters
    ----------
    submission_dir : Path
        Directory containing the participant's code submission.

    Returns
    -------
    module
        The loaded Python module.
    """
    model_path = submission_dir / "model.py"

    if not model_path.exists():
        raise FileNotFoundError(
            f"model.py not found in submission directory: {submission_dir}. "
            f"Contents: {sorted(os.listdir(submission_dir))}"
        )

    # Add submission directory to sys.path so participant can import their
    # own helper modules.
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
            "model.py must define a train(data_dir) function. "
            "See the starting kit for an example."
        )
    if not hasattr(module, "predict"):
        raise AttributeError(
            "model.py must define a predict(test_data_dir, output_dir) function. "
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

    log(f"Input data directory: {input_data_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Program directory: {program_dir}")
    log(f"Submission directory: {submission_dir}")
    log(f"Time limit: {TIME_LIMIT}s")

    # List available files
    if input_data_dir.exists():
        log(f"Input data files: {sorted(os.listdir(input_data_dir))}")
    else:
        log(f"WARNING: Input data directory does not exist: {input_data_dir}")

    if submission_dir.exists():
        log(f"Submission files: {sorted(os.listdir(submission_dir))}")
    else:
        log(f"ERROR: Submission directory does not exist: {submission_dir}")
        sys.exit(1)

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

        # Step 2: Train
        log("Calling participant's train() function ...")
        train_start = time.time()
        module.train(str(input_data_dir))
        train_elapsed = time.time() - train_start
        log(f"train() completed in {train_elapsed:.1f}s")

        # Step 3: Predict
        log("Calling participant's predict() function ...")
        predict_start = time.time()
        module.predict(str(input_data_dir), str(output_dir))
        predict_elapsed = time.time() - predict_start
        log(f"predict() completed in {predict_elapsed:.1f}s")

        # Step 4: Verify outputs exist
        log(f"Output directory contents: {sorted(os.listdir(output_dir))}")

        t1_exists = (output_dir / "track1_predictions.csv").exists()
        t2_exists = (output_dir / "track2_scores.csv").exists()

        if not t1_exists and not t2_exists:
            log(
                "WARNING: Neither track1_predictions.csv nor track2_scores.csv "
                "found in output directory. The scoring program will report an error."
            )
        else:
            if t1_exists:
                log("track1_predictions.csv found.")
            if t2_exists:
                log("track2_scores.csv found.")

    except TimeoutError as e:
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
