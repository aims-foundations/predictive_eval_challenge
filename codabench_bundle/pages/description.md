# Predictive AI Evaluation Challenge

## Overview

As AI systems proliferate, keeping evaluation infrastructure up to date is
becoming a bottleneck. New models appear faster than new benchmarks, and
benchmark contamination undermines the reliability of published scores.

This challenge asks: **Can we predict how AI models will perform on unseen
benchmarks, and can we produce reliable ability scores even when some benchmark
results are contaminated?**

We frame AI evaluation as a **measurement science** problem.  Each model is a
"subject" and each benchmark item is a "test item."  The resulting binary
response matrix (pass/fail per model-item pair) is sparse and noisy — exactly
the setting where Item Response Theory (IRT) and related psychometric models
excel.

---

## Tracks

### Track 1 — Response Prediction

**Task:** Matrix completion on a partially observed binary response matrix
(models x items).

You are given a response matrix where some entries are observed (a model was
evaluated on an item and either passed or failed) and many entries are missing.
Your goal is to predict the probability that each unobserved model-item pair
will receive a correct response.

- **Input:** Partially observed binary response matrix + metadata (model size,
  release date, item embeddings, benchmark category).
- **Output:** CSV with columns `model_id`, `item_id`, `predicted_probability`.
- **Primary metric:** AUC-ROC
- **Secondary metric:** Log-loss

### Track 2 — Robust Scoring

**Task:** Produce ability scores that are robust to benchmark contamination.

You are given a response matrix that contains both genuine responses and
artificially inflated responses (simulating benchmark contamination / data
leakage). Your goal is to produce an ability score for each model that
correlates as closely as possible with the "true" ability measured from clean
data only.

- **Input:** Response matrix with genuine + artificially inflated responses.
- **Output:** CSV with columns `model_id`, `ability_score`.
- **Primary metric:** Kendall's tau (vs. ground-truth abilities)
- **Secondary metric:** Spearman's rho

---

## Baselines

We provide baseline implementations using the
[torch_measure](https://github.com/sttruong/torch_measure) library, which
offers PyTorch-native implementations of:

| Model | Description |
|-------|-------------|
| Rasch (1PL) | Simplest IRT: P = sigmoid(theta - b) |
| 2PL | Adds item discrimination: P = sigmoid(a*(theta - b)) |
| 3PL | Adds guessing parameter |
| Beta-IRT | Beta-distribution parameterization |
| Amortized IRT | Predicts item parameters from embeddings (zero-shot capable) |
| Many-Facet Rasch | Accounts for facets beyond subject and item |
| Logistic Factor Model | Multi-dimensional latent traits |

See the **Starting Kit** for runnable baseline scripts.

---

## Timeline

| Phase | Dates | Submissions/Day |
|-------|-------|-----------------|
| Development | April 1 – July 1, 2026 | 5 |
| Final | July 1 – August 1, 2026 | 2 |

---

## Organizers

This challenge is organized by the AIMS Foundation. For questions, visit the
competition forum or contact `predictive-eval-challenge@example.org`.
