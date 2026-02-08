# GGFPS Paper Reference Repo

Standalone, readable implementation of **Gradient Guided Furthest Point Sampling (GGFPS)** and KRR evaluation.

This repository is focused on two goals:
1. Make the GGFPS method easy to understand from code.
2. Provide scripts that are immediately usable for experiments.

## Method Summary

Let:
- `X = {x_i}` be descriptor vectors,
- `g_i = ||F_i||_2` be gradient/force norms,
- `T` be the selected training set,
- `A` be the remaining candidates,
- `d_j = min_{i in T} ||x_j - x_i||_2`.

### 1. Probabilistic initialization
The first selected point is sampled with

`p_j = (g_j + eps)^{beta_0} / sum_l (g_l + eps)^{beta_0}`

where `eps` avoids zero-gradient issues.

### 2. Selection score at step `k`
For each candidate `j in A`, compute

`s_j = (g_j + eps)^{beta_k} * d_j`

and select

`j* = argmax_j s_j`.

Then update the minimum-distance vector:

`d_j <- min(d_j, ||x_j - x_{j*}||_2)`.

Repeat until the target training set size is reached.

### 3. Schedule meaning
- `beta_k > 0`: biases toward high-gradient regions.
- `beta_k < 0`: biases toward low-gradient regions.
- `beta_k = 0`: recovers standard FPS behavior.

This repo provides ascending and descending schedules directly, and switch/bounce options in the sampler API.

## Distance Strategy

This implementation follows the workflow you requested:

- **Single-B runs**: use **on-the-fly distances** (default).
- **Multi-B runs**: build one labeled distance matrix and reuse/cache it.

That logic is implemented in `src/ggfps_paper/training_set_optimization.py`.

## Why Two Demo Scripts?

Both are intentional and target different use cases:

- `scripts/run_st_simple_demo.py`
  - Minimal learning script.
  - Single split, single B, on-the-fly distances.
  - Best for understanding and quick sanity checks.

- `scripts/run_st_demo.py`
  - Experiment script.
  - Bootstraps, multi-B sweeps, and matrix caching for repeated runs.
  - Best for paper-style evaluations.

## Installation

```bash
cd ggfps_paper_repo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### Minimal demo

```bash
python3 scripts/run_st_simple_demo.py \
  --schedule ascending \
  --n-points 2000 \
  --labeled-size 1000 \
  --training-set-size 100 \
  --beta 1.5
```

### Full experiment demo (single B, on-the-fly)

```bash
python3 scripts/run_st_demo.py \
  --schedule ascending \
  --labeled-size 1000 \
  --training-set-size 100 \
  --beta 1.5 \
  --bootstraps 3
```

### Full experiment demo (multi-B, cached distance matrix)

```bash
python3 scripts/run_st_demo.py \
  --schedule ascending \
  --labeled-size 1000 \
  --training-set-size 100 \
  --betas 0.5 1.0 1.5 \
  --bootstraps 3
```

## Tests

```bash
python3 -m unittest discover -s tests
```

## Project Layout

```text
ggfps_paper_repo/
  assets/figures/
  scripts/
    run_st_demo.py
    run_st_simple_demo.py
  src/ggfps_paper/
    datasets.py
    experiment_runner.py
    ggfps_sampling.py
    kernels.py
    krr_cv.py
    simple_ggfps.py
    simple_krr.py
    training_set_optimization.py
  tests/
```

## Main API Entry Points

- `ggfps_on_the_fly(...)` and `ggfps_from_distance_matrix(...)` in `src/ggfps_paper/ggfps_sampling.py`
- `TrainingSetOptimizer` in `src/ggfps_paper/training_set_optimization.py`
- `run_ggfps_experiment(...)` in `src/ggfps_paper/experiment_runner.py`
