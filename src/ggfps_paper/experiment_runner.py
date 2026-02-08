"""Experiment runner for repeated GGFPS sampling/evaluation."""

from pathlib import Path

import numpy as np

from .training_set_optimization import TrainingSetOptimizer


def run_ggfps_experiment(
    x,
    y,
    gradient_norms,
    labeled_size,
    training_set_sizes,
    gradient_biases,
    bounds,
    num_folds=5,
    n_bootstraps=1,
    schedule="ascending",
    initializer="probabilistic",
    random_state=None,
    kernel_fn=None,
    distance_cache_dir="artifacts/distance_matrices",
):
    """Run bootstrapped labeled/training-set experiments.

    Behavior for distance handling:
    - Single-B (`len(gradient_biases) == 1`): on-the-fly GGFPS distances.
    - Multi-B (`len(gradient_biases) > 1`): build one labeled distance matrix per bootstrap,
      reuse it across all training set sizes, and save it to `distance_cache_dir`.
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    gradient_norms = np.asarray(gradient_norms, dtype=float)

    if labeled_size < 2 or labeled_size >= y.size:
        raise ValueError("labeled_size must be within [2, n_samples - 1].")

    training_set_sizes = list(training_set_sizes)
    gradient_biases = list(gradient_biases)
    if len(training_set_sizes) == 0:
        raise ValueError("training_set_sizes cannot be empty.")
    if len(gradient_biases) == 0:
        raise ValueError("gradient_biases cannot be empty.")

    multi_b = len(gradient_biases) > 1

    cache_dir_path = None
    if multi_b and distance_cache_dir is not None:
        cache_dir_path = Path(distance_cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)
    all_indices = np.arange(y.size)
    records = []

    for bootstrap_index in range(n_bootstraps):
        labeled_indices = rng.choice(all_indices, size=labeled_size, replace=False)
        test_indices = np.setdiff1d(all_indices, labeled_indices, assume_unique=False)

        distance_cache_path = None
        if multi_b and cache_dir_path is not None:
            distance_cache_path = cache_dir_path / f"labeled_{labeled_size}_bootstrap_{bootstrap_index}.npy"

        for training_set_size in training_set_sizes:
            optimizer = TrainingSetOptimizer(
                x=x,
                y=y,
                gradient_norms=gradient_norms,
                labeled_indices=labeled_indices,
                training_set_size=training_set_size,
                test_indices=test_indices,
                bounds=bounds,
                gradient_biases=gradient_biases,
                kernel_fn=kernel_fn,
                schedule=schedule,
                initializer=initializer,
                random_state=int(rng.integers(1, 1_000_000_000)),
                distance_cache_path=distance_cache_path,
            )

            result = optimizer.evaluate(num_folds=num_folds)
            result["bootstrap"] = bootstrap_index
            result["labeled_size"] = labeled_size
            if distance_cache_path is not None:
                result["distance_cache_path"] = str(distance_cache_path)
            records.append(result)

    return records
