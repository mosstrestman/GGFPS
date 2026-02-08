"""Training-set optimization for GGFPS.

Behavior
--------
- Single-B runs use on-the-fly GGFPS distances.
- Multi-B runs build one labeled distance matrix and reuse/cache it.
"""

from pathlib import Path

import numpy as np

from .ggfps_sampling import select_with_strategy
from .kernels import pairwise_l2_distance
from .krr_cv import evaluate_krr, run_cross_validation


def _flatten_points(x):
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


def _bias_to_bounds(grad_bias):
    if isinstance(grad_bias, tuple):
        if len(grad_bias) != 2:
            raise ValueError("Tuple grad_bias must have exactly two values.")
        return float(grad_bias[0]), float(grad_bias[1])

    grad_bias = float(grad_bias)
    return -grad_bias, grad_bias


class TrainingSetOptimizer:
    """Select and evaluate GGFPS training sets over one or more B values."""

    def __init__(
        self,
        x,
        y,
        gradient_norms,
        labeled_indices,
        training_set_size,
        test_indices,
        bounds,
        gradient_biases,
        kernel_fn=None,
        schedule="ascending",
        initializer="probabilistic",
        random_state=None,
        distance_matrix=None,
        distance_cache_path=None,
    ):
        self.x = np.asarray(x)
        self.y = np.asarray(y, dtype=float)
        self.gradient_norms = np.asarray(gradient_norms, dtype=float)
        self.labeled_indices = np.asarray(labeled_indices, dtype=int)
        self.training_set_size = int(training_set_size)
        self.test_indices = np.asarray(test_indices, dtype=int)
        self.bounds = dict(bounds)
        self.gradient_biases = list(gradient_biases)
        self.kernel_fn = kernel_fn
        self.schedule = schedule
        self.initializer = initializer
        self.random_state = random_state
        self.distance_matrix = None if distance_matrix is None else np.asarray(distance_matrix, dtype=float)
        self.distance_cache_path = distance_cache_path

        if self.training_set_size < 1:
            raise ValueError("training_set_size must be at least 1.")
        if self.training_set_size > self.labeled_indices.size:
            raise ValueError("training_set_size cannot exceed labeled set size.")
        if len(self.gradient_biases) == 0:
            raise ValueError("gradient_biases cannot be empty.")

        self._sampling_points = _flatten_points(self.x)[self.labeled_indices]
        self._sampling_gradients = self.gradient_norms[self.labeled_indices]

        if self.distance_matrix is not None:
            expected_size = self.labeled_indices.size
            if self.distance_matrix.shape != (expected_size, expected_size):
                raise ValueError("distance_matrix shape must match labeled set size.")

    def _get_distance_matrix_if_needed(self):
        if self.distance_matrix is not None:
            return self.distance_matrix

        # Single-B uses on-the-fly distances.
        if len(self.gradient_biases) <= 1:
            return None

        if self.distance_cache_path is not None:
            cache_path = Path(self.distance_cache_path)
            if cache_path.exists():
                self.distance_matrix = np.load(cache_path)
                return self.distance_matrix

        self.distance_matrix = pairwise_l2_distance(self._sampling_points, self._sampling_points)

        if self.distance_cache_path is not None:
            cache_path = Path(self.distance_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.distance_matrix)

        return self.distance_matrix

    def _select_training_indices(self, grad_bias):
        beta_start, beta_end = _bias_to_bounds(grad_bias)
        distance_matrix = self._get_distance_matrix_if_needed()

        relative_indices = select_with_strategy(
            points=self._sampling_points,
            gradients=self._sampling_gradients,
            n_select=self.training_set_size,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=self.schedule,
            initializer=self.initializer,
            random_state=self.random_state,
            distance_matrix=distance_matrix,
        )

        return self.labeled_indices[relative_indices]

    def select_best_training_set(self, num_folds=5):
        best_val_mae = np.inf
        best = None

        for grad_bias in self.gradient_biases:
            training_indices = self._select_training_indices(grad_bias)
            val_indices = np.setdiff1d(self.labeled_indices, training_indices, assume_unique=False)

            if val_indices.size == 0:
                raise ValueError("Validation split is empty. Use training_set_size < labeled_size.")

            val_result = run_cross_validation(
                train_val_indices=training_indices,
                test_indices=val_indices,
                x=self.x,
                y=self.y,
                bounds=self.bounds,
                kernel_fn=self.kernel_fn,
                num_folds=num_folds,
                random_state=self.random_state,
            )

            val_mae = float(val_result["test_mae"])
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best = {
                    "training_indices": training_indices,
                    "grad_bias": grad_bias,
                    "val_mae": val_mae,
                    "width": float(val_result["opt_width"]),
                    "reg": float(val_result["opt_reg"]),
                }

        return best

    def evaluate(self, num_folds=5):
        best = self.select_best_training_set(num_folds=num_folds)

        training_indices = np.asarray(best["training_indices"], dtype=int)
        test_mae, pred_test = evaluate_krr(
            x=self.x,
            y=self.y,
            train_indices=training_indices,
            test_indices=self.test_indices,
            width=best["width"],
            reg=best["reg"],
            kernel_fn=self.kernel_fn,
        )

        mode = "distance_matrix" if self._get_distance_matrix_if_needed() is not None else "on_the_fly"

        return {
            "all_test_errors": pred_test - self.y[self.test_indices],
            "test_mae": float(test_mae),
            "val_mae": float(best["val_mae"]),
            "test_indices": self.test_indices,
            "training_indices": training_indices,
            "opt_grad_bias": best["grad_bias"],
            "opt_reg": float(best["reg"]),
            "opt_width": float(best["width"]),
            "labeled_size": int(self.labeled_indices.size),
            "training_set_size": int(self.training_set_size),
            "bounds": self.bounds,
            "n_dims": int(self.x.shape[1]),
            "schedule": self.schedule,
            "initializer": self.initializer,
            "sampling_mode": mode,
        }

    def sweep_gradient_biases(self, gradient_bias_values, num_folds=5):
        records = []
        old_biases = self.gradient_biases

        try:
            self.gradient_biases = list(gradient_bias_values)
            for grad_bias in self.gradient_biases:
                training_indices = self._select_training_indices(grad_bias)
                result = run_cross_validation(
                    train_val_indices=training_indices,
                    test_indices=self.test_indices,
                    x=self.x,
                    y=self.y,
                    bounds=self.bounds,
                    kernel_fn=self.kernel_fn,
                    num_folds=num_folds,
                    random_state=self.random_state,
                )
                result["grad_bias"] = grad_bias
                records.append(result)
        finally:
            self.gradient_biases = old_biases

        return records


def run_training_set_optimization(
    x,
    y,
    gradient_norms,
    labeled_indices,
    training_set_size,
    test_indices,
    bounds,
    gradient_biases,
    kernel_fn=None,
    schedule="ascending",
    initializer="probabilistic",
    num_folds=5,
    random_state=None,
    distance_matrix=None,
    distance_cache_path=None,
):
    """Function-style wrapper around `TrainingSetOptimizer` for notebook use."""
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
        random_state=random_state,
        distance_matrix=distance_matrix,
        distance_cache_path=distance_cache_path,
    )
    return optimizer.evaluate(num_folds=num_folds)
