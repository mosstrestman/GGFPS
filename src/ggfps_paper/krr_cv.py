"""Small, readable KRR cross-validation helpers."""

import numpy as np

from .kernels import rbf_kernel


def rmse(pred_labels, target_labels):
    pred_labels = np.asarray(pred_labels, dtype=float)
    target_labels = np.asarray(target_labels, dtype=float)
    return float(np.sqrt(np.mean((pred_labels - target_labels) ** 2)))


def _resolve_grid(values_or_bounds, search_density, default_points):
    """Accept either an explicit list or a `(min, max)` tuple."""
    if isinstance(values_or_bounds, tuple):
        if len(values_or_bounds) != 2:
            raise ValueError("Tuple bounds must contain exactly two values.")
        low, high = float(values_or_bounds[0]), float(values_or_bounds[1])
        if low <= 0 or high <= 0:
            raise ValueError("Logspace bounds must be positive.")

        points = default_points if search_density is None else int(search_density)
        if points < 2:
            raise ValueError("search_density must be >= 2.")
        return np.logspace(np.log10(low), np.log10(high), points)

    values = np.asarray(list(values_or_bounds), dtype=float)
    if values.size == 0:
        raise ValueError("Hyperparameter lists cannot be empty.")
    return values


def _solve_krr(train_kernel, train_targets, reg):
    regularized = np.array(train_kernel, copy=True)
    regularized[np.diag_indices_from(regularized)] += float(reg)
    return np.linalg.solve(regularized, train_targets)


def _predict_krr(train_x, train_y, eval_x, width, reg, kernel_fn):
    train_kernel = kernel_fn(train_x, train_x, width)
    eval_kernel = kernel_fn(eval_x, train_x, width)
    alpha = _solve_krr(train_kernel, train_y, reg)
    return eval_kernel @ alpha


def _k_fold_split(indices, num_folds, rng):
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if num_folds > indices.size:
        raise ValueError("num_folds cannot exceed train set size.")

    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)

    val_folds = list(np.array_split(shuffled, num_folds))
    train_folds = [
        np.concatenate(val_folds[:fold] + val_folds[fold + 1 :])
        for fold in range(num_folds)
    ]
    return train_folds, val_folds


def tune_krr_hyperparameters(x, y, train_val_indices, bounds, num_folds=5, kernel_fn=None, random_state=None):
    """Grid-search width/reg via K-fold CV on `train_val_indices`."""
    kernel_fn = rbf_kernel if kernel_fn is None else kernel_fn
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    train_val_indices = np.asarray(train_val_indices, dtype=int)

    width_values = _resolve_grid(
        bounds["width_bounds"],
        bounds.get("search_density"),
        default_points=8,
    )
    reg_values = _resolve_grid(
        bounds["reg_bounds"],
        None,
        default_points=4,
    )

    rng = np.random.default_rng(random_state)
    train_folds, val_folds = _k_fold_split(train_val_indices, num_folds, rng)

    best_score = np.inf
    best_width = float(width_values[0])
    best_reg = float(reg_values[0])

    for width in width_values:
        for reg in reg_values:
            fold_scores = []

            for train_indices, val_indices in zip(train_folds, val_folds, strict=True):
                train_x = x[train_indices]
                val_x = x[val_indices]
                train_y = y[train_indices]
                val_y = y[val_indices]

                val_pred = _predict_krr(train_x, train_y, val_x, width, reg, kernel_fn)
                train_pred = _predict_krr(train_x, train_y, train_x, width, reg, kernel_fn)

                fold_score = 0.5 * (rmse(train_pred, train_y) + rmse(val_pred, val_y))
                fold_scores.append(fold_score)

            mean_score = float(np.mean(fold_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_width = float(width)
                best_reg = float(reg)

    return {
        "val_rmse": best_score,
        "width": best_width,
        "reg": best_reg,
    }


def evaluate_krr(x, y, train_indices, test_indices, width, reg, kernel_fn=None):
    """Fit on `train_indices` and evaluate MAE on `test_indices`."""
    kernel_fn = rbf_kernel if kernel_fn is None else kernel_fn

    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    train_x = x[train_indices]
    test_x = x[test_indices]
    train_y = y[train_indices]
    test_y = y[test_indices]

    test_pred = _predict_krr(train_x, train_y, test_x, width, reg, kernel_fn)
    test_mae = float(np.mean(np.abs(test_y - test_pred)))
    return test_mae, test_pred


def run_cross_validation(
    train_val_indices,
    test_indices,
    x,
    y,
    bounds,
    kernel_fn=None,
    num_folds=5,
    random_state=None,
):
    """Convenience wrapper used across demos and optimization utilities."""
    best = tune_krr_hyperparameters(
        x=x,
        y=y,
        train_val_indices=train_val_indices,
        bounds=bounds,
        num_folds=num_folds,
        kernel_fn=kernel_fn,
        random_state=random_state,
    )

    test_mae, test_pred = evaluate_krr(
        x=x,
        y=y,
        train_indices=train_val_indices,
        test_indices=test_indices,
        width=best["width"],
        reg=best["reg"],
        kernel_fn=kernel_fn,
    )

    return {
        "test_mae": test_mae,
        "train_val_indices": np.asarray(train_val_indices, dtype=int),
        "test_indices": np.asarray(test_indices, dtype=int),
        "all_test_errors": test_pred - np.asarray(y, dtype=float)[np.asarray(test_indices, dtype=int)],
        "opt_width": float(best["width"]),
        "opt_reg": float(best["reg"]),
        "tss": int(np.asarray(train_val_indices).size),
        "bounds": bounds,
        "n_dims": int(np.asarray(x).shape[1]),
        "val_rmse": float(best["val_rmse"]),
    }
