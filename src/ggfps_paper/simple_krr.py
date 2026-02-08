"""Minimal KRR utilities for quick experiments."""

import numpy as np

from .kernels import rbf_kernel


def _fit_and_predict(x_train, y_train, x_eval, width, reg, kernel_fn):
    train_kernel = kernel_fn(x_train, x_train, width)
    eval_kernel = kernel_fn(x_eval, x_train, width)

    train_kernel = np.array(train_kernel, copy=True)
    train_kernel[np.diag_indices_from(train_kernel)] += float(reg)

    alpha = np.linalg.solve(train_kernel, y_train)
    return eval_kernel @ alpha


def simple_kfold_krr(
    x,
    y,
    train_val_indices,
    test_indices,
    width_values,
    reg_values,
    num_folds=5,
    kernel_fn=None,
    random_state=None,
):
    """Minimal k-fold hyperparameter search + final test evaluation."""
    kernel_fn = rbf_kernel if kernel_fn is None else kernel_fn

    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    train_val_indices = np.asarray(train_val_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if num_folds > train_val_indices.size:
        raise ValueError("num_folds cannot exceed train_val_indices length.")

    width_values = np.asarray(list(width_values), dtype=float)
    reg_values = np.asarray(list(reg_values), dtype=float)
    if width_values.size == 0 or reg_values.size == 0:
        raise ValueError("width_values and reg_values must be non-empty.")

    rng = np.random.default_rng(random_state)
    shuffled = np.array(train_val_indices, copy=True)
    rng.shuffle(shuffled)
    val_folds = list(np.array_split(shuffled, num_folds))
    train_folds = [
        np.concatenate(val_folds[:fold_index] + val_folds[fold_index + 1 :])
        for fold_index in range(num_folds)
    ]

    best_score = np.inf
    best_width = float(width_values[0])
    best_reg = float(reg_values[0])

    for width in width_values:
        for reg in reg_values:
            fold_maes = []
            for train_indices, val_indices in zip(train_folds, val_folds, strict=True):
                pred_val = _fit_and_predict(
                    x[train_indices],
                    y[train_indices],
                    x[val_indices],
                    width=float(width),
                    reg=float(reg),
                    kernel_fn=kernel_fn,
                )
                mae = float(np.mean(np.abs(y[val_indices] - pred_val)))
                fold_maes.append(mae)

            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_score:
                best_score = mean_mae
                best_width = float(width)
                best_reg = float(reg)

    pred_test = _fit_and_predict(
        x[train_val_indices],
        y[train_val_indices],
        x[test_indices],
        width=best_width,
        reg=best_reg,
        kernel_fn=kernel_fn,
    )
    test_mae = float(np.mean(np.abs(y[test_indices] - pred_test)))

    return {
        "val_mae": best_score,
        "test_mae": test_mae,
        "opt_width": best_width,
        "opt_reg": best_reg,
        "pred_test": pred_test,
    }
