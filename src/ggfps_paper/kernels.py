"""Kernel and distance helpers used by the GGFPS experiments."""

import numpy as np


def pairwise_l2_distance(x_left, x_right):
    """Compute pairwise Euclidean distances between two point sets."""
    x_left = np.asarray(x_left, dtype=float)
    x_right = np.asarray(x_right, dtype=float)

    left_norm2 = np.einsum("ij,ij->i", x_left, x_left)
    right_norm2 = np.einsum("ij,ij->i", x_right, x_right)
    dist2 = left_norm2[:, None] + right_norm2[None, :] - 2.0 * (x_left @ x_right.T)
    return np.sqrt(np.maximum(dist2, 0.0))


def rbf_kernel(x_left, x_right, width):
    """Compute an isotropic Gaussian/RBF kernel matrix."""
    width = float(width)
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}.")

    distances = pairwise_l2_distance(x_left, x_right)
    return np.exp(-(distances**2) / (2.0 * width**2))
