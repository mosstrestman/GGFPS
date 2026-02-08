"""Analytical benchmark datasets used in the GGFPS paper."""

import numpy as np


class StyblinskiTangSystem:
    """Uniformly sampled Styblinski-Tang system with analytic gradients."""

    def __init__(self, n_dims, n_points, domain_low=-4.0, domain_high=4.0, seed=12345):
        self.n_dims = int(n_dims)
        self.n_points = int(n_points)
        self.domain_low = float(domain_low)
        self.domain_high = float(domain_high)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.x = rng.uniform(self.domain_low, self.domain_high, size=(self.n_points, self.n_dims))
        self.y = self.styblinski_tang(self.x)
        gradients = self.styblinski_tang_grad(self.x)
        self.gradient_norms = np.linalg.norm(gradients, axis=1)

    @staticmethod
    def styblinski_tang(x):
        return np.sum((x**4 - 16.0 * x**2 + 5.0 * x) / 2.0, axis=-1)

    @staticmethod
    def styblinski_tang_grad(x):
        return (4.0 * x**3 - 32.0 * x + 5.0) / 2.0


def random_labeled_unlabeled_split(n_samples, labeled_size, random_state=None):
    """Create a random labeled/unlabeled split of index space [0, n_samples)."""
    n_samples = int(n_samples)
    labeled_size = int(labeled_size)

    if labeled_size < 1 or labeled_size >= n_samples:
        raise ValueError("labeled_size must be within [1, n_samples - 1].")

    rng = np.random.default_rng(random_state)
    all_indices = np.arange(n_samples)
    labeled_indices = rng.choice(all_indices, size=labeled_size, replace=False)
    unlabeled_indices = np.setdiff1d(all_indices, labeled_indices, assume_unique=False)
    return labeled_indices, unlabeled_indices
