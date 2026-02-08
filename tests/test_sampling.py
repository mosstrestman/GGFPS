import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggfps_paper.ggfps_sampling import (
    ggfps_from_distance_matrix,
    ggfps_sweep,
    ggfps_sweep_descending,
    select_with_strategy,
)
from ggfps_paper.kernels import pairwise_l2_distance


class SamplingTests(unittest.TestCase):
    def test_sampling_returns_unique_indices_on_the_fly(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(50, 3))
        gradients = np.linalg.norm(x, axis=1)

        indices = ggfps_sweep(points=x, gradients=gradients, n_select=12, random_state=7)
        self.assertEqual(indices.shape, (12,))
        self.assertEqual(np.unique(indices).size, 12)

    def test_descending_sampling_returns_unique_indices(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(60, 2))
        gradients = np.linalg.norm(x, axis=1)

        indices = ggfps_sweep_descending(points=x, gradients=gradients, n_select=15, random_state=8)
        self.assertEqual(indices.shape, (15,))
        self.assertEqual(np.unique(indices).size, 15)

    def test_alternating_sampling_returns_unique_indices(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(55, 2))
        gradients = np.linalg.norm(x, axis=1)

        indices = select_with_strategy(
            points=x,
            gradients=gradients,
            n_select=14,
            beta_start=-1.0,
            beta_end=1.0,
            schedule="alternating",
            initializer="probabilistic",
            random_state=10,
        )
        self.assertEqual(indices.shape, (14,))
        self.assertEqual(np.unique(indices).size, 14)

    def test_matrix_mode_runs(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(40, 4))
        gradients = np.linalg.norm(x, axis=1)
        distance_matrix = pairwise_l2_distance(x, x)

        indices = ggfps_from_distance_matrix(
            gradients=gradients,
            distance_matrix=distance_matrix,
            n_select=10,
            beta_start=-1.0,
            beta_end=1.0,
            schedule="ascending",
            random_state=9,
        )
        self.assertEqual(indices.shape, (10,))
        self.assertEqual(np.unique(indices).size, 10)


if __name__ == "__main__":
    unittest.main()
