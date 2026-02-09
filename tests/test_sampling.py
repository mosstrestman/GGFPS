import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggfps_paper.ggfps_sampling import GGFPSampler
from ggfps_paper.kernels import pairwise_l2_distance


class SamplingTests(unittest.TestCase):
    def test_on_the_fly_sampling_returns_unique_indices(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(50, 3))
        gradients = np.linalg.norm(x, axis=1)

        sampler = GGFPSampler.ascending_on_the_fly()
        indices = sampler.sample_for_beta(points=x, gradients=gradients, n_select=12, beta=1.5, random_state=7)
        self.assertEqual(indices.shape, (12,))
        self.assertEqual(np.unique(indices).size, 12)
        self.assertEqual(sampler.mode, "on_the_fly")

    def test_descending_on_the_fly_sampling_returns_unique_indices(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(60, 2))
        gradients = np.linalg.norm(x, axis=1)

        sampler = GGFPSampler.descending_on_the_fly()
        indices = sampler.sample_for_beta(points=x, gradients=gradients, n_select=15, beta=1.5, random_state=8)
        self.assertEqual(indices.shape, (15,))
        self.assertEqual(np.unique(indices).size, 15)

    def test_alternating_on_the_fly_sampling_returns_unique_indices(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(55, 2))
        gradients = np.linalg.norm(x, axis=1)

        sampler = GGFPSampler.alternating_on_the_fly()
        indices = sampler.sample_for_beta(points=x, gradients=gradients, n_select=14, beta=1.5, random_state=10)
        self.assertEqual(indices.shape, (14,))
        self.assertEqual(np.unique(indices).size, 14)

    def test_matrix_mode_runs(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(40, 4))
        gradients = np.linalg.norm(x, axis=1)
        distance_matrix = pairwise_l2_distance(x, x)

        sampler = GGFPSampler.ascending_with_distance_matrix(distance_matrix)
        indices = sampler.sample_for_beta(points=None, gradients=gradients, n_select=10, beta=1.5, random_state=9)
        self.assertEqual(indices.shape, (10,))
        self.assertEqual(np.unique(indices).size, 10)
        self.assertEqual(sampler.mode, "distance_matrix")


if __name__ == "__main__":
    unittest.main()
