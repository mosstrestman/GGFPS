import tempfile
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggfps_paper.datasets import StyblinskiTangSystem, random_labeled_unlabeled_split
from ggfps_paper.training_set_optimization import TrainingSetOptimizer


class TrainingSetOptimizerTests(unittest.TestCase):
    def test_optimizer_single_b_uses_on_the_fly(self):
        system = StyblinskiTangSystem(n_dims=2, n_points=500, seed=11)
        labeled_indices, test_indices = random_labeled_unlabeled_split(500, 250, random_state=11)

        optimizer = TrainingSetOptimizer(
            x=system.x,
            y=system.y,
            gradient_norms=system.gradient_norms,
            labeled_indices=labeled_indices,
            training_set_size=80,
            test_indices=test_indices,
            bounds={
                "width_bounds": (0.2, 5.0),
                "reg_bounds": (1e-10, 1e-6),
                "search_density": 8,
            },
            gradient_biases=[1.5],
            schedule="ascending",
            initializer="probabilistic",
            random_state=11,
        )

        result = optimizer.evaluate(num_folds=4)
        self.assertTrue(np.isfinite(result["test_mae"]))
        self.assertGreaterEqual(result["test_mae"], 0.0)
        self.assertEqual(result["sampling_mode"], "on_the_fly")

    def test_optimizer_multi_b_uses_matrix_cache(self):
        system = StyblinskiTangSystem(n_dims=2, n_points=500, seed=12)
        labeled_indices, test_indices = random_labeled_unlabeled_split(500, 250, random_state=12)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "distances.npy"

            optimizer = TrainingSetOptimizer(
                x=system.x,
                y=system.y,
                gradient_norms=system.gradient_norms,
                labeled_indices=labeled_indices,
                training_set_size=80,
                test_indices=test_indices,
                bounds={
                    "width_bounds": (0.2, 5.0),
                    "reg_bounds": (1e-10, 1e-6),
                    "search_density": 8,
                },
                gradient_biases=[0.5, 1.5],
                schedule="ascending",
                initializer="probabilistic",
                random_state=12,
                distance_cache_path=cache_path,
            )

            result = optimizer.evaluate(num_folds=4)
            self.assertTrue(np.isfinite(result["test_mae"]))
            self.assertEqual(result["sampling_mode"], "distance_matrix")
            self.assertTrue(cache_path.exists())


if __name__ == "__main__":
    unittest.main()
