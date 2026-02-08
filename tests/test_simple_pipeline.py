import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ggfps_paper.datasets import StyblinskiTangSystem, random_labeled_unlabeled_split
from ggfps_paper.simple_ggfps import ascending_ggfps
from ggfps_paper.simple_krr import simple_kfold_krr


class SimplePipelineTests(unittest.TestCase):
    def test_simple_pipeline_runs(self):
        system = StyblinskiTangSystem(n_dims=2, n_points=400, seed=3)
        labeled_indices, test_indices = random_labeled_unlabeled_split(400, 200, random_state=3)

        training_relative = ascending_ggfps(
            points=system.x[labeled_indices],
            gradients=system.gradient_norms[labeled_indices],
            n_select=60,
            beta_max=1.5,
            random_state=3,
        )
        training_indices = labeled_indices[training_relative]

        result = simple_kfold_krr(
            x=system.x,
            y=system.y,
            train_val_indices=training_indices,
            test_indices=test_indices,
            width_values=np.logspace(np.log10(0.2), np.log10(5.0), 8),
            reg_values=np.logspace(-10, -6, 4),
            num_folds=4,
            random_state=3,
        )

        self.assertTrue(np.isfinite(result["test_mae"]))
        self.assertGreaterEqual(result["test_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
