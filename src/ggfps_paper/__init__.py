"""Reference implementation of GGFPS and KRR utilities for the GGFPS paper."""

from importlib.metadata import PackageNotFoundError, version

from .datasets import StyblinskiTangSystem, random_labeled_unlabeled_split
from .ggfps_sampling import GGFPSampler
from .krr_cv import evaluate_krr, rmse, run_cross_validation, tune_krr_hyperparameters
from .simple_krr import simple_kfold_krr
from .training_set_optimization import TrainingSetOptimizer, run_training_set_optimization

try:
    __version__ = version("ggfps-paper")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "GGFPSampler",
    "StyblinskiTangSystem",
    "TrainingSetOptimizer",
    "__version__",
    "evaluate_krr",
    "random_labeled_unlabeled_split",
    "rmse",
    "run_cross_validation",
    "run_training_set_optimization",
    "simple_kfold_krr",
    "tune_krr_hyperparameters",
]
