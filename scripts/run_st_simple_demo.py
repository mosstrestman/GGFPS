#!/usr/bin/env python3
"""Minimal teaching demo: one training set, one B value, on-the-fly GGFPS."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ggfps_paper.datasets import StyblinskiTangSystem, random_labeled_unlabeled_split
from ggfps_paper.ggfps_sampling import GGFPSampler
from ggfps_paper.simple_krr import simple_kfold_krr


def parse_args():
    parser = argparse.ArgumentParser(description="Run simplified GGFPS demo on Styblinski-Tang data.")
    parser.add_argument("--schedule", choices=["ascending", "descending", "alternating"], default="ascending")
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--labeled-size", type=int, default=1000)
    parser.add_argument("--training-set-size", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()

    st_system = StyblinskiTangSystem(n_dims=2, n_points=args.n_points, seed=args.seed)
    labeled_indices, test_indices = random_labeled_unlabeled_split(
        n_samples=args.n_points,
        labeled_size=args.labeled_size,
        random_state=args.seed,
    )

    sampler = GGFPSampler.on_the_fly(schedule=args.schedule)

    selected_relative = sampler.sample_for_beta(
        points=st_system.x[labeled_indices],
        gradients=st_system.gradient_norms[labeled_indices],
        n_select=args.training_set_size,
        beta=float(args.beta),
        random_state=args.seed,
    )

    training_indices = labeled_indices[selected_relative]

    result = simple_kfold_krr(
        x=st_system.x,
        y=st_system.y,
        train_val_indices=training_indices,
        test_indices=test_indices,
        width_values=np.logspace(np.log10(0.2), np.log10(5.0), 12),
        reg_values=np.logspace(-10, -6, 5),
        num_folds=args.num_folds,
        random_state=args.seed,
    )

    summary = {
        "schedule": args.schedule,
        "beta": args.beta,
        "training_set_size": args.training_set_size,
        "sampling_mode": sampler.mode,
        "test_mae": result["test_mae"],
        "val_mae": result["val_mae"],
        "opt_width": result["opt_width"],
        "opt_reg": result["opt_reg"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
