#!/usr/bin/env python3
"""Full experiment-style demo on the 2D Styblinski-Tang benchmark.

This script supports:
- bootstraps,
- multi-B sweeps,
- automatic distance-matrix caching for multi-B mode.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ggfps_paper.datasets import StyblinskiTangSystem
from ggfps_paper.experiment_runner import run_ggfps_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run full GGFPS demo on Styblinski-Tang data.")
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--labeled-size", type=int, default=1000)
    parser.add_argument("--training-set-size", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.5, help="Single B value (default path).")
    parser.add_argument("--betas", type=float, nargs="*", default=None, help="Optional multi-B sweep values.")
    parser.add_argument("--schedule", choices=["ascending", "descending", "alternating"], default="ascending")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--distance-cache-dir",
        type=Path,
        default=Path("artifacts/distance_matrices"),
        help="Used when multiple B values are provided.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def main():
    args = parse_args()

    st_system = StyblinskiTangSystem(n_dims=2, n_points=args.n_points, seed=args.seed)

    betas = [args.beta] if not args.betas else [float(value) for value in args.betas]

    bounds = {
        "width_bounds": (0.2, 5.0),
        "reg_bounds": (1e-10, 1e-6),
        "search_density": 12,
    }

    records = run_ggfps_experiment(
        x=st_system.x,
        y=st_system.y,
        gradient_norms=st_system.gradient_norms,
        labeled_size=args.labeled_size,
        training_set_sizes=[args.training_set_size],
        gradient_biases=betas,
        bounds=bounds,
        num_folds=args.num_folds,
        n_bootstraps=args.bootstraps,
        schedule=args.schedule,
        initializer="probabilistic",
        random_state=args.seed,
        distance_cache_dir=args.distance_cache_dir,
    )

    test_maes = np.asarray([record["test_mae"] for record in records], dtype=float)
    val_maes = np.asarray([record["val_mae"] for record in records], dtype=float)
    modes = [record.get("sampling_mode", "unknown") for record in records]

    summary = {
        "n_records": len(records),
        "test_mae_mean": float(np.mean(test_maes)),
        "test_mae_std": float(np.std(test_maes)),
        "val_mae_mean": float(np.mean(val_maes)),
        "val_mae_std": float(np.std(val_maes)),
        "schedule": args.schedule,
        "betas": betas,
        "training_set_size": args.training_set_size,
        "sampling_modes": sorted(set(modes)),
    }

    print(json.dumps(summary, indent=2))

    if args.out_json is not None:
        payload = {
            "summary": summary,
            "records": _json_ready(records),
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote detailed results to {args.out_json}")


if __name__ == "__main__":
    main()
