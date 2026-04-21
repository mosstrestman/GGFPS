"""Installable command-line interface for GGFPS demos and experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__
from .datasets import StyblinskiTangSystem, random_labeled_unlabeled_split
from .ggfps_sampling import GGFPSampler
from .simple_krr import simple_kfold_krr
from .training_set_optimization import TrainingSetOptimizer


def _configure_simple_demo_parser(parser):
    parser.description = "Run a minimal GGFPS demo on the Styblinski-Tang benchmark."
    parser.add_argument("--schedule", choices=["ascending", "descending", "alternating"], default="ascending")
    parser.add_argument("--n-points", type=int, default=2000)
    parser.add_argument("--labeled-size", type=int, default=1000)
    parser.add_argument("--training-set-size", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def _configure_demo_parser(parser):
    parser.description = "Run the full GGFPS experiment workflow on the Styblinski-Tang benchmark."
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
    return parser


def build_simple_demo_parser():
    parser = argparse.ArgumentParser(prog="ggfps-simple-demo")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return _configure_simple_demo_parser(parser)


def build_demo_parser():
    parser = argparse.ArgumentParser(prog="ggfps-demo")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return _configure_demo_parser(parser)


def build_root_parser():
    parser = argparse.ArgumentParser(
        prog="ggfps",
        description="Installable GGFPS command-line interface for demos and experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    simple_demo_parser = _configure_simple_demo_parser(
        subparsers.add_parser("simple-demo", help="Run the minimal teaching demo.")
    )
    simple_demo_parser.set_defaults(handler=_run_simple_demo_command)

    demo_parser = _configure_demo_parser(
        subparsers.add_parser("demo", help="Run the full experiment workflow.")
    )
    demo_parser.set_defaults(handler=_run_demo_command)

    return parser


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def run_simple_demo(args):
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

    return {
        "schedule": args.schedule,
        "beta": args.beta,
        "training_set_size": args.training_set_size,
        "sampling_mode": sampler.mode,
        "test_mae": result["test_mae"],
        "val_mae": result["val_mae"],
        "opt_width": result["opt_width"],
        "opt_reg": result["opt_reg"],
    }


def run_demo(args):
    st_system = StyblinskiTangSystem(n_dims=2, n_points=args.n_points, seed=args.seed)
    betas = [args.beta] if not args.betas else [float(value) for value in args.betas]

    bounds = {
        "width_bounds": (0.2, 5.0),
        "reg_bounds": (1e-10, 1e-6),
        "search_density": 12,
    }

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(st_system.y.size)

    multi_b = len(betas) > 1
    if multi_b:
        args.distance_cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for bootstrap_index in range(args.bootstraps):
        labeled_indices = rng.choice(all_indices, size=args.labeled_size, replace=False)
        test_indices = np.setdiff1d(all_indices, labeled_indices, assume_unique=False)

        distance_cache_path = None
        if multi_b:
            distance_cache_path = (
                args.distance_cache_dir / f"labeled_{args.labeled_size}_bootstrap_{bootstrap_index}.npy"
            )

        optimizer = TrainingSetOptimizer(
            x=st_system.x,
            y=st_system.y,
            gradient_norms=st_system.gradient_norms,
            labeled_indices=labeled_indices,
            training_set_size=args.training_set_size,
            test_indices=test_indices,
            bounds=bounds,
            gradient_biases=betas,
            schedule=args.schedule,
            random_state=int(rng.integers(1, 1_000_000_000)),
            distance_cache_path=distance_cache_path,
        )
        result = optimizer.evaluate(num_folds=args.num_folds)
        result["bootstrap"] = bootstrap_index
        result["labeled_size"] = args.labeled_size
        if distance_cache_path is not None:
            result["distance_cache_path"] = str(distance_cache_path)
        records.append(result)

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

    payload = {
        "summary": summary,
        "records": _json_ready(records),
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def _run_simple_demo_command(args):
    print(json.dumps(run_simple_demo(args), indent=2))
    return 0


def _run_demo_command(args):
    payload = run_demo(args)
    print(json.dumps(payload["summary"], indent=2))
    if args.out_json is not None:
        print(f"Wrote detailed results to {args.out_json}")
    return 0


def simple_demo_main(argv=None):
    parser = build_simple_demo_parser()
    args = parser.parse_args(argv)
    return _run_simple_demo_command(args)


def demo_main(argv=None):
    parser = build_demo_parser()
    args = parser.parse_args(argv)
    return _run_demo_command(args)


def main(argv=None):
    parser = build_root_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    return args.handler(args)
