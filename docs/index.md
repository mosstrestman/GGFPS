# GGFPS

**Gradient Guided Furthest Point Sampling (GGFPS)** is a Python implementation of a training-set selection method that combines:
- geometric coverage in descriptor space,
- gradient or force-norm information,
- readable experiment code for kernel ridge regression workflows.

This project is aimed at readers who want a compact, inspectable reference implementation rather than a large framework.

## Why this repository exists

The repo has two jobs:
- explain the GGFPS algorithm in code that is small enough to read end to end,
- make it easy to reproduce simple sampling and KRR experiments from an installed package.

## What you can do here

- install the package with `python -m pip install -e .`,
- run demos with `ggfps simple-demo` and `ggfps demo`,
- import `GGFPSampler` and related helpers into notebooks or scripts,
- publish these docs as a small searchable website with MkDocs or GitHub Pages.

## Key terms

If you are searching for this project, the main phrases are:
- Gradient Guided Furthest Point Sampling,
- GGFPS,
- furthest point sampling,
- training set selection,
- kernel ridge regression,
- scientific Python reference implementation.

## Next step

Go to [Getting Started](getting-started.md) for installation and runnable examples, or [API Overview](api.md) for the package layout.
