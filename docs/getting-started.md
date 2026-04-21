# Getting Started

## Install

The actual package install command is:

```bash
python -m pip install -e .
```

That tells `pip` to install the package described by the current directory in editable mode.

If you need to clone the repository first:

```bash
git clone https://github.com/mosstrestman/GGFPS.git
cd GGFPS
python -m pip install -e .
```

If you already have a local checkout:

```bash
# from the repository root
python -m pip install -e .
```

If you want an isolated environment, create a virtual environment before installing:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Upgrading `pip` is optional:

```bash
python -m pip install --upgrade pip
```

If you also want the development and docs tooling:

```bash
python -m pip install -e ".[dev,docs]"
```

## Check the CLI

```bash
ggfps --help
python -m ggfps_paper --help
```

## Run the smallest demo

```bash
ggfps simple-demo \
  --n-points 2000 \
  --labeled-size 1000 \
  --training-set-size 100 \
  --beta 1.5
```

This prints a small JSON summary with the chosen schedule, sampling mode, MAE, and tuned hyperparameters.

## Run the experiment workflow

```bash
ggfps demo \
  --schedule ascending \
  --labeled-size 1000 \
  --training-set-size 100 \
  --betas 0.5 1.0 1.5 \
  --bootstraps 3 \
  --out-json artifacts/results/demo_summary.json
```

This runs multiple bootstraps, optionally sweeps multiple `B` values, and can save a detailed JSON payload.

## Run tests

```bash
python -m unittest discover -s tests
```

## Build docs

```bash
mkdocs serve
mkdocs build
```

Publishing the built site gives search engines a clean HTML version of the documentation to index.
