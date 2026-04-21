# GGFPS Paper Reference Repo

Installable Python reference implementation of **Gradient Guided Furthest Point Sampling (GGFPS)** for training-set selection and associated kernel ridge regression experiments.

## Installation


```bash
git clone https://github.com/mosstrestman/GGFPS.git
cd GGFPS
python -m pip install -e .
```

## Quick Start

Show the installed CLI:

```bash
ggfps --help
```

Run the minimal teaching demo:

```bash
ggfps simple-demo \
  --schedule ascending \
  --n-points 2000 \
  --labeled-size 1000 \
  --training-set-size 100 \
  --beta 1.5
```

Run the full experiment workflow:

```bash
ggfps demo \
  --schedule ascending \
  --labeled-size 1000 \
  --training-set-size 100 \
  --beta 1.5 \
  --bootstraps 3
```

Run a multi-`B` experiment with cached distance matrices:

```bash
ggfps demo \
  --schedule ascending \
  --labeled-size 1000 \
  --training-set-size 100 \
  --betas 0.5 1.0 1.5 \
  --bootstraps 3
```

Equivalent Python entry point:

```bash
python -m ggfps_paper --help
```

Compatibility wrappers remain available:

```bash
python scripts/run_st_simple_demo.py --help
python scripts/run_st_demo.py --help
```

## What GGFPS Does

<p align="center">
  <img src="assets/figures/intro_plot.png" alt="GGFPS introduction figure" width="65%">
  <img src="assets/figures/st_MAE_LC_pcovfpse.png" alt="ST learning curve" width="22%">
</p>



GGFPS extends Furthest Point Sampling (FPS) by combining:
- geometric spread in descriptor space,
- gradient or force-norm information.

Given descriptor points $x_i$, gradient norms $g_i$, selected training set $T$, and remaining candidates $A$:

FPS distance term:
$d_j = \min_{i \in T} \lVert x_j - x_i \rVert_2$

GGFPS initialization probability:
$p_j = \frac{(g_j + \varepsilon)^{\beta_0}}{\sum_{\ell}(g_{\ell} + \varepsilon)^{\beta_0}}$

GGFPS score at selection step `k`:
$s_j = (g_j + \varepsilon)^{\beta_k} d_j$

Selected next point `j*`:
$j^{*} = \arg\max_{j \in A} s_j$

Distance update after selecting `j*`:
$d_j \leftarrow \min\left(d_j, \lVert x_j - x_{j^{*}} \rVert_2\right)$

Interpretation of $\beta_k$:
- $\beta_k > 0$: prefers high-gradient regions,
- $\beta_k < 0$: prefers low-gradient regions,
- $\beta_k = 0$: recovers FPS behavior.

## Schedules
<img src="assets/figures/methods_plot.png" alt="GGFPS method figure" width="500">



Implemented schedules:
- `ascending`: low-gradient bias to high-gradient bias,
- `descending`: high-gradient bias to low-gradient bias,
- `alternating`: alternates schedule endpoints across steps.


## Distance Strategy

Distance handling follows the intended workflow:
- single-`B` run: on-the-fly distances,
- multi-`B` run: one labeled distance matrix is built, reused, and optionally cached.

Code path: `src/ggfps_paper/training_set_optimization.py`.

## Python API

Import and instantiate explicit sampler variants:

```python
from ggfps_paper import GGFPSampler

# on-the-fly distances
sampler = GGFPSampler.ascending_on_the_fly()

# precomputed distance matrix
sampler = GGFPSampler.descending_with_distance_matrix(distance_matrix)
```

Run single-`B` selection:

```python
indices = sampler.sample_for_beta(
    points=X,              # not used in distance-matrix mode
    gradients=g,
    n_select=100,
    beta=1.5,
    random_state=0,
)
```

Run multi-`B` selection with one sampler instance:

```python
beta_to_indices = sampler.sample_for_betas(
    points=X,
    gradients=g,
    n_select=100,
    beta_values=[0.5, 1.0, 1.5],
    random_state=0,
)
```

## Documentation

Additional documentation lives under `docs/`:
- `docs/index.md`: project overview and search-friendly landing page,
- `docs/getting-started.md`: install and run instructions,
- `docs/api.md`: package layout and import surface.

Preview the docs locally:

```bash
mkdocs serve
```

Build the static site:

```bash
mkdocs build
```

Once the generated site is published, search engines have a clean HTML documentation surface to index.


## Tests

Run the unit test suite:

```bash
python -m unittest discover -s tests
```

## Repo Layout

- `src/ggfps_paper/`: installable package code,
- `scripts/`: thin compatibility wrappers around the packaged CLI,
- `tests/`: unit tests,
- `docs/`: user-facing documentation,
- `assets/figures/`: figures used in the paper and README,
- `.github/workflows/ci.yml`: automated installation and test checks.
