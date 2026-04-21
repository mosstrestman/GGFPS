# API Overview

## Top-level package

The installable package is `ggfps_paper`.

```python
from ggfps_paper import GGFPSampler, StyblinskiTangSystem, TrainingSetOptimizer
```

## Main modules

- `ggfps_paper.ggfps_sampling`
  Implements `GGFPSampler`, including on-the-fly and precomputed-distance modes.
- `ggfps_paper.training_set_optimization`
  Implements `TrainingSetOptimizer` and the higher-level experiment workflow.
- `ggfps_paper.krr_cv`
  Contains KRR tuning, evaluation, and cross-validation helpers.
- `ggfps_paper.simple_krr`
  Contains a smaller KRR path intended for quick experiments and teaching demos.
- `ggfps_paper.datasets`
  Provides the Styblinski-Tang benchmark and a labeled/unlabeled split helper.
- `ggfps_paper.cli`
  Provides the installable `ggfps`, `ggfps-demo`, and `ggfps-simple-demo` commands.

## Example: sampler construction

```python
from ggfps_paper import GGFPSampler

sampler = GGFPSampler.ascending_on_the_fly()
indices = sampler.sample_for_beta(
    points=X,
    gradients=g,
    n_select=100,
    beta=1.5,
    random_state=0,
)
```

## Example: optimizer workflow

```python
from ggfps_paper import TrainingSetOptimizer

optimizer = TrainingSetOptimizer(
    x=x,
    y=y,
    gradient_norms=gradient_norms,
    labeled_indices=labeled_indices,
    training_set_size=100,
    test_indices=test_indices,
    bounds={"width_bounds": (0.2, 5.0), "reg_bounds": (1e-10, 1e-6), "search_density": 12},
    gradient_biases=[0.5, 1.0, 1.5],
    schedule="ascending",
    random_state=0,
)

result = optimizer.evaluate(num_folds=5)
```
