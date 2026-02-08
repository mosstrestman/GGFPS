"""GGFPS samplers with a readable, paper-aligned API.

Design choices
--------------
- Default path uses on-the-fly distances (no NxN matrix allocation).
- Distance-matrix mode is available for repeated multi-B sweeps.
- Probabilistic initialization is the default everywhere.
"""

import numpy as np


def _normalize_probs(weights, epsilon):
    """Normalize non-negative weights into a valid probability vector."""
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    total = float(np.sum(weights))
    if total <= epsilon:
        return np.full(weights.shape, 1.0 / weights.size, dtype=float)
    return weights / total


def _validate_gradients_and_size(gradients, n_select):
    gradients = np.asarray(gradients, dtype=float)
    if gradients.ndim != 1:
        raise ValueError("gradients must be a 1D array.")
    if n_select < 1:
        raise ValueError("n_select must be at least 1.")
    if n_select > gradients.size:
        raise ValueError("n_select cannot exceed number of samples.")
    return gradients


def _beta_for_step(step, total_steps, beta_start, beta_end, schedule):
    if total_steps <= 1:
        return beta_end

    frac = step / (total_steps - 1)

    if schedule == "ascending":
        return beta_start + (beta_end - beta_start) * frac
    if schedule == "descending":
        return beta_end + (beta_start - beta_end) * frac
    if schedule == "bounce":
        return beta_start if step % 2 == 0 else beta_end
    if schedule == "switch":
        if step % 2 == 0:
            return beta_start + (beta_end - beta_start) * frac
        return beta_end - (beta_end - beta_start) * frac

    raise ValueError("Unknown schedule. Use: ascending, descending, bounce, or switch.")


def _choose_initial_index(gradients, beta0, initializer, rng, epsilon):
    if initializer == "probabilistic":
        weights = np.power(gradients + epsilon, beta0)
        probs = _normalize_probs(weights, epsilon)
        return int(rng.choice(gradients.size, p=probs))

    if initializer == "uniform":
        return int(rng.choice(gradients.size))

    if initializer == "high":
        return int(np.argmax(gradients))

    if initializer == "low":
        return int(np.argmin(gradients))

    raise ValueError("Unknown initializer. Use: probabilistic, uniform, high, or low.")


def ggfps_from_distance_matrix(
    gradients,
    distance_matrix,
    n_select,
    beta_start=-1.0,
    beta_end=1.0,
    schedule="ascending",
    initializer="probabilistic",
    random_state=None,
    epsilon=1e-8,
):
    """GGFPS using a precomputed NxN distance matrix."""
    gradients = _validate_gradients_and_size(gradients, n_select)
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array.")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if distance_matrix.shape[0] != gradients.size:
        raise ValueError("distance_matrix size must match gradients length.")

    rng = np.random.default_rng(random_state)
    selected_mask = np.zeros(gradients.size, dtype=bool)
    selected = []

    beta0 = _beta_for_step(0, n_select, beta_start, beta_end, schedule)
    initial = _choose_initial_index(gradients, beta0, initializer, rng, epsilon)

    selected.append(initial)
    selected_mask[initial] = True
    min_distances = np.array(distance_matrix[initial], copy=True)

    for step in range(1, n_select):
        beta = _beta_for_step(step, n_select, beta_start, beta_end, schedule)
        scores = np.power(gradients + epsilon, beta) * min_distances
        scores[selected_mask] = -np.inf

        next_index = int(np.argmax(scores))
        selected.append(next_index)
        selected_mask[next_index] = True

        min_distances = np.minimum(min_distances, distance_matrix[next_index])

    return np.asarray(selected, dtype=int)


def ggfps_on_the_fly(
    points,
    gradients,
    n_select,
    beta_start=-1.0,
    beta_end=1.0,
    schedule="ascending",
    initializer="probabilistic",
    random_state=None,
    epsilon=1e-8,
):
    """GGFPS with on-the-fly distances (recommended default)."""
    points = np.asarray(points, dtype=float)
    gradients = _validate_gradients_and_size(gradients, n_select)

    if points.ndim != 2:
        raise ValueError("points must be a 2D array.")
    if points.shape[0] != gradients.size:
        raise ValueError("points and gradients must have matching first dimension.")

    rng = np.random.default_rng(random_state)

    # Fast squared-Euclidean updates: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    norms2 = np.einsum("ij,ij->i", points, points)

    def distances_to_all(index):
        vec = points[index]
        dist2 = norms2 + norms2[index] - 2.0 * (points @ vec)
        return np.sqrt(np.maximum(dist2, 0.0))

    selected_mask = np.zeros(gradients.size, dtype=bool)
    selected = []

    beta0 = _beta_for_step(0, n_select, beta_start, beta_end, schedule)
    initial = _choose_initial_index(gradients, beta0, initializer, rng, epsilon)

    selected.append(initial)
    selected_mask[initial] = True
    min_distances = distances_to_all(initial)
    min_distances[initial] = 0.0

    for step in range(1, n_select):
        beta = _beta_for_step(step, n_select, beta_start, beta_end, schedule)
        scores = np.power(gradients + epsilon, beta) * min_distances
        scores[selected_mask] = -np.inf

        next_index = int(np.argmax(scores))
        selected.append(next_index)
        selected_mask[next_index] = True

        min_distances = np.minimum(min_distances, distances_to_all(next_index))
        min_distances[next_index] = 0.0

    return np.asarray(selected, dtype=int)


def ggfps_sweep(
    points,
    gradients,
    n_select,
    beta_start=-1.0,
    beta_end=1.0,
    random_state=None,
    distance_matrix=None,
):
    """Ascending GGFPS with probabilistic init.

    Uses on-the-fly distances unless `distance_matrix` is provided.
    """
    if distance_matrix is None:
        return ggfps_on_the_fly(
            points=points,
            gradients=gradients,
            n_select=n_select,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule="ascending",
            initializer="probabilistic",
            random_state=random_state,
        )

    return ggfps_from_distance_matrix(
        gradients=gradients,
        distance_matrix=distance_matrix,
        n_select=n_select,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule="ascending",
        initializer="probabilistic",
        random_state=random_state,
    )


def ggfps_sweep_descending(
    points,
    gradients,
    n_select,
    beta_start=-1.0,
    beta_end=1.0,
    random_state=None,
    distance_matrix=None,
):
    """Descending GGFPS with probabilistic init."""
    if distance_matrix is None:
        return ggfps_on_the_fly(
            points=points,
            gradients=gradients,
            n_select=n_select,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule="descending",
            initializer="probabilistic",
            random_state=random_state,
        )

    return ggfps_from_distance_matrix(
        gradients=gradients,
        distance_matrix=distance_matrix,
        n_select=n_select,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule="descending",
        initializer="probabilistic",
        random_state=random_state,
    )


def select_with_strategy(
    points,
    gradients,
    n_select,
    beta_start,
    beta_end,
    schedule="ascending",
    initializer="probabilistic",
    random_state=None,
    distance_matrix=None,
):
    """Generic selector used by optimization code.

    `distance_matrix=None` means on-the-fly mode.
    """
    if distance_matrix is None:
        return ggfps_on_the_fly(
            points=points,
            gradients=gradients,
            n_select=n_select,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=schedule,
            initializer=initializer,
            random_state=random_state,
        )

    return ggfps_from_distance_matrix(
        gradients=gradients,
        distance_matrix=distance_matrix,
        n_select=n_select,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=schedule,
        initializer=initializer,
        random_state=random_state,
    )
