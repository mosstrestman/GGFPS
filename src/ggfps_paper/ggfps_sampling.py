"""Readable class-based implementation of Gradient Guided Furthest Point Sampling (GGFPS)."""

import numpy as np


class GGFPSampler:
    """GGFPS sampler with fixed schedule and distance mode.

    Notes
    -----
    - Initialization is always probabilistic.
    - If `distance_matrix` is not provided, distances are computed on-the-fly from `points`.
    - `schedule` can be `ascending`, `descending`, or `alternating`.
    """

    def __init__(self, schedule="ascending", distance_matrix=None, epsilon=1e-8):
        self.schedule = str(schedule)
        self.epsilon = float(epsilon)
        self.distance_matrix = None if distance_matrix is None else np.asarray(distance_matrix, dtype=float)
        self._validate_schedule(self.schedule)

        if self.distance_matrix is not None:
            if self.distance_matrix.ndim != 2:
                raise ValueError("distance_matrix must be a 2D array.")
            if self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
                raise ValueError("distance_matrix must be square.")

    @classmethod
    def on_the_fly(cls, schedule="ascending", epsilon=1e-8):
        """Create a sampler that computes distances on-the-fly."""
        return cls(schedule=schedule, distance_matrix=None, epsilon=epsilon)

    @classmethod
    def with_distance_matrix(cls, distance_matrix, schedule="ascending", epsilon=1e-8):
        """Create a sampler that reuses a precomputed NxN distance matrix."""
        return cls(schedule=schedule, distance_matrix=distance_matrix, epsilon=epsilon)

    @classmethod
    def ascending_on_the_fly(cls, epsilon=1e-8):
        return cls.on_the_fly(schedule="ascending", epsilon=epsilon)

    @classmethod
    def descending_on_the_fly(cls, epsilon=1e-8):
        return cls.on_the_fly(schedule="descending", epsilon=epsilon)

    @classmethod
    def alternating_on_the_fly(cls, epsilon=1e-8):
        return cls.on_the_fly(schedule="alternating", epsilon=epsilon)

    @classmethod
    def ascending_with_distance_matrix(cls, distance_matrix, epsilon=1e-8):
        return cls.with_distance_matrix(distance_matrix=distance_matrix, schedule="ascending", epsilon=epsilon)

    @classmethod
    def descending_with_distance_matrix(cls, distance_matrix, epsilon=1e-8):
        return cls.with_distance_matrix(distance_matrix=distance_matrix, schedule="descending", epsilon=epsilon)

    @classmethod
    def alternating_with_distance_matrix(cls, distance_matrix, epsilon=1e-8):
        return cls.with_distance_matrix(distance_matrix=distance_matrix, schedule="alternating", epsilon=epsilon)

    @property
    def mode(self):
        return "distance_matrix" if self.distance_matrix is not None else "on_the_fly"

    def sample_for_beta(self, points, gradients, n_select, beta, random_state=None):
        """Convenience method for symmetric single-B selection: [-B, +B]."""
        beta = float(beta)
        return self.sample(
            points=points,
            gradients=gradients,
            n_select=n_select,
            beta_start=-beta,
            beta_end=beta,
            random_state=random_state,
        )

    def sample_for_betas(self, points, gradients, n_select, beta_values, random_state=None):
        """Run GGFPS for each B in `beta_values` and return a dict {B: indices}."""
        rng = np.random.default_rng(random_state)
        selections = {}
        for beta in beta_values:
            child_seed = None if random_state is None else int(rng.integers(1, 1_000_000_000))
            selections[float(beta)] = self.sample_for_beta(
                points=points,
                gradients=gradients,
                n_select=n_select,
                beta=beta,
                random_state=child_seed,
            )
        return selections

    def sample(self, points, gradients, n_select, beta_start=-1.0, beta_end=1.0, random_state=None):
        """Run one GGFPS selection run."""
        gradients = self._validate_gradients_and_size(gradients, n_select)

        rng = np.random.default_rng(random_state)
        selected_mask = np.zeros(gradients.size, dtype=bool)
        selected = []

        if self.distance_matrix is None:
            points = np.asarray(points, dtype=float)
            if points.ndim != 2:
                raise ValueError("points must be a 2D array in on-the-fly mode.")
            if points.shape[0] != gradients.size:
                raise ValueError("points and gradients must have matching first dimension.")

            norms2 = np.einsum("ij,ij->i", points, points)

            def distances_to_all(index):
                vector = points[index]
                dist2 = norms2 + norms2[index] - 2.0 * (points @ vector)
                return np.sqrt(np.maximum(dist2, 0.0))

            beta0 = self._beta_for_step(0, n_select, beta_start, beta_end)
            initial_index = self._probabilistic_initial_index(gradients, beta0, rng)

            selected.append(initial_index)
            selected_mask[initial_index] = True

            min_distances = distances_to_all(initial_index)
            min_distances[initial_index] = 0.0

            for step in range(1, n_select):
                beta_step = self._beta_for_step(step, n_select, beta_start, beta_end)
                scores = np.power(gradients + self.epsilon, beta_step) * min_distances
                scores[selected_mask] = -np.inf

                next_index = int(np.argmax(scores))
                selected.append(next_index)
                selected_mask[next_index] = True

                min_distances = np.minimum(min_distances, distances_to_all(next_index))
                min_distances[next_index] = 0.0

            return np.asarray(selected, dtype=int)

        distance_matrix = self.distance_matrix
        if distance_matrix.shape[0] != gradients.size:
            raise ValueError("distance_matrix size must match gradients length.")

        beta0 = self._beta_for_step(0, n_select, beta_start, beta_end)
        initial_index = self._probabilistic_initial_index(gradients, beta0, rng)

        selected.append(initial_index)
        selected_mask[initial_index] = True

        min_distances = np.array(distance_matrix[initial_index], copy=True)

        for step in range(1, n_select):
            beta_step = self._beta_for_step(step, n_select, beta_start, beta_end)
            scores = np.power(gradients + self.epsilon, beta_step) * min_distances
            scores[selected_mask] = -np.inf

            next_index = int(np.argmax(scores))
            selected.append(next_index)
            selected_mask[next_index] = True

            min_distances = np.minimum(min_distances, distance_matrix[next_index])

        return np.asarray(selected, dtype=int)

    @staticmethod
    def _validate_schedule(schedule):
        valid = {"ascending", "descending", "alternating"}
        if schedule not in valid:
            raise ValueError("Unknown schedule. Use: ascending, descending, or alternating.")

    @staticmethod
    def _validate_gradients_and_size(gradients, n_select):
        gradients = np.asarray(gradients, dtype=float)
        if gradients.ndim != 1:
            raise ValueError("gradients must be a 1D array.")
        if n_select < 1:
            raise ValueError("n_select must be at least 1.")
        if n_select > gradients.size:
            raise ValueError("n_select cannot exceed number of samples.")
        return gradients

    def _beta_for_step(self, step, total_steps, beta_start, beta_end):
        if total_steps <= 1:
            return float(beta_end)

        frac = step / (total_steps - 1)

        if self.schedule == "ascending":
            return beta_start + (beta_end - beta_start) * frac

        if self.schedule == "descending":
            return beta_end + (beta_start - beta_end) * frac

        # alternating
        if step % 2 == 0:
            return beta_start + (beta_end - beta_start) * frac
        return beta_end - (beta_end - beta_start) * frac

    def _probabilistic_initial_index(self, gradients, beta0, rng):
        weights = np.power(gradients + self.epsilon, beta0)
        probs = self._normalize_probs(weights)
        return int(rng.choice(gradients.size, p=probs))

    def _normalize_probs(self, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.where(np.isfinite(weights), weights, 0.0)

        total = float(np.sum(weights))
        if total <= self.epsilon:
            return np.full(weights.shape, 1.0 / weights.size, dtype=float)
        return weights / total
