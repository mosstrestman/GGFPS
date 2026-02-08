"""Minimal ascending/descending GGFPS wrappers.

These wrappers intentionally expose only the two schedules used most often
for the paper workflow and default to on-the-fly distances.
"""

from .ggfps_sampling import ggfps_sweep, ggfps_sweep_descending


def ascending_ggfps(
    points,
    gradients,
    n_select,
    beta_max,
    random_state=None,
    distance_matrix=None,
):
    """Ascending schedule from -beta_max to +beta_max."""
    return ggfps_sweep(
        points=points,
        gradients=gradients,
        n_select=n_select,
        beta_start=-float(beta_max),
        beta_end=float(beta_max),
        random_state=random_state,
        distance_matrix=distance_matrix,
    )


def descending_ggfps(
    points,
    gradients,
    n_select,
    beta_max,
    random_state=None,
    distance_matrix=None,
):
    """Descending schedule from +beta_max to -beta_max."""
    return ggfps_sweep_descending(
        points=points,
        gradients=gradients,
        n_select=n_select,
        beta_start=-float(beta_max),
        beta_end=float(beta_max),
        random_state=random_state,
        distance_matrix=distance_matrix,
    )
