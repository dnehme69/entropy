import numpy as np
from src.algorithms.extensive_cardinality import (
    exhaustive_extensive_cardinality,
    monte_carlo_extensive_cardinality
)


def test_exhaustive_extensive_cardinality_two_points():
    points = np.array([[0.0, 0.0], [0.1, 0.0]])
    kappa = 3  # epsilon = 0.125
    # Net includes both points (distance 0.1 < 0.125? No: 0.1 < 0.125 ? actually they are within epsilon,
    # so only one point in net!
    # Correction: epsilon-net: points within epsilon are covered ? only one point selected.
    e_card = exhaustive_extensive_cardinality(points, kappa, max_subset_size=2)
    # Net size = 1 ? subsets: {0} ? only 1 signature
    assert e_card == 1


def test_exhaustive_extensive_cardinality_far_points():
    points = np.array([[0.0, 0.0], [0.3, 0.0]])
    kappa = 3  # epsilon = 0.125
    # Distance 0.3 > 0.125 ? net includes both points
    e_card = exhaustive_extensive_cardinality(points, kappa, max_subset_size=2)
    # Subsets: {0}, {1}, {0,1}
    # {0}, {1}: same signature (isolated points) ? 1 signature
    # {0,1}: at thresholds [0.25, 0.375, 0.5], distance 0.3
    # ? at 0.25: 2*0.25=0.5 > 0.3 ? edge exists ? different from singletons
    # So total signatures = 2
    assert e_card == 2


def test_monte_carlo_extensive_cardinality():
    points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    kappa = 2  # epsilon = 0.25
    # Net includes all 3 points (min distance 0.5 > 0.25)
    e_card, coverage = monte_carlo_extensive_cardinality(
        points, kappa, n_samples=1000, random_seed=42
    )
    # Should observe at least 2 signatures: singletons and pairs
    assert e_card >= 2
    assert 0.0 <= coverage <= 1.0


def test_empty_point_cloud():
    points = np.array([]).reshape(0, 2)
    e_card = exhaustive_extensive_cardinality(points, kappa=1)
    assert e_card == 0


def test_single_point():
    points = np.array([[0.0, 0.0]])
    e_card = exhaustive_extensive_cardinality(points, kappa=1)
    assert e_card == 1
