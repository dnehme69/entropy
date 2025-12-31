import numpy as np
import pytest
from src.core.granular import greedy_epsilon_net, granular_cardinality


def test_greedy_epsilon_net_empty():
    points = np.array([]).reshape(0, 2)
    net = greedy_epsilon_net(points, 0.1)
    assert net.size == 0


def test_greedy_epsilon_net_single_point():
    points = np.array([[0.0, 0.0]])
    net = greedy_epsilon_net(points, 0.1)
    assert net.size == 1
    assert net[0] == 0


def test_greedy_epsilon_net_two_close_points():
    points = np.array([[0.0, 0.0], [0.05, 0.0]])
    net = greedy_epsilon_net(points, epsilon=0.1)
    # Should pick only one point since distance < epsilon
    assert net.size == 1


def test_greedy_epsilon_net_two_far_points():
    points = np.array([[0.0, 0.0], [0.2, 0.0]])
    net = greedy_epsilon_net(points, epsilon=0.1)
    # Should pick both points since distance > epsilon
    assert net.size == 2


def test_greedy_epsilon_net_grid():
    # 3x3 grid with spacing 0.1
    x = np.linspace(0, 0.2, 3)
    y = np.linspace(0, 0.2, 3)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    net = greedy_epsilon_net(points, epsilon=0.15)
    # With epsilon=0.15 > spacing=0.1, should get fewer than 9 points
    assert net.size < 9
    # But at least 1
    assert net.size >= 1


def test_granular_cardinality():
    points = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]])
    kappa = 3  # epsilon = 1/8 = 0.125
    g_card = granular_cardinality(points, kappa)
    # Distances: 0.2, 0.4, 0.2 ? all > 0.125, so all 3 points selected
    assert g_card == 3


def test_granular_cardinality_close_points():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    kappa = 3  # epsilon = 0.125
    g_card = granular_cardinality(points, kappa)
    # Adjacent distance = 0.1 < 0.125, so only 2 points (e.g., indices 0 and 2)
    assert g_card == 2


def test_custom_distance_matrix():
    points = np.array([[0.0, 0.0], [0.1, 0.0]])
    # Override with larger distance
    dist_matrix = np.array([[0.0, 0.3], [0.3, 0.0]])
    net = greedy_epsilon_net(points, epsilon=0.2, distance_matrix=dist_matrix)
    # Even though Euclidean distance is 0.1, we tell it it's 0.3 > 0.2
    assert net.size == 2


def test_invalid_distance_matrix_shape():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    bad_dist = np.array([[0.0, 1.0]])  # wrong shape
    with pytest.raises(ValueError, match="distance_matrix must be"):
        greedy_epsilon_net(points, 0.1, distance_matrix=bad_dist)


def test_unsupported_metric():
    points = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match="Unsupported metric"):
        greedy_epsilon_net(points, 0.1, metric="hamming")
