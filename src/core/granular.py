import numpy as np
from typing import Optional

def greedy_epsilon_net(
    points: np.ndarray,
    epsilon: float,
    distance_matrix: Optional[np.ndarray] = None,
    metric: str = 'euclidean'
) -> np.ndarray:
    r"""
    Construct a maximal epsilon-net from a point cloud using the greedy algorithm.
    
    An epsilon-net is a subset N of the point cloud such that:
    (1) (epsilon-separated) d(x, y) > epsilon for all distinct x, y in N
    (2) (epsilon-dense) for every point z in the cloud, there exists x in N with d(z, x) <= epsilon
    
    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (n_points, n_dim).
    epsilon : float
        The scale parameter (epsilon > 0).
    distance_matrix : np.ndarray, optional
        Precomputed pairwise distance matrix of shape (n_points, n_points).
        If not provided, computed on-the-fly using the specified metric.
    metric : str, default='euclidean'
        Distance metric to use if distance_matrix is not provided.
        Supported: 'euclidean', 'manhattan', 'chebyshev'.
    
    Returns
    -------
    net_indices : np.ndarray
        Array of indices (into `points`) of the selected net points.
    """
    if points.size == 0:
        return np.array([], dtype=int)
    
    n_points = points.shape[0]
    if distance_matrix is None:
        if metric == 'euclidean':
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
        elif metric == 'manhattan':
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sum(np.abs(diff), axis=2)
        elif metric == 'chebyshev':
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.max(np.abs(diff), axis=2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    else:
        if distance_matrix.shape != (n_points, n_points):
            raise ValueError("distance_matrix must be (n_points, n_points)")
        dists = distance_matrix

    np.fill_diagonal(dists, 0.0)
    dists = np.maximum(dists, dists.T)

    selected = []
    covered = np.zeros(n_points, dtype=bool)
    remaining_indices = np.arange(n_points)

    while not covered.all():
        uncovered = remaining_indices[~covered[remaining_indices]]
        candidate = uncovered[0]
        selected.append(candidate)
        covered = covered | (dists[candidate] <= epsilon)

    return np.array(selected, dtype=int)


def granular_cardinality(
    points: np.ndarray,
    kappa: int,
    distance_matrix: Optional[np.ndarray] = None,
    metric: str = 'euclidean'
) -> int:
    r"""
    Compute the granular cardinality g-card(X, kappa) = packing number at scale epsilon_kappa.
    
    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (n_points, n_dim).
    kappa : int
        Scale parameter (epsilon_kappa = 2^(-kappa)).
    distance_matrix : np.ndarray, optional
        Precomputed pairwise distances.
    metric : str, default='euclidean'
        Distance metric.
    
    Returns
    -------
    g_card : int
        The granular cardinality at scale kappa.
    """
    epsilon_kappa = 2.0 ** (-kappa)
    net_indices = greedy_epsilon_net(points, epsilon_kappa, distance_matrix, metric)
    return int(net_indices.size)
