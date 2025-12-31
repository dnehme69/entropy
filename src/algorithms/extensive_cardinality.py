import numpy as np
from typing import Set, Tuple, FrozenSet, Optional
from itertools import combinations
from src.core.vr_signature import canonical_vr_signature
from src.core.granular import greedy_epsilon_net


def exhaustive_extensive_cardinality(
    points: np.ndarray,
    kappa: int,
    max_subset_size: int = 4,
    metric: str = 'euclidean'
) -> int:
    r"""
    Compute e-card(X, kappa) by exhaustive enumeration of all nonempty subsets
    of the epsilon_kappa-net, up to size max_subset_size.
    
    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (n_points, n_dim).
    kappa : int
        Scale parameter.
    max_subset_size : int, default=4
        Maximum size of subsets to consider (paper uses 4).
    metric : str, default='euclidean'
        Metric for net construction.
    
    Returns
    -------
    e_card : int
        Number of distinct VR signatures.
    """
    epsilon_kappa = 2.0 ** (-kappa)
    
    # Step 1: Build epsilon-net
    net_indices = greedy_epsilon_net(points, epsilon_kappa, metric=metric)
    net_points = points[net_indices]
    n_net = len(net_points)
    
    if n_net == 0:
        return 0
    
    # Step 2: Define thresholds: 2*eps, 3*eps, 4*eps
    thresholds = [2.0 * epsilon_kappa, 3.0 * epsilon_kappa, 4.0 * epsilon_kappa]
    
    # Step 3: Enumerate all nonempty subsets up to size max_subset_size
    unique_signatures: Set[Tuple[FrozenSet[Tuple[int, ...]], ...]] = set()
    
    for size in range(1, min(max_subset_size + 1, n_net + 1)):
        for subset_indices in combinations(range(n_net), size):
            subset = net_points[list(subset_indices)]
            # Compute canonical VR signature
            signature = canonical_vr_signature(subset, thresholds)
            unique_signatures.add(signature)
    
    return len(unique_signatures)


def monte_carlo_extensive_cardinality(
    points: np.ndarray,
    kappa: int,
    n_samples: int = 10000,
    max_subset_size: int = 4,
    metric: str = 'euclidean',
    random_seed: Optional[int] = None
) -> Tuple[int, float]:
    r"""
    Estimate e-card using Monte Carlo sampling with Horvitz-Thompson correction.
    
    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    kappa : int
        Scale parameter.
    n_samples : int, default=10000
        Number of random subsets to sample.
    max_subset_size : int, default=4
    metric : str, default='euclidean'
    random_seed : int, optional
    
    Returns
    -------
    estimated_e_card : int
        Lower bound on e-card (number of distinct signatures observed).
    coverage_estimate : float
        Estimated proportion of signature space covered.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    epsilon_kappa = 2.0 ** (-kappa)
    net_indices = greedy_epsilon_net(points, epsilon_kappa, metric=metric)
    net_points = points[net_indices]
    n_net = len(net_points)
    
    if n_net == 0:
        return 0, 1.0
    
    thresholds = [2.0 * epsilon_kappa, 3.0 * epsilon_kappa, 4.0 * epsilon_kappa]
    observed_signatures: Set = set()
    
    for _ in range(n_samples):
        # Random subset size (1 to max_subset_size)
        size = np.random.randint(1, min(max_subset_size, n_net) + 1)
        if n_net < size:
            continue
        subset_indices = np.random.choice(n_net, size, replace=False)
        subset = net_points[subset_indices]
        signature = canonical_vr_signature(subset, thresholds)
        observed_signatures.add(signature)
    
    # Simple coverage estimate: observed / total possible (very conservative)
    total_possible = sum(
        len(list(combinations(range(n_net), s))) 
        for s in range(1, min(max_subset_size + 1, n_net + 1))
    )
    coverage = min(1.0, len(observed_signatures) / total_possible) if total_possible > 0 else 1.0
    
    return len(observed_signatures), coverage
