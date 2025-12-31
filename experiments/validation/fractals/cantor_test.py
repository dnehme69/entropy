import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from src.core.granular import granular_cardinality
from src.algorithms.extensive_cardinality import exhaustive_extensive_cardinality


def generate_cantor_set(depth: int, scale: float = 1.0) -> np.ndarray:
    """
    Generate Cantor set iteratively (avoids recursion issues).
    Returns 2^depth points in [0, scale], embedded in R^2 as (x, 0).
    """
    points = np.array([0.0])
    for _ in range(depth):
        # At each step: map x -> [x/3, x/3 + 2/3]
        left = points / 3.0
        right = points / 3.0 + 2.0 / 3.0
        points = np.concatenate([left, right])
    points = scale * points
    return np.column_stack([points, np.zeros_like(points)])


def run_cantor_validation():
    """Validate entropy on a small Cantor set (depth=6 → 64 points)."""
    depth = 6  # 2^6 = 64 points — safe for exact computation
    points = generate_cantor_set(depth, scale=1.0)
    
    print("Fractal validation (Cantor set)")
    print("--------------------------------")
    print(f"Depth: {depth} | Points: {len(points)}")
    print(f"Hausdorff dimension: log(2)/log(3) ≈ 0.6309")
    print()
    print(f"{'kappa':<6} | {'epsilon':<10} | {'g-card':<8} | {'e-card':<8} | {'entropy':<9}")
    print("-" * 60)
    
    # Choose kappas where epsilon ~ 3^{-k} (fractal scales)
    kappas = [2, 3, 4, 5, 6]
    
    for kappa in kappas:
        epsilon = 2.0 ** (-kappa)
        g_card = granular_cardinality(points, kappa)
        
        # Exact feasible since g-card will be small (Cantor is sparse)
        if g_card <= 100:
            e_card = exhaustive_extensive_cardinality(
                points, kappa, max_subset_size=4
            )
            entropy = np.log(e_card) / np.log(g_card) if g_card > 1 else 0.0
            print(f"{kappa:<6} | {epsilon:<10.4f} | {g_card:<8} | {e_card:<8} | {entropy:<9.3f}")
        else:
            print(f"{kappa:<6} | {epsilon:<10.4f} | {g_card:<8} | —        | —         (net too large)")
    
    print("\nExpected: entropy > 0.63, possibly > 1 due to hierarchical gaps.")


if __name__ == "__main__":
    run_cantor_validation()
