import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from src.core.granular import granular_cardinality
from src.algorithms.extensive_cardinality import exhaustive_extensive_cardinality


def generate_uniform_noise(n_points: int, seed: int = 42) -> np.ndarray:
    """Generate i.i.d. uniform points in [0,1]^2."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n_points, 2))


def run_uniform_noise_validation():
    """Validate entropy on generic 2D noise (expected entropy ≈ 2)."""
    n_points = 100
    points = generate_uniform_noise(n_points, seed=42)
    
    print("Generic validation (uniform noise in [0,1]^2)")
    print("---------------------------------------------")
    print(f"Point cloud size: {n_points}")
    print("Expected: entropy → 2 (ambient dimension) due to lack of structure.")
    print()
    print(f"{'kappa':<6} | {'epsilon':<10} | {'g-card':<8} | {'e-card':<8} | {'entropy':<9}")
    print("-" * 60)
    
    kappas = [3, 4, 5, 6]
    
    for kappa in kappas:
        epsilon = 2.0 ** (-kappa)
        g_card = granular_cardinality(points, kappa)
        
        if g_card <= 100:
            e_card = exhaustive_extensive_cardinality(
                points, kappa, max_subset_size=4
            )
            entropy = np.log(e_card) / np.log(g_card) if g_card > 1 else 0.0
            print(f"{kappa:<6} | {epsilon:<10.4f} | {g_card:<8} | {e_card:<8} | {entropy:<9.3f}")
        else:
            print(f"{kappa:<6} | {epsilon:<10.4f} | {g_card:<8} | —        | —         (net too large)")


if __name__ == "__main__":
    run_uniform_noise_validation()
