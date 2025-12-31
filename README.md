# Geometric Entropy Numerical Validation

Exact computation of $\eta_\kappa$ on three canonical regimes:

- **Smooth**: unit circle ? $\eta_\kappa \to 0$
- **Fractal**: Cantor set ? $\eta_\kappa = \Theta(1/\kappa)$
- **Generic**: uniform noise ? $\eta_\kappa \gg 0$

Reproduces Figure 1 from *Geometric Entropy Density and Configuration Decay* (Nehme, 2025).

## Structure
- src/: core cardinality and VR signature modules
- alidation/: scripts to recompute results
- data/: exact $\eta_\kappa$ values
- igures/: publication-ready TikZ-compatible plots

All results are **fully reproducible** from this repository.
