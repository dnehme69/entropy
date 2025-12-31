import numpy as np
import pytest
from src.core.vr_signature import vr_complex_signature, canonical_vr_signature


def test_vr_complex_signature_two_points():
    points = np.array([[0.0, 0.0], [0.1, 0.0]])
    
    # At epsilon = 0.06, distance 0.1 > 2*0.06 = 0.12? No, 0.1 < 0.12 ? edge exists
    edges = vr_complex_signature(points, epsilon=0.06)
    assert len(edges) == 1
    assert (0, 1) in edges
    
    # At epsilon = 0.04, 2*epsilon = 0.08 < 0.1 ? no edge
    edges = vr_complex_signature(points, epsilon=0.04)
    assert len(edges) == 0


def test_vr_complex_signature_three_points():
    # Equilateral triangle with side 0.1
    points = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.05, 0.0866]  # approx height
    ])
    
    # At epsilon = 0.06 (2*eps = 0.12 > 0.1), all edges present
    edges = vr_complex_signature(points, epsilon=0.06)
    assert len(edges) == 3
    expected = {(0,1), (0,2), (1,2)}
    assert edges == expected


def test_canonical_vr_signature():
    points = np.array([[0.0, 0.0], [0.1, 0.0]])
    thresholds = [0.08, 0.12, 0.16]  # corresponding to 2e, 3e, 4e with e=0.04
    
    signature = canonical_vr_signature(points, thresholds)
    assert len(signature) == 3
    
    # At 0.08: 2*0.08=0.16 > 0.1 ? edge
    # Actually: VR radius = threshold, so edge if d <= 2*threshold?
    # Correction: In GUDHI, RipsComplex(points, max_edge_length=L) includes edge if d <= L
    # So we pass L = 2 * epsilon ? correct
    # Thus at threshold=0.08, max_edge=0.16 ? includes edge
    assert len(signature[0]) == 1  # 0.08
    assert len(signature[1]) == 1  # 0.12 ? max_edge=0.24
    assert len(signature[2]) == 1  # 0.16 ? max_edge=0.32


def test_empty_point_cloud():
    points = np.array([]).reshape(0, 2)
    edges = vr_complex_signature(points, epsilon=1.0)
    assert len(edges) == 0


def test_single_point():
    points = np.array([[0.0, 0.0]])
    edges = vr_complex_signature(points, epsilon=1.0)
    assert len(edges) == 0
