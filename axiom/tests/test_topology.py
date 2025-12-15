"""Tests for topology.py - CLAUDEME LAW_2: No test -> not shipped."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_compute_persistence_returns_diagrams():
    """Test that compute_persistence returns H0, H1, H2 diagrams."""
    from src.topology import compute_persistence

    vf = np.random.rand(50, 2) * 10
    d = compute_persistence(vf)
    assert "H0" in d and "H1" in d and "H2" in d


def test_compute_persistence_small_input():
    """Test handling of very small inputs."""
    from src.topology import compute_persistence

    vf = np.array([[0, 0]])  # Only 1 point
    d = compute_persistence(vf)
    assert "H0" in d


def test_classify_features_filters_noise():
    """Test that classify_features filters low-persistence features."""
    from src.topology import classify_features

    # Create diagram with noise (low persistence) and signal
    diagram = np.array([[0.0, 0.005], [0.1, 0.5], [0.2, 0.3]])  # pers: 0.005, 0.4, 0.1
    features = classify_features(diagram, threshold=0.05)
    # Should only keep features with persistence >= 0.05
    assert len(features) == 2
    assert all(f[2] >= 0.05 for f in features)


def test_h1_interpretation():
    """Test H1 interpretation returns expected fields."""
    from src.topology import h1_interpretation

    features = [(0.1, 0.5, 0.4)]
    result = h1_interpretation(features)
    assert "hole_count" in result
    assert "max_persistence" in result
    assert "interpretation" in result
    assert result["hole_count"] == 1


def test_h1_interpretation_empty():
    """Test H1 interpretation with no features."""
    from src.topology import h1_interpretation

    result = h1_interpretation([])
    assert result["hole_count"] == 0
    assert "trivial" in result["interpretation"].lower()


def test_h2_interpretation():
    """Test H2 interpretation returns expected fields."""
    from src.topology import h2_interpretation

    features = [(0.1, 0.6, 0.5)]
    result = h2_interpretation(features)
    assert "void_count" in result
    assert "max_persistence" in result
    assert result["void_count"] == 1


def test_trivial_check_empty():
    """Test trivial topology check with empty diagrams."""
    from src.topology import trivial_topology_check

    empty = {"H0": np.array([]), "H1": np.array([]), "H2": np.array([])}
    assert trivial_topology_check(empty) is True


def test_trivial_check_with_features():
    """Test trivial topology check with significant features."""
    from src.topology import trivial_topology_check

    # H1 with significant feature
    diagram = {
        "H0": np.array([[0, np.inf]]),
        "H1": np.array([[0.1, 0.5]]),  # persistence 0.4 > threshold
        "H2": np.array([]),
    }
    assert trivial_topology_check(diagram) is False


def test_topology_loss_term():
    """Test Wasserstein distance computation."""
    from src.topology import topology_loss_term

    obs = {"H1": np.array([[0.1, 0.5], [0.2, 0.6]])}
    pred = {"H1": np.array([[0.1, 0.5], [0.2, 0.6]])}
    # Identical diagrams should have distance near 0
    loss = topology_loss_term(obs, pred)
    assert loss < 0.01


def test_topology_loss_term_empty():
    """Test Wasserstein with empty diagrams."""
    from src.topology import topology_loss_term

    obs = {"H1": np.array([])}
    pred = {"H1": np.array([])}
    loss = topology_loss_term(obs, pred)
    assert loss == 0.0


def test_topology_receipt_emit(capsys):
    """Test that topology_receipt_emit produces valid receipt."""
    from src.topology import topology_receipt_emit, compute_persistence, h1_interpretation

    vf = np.random.rand(30, 2)
    diagram = compute_persistence(vf)
    h1_features = [(0.1, 0.5, 0.4)]
    interp = h1_interpretation(h1_features)

    receipt = topology_receipt_emit("NGC1234", diagram, interp)

    assert receipt["receipt_type"] == "topology"
    assert receipt["galaxy_id"] == "NGC1234"
    assert "diagrams_b64" in receipt
    assert "payload_hash" in receipt
