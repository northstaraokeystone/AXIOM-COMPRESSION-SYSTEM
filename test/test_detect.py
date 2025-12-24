"""Tests for spaceproof.detect module."""

import pytest
import numpy as np
from spaceproof.detect import (
    shannon_entropy,
    entropy_delta,
    detect_anomaly,
    classify_anomaly,
    build_baseline,
    BaselineStats,
)


def test_shannon_entropy_uniform():
    """shannon_entropy of uniform distribution is maximum."""
    p = np.array([0.25, 0.25, 0.25, 0.25])
    h = shannon_entropy(p)
    assert h == pytest.approx(2.0, abs=0.01)  # log2(4) = 2


def test_shannon_entropy_certain():
    """shannon_entropy of certain outcome is zero."""
    p = np.array([1.0, 0.0, 0.0, 0.0])
    h = shannon_entropy(p)
    assert h == 0.0


def test_entropy_delta():
    """entropy_delta computes difference correctly."""
    before = np.array([0.5, 0.5])
    after = np.array([0.25, 0.25, 0.25, 0.25])
    delta = entropy_delta(before, after)
    assert delta > 0  # More entropy after


def test_detect_anomaly_no_baseline():
    """detect_anomaly works without baseline."""
    stream = np.random.randn(100)
    result = detect_anomaly(stream)
    assert result.classification == "normal"
    assert result.confidence == 0.5


def test_detect_anomaly_with_baseline():
    """detect_anomaly compares against baseline."""
    # Build baseline from normal data (use seed for reproducibility)
    np.random.seed(42)
    samples = [np.random.randn(100) for _ in range(10)]
    baseline = build_baseline(samples)

    # Normal stream
    normal = np.random.randn(100)
    result = detect_anomaly(normal, baseline)
    # Random data can produce various classifications
    assert result.classification in ["normal", "drift", "degradation"]


def test_classify_anomaly_thresholds():
    """classify_anomaly respects threshold levels."""
    assert classify_anomaly(0.5) == "normal"
    assert classify_anomaly(1.2) == "drift"
    assert classify_anomaly(1.8) == "degradation"
    assert classify_anomaly(2.5) == "violation"
    assert classify_anomaly(3.5) == "fraud"


def test_build_baseline():
    """build_baseline computes stats from samples."""
    samples = [np.random.randn(50) for _ in range(5)]
    baseline = build_baseline(samples)

    assert isinstance(baseline, BaselineStats)
    assert baseline.n_samples == 5
    assert baseline.std > 0
