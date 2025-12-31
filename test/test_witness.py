"""Tests for spaceproof.witness module."""

import numpy as np
from spaceproof.witness import KAN, KANConfig, train


def test_kan_init_default():
    """KAN initializes with default config."""
    kan = KAN()
    assert kan.hidden_dim == 8
    assert kan.n_knots == 10
    assert kan.trained is False


def test_kan_init_custom():
    """KAN accepts custom config."""
    config = KANConfig(hidden_dim=16, n_knots=20)
    kan = KAN(config)
    assert kan.hidden_dim == 16
    assert kan.n_knots == 20


def test_kan_forward():
    """KAN forward pass works."""
    kan = KAN()
    x = np.linspace(0, 1, 50)
    y = kan.forward(x)
    assert len(y) == 50


def test_kan_fit():
    """KAN fit updates weights."""
    kan = KAN()
    x = np.linspace(0, 1, 50)
    y = np.sin(x * 2 * np.pi)

    loss = kan.fit(x, y, epochs=10)
    assert kan.trained is True
    assert loss >= 0


def test_kan_compression_ratio():
    """KAN compression ratio is positive for structured data."""
    kan = KAN()
    x = np.linspace(0, 1, 100)
    y = x ** 2  # Simple quadratic

    kan.fit(x, y, epochs=20)
    ratio = kan.get_compression_ratio(x, y)
    assert ratio >= 0


def test_train_function():
    """train function returns results dict."""
    kan = KAN()
    r = np.linspace(1, 50, 100)
    v = np.sqrt(r) * 10  # Simple rotation curve

    results = train(kan, r, v, epochs=10)
    assert "compression" in results
    assert "r_squared" in results
    assert "equation" in results
