"""Test real bandwidth data ingest."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest_real import load_bandwidth_data, load_delay_data, sample_bandwidth


def test_load_bandwidth_data():
    """Load bandwidth data and verify structure."""
    data = load_bandwidth_data()

    assert "source" in data, "Missing source field"
    assert "values" in data, "Missing values field"
    assert data["values"]["minimum_mbps"] == 2.0, "Wrong minimum bandwidth"
    assert data["values"]["maximum_mbps"] == 10.0, "Wrong maximum bandwidth"

    print(f"PASS: Bandwidth data loaded - {data['source']}")


def test_load_delay_data():
    """Load delay data and verify structure."""
    data = load_delay_data()

    assert "source" in data, "Missing source field"
    assert "values" in data, "Missing values field"
    assert data["values"]["opposition_min_s"] == 180, "Wrong min delay"
    assert data["values"]["conjunction_max_s"] == 1320, "Wrong max delay"

    print(f"PASS: Delay data loaded - {data['source']}")


def test_sample_bandwidth():
    """Sample bandwidth values and verify range."""
    samples = sample_bandwidth(100, 42)

    assert len(samples) == 100, f"Expected 100 samples, got {len(samples)}"
    assert all(2 <= s <= 10 for s in samples), "Samples out of range"

    print(f"PASS: Bandwidth samples - min={min(samples):.1f}, max={max(samples):.1f}")


if __name__ == "__main__":
    test_load_bandwidth_data()
    test_load_delay_data()
    test_sample_bandwidth()
