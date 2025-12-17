"""Test null hypothesis behavior."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validate import test_null_hypothesis


def test_null_hypothesis_passes():
    """With infinite bandwidth, threshold should be 1 crew."""
    result = test_null_hypothesis()

    assert result["threshold"] <= 1, f"Expected threshold <= 1, got {result['threshold']}"
    assert result["passed"], "Null hypothesis test should pass"
    print(f"PASS: Null hypothesis - threshold={result['threshold']}")


if __name__ == "__main__":
    test_null_hypothesis_passes()
