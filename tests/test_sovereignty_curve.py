"""Test sovereignty curve generation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plot_curve import generate_curve_data, find_knee


def test_generate_curve():
    """Curve should have correct structure."""
    data = generate_curve_data((10, 100), 4.0, 480)

    assert len(data) == 91, f"Expected 91 points, got {len(data)}"
    assert all(isinstance(d, tuple) and len(d) == 2 for d in data), (
        "Invalid data structure"
    )

    print(f"PASS: Curve generated with {len(data)} points")


def test_find_knee():
    """Knee should be in reasonable range."""
    data = generate_curve_data((10, 100), 4.0, 480)
    knee = find_knee(data)

    assert 30 < knee < 80, f"Knee {knee} outside expected range [30, 80]"

    print(f"PASS: Knee at {knee} crew")


def test_curve_crosses_zero():
    """Curve should cross zero at knee."""
    data = generate_curve_data((10, 100), 4.0, 480)
    knee = find_knee(data)

    # Find advantage at knee-1 and knee
    advantages = {crew: adv for crew, adv in data}

    if knee > 10:
        assert advantages.get(knee - 1, 1) <= 0, (
            "Advantage should be negative before knee"
        )
    assert advantages.get(knee, -1) > 0, "Advantage should be positive at knee"

    print(f"PASS: Curve crosses zero at knee={knee}")


if __name__ == "__main__":
    test_generate_curve()
    test_find_knee()
    test_curve_crosses_zero()
