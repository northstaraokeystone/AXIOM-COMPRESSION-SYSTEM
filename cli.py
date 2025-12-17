#!/usr/bin/env python3
"""AXIOM-CORE CLI - The Sovereignty Calculator

One equation. One curve. One finding.

Usage:
    python cli.py baseline      # Run baseline test
    python cli.py bootstrap     # Run bootstrap analysis
    python cli.py curve         # Generate sovereignty curve
    python cli.py full          # Run full integration test
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sovereignty import compute_sovereignty, find_threshold, SovereigntyConfig
from src.validate import test_null_hypothesis, test_baseline, bootstrap_threshold, generate_falsifiable_prediction
from src.plot_curve import generate_curve_data, find_knee, plot_sovereignty_curve, format_finding
from src.entropy_shannon import (
    HUMAN_DECISION_RATE_BPS,
    STARLINK_MARS_BANDWIDTH_MIN_MBPS,
    STARLINK_MARS_BANDWIDTH_MAX_MBPS,
)


def cmd_baseline():
    """Run baseline sovereignty test."""
    print("=" * 60)
    print("BASELINE SOVEREIGNTY TEST")
    print("=" * 60)

    result = test_baseline()
    print(f"\nConfiguration:")
    print(f"  Bandwidth: {result['bandwidth_mbps']} Mbps")
    print(f"  Delay: {result['delay_s']} seconds")
    print(f"  Compute: {result['compute_flops']} FLOPS (no AI assist)")

    print(f"\nRESULT: Threshold = {result['threshold']} crew")
    print("=" * 60)


def cmd_bootstrap():
    """Run bootstrap statistical analysis."""
    print("=" * 60)
    print("BOOTSTRAP STATISTICAL ANALYSIS")
    print("=" * 60)

    print("\nRunning 100 bootstrap iterations...")
    result = bootstrap_threshold(100, 42)

    print(f"\nRESULTS:")
    print(f"  Mean threshold: {result['mean']:.1f} crew")
    print(f"  Std deviation:  {result['std']:.1f} crew")
    print(f"  Range: [{result['min']}, {result['max']}] crew")
    print(f"  P-value: {result['p_value']:.6f}")

    print("\n" + generate_falsifiable_prediction(result))
    print("=" * 60)


def cmd_curve():
    """Generate sovereignty curve."""
    print("=" * 60)
    print("SOVEREIGNTY CURVE GENERATION")
    print("=" * 60)

    # Parameters
    bandwidth = 4.0  # Expected Mbps
    delay = 480.0    # 8 min average

    print(f"\nConfiguration:")
    print(f"  Bandwidth: {bandwidth} Mbps")
    print(f"  Delay: {delay} seconds ({delay/60:.0f} minutes)")

    # Generate curve
    data = generate_curve_data((10, 100), bandwidth, delay)
    knee = find_knee(data)

    # Bootstrap for uncertainty
    bootstrap = bootstrap_threshold(100, 42)
    uncertainty = bootstrap["std"]

    print(f"\nTHRESHOLD: {knee} +/- {uncertainty:.0f} crew")

    # Plot
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "sovereignty_curve.png")
    plot_sovereignty_curve(data, knee, output_path, uncertainty=uncertainty)
    print(f"\nCurve saved to: {output_path}")

    # The finding
    print("\n" + "-" * 60)
    print("THE FINDING:")
    print("-" * 60)
    print(format_finding(knee, bandwidth, delay / 60, uncertainty))
    print("=" * 60)


def cmd_full():
    """Run full integration test."""
    print("=" * 60)
    print("AXIOM-CORE v1 INTEGRATION TEST")
    print("=" * 60)

    # 1. Null hypothesis
    print("\n[1] NULL HYPOTHESIS TEST")
    null_result = test_null_hypothesis()
    status = "PASS" if null_result["passed"] else "FAIL"
    print(f"    Status: {status}")
    print(f"    Threshold with infinite bandwidth: {null_result['threshold']}")

    # 2. Baseline
    print("\n[2] BASELINE TEST")
    baseline = test_baseline()
    print(f"    Threshold (no tech assist): {baseline['threshold']} crew")

    # 3. Bootstrap
    print("\n[3] BOOTSTRAP ANALYSIS")
    bootstrap = bootstrap_threshold(100, 42)
    print(f"    Mean: {bootstrap['mean']:.1f} +/- {bootstrap['std']:.1f} crew")
    print(f"    P-value: {bootstrap['p_value']:.6f}")

    # 4. Curve
    print("\n[4] SOVEREIGNTY CURVE")
    data = generate_curve_data((10, 100), 4.0, 480)
    knee = find_knee(data)
    print(f"    Knee at: {knee} crew")

    # Generate plot
    output_path = os.path.join(os.path.dirname(__file__), "outputs", "sovereignty_curve.png")
    plot_sovereignty_curve(data, knee, output_path, uncertainty=bootstrap["std"])
    print(f"    Saved: {output_path}")

    # 5. The finding
    print("\n" + "=" * 60)
    print("THE FINDING:")
    print("=" * 60)
    print(format_finding(
        knee=int(bootstrap["mean"]),
        bandwidth=4.0,
        delay=8.0,  # minutes
        uncertainty=bootstrap["std"]
    ))

    # 6. Falsifiable prediction
    print("\n" + "-" * 60)
    print(generate_falsifiable_prediction(bootstrap))

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "baseline":
        cmd_baseline()
    elif cmd == "bootstrap":
        cmd_bootstrap()
    elif cmd == "curve":
        cmd_curve()
    elif cmd == "full":
        cmd_full()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
