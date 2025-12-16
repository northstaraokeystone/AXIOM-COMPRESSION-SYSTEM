#!/usr/bin/env python3
"""AXIOM-SYSTEM v2 CLI - Unified solar entropy simulation.

Usage:
    python cli.py --test                    # Run quick test
    python cli.py --baseline                # Run baseline simulation
    python cli.py --relay                   # Run with Moon relay
    python cli.py --neuralink               # Run with Neuralink
    python cli.py --full                    # Run all configurations
    python cli.py --observe <view> [body]   # Observe system state
    python cli.py --prove                   # Show proof/findings
"""

import argparse
import json
import sys
import time

# Add src to path
sys.path.insert(0, '.')

from src.system import SystemConfig, run_simulation, initialize_system
from src.observe import observe, observe_all
from src.prove import (
    format_system_discovery,
    format_neuralink_impact,
    format_relay_impact,
    format_paradigm_shift,
    format_tweet,
    bits_to_mass_equivalence,
)
from src.entropy import (
    NEURALINK_MULTIPLIER,
    MDL_BETA,
    SOVEREIGNTY_THRESHOLD_NEURALINK,
)


def run_test():
    """Quick validation test."""
    print("AXIOM-SYSTEM v2 Quick Test")
    print("=" * 50)

    # Test 1: Initialize system
    config = SystemConfig(duration_sols=10, emit_receipts=False)
    state = initialize_system(config)
    assert "earth" in state.bodies, "FAIL: earth not in bodies"
    assert "mars" in state.bodies, "FAIL: mars not in bodies"
    print("[PASS] System initialization")

    # Test 2: Run short simulation
    result = run_simulation(config)
    assert result.sol == 10, f"FAIL: expected sol 10, got {result.sol}"
    print("[PASS] Simulation run")

    # Test 3: Verify constants
    assert NEURALINK_MULTIPLIER == 1e5, "FAIL: NEURALINK_MULTIPLIER"
    assert MDL_BETA == 0.09, "FAIL: MDL_BETA"
    assert SOVEREIGNTY_THRESHOLD_NEURALINK == 5, "FAIL: SOVEREIGNTY_THRESHOLD_NEURALINK"
    print("[PASS] Constants verified")

    # Test 4: Emit receipt
    from src.core import emit_receipt
    receipt = emit_receipt("test", {"value": 42})
    assert "receipt_type" in receipt, "FAIL: receipt structure"
    assert "payload_hash" in receipt, "FAIL: receipt hash"
    print("[PASS] Receipt emission")

    print("=" * 50)
    print("ALL TESTS PASSED")
    return 0


def run_baseline(duration_sols=500):
    """Run baseline simulation (no relay, no Neuralink)."""
    print(f"Running BASELINE simulation ({duration_sols} sols)...")
    config = SystemConfig(
        duration_sols=duration_sols,
        moon_relay_enabled=False,
        neuralink_enabled=False,
        random_seed=42,
        emit_receipts=False,
    )
    start = time.time()
    result = run_simulation(config)
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.1f}s")
    print(observe(result, "meta"))
    return result


def run_relay(duration_sols=500):
    """Run simulation with Moon relay enabled."""
    print(f"Running MOON RELAY simulation ({duration_sols} sols)...")
    config = SystemConfig(
        duration_sols=duration_sols,
        moon_relay_enabled=True,
        neuralink_enabled=False,
        random_seed=42,
        emit_receipts=False,
    )
    start = time.time()
    result = run_simulation(config)
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.1f}s")
    print(observe(result, "meta"))
    return result


def run_neuralink(duration_sols=500):
    """Run simulation with Neuralink enabled."""
    print(f"Running NEURALINK simulation ({duration_sols} sols)...")
    config = SystemConfig(
        duration_sols=duration_sols,
        moon_relay_enabled=False,
        neuralink_enabled=True,
        neuralink_mbps=1.0,
        random_seed=42,
        emit_receipts=False,
    )
    start = time.time()
    result = run_simulation(config)
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.1f}s")
    print(observe(result, "meta"))
    return result


def run_full(duration_sols=500):
    """Run all configurations and compare."""
    print("=" * 70)
    print("AXIOM-SYSTEM v2 FULL COMPARISON")
    print("=" * 70)

    # Baseline
    print("\n[1/3] BASELINE")
    baseline = run_baseline(duration_sols)

    # Moon relay
    print("\n[2/3] MOON RELAY")
    relay = run_relay(duration_sols)

    # Neuralink
    print("\n[3/3] NEURALINK")
    neuralink = run_neuralink(duration_sols)

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    from src.system import find_sovereignty_sol

    baseline_mars_sol = find_sovereignty_sol(baseline, "mars")
    relay_mars_sol = find_sovereignty_sol(relay, "mars")
    neuralink_mars_sol = find_sovereignty_sol(neuralink, "mars")

    print(f"\nMars Sovereignty Sol:")
    print(f"  Baseline:  {baseline_mars_sol or 'Never'}")
    print(f"  Relay:     {relay_mars_sol or 'Never'}")
    print(f"  Neuralink: {neuralink_mars_sol or 'Never'}")

    print(f"\nFinal System Sovereignty:")
    print(f"  Baseline:  {'YES' if baseline.system_sovereign else 'NO'}")
    print(f"  Relay:     {'YES' if relay.system_sovereign else 'NO'}")
    print(f"  Neuralink: {'YES' if neuralink.system_sovereign else 'NO'}")

    print("\n" + format_neuralink_impact())
    print(format_relay_impact(baseline_mars_sol, relay_mars_sol))

    return baseline, relay, neuralink


def main():
    parser = argparse.ArgumentParser(description="AXIOM-SYSTEM v2 CLI")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--baseline", action="store_true", help="Run baseline simulation")
    parser.add_argument("--relay", action="store_true", help="Run with Moon relay")
    parser.add_argument("--neuralink", action="store_true", help="Run with Neuralink")
    parser.add_argument("--full", action="store_true", help="Run all configurations")
    parser.add_argument("--observe", type=str, choices=["micro", "macro", "meta", "all"],
                        help="Observe system state")
    parser.add_argument("--body", type=str, default="mars", help="Body for micro view")
    parser.add_argument("--prove", action="store_true", help="Show proof/findings")
    parser.add_argument("--tweet", action="store_true", help="Output tweet-length summary")
    parser.add_argument("--sols", type=int, default=500, help="Simulation duration")

    args = parser.parse_args()

    if args.test:
        return run_test()

    if args.baseline:
        run_baseline(args.sols)
        return 0

    if args.relay:
        run_relay(args.sols)
        return 0

    if args.neuralink:
        run_neuralink(args.sols)
        return 0

    if args.full:
        run_full(args.sols)
        return 0

    if args.observe:
        config = SystemConfig(duration_sols=100, emit_receipts=False)
        result = run_simulation(config)
        if args.observe == "all":
            print(observe_all(result, args.body))
        else:
            print(observe(result, args.observe, args.body))
        return 0

    if args.prove:
        config = SystemConfig(duration_sols=args.sols, emit_receipts=False)
        result = run_simulation(config)
        print(format_paradigm_shift(result))
        return 0

    if args.tweet:
        config = SystemConfig(duration_sols=args.sols, emit_receipts=False)
        result = run_simulation(config)
        print(format_tweet(result))
        return 0

    # Default: run test
    return run_test()


if __name__ == "__main__":
    sys.exit(main())
