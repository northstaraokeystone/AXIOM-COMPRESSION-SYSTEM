#!/usr/bin/env python3
"""AXIOM-CORE CLI - The Sovereignty Calculator

One equation. One curve. One finding.

Usage:
    python cli.py baseline      # Run baseline test
    python cli.py bootstrap     # Run bootstrap analysis
    python cli.py curve         # Generate sovereignty curve
    python cli.py full          # Run full integration test
    python cli.py --simulate_timeline --c_base 50 --p_factor 1.8 --tau 0     # Earth timeline
    python cli.py --simulate_timeline --c_base 50 --p_factor 1.8 --tau 1200  # Mars timeline

    # Partition resilience testing (Dec 2025)
    python cli.py --partition 0.4 --nodes 5 --simulate    # Single partition test
    python cli.py --stress_quorum                          # Full 1000-iteration stress test

    # Adaptive rerouting and blackout testing (Dec 2025)
    python cli.py --reroute --simulate                     # Single reroute test
    python cli.py --blackout 43 --reroute --simulate       # Blackout with reroute
    python cli.py --blackout_sweep --reroute               # Full blackout sweep (43-60d, 1000 iterations)
    python cli.py --algo_info                              # Output reroute algorithm spec

    # Extended blackout sweep and retention curve (Dec 2025)
    python cli.py --extended_sweep 43 90 --simulate        # Extended sweep (43-90d)
    python cli.py --retention_curve                        # Output retention curve as JSON
    python cli.py --blackout_sweep 60 --simulate           # Single-point extended blackout test
    python cli.py --gnn_stub                               # Echo GNN sensitivity stub config

    # GNN nonlinear caching (Dec 2025)
    python cli.py --gnn_nonlinear --blackout 150 --simulate  # GNN nonlinear at 150d
    python cli.py --cache_depth 1000000000 --blackout 200 --gnn_nonlinear  # Custom cache depth
    python cli.py --cache_sweep --simulate                   # Cache depth sensitivity sweep
    python cli.py --extreme_sweep 200 --simulate             # Extreme sweep to 200d
    python cli.py --overflow_test --simulate                 # Test cache overflow detection
    python cli.py --innovation_stubs                         # Echo innovation stub status

    # Entropy pruning (Dec 2025)
    python cli.py --entropy_prune --blackout 150 --simulate     # Single pruning test
    python cli.py --trim_factor 0.4 --entropy_prune --simulate  # Custom trim factor
    python cli.py --hybrid_prune --blackout 200 --simulate      # Hybrid dedup + predictive
    python cli.py --pruning_sweep --simulate                    # Pruning sensitivity sweep
    python cli.py --extended_250d --simulate                    # 250d with pruning
    python cli.py --verify_chain --entropy_prune --simulate     # Verify chain integrity
    python cli.py --pruning_info                                # Echo pruning configuration
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.args import create_parser
from cli.dispatch import dispatch


def main():
    """Main entry point for AXIOM CLI."""
    parser = create_parser()
    args = parser.parse_args()
    dispatch(args, __doc__)


if __name__ == "__main__":
    main()
