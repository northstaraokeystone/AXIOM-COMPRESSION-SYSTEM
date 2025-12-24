#!/usr/bin/env python3
"""SpaceProof CLI - Space-grade proof infrastructure.

No receipt, not real.

Usage:
    space init                              # Initialize project
    space compress --domain tesla           # Compress telemetry
    space witness --domain galaxy           # Discover physics laws
    space sovereignty --crew 10             # Calculate autonomy threshold
    space detect --config doge              # Entropy anomaly detection
    space anchor --batch receipts.jsonl     # Anchor receipts
    space loop --cycles 100                 # Run SENSE→ACTUATE cycle
    space audit --from 2024-01-01           # Query ledger

    # Sovereignty calculations
    space baseline                           # Run baseline test
    space bootstrap                          # Run bootstrap analysis
    space curve                              # Generate sovereignty curve
    space full                               # Run full integration test

    # Partition resilience testing
    space --partition 0.4 --nodes 5 --simulate    # Single partition test
    space --stress_quorum                          # Full 1000-iteration stress test

    # Blackout testing
    space --reroute --simulate                     # Single reroute test
    space --blackout 43 --reroute --simulate       # Blackout with reroute
    space --blackout_sweep --reroute               # Full blackout sweep

    # Extended testing
    space --extended_sweep 43 90 --simulate        # Extended sweep
    space --gnn_nonlinear --blackout 150 --simulate  # GNN nonlinear test

    # Test receipts
    space --test                                   # Emit test receipt
    space --config xai --test                      # Test with xAI config
    space --config doge --test                     # Test with DOGE config

Stakeholder Configs:
    xai.yaml     - Elon/xAI: compression + sovereignty
    doge.yaml    - DOGE: fraud detection + audit
    dot.yaml     - DOT: compliance + safety
    defense.yaml - Defense: fire-control + lineage
    nro.yaml     - NRO: constellation governance
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spaceproof.cli.args import create_parser
from spaceproof.cli.dispatch import dispatch


def main():
    """Main entry point for SpaceProof CLI."""
    parser = create_parser()
    args = parser.parse_args()
    dispatch(args, __doc__)


if __name__ == "__main__":
    main()
