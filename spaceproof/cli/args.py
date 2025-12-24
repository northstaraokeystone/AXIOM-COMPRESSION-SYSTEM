"""SpaceProof CLI argument parser."""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for SpaceProof CLI."""
    parser = argparse.ArgumentParser(
        prog="spaceproof",
        description="Space-grade proof infrastructure. No receipt, not real.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === CORE COMMANDS ===

    # init
    subparsers.add_parser("init", help="Initialize project")

    # compress
    compress_p = subparsers.add_parser("compress", help="Compress telemetry")
    compress_p.add_argument("--domain", type=str, default="tesla", help="Domain")

    # witness
    witness_p = subparsers.add_parser("witness", help="Discover physics laws")
    witness_p.add_argument("--domain", type=str, default="galaxy", help="Domain")

    # detect
    detect_p = subparsers.add_parser("detect", help="Entropy anomaly detection")
    detect_p.add_argument("--config", type=str, default="doge", help="Config name")

    # anchor
    anchor_p = subparsers.add_parser("anchor", help="Anchor receipts")
    anchor_p.add_argument("--batch", type=str, help="Batch file path")

    # loop
    loop_p = subparsers.add_parser("loop", help="Run SENSE->ACTUATE cycle")
    loop_p.add_argument("--cycles", type=int, default=1, help="Number of cycles")

    # audit
    audit_p = subparsers.add_parser("audit", help="Query ledger")
    audit_p.add_argument("--from", dest="from_date", type=str, help="Start date")

    # === SOVEREIGNTY COMMAND ===
    sov_p = subparsers.add_parser("sovereignty", help="Autonomy threshold calculation")
    sov_sub = sov_p.add_subparsers(dest="sov_command", help="Sovereignty subcommands")

    # sovereignty basic
    sov_p.add_argument("--crew", type=int, help="Crew size for calculation")

    # sovereignty mars
    mars_p = sov_sub.add_parser("mars", help="Mars computational sovereignty simulator")
    mars_p.add_argument("--config", type=str, help="Path to colony config YAML")
    mars_p.add_argument("--find-threshold", action="store_true", help="Find minimum crew for target")
    mars_p.add_argument("--target", type=float, default=95.0, help="Target sovereignty score (%%)")
    mars_p.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo validation")
    mars_p.add_argument("--iterations", type=int, default=1000, help="Monte Carlo iterations")
    mars_p.add_argument("--compare", nargs=2, metavar="CONFIG", help="Compare two configurations")
    mars_p.add_argument("--report", type=str, help="Generate markdown report to file")
    mars_p.add_argument("--scenario", type=str, help="Run specific scenario")

    # === SIMULATION COMMANDS ===
    parser.add_argument("--partition", type=float, help="Partition loss fraction")
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes")
    parser.add_argument("--simulate", action="store_true", help="Enable simulation")
    parser.add_argument("--stress_quorum", action="store_true", help="Stress test quorum")
    parser.add_argument("--reroute", action="store_true", help="Enable rerouting")
    parser.add_argument("--blackout", type=int, help="Blackout duration days")
    parser.add_argument("--blackout_sweep", action="store_true", help="Blackout sweep test")
    parser.add_argument("--test", action="store_true", help="Emit test receipt")
    parser.add_argument("--config", type=str, help="Config name")

    return parser
