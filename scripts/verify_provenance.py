#!/usr/bin/env python3
"""verify_provenance.py - CLI for Receipt Verification

Usage:
    python scripts/verify_provenance.py receipts.jsonl
    python scripts/verify_provenance.py --help

THE VERIFICATION INSIGHT:
    Trust but verify. Every receipt has a payload_hash.
    This script recomputes hashes and validates chain integrity.

Source: AXIOM Validation Lock v1
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prove import verify_provenance, verify_real_data_provenance


def format_results(results: dict) -> str:
    """Format verification results for display.

    Args:
        results: Results from verify_provenance()

    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "AXIOM PROVENANCE VERIFICATION",
        "=" * 60,
        "",
        f"Receipts checked: {results['receipts_checked']}",
        f"Valid:            {results['valid_count']}",
        f"Invalid:          {results['invalid_count']}",
        f"Chain breaks:     {len(results['broken_chains'])}",
        "",
    ]

    if results['hash_mismatches']:
        lines.append("HASH MISMATCHES:")
        for mismatch in results['hash_mismatches']:
            lines.append(f"  Line {mismatch.get('line', '?')}: {mismatch.get('receipt_type', 'unknown')}")
            if 'error' in mismatch:
                lines.append(f"    Error: {mismatch['error']}")
            else:
                lines.append(f"    Expected: {mismatch.get('expected', 'N/A')[:32]}...")
                lines.append(f"    Computed: {mismatch.get('computed', 'N/A')[:32]}...")
        lines.append("")

    if results['broken_chains']:
        lines.append("CHAIN BREAKS:")
        for chain_break in results['broken_chains']:
            if 'gap_seconds' in chain_break:
                gap_hours = chain_break['gap_seconds'] / 3600
                lines.append(f"  Gap: {gap_hours:.1f} hours")
                lines.append(f"    Before: {chain_break.get('before', 'N/A')}")
                lines.append(f"    After:  {chain_break.get('after', 'N/A')}")
            elif 'line' in chain_break:
                lines.append(f"  Line {chain_break['line']}: {chain_break.get('error', 'Unknown error')}")
        lines.append("")

    # Overall status
    lines.append("-" * 60)
    if results['verification_passed']:
        lines.append("STATUS: ✓ VERIFICATION PASSED")
    else:
        lines.append("STATUS: ✗ VERIFICATION FAILED")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Verify AXIOM receipt chain integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s receipts.jsonl
    %(prog)s receipts.jsonl --verbose
    %(prog)s receipts.jsonl --json

The verification checks:
    1. Each receipt has a valid payload_hash
    2. Recomputed hash matches stored hash
    3. No large gaps (>1 hour) in receipt timestamps
    4. All JSON is valid
        """
    )

    parser.add_argument(
        "receipts_file",
        nargs="?",
        default="receipts.jsonl",
        help="Path to receipts.jsonl file (default: receipts.jsonl)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--real-data", "-r",
        action="store_true",
        help="Also verify real_data receipt provenance"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on any verification failure"
    )

    args = parser.parse_args()

    # Verify provenance
    results = verify_provenance(args.receipts_file)

    # Output results
    if args.json:
        # Remove non-serializable receipt object
        output = {k: v for k, v in results.items() if k != 'verification_receipt'}
        print(json.dumps(output, indent=2, default=str))
    else:
        print(format_results(results))

    # Optionally verify real data provenance
    if args.real_data:
        real_results = verify_real_data_provenance(args.receipts_file)
        print("\n" + "=" * 60)
        print("REAL DATA PROVENANCE")
        print("=" * 60)
        print(f"Real data receipts: {real_results['real_data_receipts']}")
        print(f"Verified:          {real_results['verified']}")
        print(f"Unverified:        {real_results['unverified']}")
        if real_results['errors']:
            print("\nErrors:")
            for err in real_results['errors']:
                if isinstance(err, dict):
                    print(f"  {err.get('dataset_id', 'unknown')}: {err.get('error', 'Unknown')}")
                else:
                    print(f"  {err}")

    # Exit code
    if args.strict and not results['verification_passed']:
        sys.exit(1)
    elif results.get('error'):
        # File not found or similar - exit with 0 (expected on clean repo)
        print(f"\nNote: {results['error']}")
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
