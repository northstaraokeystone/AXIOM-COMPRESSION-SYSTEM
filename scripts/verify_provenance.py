#!/usr/bin/env python3
"""verify_provenance.py - Verify receipt chain integrity.

Usage:
    python scripts/verify_provenance.py receipts.jsonl

Verifies:
    - Each receipt has valid payload_hash
    - No broken chains
    - Computes merkle root
    - Emits verification receipt
"""

import argparse
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prove import verify_provenance


def main():
    parser = argparse.ArgumentParser(
        description="Verify receipt chain provenance"
    )
    parser.add_argument(
        "receipts_path",
        help="Path to receipts.jsonl file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed verification results"
    )
    args = parser.parse_args()

    # Check file exists
    if not os.path.exists(args.receipts_path):
        print(f"File not found: {args.receipts_path}")
        # If file doesn't exist and is empty, that's OK
        if os.path.getsize(args.receipts_path) if os.path.exists(args.receipts_path) else 0 == 0:
            print("No receipts to verify (empty file)")
            sys.exit(0)
        sys.exit(1)

    # Check if file is empty
    if os.path.getsize(args.receipts_path) == 0:
        print("No receipts to verify (empty file)")
        sys.exit(0)

    print(f"Verifying provenance: {args.receipts_path}")
    print("=" * 60)

    result = verify_provenance(args.receipts_path)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    # Summary
    print(f"Total receipts:  {result['valid_count'] + result['invalid_count']}")
    print(f"Valid receipts:  {result['valid_count']}")
    print(f"Invalid:         {result['invalid_count']}")
    print(f"Broken chains:   {len(result['broken_chains'])}")
    print(f"Merkle root:     {result['merkle_root'][:32]}...")
    print("=" * 60)

    # Detailed output
    if args.verbose and result.get('details'):
        print("\nDetailed verification:")
        for detail in result['details']:
            status = "OK" if detail.get('valid') else "FAIL"
            line = detail.get('line', '?')
            rtype = detail.get('receipt_type', 'unknown')
            error = detail.get('error', '')
            print(f"  Line {line}: [{status}] {rtype} {error}")

    # Broken chains detail
    if result['broken_chains']:
        print("\nBroken chain details:")
        for chain in result['broken_chains']:
            print(f"  Line {chain['line']}:")
            print(f"    Expected: {chain['expected_hash'][:32]}...")
            print(f"    Computed: {chain['computed_hash'][:32]}...")

    # Exit status
    if result['invalid_count'] == 0:
        print("\nVERIFICATION PASSED")
        sys.exit(0)
    else:
        print(f"\nVERIFICATION FAILED: {result['invalid_count']} invalid receipts")
        sys.exit(1)


if __name__ == "__main__":
    main()
