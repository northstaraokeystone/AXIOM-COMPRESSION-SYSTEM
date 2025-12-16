#!/bin/bash
# AXIOM-SYSTEM v2 Gate T+2h: SKELETON
# RUN THIS OR KILL PROJECT

set -e

echo "=== AXIOM-SYSTEM v2 Gate T+2h ==="

[ -f spec.md ]            || { echo "FAIL: no spec"; exit 1; }
[ -f ledger_schema.json ] || { echo "FAIL: no schema"; exit 1; }
[ -f cli.py ]             || { echo "FAIL: no cli"; exit 1; }

# Test CLI emits receipt
cd "$(dirname "$0")"
python cli.py --test 2>&1 | grep -q "PASS" || { echo "FAIL: cli test failed"; exit 1; }

echo "PASS: T+2h gate"
