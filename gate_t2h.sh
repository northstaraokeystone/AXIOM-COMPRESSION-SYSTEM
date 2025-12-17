#!/bin/bash
# gate_t2h.sh - RUN THIS OR KILL PROJECT
# T+2h SKELETON gate per CLAUDEME §3

set -e

echo "=== T+2h GATE CHECK ==="

# 1. Check required files exist
[ -f spec.md ] || { echo "FAIL: no spec.md (optional for existing project)"; }
[ -f ledger_schema.json ] || { echo "FAIL: no ledger_schema.json"; exit 1; }
[ -f cli.py ] || { echo "FAIL: no cli.py"; exit 1; }

# 2. Test CLI emits receipt
python cli.py --partition 0.4 --nodes 5 --simulate 2>&1 | grep -q '"receipt_type"' || { echo "FAIL: no receipt"; exit 1; }

# 3. Check partition spec file
[ -f data/node_partition_spec.json ] || { echo "FAIL: no partition spec"; exit 1; }

echo "PASS: T+2h gate"
