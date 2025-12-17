#!/bin/bash
# gate_t24h.sh - RUN THIS OR KILL PROJECT
# T+24h MVP gate per CLAUDEME §3

set -e

echo "=== T+24h GATE CHECK ==="

# 1. Run pytest on partition tests
python -m pytest tests/test_partition.py -q || { echo "FAIL: partition tests"; exit 1; }

# 2. Check for emit_receipt in source files
grep -rq "emit_receipt" src/partition.py || { echo "FAIL: no receipts in partition.py"; exit 1; }
grep -rq "emit_receipt" src/ledger.py || { echo "FAIL: no receipts in ledger.py"; exit 1; }
grep -rq "emit_receipt" src/reasoning.py || { echo "FAIL: no receipts in reasoning.py"; exit 1; }
grep -rq "emit_receipt" src/mitigation.py || { echo "FAIL: no receipts in mitigation.py"; exit 1; }

# 3. Check for assertions in tests
grep -rq "assert" tests/test_partition.py || { echo "FAIL: no assertions in partition tests"; exit 1; }

# 4. Verify constants
python -c "from src.partition import NODE_BASELINE, QUORUM_THRESHOLD; print(f'Baseline: {NODE_BASELINE}, Quorum: {QUORUM_THRESHOLD}')" || { echo "FAIL: constants check"; exit 1; }

echo "PASS: T+24h gate"
