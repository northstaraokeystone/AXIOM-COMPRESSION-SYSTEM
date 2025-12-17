#!/bin/bash
# gate_t48h.sh - RUN THIS OR KILL PROJECT
# T+48h HARDENED gate per CLAUDEME §3

set -e

echo "=== T+48h GATE CHECK ==="

# 1. Check for anomaly detection
grep -rq "anomaly" src/partition.py || { echo "FAIL: no anomaly detection in partition.py"; exit 1; }

# 2. Check for stoprule
grep -rq "StopRule" src/partition.py || { echo "FAIL: no StopRule in partition.py"; exit 1; }
grep -rq "stoprule\|StopRule" src/core.py || { echo "FAIL: no stoprules in core.py"; exit 1; }

# 3. Run full test suite
python -m pytest tests/ -q || { echo "FAIL: test suite"; exit 1; }

# 4. Verify partition stress at 1000 iterations completes
python -c "
from src.partition import stress_sweep, NODE_BASELINE, PARTITION_MAX_TEST_PCT, BASE_ALPHA
results = stress_sweep(NODE_BASELINE, (0.0, PARTITION_MAX_TEST_PCT), 1000, BASE_ALPHA, 42)
successes = len([r for r in results if r['quorum_status']])
assert successes == 1000, f'Expected 1000 successes, got {successes}'
print(f'Stress test: {successes}/1000 quorum successes')
" || { echo "FAIL: stress test"; exit 1; }

echo "PASS: T+48h gate — SHIP IT"
