#!/bin/bash
# AXIOM-SYSTEM v2 Gate T+24h: MVP
# RUN THIS OR KILL PROJECT

set -e

echo "=== AXIOM-SYSTEM v2 Gate T+24h ==="

cd "$(dirname "$0")"

# Run pytest if tests exist
if [ -d tests ] && [ -n "$(ls tests/test_*.py 2>/dev/null)" ]; then
    python -m pytest tests/ -q || { echo "FAIL: tests"; exit 1; }
fi

# Check emit_receipt in src
grep -rq "emit_receipt" src/*.py || { echo "FAIL: no receipts in src"; exit 1; }

# Check for assertions in test files
if [ -d tests ] && [ -n "$(ls tests/test_*.py 2>/dev/null)" ]; then
    grep -rq "assert" tests/*.py || { echo "FAIL: no assertions"; exit 1; }
fi

# Verify constants
python -c "
from src.entropy import NEURALINK_MULTIPLIER, MDL_BETA, SOVEREIGNTY_THRESHOLD_NEURALINK
assert NEURALINK_MULTIPLIER == 1e5, 'NEURALINK_MULTIPLIER wrong'
assert MDL_BETA == 0.09, 'MDL_BETA wrong'
assert SOVEREIGNTY_THRESHOLD_NEURALINK == 5, 'Threshold wrong'
print('Constants verified')
"

# Test simulation runs
python -c "
from src.system import run_simulation, SystemConfig
cfg = SystemConfig(duration_sols=100, emit_receipts=False)
result = run_simulation(cfg)
assert result.sol == 100, f'Expected sol 100, got {result.sol}'
print(f'Simulation: {result.sol} sols, sovereign={result.system_sovereign}')
"

echo "PASS: T+24h gate"
