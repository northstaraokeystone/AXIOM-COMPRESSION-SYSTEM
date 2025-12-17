#!/bin/bash
# AXIOM-COMPRESSION-SYSTEM Local Validation Gate
# This script MUST pass before any git push
# Run: ./validate.sh

set -e  # Exit on first failure

echo "========================================="
echo "AXIOM VALIDATION GATE"
echo "========================================="
echo ""

# Test 1: Core imports
echo "[1/4] Testing core module imports..."
python -c "import sys; sys.path.insert(0,'.'); from src import core; print('✓ imports: OK')" || {
    echo "✗ FAIL: Core imports failed"
    exit 1
}
echo ""

# Test 2: CLI responds
echo "[2/4] Testing CLI --help..."
python cli.py --help > /dev/null 2>&1 && echo "✓ cli: OK" || {
    echo "✗ FAIL: CLI does not respond"
    python cli.py --help 2>&1
    exit 1
}
echo ""

# Test 3: Additional module imports
echo "[3/4] Testing additional module imports..."
python -c "from cli import base; print('✓ cli.base: OK')" || {
    echo "✗ FAIL: CLI base imports failed"
    exit 1
}
echo ""

# Test 4: Run tests
echo "[4/4] Running test suite..."
python -m pytest tests/ -x -q || {
    echo "✗ FAIL: Tests failed"
    exit 1
}
echo ""

echo "========================================="
echo "✓ ALL VALIDATION GATES PASSED"
echo "========================================="
echo ""
echo "Safe to push to remote."
exit 0
