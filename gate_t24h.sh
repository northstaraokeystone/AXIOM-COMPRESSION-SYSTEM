#!/bin/bash
# gate_t24h.sh - RUN THIS OR KILL PROJECT
# T+24h MVP gate per CLAUDEME §3

set -e

echo "=== T+24h GATE CHECK ==="

# 1. Run pytest on core module tests
python -m pytest test/test_core.py test/test_ledger.py test/test_detect.py -q || { echo "FAIL: core tests"; exit 1; }

# 2. Check for emit_receipt in core source files
grep -rq "emit_receipt" spaceproof/ledger.py || { echo "FAIL: no receipts in ledger.py"; exit 1; }
grep -rq "emit_receipt" spaceproof/detect.py || { echo "FAIL: no receipts in detect.py"; exit 1; }
grep -rq "emit_receipt" spaceproof/anchor.py || { echo "FAIL: no receipts in anchor.py"; exit 1; }

# 3. Check for assertions in tests
grep -rq "assert" test/test_core.py || { echo "FAIL: no assertions in core tests"; exit 1; }
grep -rq "assert" test/test_detect.py || { echo "FAIL: no assertions in detect tests"; exit 1; }

# 4. Verify core imports work
python -c "from spaceproof.core import emit_receipt, dual_hash; print('✓ Core imports OK')" || { echo "FAIL: core imports"; exit 1; }

echo "PASS: T+24h gate"

echo ""
echo "=== HARDWARE PROVENANCE GATES (v3.0) ==="

# 5. Verify hardware detection functions exist
python -c "
from spaceproof.detect import (
    compute_component_entropy,
    detect_counterfeit_signature,
    detect_rework_accumulation,
    compute_supply_chain_compression,
    detect_hardware_fraud,
    HARDWARE_ENTROPY_THRESHOLDS,
    REWORK_THRESHOLDS,
)
print('✓ Hardware detection functions loaded')
" || { echo "FAIL: hardware detection functions"; exit 1; }

# 6. Verify hardware anchor functions exist
python -c "
from spaceproof.anchor import (
    anchor_component_provenance,
    validate_provenance_chain,
    merge_component_chains,
    create_manufacturer_receipt,
    create_rework_receipt,
)
print('✓ Hardware anchor functions loaded')
" || { echo "FAIL: hardware anchor functions"; exit 1; }

# 7. Verify META-LOOP hardware patterns
python -c "
from spaceproof.meta_integration import (
    ESCAPE_VELOCITY,
    HARDWARE_PATTERN_TYPES,
    discover_hardware_patterns,
    run_hardware_meta_loop,
)
assert 'counterfeit_detection' in ESCAPE_VELOCITY
assert ESCAPE_VELOCITY['counterfeit_detection'] == 0.90
assert 'COUNTERFEIT_HUNTER' in HARDWARE_PATTERN_TYPES
print('✓ Hardware escape velocities configured')
" || { echo "FAIL: META-LOOP hardware configuration"; exit 1; }

# 8. Test legitimate component passes
python -c "
from spaceproof.detect import detect_counterfeit_signature
legitimate = {
    'id': 'CAP001',
    'manufacturer': 'Vishay',
    'visual_hash': 'abc123',
    'electrical_hash': 'def456',
    'provenance_chain': [
        {'receipt_type': 'manufacturer', 'manufacturer': 'Vishay'},
        {'receipt_type': 'distributor', 'distributor': 'Digi-Key'},
    ],
    'manufacturer_baseline': {'entropy': 0.28, 'manufacturer': 'Vishay'},
}
result = detect_counterfeit_signature(legitimate, legitimate['manufacturer_baseline'])
assert result.classification == 'legitimate' or result.entropy < 0.70, f'Expected legitimate, got {result.classification}'
print('✓ Legitimate component validated')
" || { echo "FAIL: legitimate component test"; exit 1; }

# 9. Test counterfeit component fails
python -c "
from spaceproof.detect import detect_counterfeit_signature
counterfeit = {
    'id': 'CAP002',
    'manufacturer': 'Unknown',
    'provenance_chain': [],  # Missing chain
}
result = detect_counterfeit_signature(counterfeit, {'entropy': 0.28})
assert result.entropy > 0.70, f'Expected high entropy, got {result.entropy}'
print('✓ Counterfeit component detected')
" || { echo "FAIL: counterfeit component test"; exit 1; }

# 10. Test excessive rework detection
python -c "
from spaceproof.detect import detect_rework_accumulation
rework_history = [
    {'cycle': 1, 'entropy': 0.30},
    {'cycle': 2, 'entropy': 0.32},
    {'cycle': 3, 'entropy': 0.35},
    {'cycle': 4, 'entropy': 0.45},
    {'cycle': 5, 'entropy': 0.52},
]
result = detect_rework_accumulation('IC001', rework_history)
assert result.degradation_detected, 'Expected degradation detected'
assert result.entropy_trend == 'increasing', f'Expected increasing, got {result.entropy_trend}'
print('✓ Excessive rework detected')
" || { echo "FAIL: rework detection test"; exit 1; }

# 11. Verify hardware scenarios loaded
python -c "
from spaceproof.sim.scenarios import HardwareSupplyChainScenario, PowerSupplyPrototypeScenario
print('✓ Hardware scenarios loaded')
" || { echo "FAIL: hardware scenarios"; exit 1; }

echo ""
echo "=== HARDWARE GATES PASSED ==="
