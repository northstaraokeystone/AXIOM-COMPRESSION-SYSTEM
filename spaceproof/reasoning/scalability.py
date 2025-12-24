"""reasoning/scalability.py - Multi-Scale Scalability Gate Functions.

Functions for scalability validation and 3.1 push readiness.
"""

from typing import Any, Dict
from datetime import datetime
import json

from ..core import emit_receipt, StopRule, dual_hash


def enforce_scalability_gate(sweep_results: Dict[str, Any]) -> bool:
    """Enforce scalability gate before 3.1 push.

    Blocks 3.1 push if:
    - alpha at 10^9 < 3.06
    - instability > 0 at any scale
    - degradation > 1%

    Args:
        sweep_results: Output from multi_scale_sweep.run_scale_sweep()

    Returns:
        True if gate passes, raises StopRule if fails

    Receipt: scalability_gate_enforcement

    Gate Logic:
        IF scalability_gate_passed AND alpha_at_10e9 >= 3.06 AND instability == 0.00:
            ready_for_31_push = True
            emit scalability_gate_receipt(passed=True)
        ELSE:
            ready_for_31_push = False
            emit scalability_gate_receipt(passed=False)
            raise StopRule("Scalability gate failed")
    """
    from ..multi_scale_sweep import (
        scalability_gate,
        SCALABILITY_GATE_THRESHOLD as GATE_THRESHOLD,
    )

    # Run gate check
    gate_result = scalability_gate(sweep_results)

    gate_passed = gate_result.get("gate_passed", False)
    alpha_at_10e9 = gate_result.get("alpha_at_10e9", 0)
    instability_at_10e9 = gate_result.get("instability_at_10e9", 1.0)

    enforcement_result = {
        "gate_passed": gate_passed,
        "alpha_at_10e9": alpha_at_10e9,
        "instability_at_10e9": instability_at_10e9,
        "threshold": GATE_THRESHOLD,
        "ready_for_31_push": gate_passed,
    }

    emit_receipt(
        "scalability_gate_enforcement",
        {
            "receipt_type": "scalability_gate_enforcement",
            "tenant_id": "spaceproof-reasoning",
            "ts": datetime.utcnow().isoformat() + "Z",
            **enforcement_result,
            "payload_hash": dual_hash(json.dumps(enforcement_result, sort_keys=True)),
        },
    )

    if not gate_passed:
        raise StopRule(
            f"Scalability gate failed: alpha={alpha_at_10e9}, "
            f"instability={instability_at_10e9}, threshold={GATE_THRESHOLD}"
        )

    return True


def get_31_push_readiness() -> Dict[str, Any]:
    """Return status of all prerequisites for 3.1 push.

    Checks:
    1. Multi-scale validation completed
    2. Scalability gate passed
    3. Alpha at 10^9 >= 3.06
    4. Instability == 0.00 at all scales
    5. Degradation < 1%

    Returns:
        Dict with readiness status and details

    Receipt: push_31_readiness
    """
    from ..multi_scale_sweep import (
        run_scale_sweep,
        check_degradation,
        scalability_gate,
        TREE_SCALES,
        ALPHA_BASELINE,
        SCALABILITY_GATE_THRESHOLD as GATE_THRESHOLD,
    )

    # Run validation
    try:
        sweep_results = run_scale_sweep(TREE_SCALES)
        sweep_completed = True
    except Exception as e:
        sweep_completed = False
        sweep_results = {"error": str(e)}

    # Check degradation
    try:
        degradation = check_degradation(sweep_results)
        degradation_ok = degradation.get("degradation_acceptable", False)
        degradation_pct = degradation.get("degradation_pct", 1.0)
    except Exception:
        degradation_ok = False
        degradation_pct = None

    # Check gate
    try:
        gate_result = scalability_gate(sweep_results)
        gate_passed = gate_result.get("gate_passed", False)
        alpha_at_10e9 = gate_result.get("alpha_at_10e9", 0)
        instability_at_10e9 = gate_result.get("instability_at_10e9", 1.0)
    except Exception:
        gate_passed = False
        alpha_at_10e9 = None
        instability_at_10e9 = None

    # Determine overall readiness
    ready = (
        sweep_completed
        and degradation_ok
        and gate_passed
        and (alpha_at_10e9 is not None and alpha_at_10e9 >= GATE_THRESHOLD)
        and (instability_at_10e9 is not None and instability_at_10e9 == 0.00)
    )

    result = {
        "ready_for_31_push": ready,
        "prerequisites": {
            "multi_scale_sweep_completed": sweep_completed,
            "degradation_under_1pct": degradation_ok,
            "degradation_pct": degradation_pct,
            "scalability_gate_passed": gate_passed,
            "alpha_at_10e9_ge_306": alpha_at_10e9 is not None
            and alpha_at_10e9 >= GATE_THRESHOLD,
            "alpha_at_10e9": alpha_at_10e9,
            "instability_zero": instability_at_10e9 is not None
            and instability_at_10e9 == 0.00,
            "instability_at_10e9": instability_at_10e9,
        },
        "gate_threshold": GATE_THRESHOLD,
        "alpha_baseline": ALPHA_BASELINE,
        "scales_tested": TREE_SCALES,
    }

    emit_receipt(
        "push_31_readiness",
        {
            "receipt_type": "push_31_readiness",
            "tenant_id": "spaceproof-reasoning",
            "ts": datetime.utcnow().isoformat() + "Z",
            "ready_for_31_push": ready,
            "gate_passed": gate_passed,
            "alpha_at_10e9": alpha_at_10e9,
            "degradation_ok": degradation_ok,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


__all__ = [
    "enforce_scalability_gate",
    "get_31_push_readiness",
]
