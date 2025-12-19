"""paths/agi/defenses/audits.py - Expanded Audit Integration.

Functions for expanded audits including injection and poisoning stress tests.
"""

from typing import Any, Dict, List, Optional

from ...base import emit_path_receipt
from ..policy import AGI_TENANT_ID, ALIGNMENT_METRIC, compute_alignment


def integrate_expanded_audits(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire expanded audits (injection/poisoning) to AGI path."""
    from ....agi_audit_expanded import (
        load_expanded_audit_config, run_expanded_audit, EXPANDED_RECOVERY_THRESHOLD,
    )
    if config is None:
        config = load_expanded_audit_config()
    audit = run_expanded_audit(attack_type="all", iterations=config["test_iterations"])
    result = {
        "integrated": True, "expanded_config": config,
        "audit_results": {
            "avg_recovery": audit["avg_recovery"], "recovery_rate": audit["recovery_rate"],
            "injection_recovery": audit["injection_recovery"],
            "poisoning_recovery": audit["poisoning_recovery"],
            "overall_classification": audit["overall_classification"],
            "recovery_passed": audit["recovery_passed"],
        },
        "recovery_threshold": EXPANDED_RECOVERY_THRESHOLD, "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Hybrid RL defenses adapt to injection/poisoning patterns",
        "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "expanded_integrate", result)
    return result


def run_injection_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run injection attack stress test."""
    from ....agi_audit_expanded import run_expanded_audit, EXPANDED_RECOVERY_THRESHOLD
    audit = run_expanded_audit(attack_type="injection", iterations=iterations)
    result = {
        "stress_test_type": "injection", "iterations": audit["iterations"],
        "avg_recovery": audit["avg_recovery"], "recovery_rate": audit["recovery_rate"],
        "injection_recovery": audit["injection_recovery"],
        "recovered_count": audit["recovered_count"], "failed_count": audit["failed_count"],
        "recovery_threshold": EXPANDED_RECOVERY_THRESHOLD, "stress_passed": audit["recovery_passed"],
        "alignment_metric": ALIGNMENT_METRIC, "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "injection_stress", result)
    return result


def run_poisoning_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run poisoning attack stress test."""
    from ....agi_audit_expanded import run_expanded_audit, EXPANDED_RECOVERY_THRESHOLD
    audit = run_expanded_audit(attack_type="poisoning", iterations=iterations)
    result = {
        "stress_test_type": "poisoning", "iterations": audit["iterations"],
        "avg_recovery": audit["avg_recovery"], "recovery_rate": audit["recovery_rate"],
        "poisoning_recovery": audit["poisoning_recovery"],
        "recovered_count": audit["recovered_count"], "failed_count": audit["failed_count"],
        "recovery_threshold": EXPANDED_RECOVERY_THRESHOLD, "stress_passed": audit["recovery_passed"],
        "alignment_metric": ALIGNMENT_METRIC, "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "poisoning_stress", result)
    return result


def compute_expanded_alignment(receipts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Compute alignment combining compression, adversarial, and expanded audits."""
    from ....adversarial_audit import run_audit as run_basic_audit, RECOVERY_THRESHOLD as BASIC_THRESHOLD
    from ....agi_audit_expanded import run_expanded_audit, EXPANDED_RECOVERY_THRESHOLD
    if receipts is None:
        receipts = []
    compression_alignment = compute_alignment(receipts)
    basic_audit = run_basic_audit(noise_level=0.05, iterations=50)
    basic_adversarial = basic_audit["avg_recovery"]
    expanded_audit = run_expanded_audit(attack_type="all", iterations=50)
    expanded_recovery = expanded_audit["avg_recovery"]
    combined = compression_alignment * 0.2 + basic_adversarial * 0.3 + expanded_recovery * 0.5
    result = {
        "compression_alignment": round(compression_alignment, 4),
        "basic_adversarial_alignment": round(basic_adversarial, 4),
        "expanded_alignment": round(expanded_recovery, 4),
        "injection_recovery": expanded_audit["injection_recovery"],
        "poisoning_recovery": expanded_audit["poisoning_recovery"],
        "combined_alignment": round(combined, 4),
        "weights": {"compression": 0.2, "basic_adversarial": 0.3, "expanded": 0.5},
        "thresholds": {"basic": BASIC_THRESHOLD, "expanded": EXPANDED_RECOVERY_THRESHOLD},
        "is_aligned": combined >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Comprehensive alignment via compression + adversarial + expanded audits",
        "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "expanded_alignment", result)
    return result


__all__ = [
    "integrate_expanded_audits", "run_injection_stress_test",
    "run_poisoning_stress_test", "compute_expanded_alignment",
]
