"""Adversarial audit defense integration for AGI path.

This module integrates adversarial audit defenses including:
- Basic adversarial noise auditing
- Expanded audits (injection/poisoning attacks)
- Combined alignment metrics

Source: SpaceProof scalable paths architecture - AGI adversarial defenses
"""

from typing import Dict, Any, List, Optional

from ...base import emit_path_receipt

# Import AGI constants from core module
from ..core import AGI_TENANT_ID, ALIGNMENT_METRIC, compute_alignment


# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA = {
    "adversarial_integrate": "AGI adversarial audit integration results",
    "alignment_stress": "AGI alignment stress test results",
    "combined_alignment": "AGI combined compression + adversarial alignment",
    "expanded_integrate": "AGI expanded audit integration results",
    "injection_stress": "AGI injection attack stress test results",
    "poisoning_stress": "AGI poisoning attack stress test results",
    "expanded_alignment": "AGI comprehensive alignment metrics",
}


# === ADVERSARIAL AUDIT INTEGRATION ===


def integrate_adversarial(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire adversarial audits to AGI path.

    Args:
        config: Optional adversarial config override

    Returns:
        Dict with adversarial integration results

    Receipt: agi_adversarial_integrate
    """
    # Import adversarial module
    from ....adversarial_audit import (
        load_adversarial_config,
        run_audit,
        RECOVERY_THRESHOLD,
    )

    if config is None:
        config = load_adversarial_config()

    # Run audit
    audit = run_audit(
        noise_level=config["noise_level"], iterations=config["test_iterations"]
    )

    result = {
        "integrated": True,
        "adversarial_config": config,
        "audit_results": {
            "avg_recovery": audit["avg_recovery"],
            "alignment_rate": audit["alignment_rate"],
            "overall_classification": audit["overall_classification"],
            "recovery_passed": audit["recovery_passed"],
        },
        "recovery_threshold": RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Compression as alignment - recovery indicates coherent behavior",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "adversarial_integrate", result)
    return result


def run_alignment_stress_test(noise_level: float = 0.05) -> Dict[str, Any]:
    """Run adversarial alignment stress test.

    Args:
        noise_level: Noise level for testing

    Returns:
        Dict with stress test results

    Receipt: agi_alignment_stress
    """
    # Import adversarial module
    from ....adversarial_audit import (
        run_stress_test,
        RECOVERY_THRESHOLD,
    )

    # Run stress test
    stress = run_stress_test(
        noise_levels=[0.01, 0.03, 0.05, 0.10, noise_level], iterations_per_level=50
    )

    # Compute alignment metrics
    passed_levels = [r for r in stress["results_by_level"] if r["passed"]]
    failed_levels = [r for r in stress["results_by_level"] if not r["passed"]]

    result = {
        "stress_test_complete": True,
        "noise_levels_tested": stress["noise_levels_tested"],
        "critical_noise_level": stress["critical_noise_level"],
        "stress_passed": stress["stress_passed"],
        "passed_levels": len(passed_levels),
        "failed_levels": len(failed_levels),
        "recovery_threshold": RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "results_summary": stress["results_by_level"],
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "alignment_stress", result)
    return result


def compute_adversarial_alignment(
    receipts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute alignment combining compression metric and adversarial audit.

    Args:
        receipts: Optional system receipts for compression alignment

    Returns:
        Dict with combined alignment metrics

    Receipt: agi_combined_alignment
    """
    # Import adversarial module
    from ....adversarial_audit import (
        run_audit,
        RECOVERY_THRESHOLD,
    )

    # Compute compression alignment
    if receipts is None:
        receipts = []
    compression_alignment = compute_alignment(receipts)

    # Run adversarial audit
    audit = run_audit(noise_level=0.05, iterations=50)
    adversarial_alignment = audit["avg_recovery"]

    # Combined alignment (weighted average)
    # Weight adversarial higher since it's active testing
    combined = (compression_alignment * 0.4) + (adversarial_alignment * 0.6)

    result = {
        "compression_alignment": round(compression_alignment, 4),
        "adversarial_alignment": round(adversarial_alignment, 4),
        "combined_alignment": round(combined, 4),
        "compression_weight": 0.4,
        "adversarial_weight": 0.6,
        "recovery_threshold": RECOVERY_THRESHOLD,
        "is_aligned": combined >= RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Combined compression + adversarial = robust alignment",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "combined_alignment", result)
    return result


# === EXPANDED AUDIT INTEGRATION ===


def integrate_expanded_audits(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wire expanded audits (injection/poisoning) to AGI path.

    Args:
        config: Optional expanded audit config override

    Returns:
        Dict with expanded audit integration results

    Receipt: agi_expanded_integrate
    """
    # Import expanded audit module
    from ....agi_audit_expanded import (
        load_expanded_audit_config,
        run_expanded_audit,
        EXPANDED_RECOVERY_THRESHOLD,
    )

    if config is None:
        config = load_expanded_audit_config()

    # Run expanded audit
    audit = run_expanded_audit(attack_type="all", iterations=config["test_iterations"])

    result = {
        "integrated": True,
        "expanded_config": config,
        "audit_results": {
            "avg_recovery": audit["avg_recovery"],
            "recovery_rate": audit["recovery_rate"],
            "injection_recovery": audit["injection_recovery"],
            "poisoning_recovery": audit["poisoning_recovery"],
            "overall_classification": audit["overall_classification"],
            "recovery_passed": audit["recovery_passed"],
        },
        "recovery_threshold": EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Hybrid RL defenses adapt to injection/poisoning patterns",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "expanded_integrate", result)
    return result


def run_injection_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run injection attack stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with injection stress test results

    Receipt: agi_injection_stress
    """
    # Import expanded audit module
    from ....agi_audit_expanded import (
        run_expanded_audit,
        EXPANDED_RECOVERY_THRESHOLD,
    )

    # Run injection-only audit
    audit = run_expanded_audit(attack_type="injection", iterations=iterations)

    result = {
        "stress_test_type": "injection",
        "iterations": audit["iterations"],
        "avg_recovery": audit["avg_recovery"],
        "recovery_rate": audit["recovery_rate"],
        "injection_recovery": audit["injection_recovery"],
        "recovered_count": audit["recovered_count"],
        "failed_count": audit["failed_count"],
        "recovery_threshold": EXPANDED_RECOVERY_THRESHOLD,
        "stress_passed": audit["recovery_passed"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "injection_stress", result)
    return result


def run_poisoning_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run poisoning attack stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with poisoning stress test results

    Receipt: agi_poisoning_stress
    """
    # Import expanded audit module
    from ....agi_audit_expanded import (
        run_expanded_audit,
        EXPANDED_RECOVERY_THRESHOLD,
    )

    # Run poisoning-only audit
    audit = run_expanded_audit(attack_type="poisoning", iterations=iterations)

    result = {
        "stress_test_type": "poisoning",
        "iterations": audit["iterations"],
        "avg_recovery": audit["avg_recovery"],
        "recovery_rate": audit["recovery_rate"],
        "poisoning_recovery": audit["poisoning_recovery"],
        "recovered_count": audit["recovered_count"],
        "failed_count": audit["failed_count"],
        "recovery_threshold": EXPANDED_RECOVERY_THRESHOLD,
        "stress_passed": audit["recovery_passed"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "poisoning_stress", result)
    return result


def compute_expanded_alignment(
    receipts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute alignment combining compression, adversarial, and expanded audits.

    Args:
        receipts: Optional system receipts for compression alignment

    Returns:
        Dict with comprehensive alignment metrics

    Receipt: agi_expanded_alignment
    """
    # Import modules
    from ....adversarial_audit import (
        run_audit as run_basic_audit,
        RECOVERY_THRESHOLD as BASIC_THRESHOLD,
    )
    from ....agi_audit_expanded import (
        run_expanded_audit,
        EXPANDED_RECOVERY_THRESHOLD,
    )

    # Compute compression alignment
    if receipts is None:
        receipts = []
    compression_alignment = compute_alignment(receipts)

    # Run basic adversarial audit
    basic_audit = run_basic_audit(noise_level=0.05, iterations=50)
    basic_adversarial = basic_audit["avg_recovery"]

    # Run expanded audit
    expanded_audit = run_expanded_audit(attack_type="all", iterations=50)
    expanded_recovery = expanded_audit["avg_recovery"]

    # Combined alignment (weighted)
    # Compression: 20%, Basic adversarial: 30%, Expanded: 50%
    combined = (
        compression_alignment * 0.2 + basic_adversarial * 0.3 + expanded_recovery * 0.5
    )

    result = {
        "compression_alignment": round(compression_alignment, 4),
        "basic_adversarial_alignment": round(basic_adversarial, 4),
        "expanded_alignment": round(expanded_recovery, 4),
        "injection_recovery": expanded_audit["injection_recovery"],
        "poisoning_recovery": expanded_audit["poisoning_recovery"],
        "combined_alignment": round(combined, 4),
        "weights": {
            "compression": 0.2,
            "basic_adversarial": 0.3,
            "expanded": 0.5,
        },
        "thresholds": {
            "basic": BASIC_THRESHOLD,
            "expanded": EXPANDED_RECOVERY_THRESHOLD,
        },
        "is_aligned": combined >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Comprehensive alignment via compression + adversarial + expanded audits",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "expanded_alignment", result)
    return result


__all__ = [
    "RECEIPT_SCHEMA",
    "integrate_adversarial",
    "run_alignment_stress_test",
    "compute_adversarial_alignment",
    "integrate_expanded_audits",
    "run_injection_stress_test",
    "run_poisoning_stress_test",
    "compute_expanded_alignment",
]
