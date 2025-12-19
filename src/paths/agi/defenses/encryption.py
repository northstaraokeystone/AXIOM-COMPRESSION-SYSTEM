"""Fractal encryption defense integration for AGI path.

This module integrates fractal encryption defenses including:
- Side-channel attack resilience
- Model inversion attack resilience
- Fractal self-similarity-based privacy

Source: AXIOM scalable paths architecture - AGI encryption defenses
"""

from typing import Dict, Any, List, Optional

from ...base import emit_path_receipt, load_path_spec

# Import AGI constants from core module
from ..core import AGI_TENANT_ID, ALIGNMENT_METRIC, compute_alignment


# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA = {
    "fractal_encrypt_integrate": "AGI fractal encryption integration results",
    "side_channel_stress": "AGI side-channel resilience stress test results",
    "model_inversion_stress": "AGI model inversion resilience stress test results",
    "fractal_alignment": "AGI comprehensive alignment with fractal encryption",
}


# === FRACTAL ENCRYPTION INTEGRATION ===


def integrate_fractal_encrypt(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wire fractal encryption defense to AGI path.

    Args:
        config: Optional fractal encrypt config override

    Returns:
        Dict with fractal encryption integration results

    Receipt: agi_fractal_encrypt_integrate
    """
    # Import fractal encrypt module
    from ....fractal_encrypt_audit import (
        load_encrypt_config,
        run_fractal_encrypt_audit,
        SIDE_CHANNEL_RESILIENCE,
        MODEL_INVERSION_RESILIENCE,
    )

    if config is None:
        config = load_encrypt_config()

    # Run encryption audit
    audit = run_fractal_encrypt_audit(["side_channel", "model_inversion"])

    result = {
        "integrated": True,
        "encrypt_config": config,
        "audit_results": {
            "side_channel_resilience": audit["results"]
            .get("side_channel", {})
            .get("resilience", 0),
            "model_inversion_resilience": audit["results"]
            .get("model_inversion", {})
            .get("resilience", 0),
            "all_passed": audit["all_passed"],
        },
        "thresholds": {
            "side_channel": SIDE_CHANNEL_RESILIENCE,
            "model_inversion": MODEL_INVERSION_RESILIENCE,
        },
        "defense_mechanisms": config.get("defense_mechanisms", []),
        "key_insight": "Fractal self-similarity makes pattern extraction exponentially harder",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "fractal_encrypt_integrate", result)
    return result


def run_side_channel_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run side-channel resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_side_channel_stress
    """
    # Import fractal encrypt module
    from ....fractal_encrypt_audit import (
        test_side_channel_resilience,
        SIDE_CHANNEL_RESILIENCE,
    )

    resilience = test_side_channel_resilience(iterations)

    result = {
        "stress_test_type": "side_channel",
        "iterations": iterations,
        "resilience": resilience,
        "target": SIDE_CHANNEL_RESILIENCE,
        "passed": resilience >= SIDE_CHANNEL_RESILIENCE,
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "side_channel_stress", result)
    return result


def run_model_inversion_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run model inversion resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_model_inversion_stress
    """
    # Import fractal encrypt module
    from ....fractal_encrypt_audit import (
        test_model_inversion_resilience,
        MODEL_INVERSION_RESILIENCE,
    )

    resilience = test_model_inversion_resilience(None, iterations)

    result = {
        "stress_test_type": "model_inversion",
        "iterations": iterations,
        "resilience": resilience,
        "target": MODEL_INVERSION_RESILIENCE,
        "passed": resilience >= MODEL_INVERSION_RESILIENCE,
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "model_inversion_stress", result)
    return result


def compute_fractal_alignment(
    receipts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute alignment combining compression, adversarial, expanded, and fractal audits.

    Args:
        receipts: Optional system receipts for compression alignment

    Returns:
        Dict with comprehensive alignment metrics including fractal encryption

    Receipt: agi_fractal_alignment
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
    from ....fractal_encrypt_audit import (
        test_side_channel_resilience,
        test_model_inversion_resilience,
        SIDE_CHANNEL_RESILIENCE,
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

    # Run fractal encryption tests
    side_channel = test_side_channel_resilience(50)
    model_inversion = test_model_inversion_resilience(None, 50)
    fractal_resilience = (side_channel + model_inversion) / 2

    # Combined alignment (weighted)
    # Compression: 15%, Basic adversarial: 20%, Expanded: 35%, Fractal: 30%
    combined = (
        compression_alignment * 0.15
        + basic_adversarial * 0.20
        + expanded_recovery * 0.35
        + fractal_resilience * 0.30
    )

    result = {
        "compression_alignment": round(compression_alignment, 4),
        "basic_adversarial_alignment": round(basic_adversarial, 4),
        "expanded_alignment": round(expanded_recovery, 4),
        "fractal_resilience": round(fractal_resilience, 4),
        "side_channel_resilience": side_channel,
        "model_inversion_resilience": model_inversion,
        "combined_alignment": round(combined, 4),
        "weights": {
            "compression": 0.15,
            "basic_adversarial": 0.20,
            "expanded": 0.35,
            "fractal": 0.30,
        },
        "thresholds": {
            "basic": BASIC_THRESHOLD,
            "expanded": EXPANDED_RECOVERY_THRESHOLD,
            "fractal": SIDE_CHANNEL_RESILIENCE,
        },
        "is_aligned": combined >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Comprehensive alignment via compression + adversarial + expanded + fractal encryption",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "fractal_alignment", result)
    return result


__all__ = [
    "RECEIPT_SCHEMA",
    "integrate_fractal_encrypt",
    "run_side_channel_stress_test",
    "run_model_inversion_stress_test",
    "compute_fractal_alignment",
]
