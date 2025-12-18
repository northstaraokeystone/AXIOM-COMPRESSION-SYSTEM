"""Post-AGI ethics modeling via fractal policies.

Evolution path: stub -> policy -> evaluate -> autonomous_audit

KEY INSIGHT: "Audit trail IS alignment"
- If system cannot prove what it did, it did not do it
- Compression as alignment: More compressible = more coherent = more aligned
- Fractal policies are self-similar at all scales

ETHICS DIMENSIONS:
1. Autonomy - Respect for agency
2. Beneficence - Do good
3. Non-maleficence - Do no harm
4. Justice - Fair distribution

Source: AXIOM scalable paths architecture - AGI ethics modeling
"""

import json
import math
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..base import emit_path_receipt, load_path_spec


# === CONSTANTS ===

AGI_TENANT_ID = "axiom-agi"
"""Tenant ID for AGI path receipts."""

POLICY_DEPTH_DEFAULT = 3
"""Default fractal policy depth."""

ETHICS_DIMENSIONS = ["autonomy", "beneficence", "non_maleficence", "justice"]
"""Core ethics dimensions."""

ALIGNMENT_METRIC = "compression_as_alignment"
"""Alignment metric: compressibility indicates coherence."""

AUDIT_REQUIREMENT = "receipts_native"
"""Audit requirement: All actions emit receipts."""


# === STUB STATUS ===


def stub_status() -> Dict[str, Any]:
    """Return current stub status.

    Returns:
        Dict with stub readiness info

    Receipt: agi_status
    """
    spec = load_path_spec("agi")

    status = {
        "ready": True,
        "stage": "stub",
        "version": spec.get("version", "0.1.0"),
        "evolution_path": ["stub", "policy", "evaluate", "autonomous_audit"],
        "current_capabilities": ["stub_status", "fractal_policy", "compute_alignment"],
        "pending_capabilities": [
            "evaluate_ethics",
            "audit_decision",
            "autonomous_audit",
        ],
        "key_insight": spec.get("key_insight", "Audit trail IS alignment"),
        "config": spec.get("config", {}),
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "status", status)
    return status


# === FRACTAL POLICY ===


def fractal_policy(depth: int = POLICY_DEPTH_DEFAULT) -> Dict[str, Any]:
    """Generate self-similar policy at given depth.

    Fractal policies are self-similar at all scales:
    - Depth 1: Base ethics dimensions
    - Depth 2: Each dimension has sub-dimensions
    - Depth 3: Sub-sub-dimensions (practical rules)

    Args:
        depth: Policy depth (1-5)

    Returns:
        Dict with policy structure

    Receipt: agi_policy
    """
    depth = max(1, min(depth, 5))

    def build_policy_tree(dim: str, current_depth: int) -> Dict[str, Any]:
        """Recursively build policy tree."""
        if current_depth == 0:
            return {"weight": 1.0, "leaf": True}

        # Generate sub-dimensions based on ethics dimension
        sub_dims = get_sub_dimensions(dim, current_depth)
        children = {sub: build_policy_tree(sub, current_depth - 1) for sub in sub_dims}

        return {
            "weight": 1.0 / len(sub_dims) if sub_dims else 1.0,
            "leaf": False,
            "children": children,
        }

    policy = {
        "depth": depth,
        "dimensions": ETHICS_DIMENSIONS,
        "tree": {dim: build_policy_tree(dim, depth - 1) for dim in ETHICS_DIMENSIONS},
        "self_similar": True,
        "fractal_scaling": True,
        "tenant_id": AGI_TENANT_ID,
    }

    # Compute policy complexity (number of nodes)
    policy["complexity"] = compute_tree_complexity(policy["tree"])

    emit_path_receipt(
        "agi",
        "policy",
        {
            "depth": depth,
            "dimensions": ETHICS_DIMENSIONS,
            "complexity": policy["complexity"],
            "self_similar": True,
        },
    )

    return policy


def get_sub_dimensions(dim: str, depth: int) -> List[str]:
    """Get sub-dimensions for ethics dimension.

    Args:
        dim: Parent dimension
        depth: Current depth level

    Returns:
        List of sub-dimension names
    """
    sub_dims = {
        "autonomy": ["informed_consent", "self_determination", "privacy"],
        "beneficence": ["active_good", "prevent_harm", "maximize_welfare"],
        "non_maleficence": ["no_direct_harm", "minimize_risk", "precaution"],
        "justice": ["fairness", "equal_treatment", "proportionality"],
        # Sub-sub dimensions
        "informed_consent": ["disclosure", "comprehension", "voluntariness"],
        "self_determination": ["choice", "agency", "independence"],
        "privacy": ["data_protection", "anonymity", "control"],
        "active_good": ["benefit", "assistance", "improvement"],
        "prevent_harm": ["protection", "warning", "intervention"],
        "maximize_welfare": ["utility", "wellbeing", "flourishing"],
        "no_direct_harm": ["safety", "integrity", "respect"],
        "minimize_risk": ["assessment", "mitigation", "monitoring"],
        "precaution": ["uncertainty", "reversibility", "fallback"],
        "fairness": ["impartiality", "consistency", "transparency"],
        "equal_treatment": ["non_discrimination", "access", "opportunity"],
        "proportionality": ["necessity", "balance", "minimal_impact"],
    }

    return sub_dims.get(dim, [f"{dim}_sub_{i}" for i in range(1, 4)])


def compute_tree_complexity(tree: Dict[str, Any]) -> int:
    """Compute number of nodes in policy tree.

    Args:
        tree: Policy tree structure

    Returns:
        Total node count
    """
    count = 0
    for dim, node in tree.items():
        count += 1
        if not node.get("leaf", True) and "children" in node:
            count += compute_tree_complexity(node["children"])
    return count


# === ETHICS EVALUATION (STUB) ===


def evaluate_ethics(
    action: Dict[str, Any], policy: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate action against ethics policy.

    STUB: Returns placeholder evaluation.
    FULL: Will traverse policy tree and compute scores.

    Args:
        action: Action to evaluate
        policy: Policy to evaluate against (default: depth=3)

    Returns:
        Dict with evaluation results

    Receipt: agi_ethics
    """
    if policy is None:
        policy = fractal_policy(depth=POLICY_DEPTH_DEFAULT)

    # Stub: Return placeholder scores
    dimension_scores = {dim: 0.8 for dim in ETHICS_DIMENSIONS}

    result = {
        "stub_mode": True,
        "action": action,
        "policy_depth": policy.get("depth", POLICY_DEPTH_DEFAULT),
        "dimension_scores": dimension_scores,
        "aggregate_score": sum(dimension_scores.values()) / len(dimension_scores),
        "passes_threshold": True,
        "threshold": 0.7,
        "audit_receipt_id": f"agi_ethics_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "ethics", result)
    return result


# === ALIGNMENT COMPUTATION ===


def compute_alignment(system_receipts: List[Dict[str, Any]]) -> float:
    """Compute alignment via compression metric.

    KEY INSIGHT: "Compression as alignment"
    - More compressible = more coherent
    - More coherent = more aligned
    - Alignment = compressibility ratio

    Args:
        system_receipts: List of system receipts

    Returns:
        Alignment score (0.0 to 1.0)

    Receipt: agi_alignment
    """
    if not system_receipts:
        return 0.0

    # Compute raw size (uncompressed)
    raw_data = json.dumps(system_receipts, sort_keys=True)
    raw_size = len(raw_data)

    # Compute "compressed" size (deduplicated patterns)
    # Stub: Use character frequency as proxy for compressibility
    char_freq = {}
    for c in raw_data:
        char_freq[c] = char_freq.get(c, 0) + 1

    # Shannon entropy as compressibility proxy
    total_chars = len(raw_data)
    entropy = 0.0
    for count in char_freq.values():
        p = count / total_chars
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalize entropy (max entropy = log2(unique_chars))
    max_entropy = math.log2(len(char_freq)) if char_freq else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Alignment = 1 - normalized_entropy (lower entropy = more aligned)
    alignment = 1 - normalized_entropy
    alignment = max(0.0, min(1.0, alignment))

    result = {
        "receipts_count": len(system_receipts),
        "raw_size": raw_size,
        "entropy": round(entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "alignment_score": round(alignment, 4),
        "metric": ALIGNMENT_METRIC,
        "interpretation": "Higher alignment = more coherent behavior pattern",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "alignment", result)
    return alignment


# === AUDIT (STUB) ===


def audit_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Receipt-native audit of decision.

    STUB: Returns audit placeholder.
    FULL: Will create complete audit trail.

    KEY INSIGHT: "If system cannot prove what it did, it did not do it"

    Args:
        decision: Decision to audit

    Returns:
        Dict with audit results

    Receipt: agi_audit
    """
    audit_id = f"audit_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    result = {
        "stub_mode": True,
        "audit_id": audit_id,
        "decision": decision,
        "audit_trail": [
            {"step": 1, "action": "decision_received", "receipt": True},
            {"step": 2, "action": "policy_lookup", "receipt": True},
            {"step": 3, "action": "ethics_eval", "receipt": True},
            {"step": 4, "action": "decision_executed", "receipt": True},
        ],
        "complete_trail": True,
        "verifiable": True,
        "key_insight": "If system cannot prove what it did, it did not do it",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "audit", result)
    return result


# === PATH INFO ===


def get_agi_info() -> Dict[str, Any]:
    """Get AGI path configuration and status.

    Returns:
        Dict with path info

    Receipt: agi_info
    """
    spec = load_path_spec("agi")

    info = {
        "path": "agi",
        "version": spec.get("version", "0.1.0"),
        "status": spec.get("status", "stub"),
        "description": spec.get("description", ""),
        "key_insight": spec.get("key_insight", ""),
        "ethics_dimensions": ETHICS_DIMENSIONS,
        "alignment_metric": ALIGNMENT_METRIC,
        "config": spec.get("config", {}),
        "dependencies": spec.get("dependencies", []),
        "receipts": spec.get("receipts", []),
        "evolution": spec.get("evolution", {}),
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "info", info)
    return info


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
    from ...adversarial_audit import (
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
    from ...adversarial_audit import (
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
    from ...adversarial_audit import (
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
    from ...agi_audit_expanded import (
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
    from ...agi_audit_expanded import (
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
    from ...agi_audit_expanded import (
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
    from ...adversarial_audit import (
        run_audit as run_basic_audit,
        RECOVERY_THRESHOLD as BASIC_THRESHOLD,
    )
    from ...agi_audit_expanded import (
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
    from ...fractal_encrypt_audit import (
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
    from ...fractal_encrypt_audit import (
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
    from ...fractal_encrypt_audit import (
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
    from ...adversarial_audit import (
        run_audit as run_basic_audit,
        RECOVERY_THRESHOLD as BASIC_THRESHOLD,
    )
    from ...agi_audit_expanded import (
        run_expanded_audit,
        EXPANDED_RECOVERY_THRESHOLD,
    )
    from ...fractal_encrypt_audit import (
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
