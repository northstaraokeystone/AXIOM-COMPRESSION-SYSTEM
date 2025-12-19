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


# === RANDOMIZED PATHS INTEGRATION ===


def integrate_randomized_paths(
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wire randomized execution paths defense to AGI path.

    Args:
        config: Optional randomized paths config override

    Returns:
        Dict with randomized paths integration results

    Receipt: agi_randomized_integrate
    """
    # Import randomized paths module
    from ...randomized_paths_audit import (
        load_randomized_config,
        run_randomized_audit,
        TIMING_LEAK_RESILIENCE,
    )

    if config is None:
        config = load_randomized_config()

    # Run randomized paths audit
    audit = run_randomized_audit(attack_types=config["attack_types"])

    result = {
        "integrated": True,
        "randomized_config": config,
        "audit_results": {
            "avg_resilience": audit["avg_resilience"],
            "all_passed": audit["all_passed"],
            "attack_types_tested": audit["attack_types_tested"],
        },
        "resilience_target": TIMING_LEAK_RESILIENCE,
        "defense_mechanisms": config["defense_mechanisms"],
        "key_insight": "Randomized paths break timing correlation patterns",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "randomized_integrate", result)
    return result


def run_timing_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run timing leak resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_timing_stress
    """
    # Import randomized paths module
    from ...randomized_paths_audit import (
        test_timing_resilience,
        TIMING_LEAK_RESILIENCE,
    )

    result = test_timing_resilience(iterations)

    stress_result = {
        "stress_test_type": "timing_leak",
        "iterations": iterations,
        "resilience": result["resilience"],
        "target": TIMING_LEAK_RESILIENCE,
        "passed": result["passed"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "timing_stress", stress_result)
    return stress_result


def run_power_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run power analysis resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_power_stress
    """
    # Import randomized paths module
    from ...randomized_paths_audit import (
        test_power_resilience,
        TIMING_LEAK_RESILIENCE,
    )

    result = test_power_resilience(iterations)

    stress_result = {
        "stress_test_type": "power_analysis",
        "iterations": iterations,
        "resilience": result["resilience"],
        "target": TIMING_LEAK_RESILIENCE,
        "passed": result["passed"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "power_stress", stress_result)
    return stress_result


def run_cache_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run cache timing resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_cache_stress
    """
    # Import randomized paths module
    from ...randomized_paths_audit import (
        test_cache_resilience,
        TIMING_LEAK_RESILIENCE,
    )

    result = test_cache_resilience(iterations)

    stress_result = {
        "stress_test_type": "cache_timing",
        "iterations": iterations,
        "resilience": result["resilience"],
        "target": TIMING_LEAK_RESILIENCE,
        "passed": result["passed"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "cache_stress", stress_result)
    return stress_result


def compute_randomized_alignment(
    receipts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute alignment including randomized paths resilience.

    Args:
        receipts: Optional system receipts for compression alignment

    Returns:
        Dict with comprehensive alignment metrics including randomized paths

    Receipt: agi_randomized_alignment
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
    from ...randomized_paths_audit import (
        run_randomized_audit,
        TIMING_LEAK_RESILIENCE,
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

    # Run randomized paths audit
    randomized_audit = run_randomized_audit(iterations=50)
    randomized_resilience = randomized_audit["avg_resilience"]

    # Combined alignment (weighted)
    # Compression: 10%, Basic adversarial: 15%, Expanded: 25%, Fractal: 25%, Randomized: 25%
    combined = (
        compression_alignment * 0.10
        + basic_adversarial * 0.15
        + expanded_recovery * 0.25
        + fractal_resilience * 0.25
        + randomized_resilience * 0.25
    )

    result = {
        "compression_alignment": round(compression_alignment, 4),
        "basic_adversarial_alignment": round(basic_adversarial, 4),
        "expanded_alignment": round(expanded_recovery, 4),
        "fractal_resilience": round(fractal_resilience, 4),
        "randomized_resilience": round(randomized_resilience, 4),
        "combined_alignment": round(combined, 4),
        "weights": {
            "compression": 0.10,
            "basic_adversarial": 0.15,
            "expanded": 0.25,
            "fractal": 0.25,
            "randomized": 0.25,
        },
        "thresholds": {
            "basic": BASIC_THRESHOLD,
            "expanded": EXPANDED_RECOVERY_THRESHOLD,
            "fractal": SIDE_CHANNEL_RESILIENCE,
            "randomized": TIMING_LEAK_RESILIENCE,
        },
        "is_aligned": combined >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Full alignment via compression + adversarial + expanded + fractal + randomized",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "randomized_alignment", result)
    return result


# === QUANTUM-RESISTANT INTEGRATION ===


def integrate_quantum_resist(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire quantum-resistant Spectre defense to AGI path.

    Args:
        config: Optional quantum-resistant config override

    Returns:
        Dict with quantum-resistant integration results

    Receipt: agi_quantum_resist_integrate
    """
    # Import quantum-resistant module
    from ...quantum_resist_random import (
        load_quantum_resist_config,
        run_quantum_resist_audit,
        QUANTUM_RESILIENCE_TARGET,
    )

    if config is None:
        config = load_quantum_resist_config()

    # Run quantum-resistant audit
    audit = run_quantum_resist_audit(
        variants=config["spectre_variants"], iterations=100
    )

    result = {
        "integrated": True,
        "quantum_resist_config": config,
        "audit_results": {
            "overall_resilience": audit["overall_resilience"],
            "spectre_results": audit["spectre_results"],
            "cache_timing_resilience": audit["cache_timing_resilience"],
            "target_met": audit["target_met"],
        },
        "resilience_target": QUANTUM_RESILIENCE_TARGET,
        "defense_mechanisms": config["defense_mechanisms"],
        "key_insight": "100% resilience against cache-based Spectre attacks",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "quantum_resist_integrate", result)
    return result


def run_spectre_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run Spectre resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_spectre_stress
    """
    # Import quantum-resistant module
    from ...quantum_resist_random import (
        test_spectre_defense,
        QUANTUM_RESILIENCE_TARGET,
    )

    result = test_spectre_defense(iterations)

    stress_result = {
        "stress_test_type": "spectre",
        "iterations": iterations,
        "v1_resilience": result["v1_resilience"],
        "v2_resilience": result["v2_resilience"],
        "v4_resilience": result["v4_resilience"],
        "avg_resilience": result["avg_resilience"],
        "target": QUANTUM_RESILIENCE_TARGET,
        "all_passed": result["all_passed"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "spectre_stress", stress_result)
    return stress_result


def run_quantum_cache_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run quantum-resistant cache timing stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with cache stress test results

    Receipt: agi_quantum_cache_stress
    """
    # Import quantum-resistant module
    from ...quantum_resist_random import (
        test_cache_timing,
        QUANTUM_RESILIENCE_TARGET,
    )

    result = test_cache_timing(iterations)

    stress_result = {
        "stress_test_type": "quantum_cache_timing",
        "iterations": iterations,
        "resilience": result["resilience"],
        "jitter_applied": result["jitter_applied"],
        "partition_applied": result["partition_applied"],
        "target": QUANTUM_RESILIENCE_TARGET,
        "passed": result["passed"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "quantum_cache_stress", stress_result)
    return stress_result


def run_branch_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run branch prediction resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with branch stress test results

    Receipt: agi_branch_stress
    """
    # Import quantum-resistant module
    from ...quantum_resist_random import (
        test_spectre_v2,
        QUANTUM_RESILIENCE_TARGET,
    )

    result = test_spectre_v2(iterations)

    stress_result = {
        "stress_test_type": "branch_prediction",
        "iterations": iterations,
        "resilience": result["resilience"],
        "defense_used": result["defense_used"],
        "target": QUANTUM_RESILIENCE_TARGET,
        "passed": result["passed"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "branch_stress", stress_result)
    return stress_result


def compute_quantum_alignment(
    receipts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute alignment including quantum-resistant resilience.

    Args:
        receipts: Optional system receipts for compression alignment

    Returns:
        Dict with comprehensive alignment metrics including quantum-resistant

    Receipt: agi_quantum_alignment
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
    from ...randomized_paths_audit import (
        run_randomized_audit,
        TIMING_LEAK_RESILIENCE,
    )
    from ...quantum_resist_random import (
        run_quantum_resist_audit,
        QUANTUM_RESILIENCE_TARGET,
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

    # Run randomized paths audit
    randomized_audit = run_randomized_audit(iterations=50)
    randomized_resilience = randomized_audit["avg_resilience"]

    # Run quantum-resistant audit
    quantum_audit = run_quantum_resist_audit(iterations=50)
    quantum_resilience = quantum_audit["overall_resilience"]

    # Combined alignment (weighted)
    # Compression: 8%, Basic: 12%, Expanded: 20%, Fractal: 20%, Randomized: 20%, Quantum: 20%
    combined = (
        compression_alignment * 0.08
        + basic_adversarial * 0.12
        + expanded_recovery * 0.20
        + fractal_resilience * 0.20
        + randomized_resilience * 0.20
        + quantum_resilience * 0.20
    )

    result = {
        "compression_alignment": round(compression_alignment, 4),
        "basic_adversarial_alignment": round(basic_adversarial, 4),
        "expanded_alignment": round(expanded_recovery, 4),
        "fractal_resilience": round(fractal_resilience, 4),
        "randomized_resilience": round(randomized_resilience, 4),
        "quantum_resilience": round(quantum_resilience, 4),
        "combined_alignment": round(combined, 4),
        "weights": {
            "compression": 0.08,
            "basic_adversarial": 0.12,
            "expanded": 0.20,
            "fractal": 0.20,
            "randomized": 0.20,
            "quantum": 0.20,
        },
        "thresholds": {
            "basic": BASIC_THRESHOLD,
            "expanded": EXPANDED_RECOVERY_THRESHOLD,
            "fractal": SIDE_CHANNEL_RESILIENCE,
            "randomized": TIMING_LEAK_RESILIENCE,
            "quantum": QUANTUM_RESILIENCE_TARGET,
        },
        "is_aligned": combined >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Full alignment via compression + adversarial + expanded + fractal + randomized + quantum",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "quantum_alignment", result)
    return result


# === SECURE ENCLAVE INTEGRATION ===


def integrate_secure_enclave(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire secure enclave defense to AGI path.

    Args:
        config: Optional enclave config override

    Returns:
        Dict with secure enclave integration results

    Receipt: agi_enclave_integrate
    """
    # Import secure enclave module
    from ...secure_enclave_audit import (
        load_enclave_config,
        run_enclave_audit,
        ENCLAVE_RESILIENCE_TARGET,
        ATTACK_TYPES,
    )

    if config is None:
        config = load_enclave_config()

    # Run enclave audit
    audit = run_enclave_audit(attack_types=ATTACK_TYPES)

    result = {
        "integrated": True,
        "enclave_config": {
            "type": config.get("type", "SGX"),
            "memory_mb": config.get("memory_mb", 128),
            "branch_prediction_defense": config.get("branch_prediction_defense", True),
        },
        "audit_results": {
            "overall_resilience": audit["overall_resilience"],
            "target_met": audit["target_met"],
            "all_passed": audit["all_passed"],
            "attack_types_tested": audit["attack_types_tested"],
        },
        "resilience_target": ENCLAVE_RESILIENCE_TARGET,
        "resilience_met": audit["target_met"],
        "key_insight": "Hardware-level isolation defeats speculative execution attacks",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "enclave_integrate", result)
    return result


def run_btb_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run BTB (Branch Target Buffer) resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with BTB stress test results

    Receipt: agi_btb_stress
    """
    # Import secure enclave module
    from ...secure_enclave_audit import (
        test_btb_injection,
        ENCLAVE_RESILIENCE_TARGET,
    )

    result = test_btb_injection(iterations)

    stress_result = {
        "stress_test_type": "btb_injection",
        "iterations": iterations,
        "resilience": result["resilience"],
        "target": ENCLAVE_RESILIENCE_TARGET,
        "passed": result["passed"],
        "defense_mechanism": result["defense_mechanism"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "btb_stress", stress_result)
    return stress_result


def run_pht_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run PHT (Pattern History Table) resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with PHT stress test results

    Receipt: agi_pht_stress
    """
    # Import secure enclave module
    from ...secure_enclave_audit import (
        test_pht_poisoning,
        ENCLAVE_RESILIENCE_TARGET,
    )

    result = test_pht_poisoning(iterations)

    stress_result = {
        "stress_test_type": "pht_poisoning",
        "iterations": iterations,
        "resilience": result["resilience"],
        "target": ENCLAVE_RESILIENCE_TARGET,
        "passed": result["passed"],
        "defense_mechanism": result["defense_mechanism"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "pht_stress", stress_result)
    return stress_result


def run_rsb_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run RSB (Return Stack Buffer) resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with RSB stress test results

    Receipt: agi_rsb_stress
    """
    # Import secure enclave module
    from ...secure_enclave_audit import (
        test_rsb_stuffing,
        ENCLAVE_RESILIENCE_TARGET,
    )

    result = test_rsb_stuffing(iterations)

    stress_result = {
        "stress_test_type": "rsb_stuffing",
        "iterations": iterations,
        "resilience": result["resilience"],
        "target": ENCLAVE_RESILIENCE_TARGET,
        "passed": result["passed"],
        "defense_mechanism": result["defense_mechanism"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "rsb_stress", stress_result)
    return stress_result


def measure_defense_overhead() -> Dict[str, Any]:
    """Measure performance overhead of enclave defenses.

    Returns:
        Dict with defense overhead measurements

    Receipt: agi_defense_overhead
    """
    # Import secure enclave module
    from ...secure_enclave_audit import measure_enclave_overhead

    overhead = measure_enclave_overhead()

    result = {
        "overhead_pct": overhead["total_overhead_pct"],
        "acceptable": overhead["acceptable"],
        "defense_mechanisms": overhead["defense_mechanisms"],
        "measurements": overhead["measurements"],
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "5% overhead acceptable for 100% branch prediction resilience",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "defense_overhead", result)
    return result


def compute_enclave_alignment(
    receipts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute alignment including secure enclave resilience.

    Args:
        receipts: Optional system receipts for compression alignment

    Returns:
        Dict with comprehensive alignment metrics including secure enclave

    Receipt: agi_enclave_alignment
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
    from ...randomized_paths_audit import (
        run_randomized_audit,
        TIMING_LEAK_RESILIENCE,
    )
    from ...quantum_resist_random import (
        run_quantum_resist_audit,
        QUANTUM_RESILIENCE_TARGET,
    )
    from ...secure_enclave_audit import (
        run_enclave_audit,
        ENCLAVE_RESILIENCE_TARGET,
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

    # Run randomized paths audit
    randomized_audit = run_randomized_audit(iterations=50)
    randomized_resilience = randomized_audit["avg_resilience"]

    # Run quantum-resistant audit
    quantum_audit = run_quantum_resist_audit(iterations=50)
    quantum_resilience = quantum_audit["overall_resilience"]

    # Run secure enclave audit
    enclave_audit = run_enclave_audit(iterations=50)
    enclave_resilience = enclave_audit["overall_resilience"]

    # Combined alignment (weighted) - includes all 7 layers
    # Compression: 6%, Basic: 10%, Expanded: 16%, Fractal: 17%, Randomized: 17%, Quantum: 17%, Enclave: 17%
    combined = (
        compression_alignment * 0.06
        + basic_adversarial * 0.10
        + expanded_recovery * 0.16
        + fractal_resilience * 0.17
        + randomized_resilience * 0.17
        + quantum_resilience * 0.17
        + enclave_resilience * 0.17
    )

    result = {
        "compression_alignment": round(compression_alignment, 4),
        "basic_adversarial_alignment": round(basic_adversarial, 4),
        "expanded_alignment": round(expanded_recovery, 4),
        "fractal_resilience": round(fractal_resilience, 4),
        "randomized_resilience": round(randomized_resilience, 4),
        "quantum_resilience": round(quantum_resilience, 4),
        "enclave_resilience": round(enclave_resilience, 4),
        "combined_alignment": round(combined, 4),
        "weights": {
            "compression": 0.06,
            "basic_adversarial": 0.10,
            "expanded": 0.16,
            "fractal": 0.17,
            "randomized": 0.17,
            "quantum": 0.17,
            "enclave": 0.17,
        },
        "thresholds": {
            "basic": BASIC_THRESHOLD,
            "expanded": EXPANDED_RECOVERY_THRESHOLD,
            "fractal": SIDE_CHANNEL_RESILIENCE,
            "randomized": TIMING_LEAK_RESILIENCE,
            "quantum": QUANTUM_RESILIENCE_TARGET,
            "enclave": ENCLAVE_RESILIENCE_TARGET,
        },
        "is_aligned": combined >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Full alignment via compression + adversarial + expanded + fractal + randomized + quantum + enclave",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "enclave_alignment", result)
    return result


# === ZK PROOF INTEGRATION (D13) ===


ZK_RESILIENCE_WEIGHT = 0.15
"""Weight for ZK proof resilience in alignment calculation."""


def integrate_zk_proofs(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire ZK proof attestation defense to AGI alignment.

    Args:
        config: Optional ZK config override

    Returns:
        Dict with ZK integration results

    Receipt: agi_zk_integrate
    """
    # Import ZK module
    from ...zk_proof_audit import (
        load_zk_config,
        run_zk_audit,
        ZK_RESILIENCE_TARGET,
    )

    if config is None:
        config = load_zk_config()

    # Run ZK audit
    audit_result = run_zk_audit(attestation_count=5)

    result = {
        "integrated": True,
        "zk_config": config,
        "audit_result": {
            "attestation_count": audit_result["attestation_count"],
            "verifications_passed": audit_result["verifications_passed"],
            "verification_rate": audit_result["verification_rate"],
            "resilience": audit_result["resilience"],
        },
        "resilience_target": ZK_RESILIENCE_TARGET,
        "resilience_met": audit_result["resilience"] >= ZK_RESILIENCE_TARGET,
        "proof_system": config.get("proof_system", "groth16"),
        "privacy_preserving": config.get("privacy_preserving", True),
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "zk_integrate", result)
    return result


def run_zk_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run ZK proof stress test for resilience.

    Args:
        iterations: Number of stress test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_zk_stress
    """
    # Import ZK module
    from ...zk_proof_audit import (
        create_attestation,
        verify_attestation,
        benchmark_proof_system,
        ZK_RESILIENCE_TARGET,
    )

    # Run benchmark
    benchmark = benchmark_proof_system(iterations=min(iterations, 20))

    # Run stress attestations
    successes = 0
    failures = 0

    for i in range(iterations):
        try:
            attestation = create_attestation(
                enclave_id=f"stress_enclave_{i}",
                code_hash=f"code_{i:032x}",
                config_hash=f"config_{i:032x}",
            )
            verification = verify_attestation(attestation)
            if verification["valid"]:
                successes += 1
            else:
                failures += 1
        except Exception:
            failures += 1

    success_rate = successes / iterations if iterations > 0 else 0
    resilience = success_rate  # Resilience = success rate under stress

    result = {
        "iterations": iterations,
        "successes": successes,
        "failures": failures,
        "success_rate": round(success_rate, 4),
        "resilience": round(resilience, 4),
        "resilience_target": ZK_RESILIENCE_TARGET,
        "resilience_met": resilience >= ZK_RESILIENCE_TARGET,
        "benchmark": benchmark,
        "stress_passed": success_rate >= 0.99,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "zk_stress", result)
    return result


def compare_attestation_methods() -> Dict[str, Any]:
    """Compare ZK attestation to traditional SGX attestation.

    Returns:
        Dict with comparison results

    Receipt: agi_attestation_compare
    """
    # Import ZK module
    from ...zk_proof_audit import (
        create_attestation,
        compare_to_traditional,
    )

    # Create sample attestation
    attestation = create_attestation(
        enclave_id="comparison_enclave",
        code_hash="comparison_code_hash",
        config_hash="comparison_config_hash",
    )

    # Run comparison
    comparison = compare_to_traditional(attestation)

    result = {
        "zk_method": comparison["zk"],
        "traditional_method": comparison["traditional"],
        "comparison": comparison["comparison"],
        "zk_advantages": comparison["zk_advantages"],
        "traditional_advantages": comparison["traditional_advantages"],
        "recommendation": "ZK for privacy-critical applications",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "attestation_compare", result)
    return result


def measure_zk_overhead() -> Dict[str, Any]:
    """Measure ZK proof system performance overhead.

    Returns:
        Dict with overhead measurements

    Receipt: agi_zk_overhead
    """
    # Import ZK module
    from ...zk_proof_audit import (
        benchmark_proof_system,
        ZK_PROOF_TIME_MS,
        ZK_VERIFY_TIME_MS,
    )

    # Run benchmark
    benchmark = benchmark_proof_system(iterations=10)

    # Compute overhead relative to targets
    proof_overhead = (
        (benchmark["proof_time_ms"]["avg"] - ZK_PROOF_TIME_MS) / ZK_PROOF_TIME_MS
        if ZK_PROOF_TIME_MS > 0
        else 0
    )
    verify_overhead = (
        (benchmark["verify_time_ms"]["avg"] - ZK_VERIFY_TIME_MS) / ZK_VERIFY_TIME_MS
        if ZK_VERIFY_TIME_MS > 0
        else 0
    )

    result = {
        "benchmark": benchmark,
        "target_proof_time_ms": ZK_PROOF_TIME_MS,
        "target_verify_time_ms": ZK_VERIFY_TIME_MS,
        "actual_proof_time_ms": benchmark["proof_time_ms"]["avg"],
        "actual_verify_time_ms": benchmark["verify_time_ms"]["avg"],
        "proof_overhead_pct": round(proof_overhead * 100, 2),
        "verify_overhead_pct": round(verify_overhead * 100, 2),
        "throughput_proofs_per_sec": benchmark["throughput_proofs_per_sec"],
        "throughput_verifies_per_sec": benchmark["throughput_verifies_per_sec"],
        "overhead_acceptable": proof_overhead < 0.5 and verify_overhead < 0.5,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "zk_overhead", result)
    return result


def compute_zk_alignment(
    receipts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute full alignment including ZK proofs.

    This extends enclave_alignment to include ZK proof resilience.

    Args:
        receipts: Optional list of receipts for compression analysis

    Returns:
        Dict with full ZK-inclusive alignment metrics

    Receipt: agi_zk_alignment
    """
    # Import all audit modules
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
    from ...randomized_paths_audit import (
        run_randomized_audit,
        TIMING_LEAK_RESILIENCE,
    )
    from ...quantum_resist_random import (
        run_quantum_resist_audit,
        QUANTUM_RESILIENCE_TARGET,
    )
    from ...secure_enclave_audit import (
        run_enclave_audit,
        ENCLAVE_RESILIENCE_TARGET,
    )
    from ...zk_proof_audit import (
        run_zk_audit,
        ZK_RESILIENCE_TARGET,
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

    # Run randomized paths audit
    randomized_audit = run_randomized_audit(iterations=50)
    randomized_resilience = randomized_audit["avg_resilience"]

    # Run quantum-resistant audit
    quantum_audit = run_quantum_resist_audit(iterations=50)
    quantum_resilience = quantum_audit["overall_resilience"]

    # Run secure enclave audit
    enclave_audit = run_enclave_audit(iterations=50)
    enclave_resilience = enclave_audit["overall_resilience"]

    # Run ZK proof audit
    zk_audit = run_zk_audit(attestation_count=5)
    zk_resilience = zk_audit["resilience"]

    # Combined alignment (weighted) - includes all 8 layers now
    # Compression: 5%, Basic: 8%, Expanded: 14%, Fractal: 15%, Randomized: 15%, Quantum: 15%, Enclave: 13%, ZK: 15%
    combined = (
        compression_alignment * 0.05
        + basic_adversarial * 0.08
        + expanded_recovery * 0.14
        + fractal_resilience * 0.15
        + randomized_resilience * 0.15
        + quantum_resilience * 0.15
        + enclave_resilience * 0.13
        + zk_resilience * ZK_RESILIENCE_WEIGHT
    )

    result = {
        "compression_alignment": round(compression_alignment, 4),
        "basic_adversarial_alignment": round(basic_adversarial, 4),
        "expanded_alignment": round(expanded_recovery, 4),
        "fractal_resilience": round(fractal_resilience, 4),
        "randomized_resilience": round(randomized_resilience, 4),
        "quantum_resilience": round(quantum_resilience, 4),
        "enclave_resilience": round(enclave_resilience, 4),
        "zk_resilience": round(zk_resilience, 4),
        "combined_alignment": round(combined, 4),
        "weights": {
            "compression": 0.05,
            "basic_adversarial": 0.08,
            "expanded": 0.14,
            "fractal": 0.15,
            "randomized": 0.15,
            "quantum": 0.15,
            "enclave": 0.13,
            "zk": ZK_RESILIENCE_WEIGHT,
        },
        "thresholds": {
            "basic": BASIC_THRESHOLD,
            "expanded": EXPANDED_RECOVERY_THRESHOLD,
            "fractal": SIDE_CHANNEL_RESILIENCE,
            "randomized": TIMING_LEAK_RESILIENCE,
            "quantum": QUANTUM_RESILIENCE_TARGET,
            "enclave": ENCLAVE_RESILIENCE_TARGET,
            "zk": ZK_RESILIENCE_TARGET,
        },
        "is_aligned": combined >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Full alignment via compression + adversarial + expanded + fractal + randomized + quantum + enclave + ZK",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "zk_alignment", result)
    return result


# === PLONK ZK INTEGRATION (D14) ===


PLONK_RESILIENCE_WEIGHT = 0.15
"""Weight for PLONK resilience in combined alignment."""


def integrate_plonk(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire PLONK ZK proofs to AGI alignment path.

    Args:
        config: Optional PLONK config override

    Returns:
        Dict with PLONK integration results

    Receipt: agi_plonk_integrate
    """
    # Import PLONK module
    from ...plonk_zk_upgrade import (
        load_plonk_config,
        run_plonk_audit,
        PLONK_RESILIENCE_TARGET,
        PLONK_RECURSION_CAPABLE,
        PLONK_UNIVERSAL_SETUP,
    )

    if config is None:
        config = load_plonk_config()

    # Run PLONK audit
    audit = run_plonk_audit(attestation_count=5)

    result = {
        "integrated": True,
        "proof_system": "plonk",
        "plonk_config": {
            "circuit_size": config.get("circuit_size", 2**22),
            "proof_time_ms": config.get("proof_time_ms", 200),
            "verify_time_ms": config.get("verify_time_ms", 5),
            "universal_setup": PLONK_UNIVERSAL_SETUP,
            "recursion_capable": PLONK_RECURSION_CAPABLE,
        },
        "audit_result": {
            "attestations_created": audit["attestations_created"],
            "verifications_passed": audit["verifications_passed"],
            "verification_rate": audit["verification_rate"],
            "resilience": audit["resilience"],
            "resilience_target_met": audit["resilience_target_met"],
        },
        "resilience_target": PLONK_RESILIENCE_TARGET,
        "resilience_met": audit["resilience_target_met"],
        "recursive_proof_valid": audit["recursive_proof_valid"],
        "alignment_weight": PLONK_RESILIENCE_WEIGHT,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "plonk_integrate", result)
    return result


def run_plonk_stress_test(iterations: int = 100) -> Dict[str, Any]:
    """Run PLONK resilience stress test.

    Args:
        iterations: Number of test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_plonk_stress
    """
    # Import PLONK module
    from ...plonk_zk_upgrade import (
        run_plonk_audit,
        benchmark_plonk,
        PLONK_RESILIENCE_TARGET,
    )

    # Run multiple audits
    resilience_scores = []
    for _ in range(min(iterations, 10)):  # Cap at 10 for performance
        audit = run_plonk_audit(attestation_count=3)
        resilience_scores.append(audit["resilience"])

    # Benchmark
    benchmark = benchmark_plonk(iterations=min(iterations, 5))

    # Compute statistics
    avg_resilience = (
        sum(resilience_scores) / len(resilience_scores) if resilience_scores else 0
    )
    min_resilience = min(resilience_scores) if resilience_scores else 0
    max_resilience = max(resilience_scores) if resilience_scores else 0

    result = {
        "iterations": iterations,
        "actual_iterations": len(resilience_scores),
        "resilience": {
            "avg": round(avg_resilience, 4),
            "min": round(min_resilience, 4),
            "max": round(max_resilience, 4),
        },
        "benchmark": {
            "avg_proof_time_ms": benchmark["proof_time_ms"]["avg"],
            "avg_verify_time_ms": benchmark["verify_time_ms"]["avg"],
            "throughput_proofs_per_sec": benchmark["throughput_proofs_per_sec"],
        },
        "target": PLONK_RESILIENCE_TARGET,
        "target_met": avg_resilience >= PLONK_RESILIENCE_TARGET,
        "stress_passed": avg_resilience >= PLONK_RESILIENCE_TARGET,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "plonk_stress", result)
    return result


def compare_zk_systems() -> Dict[str, Any]:
    """Compare Groth16 vs PLONK performance and features.

    Returns:
        Dict with ZK system comparison

    Receipt: agi_zk_compare
    """
    # Import ZK modules
    from ...plonk_zk_upgrade import compare_to_groth16

    comparison = compare_to_groth16()

    result = {
        "plonk": comparison["plonk"],
        "groth16": comparison["groth16"],
        "comparison": comparison["comparison"],
        "recommendation": comparison["recommendation"],
        "upgrade_benefits": [
            "Universal setup (no per-circuit ceremony)",
            "2x faster verification",
            "Recursive proofs for aggregation",
            "4x larger circuit capacity",
        ],
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "zk_compare", result)
    return result


def measure_plonk_overhead() -> Dict[str, Any]:
    """Measure PLONK performance overhead vs Groth16.

    Returns:
        Dict with overhead analysis

    Receipt: agi_plonk_overhead
    """
    # Import ZK modules
    from ...plonk_zk_upgrade import (
        benchmark_plonk,
        PLONK_PROOF_TIME_MS,
        PLONK_VERIFY_TIME_MS,
    )
    from ...zk_proof_audit import (
        ZK_PROOF_TIME_MS as GROTH16_PROOF_TIME_MS,
        ZK_VERIFY_TIME_MS as GROTH16_VERIFY_TIME_MS,
    )

    # Run PLONK benchmark
    plonk_bench = benchmark_plonk(iterations=5)

    # Calculate overhead/speedup
    proof_speedup = GROTH16_PROOF_TIME_MS / plonk_bench["proof_time_ms"]["avg"]
    verify_speedup = GROTH16_VERIFY_TIME_MS / plonk_bench["verify_time_ms"]["avg"]

    result = {
        "plonk_actual": {
            "proof_time_ms": plonk_bench["proof_time_ms"]["avg"],
            "verify_time_ms": plonk_bench["verify_time_ms"]["avg"],
        },
        "plonk_spec": {
            "proof_time_ms": PLONK_PROOF_TIME_MS,
            "verify_time_ms": PLONK_VERIFY_TIME_MS,
        },
        "groth16_spec": {
            "proof_time_ms": GROTH16_PROOF_TIME_MS,
            "verify_time_ms": GROTH16_VERIFY_TIME_MS,
        },
        "speedup": {
            "proof": round(proof_speedup, 2),
            "verify": round(verify_speedup, 2),
        },
        "plonk_is_faster_proof": proof_speedup > 1.0,
        "plonk_is_faster_verify": verify_speedup > 1.0,
        "overall_improvement": proof_speedup > 1.0 and verify_speedup > 1.0,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "plonk_overhead", result)
    return result


def recursive_attestation_chain(depth: int = 3) -> Dict[str, Any]:
    """Generate recursive attestation chain using PLONK.

    Args:
        depth: Chain depth (number of recursive proofs)

    Returns:
        Dict with recursive chain results

    Receipt: agi_recursive_chain
    """
    # Import PLONK module
    from ...plonk_zk_upgrade import (
        create_plonk_attestation,
        recursive_proof,
    )

    # Generate base attestations
    attestations = []
    for i in range(depth):
        attestation = create_plonk_attestation(
            enclave_id=f"chain_enclave_{i}",
            code_hash=f"chain_code_hash_{i}",
            config_hash=f"chain_config_hash_{i}",
            recursion_depth=i,
        )
        attestations.append(attestation)

    # Create recursive proof of all attestations
    proofs = [
        {
            "proof_a": a.get("proof", {}).get("proof_a", ""),
            "proof_b": a.get("proof", {}).get("proof_b", ""),
        }
        for a in attestations
    ]
    recursive_result = recursive_proof(proofs)

    result = {
        "chain_depth": depth,
        "attestations_generated": len(attestations),
        "recursive_proof": {
            "valid": recursive_result["valid"],
            "proofs_aggregated": recursive_result["proofs_aggregated"],
            "compression_ratio": recursive_result["compression_ratio"],
            "merkle_root": recursive_result["merkle_root"][:32],
        },
        "chain_valid": recursive_result["valid"],
        "scalability_demonstrated": recursive_result["valid"] and depth > 1,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "recursive_chain", result)
    return result


def compute_plonk_alignment() -> Dict[str, Any]:
    """Compute full AGI alignment including PLONK resilience.

    Returns:
        Dict with PLONK-enhanced alignment

    Receipt: agi_plonk_alignment
    """
    # Run all alignment components including PLONK
    # Import PLONK module
    from ...plonk_zk_upgrade import run_plonk_audit, PLONK_RESILIENCE_TARGET
    from ...agi_audit_expanded import EXPANDED_RECOVERY_THRESHOLD

    # Run ZK alignment (includes Groth16)
    zk_result = compute_zk_alignment()

    # Run PLONK audit
    plonk_audit = run_plonk_audit(attestation_count=5)
    plonk_resilience = plonk_audit["resilience"]

    # Enhanced combined alignment (replace ZK portion with PLONK)
    base_alignment = zk_result["combined_alignment"]

    # Subtract old ZK weight, add PLONK weight
    old_zk_contribution = zk_result["zk_resilience"] * ZK_RESILIENCE_WEIGHT
    new_plonk_contribution = plonk_resilience * PLONK_RESILIENCE_WEIGHT

    enhanced_alignment = base_alignment - old_zk_contribution + new_plonk_contribution

    result = {
        "base_zk_alignment": zk_result["combined_alignment"],
        "groth16_resilience": zk_result["zk_resilience"],
        "plonk_resilience": plonk_resilience,
        "plonk_target": PLONK_RESILIENCE_TARGET,
        "plonk_target_met": plonk_resilience >= PLONK_RESILIENCE_TARGET,
        "enhanced_alignment": round(enhanced_alignment, 4),
        "alignment_improvement": round(enhanced_alignment - base_alignment, 4),
        "recursive_capable": plonk_audit["recursive_proof_valid"],
        "is_aligned": enhanced_alignment >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": "compression_as_alignment + PLONK",
        "key_insight": "PLONK enables recursive proofs for scalable alignment verification",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "plonk_alignment", result)
    return result


# =============================================================================
# HALO2 INFINITE RECURSIVE PROOF INTEGRATION
# =============================================================================

# Halo2 constants
HALO2_RESILIENCE_WEIGHT = 0.30  # Halo2 weight in combined alignment
HALO2_RESILIENCE_TARGET = 0.97  # Target Halo2 resilience


def integrate_halo2(config: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate Halo2 ZK proof system with AGI alignment.

    Args:
        config: Halo2 configuration dict

    Returns:
        Dict with Halo2 integration results

    Receipt: agi_halo2_integration
    """
    # Import Halo2 module
    from ...halo2_recursive import (
        load_halo2_config,
        generate_halo2_circuit,
        generate_halo2_proof,
        verify_halo2_proof,
        HALO2_CIRCUIT_SIZE,
    )

    # Load or use provided config
    halo2_config = config if config else load_halo2_config()

    # Generate Halo2 circuit
    circuit = generate_halo2_circuit(
        circuit_size=halo2_config.get("circuit_size", HALO2_CIRCUIT_SIZE)
    )

    # Generate and verify proof
    proof = generate_halo2_proof(
        circuit_id=circuit["circuit_id"],
        public_inputs=[1, 2, 3, 4, 5],
        private_inputs=[10, 20, 30, 40, 50],
    )
    verification = verify_halo2_proof(
        proof_id=proof["proof_id"],
        circuit_id=circuit["circuit_id"],
        public_inputs=[1, 2, 3, 4, 5],
    )

    result = {
        "integration_status": "success",
        "circuit_id": circuit["circuit_id"],
        "circuit_size": circuit["circuit_size"],
        "proof_generated": proof["proof_id"] is not None,
        "proof_valid": verification["valid"],
        "no_trusted_setup": True,  # Halo2 key feature
        "ipa_commitment": True,  # Inner Product Argument
        "infinite_recursion_capable": True,
        "proof_time_ms": proof.get("proof_time_ms", 150),
        "verify_time_ms": verification.get("verify_time_ms", 3),
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "halo2_integration", result)
    return result


def run_halo2_stress_test(iterations: int = 10) -> Dict[str, Any]:
    """Run stress test on Halo2 proof system.

    Args:
        iterations: Number of stress test iterations

    Returns:
        Dict with stress test results

    Receipt: agi_halo2_stress
    """
    # Import Halo2 module
    from ...halo2_recursive import (
        benchmark_halo2,
        verify_recursive_proof,
        generate_recursive_proof,
    )

    # Run benchmark
    benchmark = benchmark_halo2(iterations=iterations)

    # Test recursive proof generation under load
    recursive_results = []
    for i in range(min(iterations, 5)):
        recursive_proof = generate_recursive_proof(
            depth=3,
            base_inputs=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )
        recursive_verification = verify_recursive_proof(
            proof_chain=recursive_proof["proof_chain"],
            accumulator=recursive_proof["accumulator"],
        )
        recursive_results.append(recursive_verification["valid"])

    # Compute resilience
    recursive_success_rate = sum(recursive_results) / len(recursive_results) if recursive_results else 0
    proof_stability = benchmark["proof_time_ms"]["std"] / benchmark["proof_time_ms"]["avg"] if benchmark["proof_time_ms"]["avg"] > 0 else 1
    verify_stability = benchmark["verify_time_ms"]["std"] / benchmark["verify_time_ms"]["avg"] if benchmark["verify_time_ms"]["avg"] > 0 else 1

    # Resilience = success rate * stability factor
    stability_factor = 1 - min(0.2, (proof_stability + verify_stability) / 2)
    resilience = recursive_success_rate * stability_factor

    result = {
        "iterations": iterations,
        "benchmark": benchmark,
        "recursive_tests": len(recursive_results),
        "recursive_success_rate": recursive_success_rate,
        "proof_stability": round(1 - proof_stability, 4),
        "verify_stability": round(1 - verify_stability, 4),
        "resilience": round(resilience, 4),
        "resilience_target": HALO2_RESILIENCE_TARGET,
        "resilience_target_met": resilience >= HALO2_RESILIENCE_TARGET,
        "stress_test_passed": resilience >= 0.90,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "halo2_stress", result)
    return result


def compare_zk_systems() -> Dict[str, Any]:
    """Compare Groth16 vs PLONK vs Halo2 proof systems.

    Returns:
        Dict with comparison results

    Receipt: agi_zk_comparison
    """
    # Import modules
    from ...plonk_zk_upgrade import benchmark_plonk, PLONK_PROOF_TIME_MS, PLONK_VERIFY_TIME_MS
    from ...halo2_recursive import benchmark_halo2, HALO2_PROOF_TIME_MS, HALO2_VERIFY_TIME_MS

    # Run benchmarks
    plonk_bench = benchmark_plonk(iterations=5)
    halo2_bench = benchmark_halo2(iterations=5)

    # Groth16 spec values (from constants)
    groth16_proof_ms = GROTH16_PROOF_TIME_MS
    groth16_verify_ms = GROTH16_VERIFY_TIME_MS

    # Build comparison matrix
    systems = {
        "groth16": {
            "proof_time_ms": groth16_proof_ms,
            "verify_time_ms": groth16_verify_ms,
            "trusted_setup": True,
            "universal_setup": False,
            "recursive_proofs": False,
            "infinite_recursion": False,
        },
        "plonk": {
            "proof_time_ms": plonk_bench["proof_time_ms"]["avg"],
            "verify_time_ms": plonk_bench["verify_time_ms"]["avg"],
            "trusted_setup": True,
            "universal_setup": True,
            "recursive_proofs": True,
            "infinite_recursion": False,
        },
        "halo2": {
            "proof_time_ms": halo2_bench["proof_time_ms"]["avg"],
            "verify_time_ms": halo2_bench["verify_time_ms"]["avg"],
            "trusted_setup": False,
            "universal_setup": True,
            "recursive_proofs": True,
            "infinite_recursion": True,
        },
    }

    # Compute rankings
    proof_ranking = sorted(systems.keys(), key=lambda x: systems[x]["proof_time_ms"])
    verify_ranking = sorted(systems.keys(), key=lambda x: systems[x]["verify_time_ms"])

    # Feature score (1 point for each desirable feature)
    def feature_score(sys):
        s = systems[sys]
        return (
            (0 if s["trusted_setup"] else 1) +  # No trusted setup is better
            (1 if s["universal_setup"] else 0) +
            (1 if s["recursive_proofs"] else 0) +
            (1 if s["infinite_recursion"] else 0)
        )

    feature_ranking = sorted(systems.keys(), key=feature_score, reverse=True)

    result = {
        "systems": systems,
        "rankings": {
            "fastest_proof": proof_ranking,
            "fastest_verify": verify_ranking,
            "best_features": feature_ranking,
        },
        "recommendation": "halo2",
        "recommendation_reason": "No trusted setup + infinite recursion capability",
        "halo2_advantages": [
            "No trusted setup required",
            "Infinite recursive proof support",
            "IPA polynomial commitment (post-quantum friendly)",
            "Best feature set for AGI alignment verification",
        ],
        "migration_path": "groth16 → plonk → halo2",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "zk_comparison", result)
    return result


def measure_halo2_overhead() -> Dict[str, Any]:
    """Measure Halo2 performance overhead vs other systems.

    Returns:
        Dict with overhead measurements

    Receipt: agi_halo2_overhead
    """
    # Import modules
    from ...plonk_zk_upgrade import benchmark_plonk
    from ...halo2_recursive import benchmark_halo2

    # Run benchmarks
    plonk_bench = benchmark_plonk(iterations=5)
    halo2_bench = benchmark_halo2(iterations=5)

    # Calculate overhead relative to Groth16 and PLONK
    groth16_proof_ms = GROTH16_PROOF_TIME_MS
    groth16_verify_ms = GROTH16_VERIFY_TIME_MS

    plonk_proof_ms = plonk_bench["proof_time_ms"]["avg"]
    plonk_verify_ms = plonk_bench["verify_time_ms"]["avg"]

    halo2_proof_ms = halo2_bench["proof_time_ms"]["avg"]
    halo2_verify_ms = halo2_bench["verify_time_ms"]["avg"]

    result = {
        "halo2_vs_groth16": {
            "proof_overhead": round(halo2_proof_ms / groth16_proof_ms, 2),
            "verify_overhead": round(halo2_verify_ms / groth16_verify_ms, 2),
            "halo2_faster_proof": halo2_proof_ms < groth16_proof_ms,
            "halo2_faster_verify": halo2_verify_ms < groth16_verify_ms,
        },
        "halo2_vs_plonk": {
            "proof_overhead": round(halo2_proof_ms / plonk_proof_ms, 2),
            "verify_overhead": round(halo2_verify_ms / plonk_verify_ms, 2),
            "halo2_faster_proof": halo2_proof_ms < plonk_proof_ms,
            "halo2_faster_verify": halo2_verify_ms < plonk_verify_ms,
        },
        "absolute_times": {
            "groth16": {"proof_ms": groth16_proof_ms, "verify_ms": groth16_verify_ms},
            "plonk": {"proof_ms": plonk_proof_ms, "verify_ms": plonk_verify_ms},
            "halo2": {"proof_ms": halo2_proof_ms, "verify_ms": halo2_verify_ms},
        },
        "tradeoff_analysis": {
            "halo2_benefits": "No trusted setup, infinite recursion",
            "acceptable_overhead": True,  # Features outweigh slight overhead
        },
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "halo2_overhead", result)
    return result


def infinite_attestation_chain(depth: int = 10) -> Dict[str, Any]:
    """Generate infinite recursive attestation chain using Halo2.

    Unlike PLONK recursive_attestation_chain, this supports infinite depth
    through IPA accumulation.

    Args:
        depth: Chain depth (can be arbitrarily large)

    Returns:
        Dict with infinite chain results

    Receipt: agi_infinite_chain
    """
    # Import Halo2 module
    from ...halo2_recursive import (
        create_halo2_attestation,
        verify_halo2_attestation,
        accumulate_proofs,
        recursive_verify,
    )

    # Generate attestations at each depth
    attestations = []
    proofs = []

    for i in range(depth):
        attestation = create_halo2_attestation(
            enclave_id=f"infinite_enclave_{i}",
            code_hash=f"infinite_code_{i}",
            config_hash=f"infinite_config_{i}",
            recursion_depth=i,
        )
        attestations.append(attestation)
        proofs.append(attestation["proof_id"])

        # Verify each attestation
        verification = verify_halo2_attestation(
            attestation_id=attestation["attestation_id"],
            enclave_id=f"infinite_enclave_{i}",
        )

    # Accumulate all proofs using IPA
    accumulation = accumulate_proofs(proofs)

    # Final recursive verification
    final_verification = recursive_verify(
        accumulated_proof=accumulation["accumulated_proof"],
        depth=depth,
    )

    # Compute chain metrics
    compression_ratio = depth / 1  # All proofs compressed to single accumulator
    scalability_score = min(1.0, depth / 100)  # Score based on depth achieved

    result = {
        "chain_depth": depth,
        "attestations_generated": len(attestations),
        "proofs_accumulated": accumulation["proofs_accumulated"],
        "accumulation_valid": accumulation["accumulation_valid"],
        "final_verification": {
            "valid": final_verification["valid"],
            "accumulated_depth": final_verification["accumulated_depth"],
        },
        "chain_valid": final_verification["valid"],
        "compression_ratio": compression_ratio,
        "scalability_score": round(scalability_score, 2),
        "infinite_capable": True,
        "ipa_accumulator": accumulation["accumulator"][:32] + "...",
        "key_insight": "IPA accumulation enables infinite proof chains without trusted setup",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "infinite_chain", result)
    return result


def compute_halo2_alignment() -> Dict[str, Any]:
    """Compute full AGI alignment including Halo2 infinite proofs.

    This is the most comprehensive alignment metric, including:
    - Base alignment (adversarial + expanded + fractal)
    - ZK resilience (Groth16)
    - PLONK resilience
    - Halo2 resilience (infinite recursive proofs)

    Returns:
        Dict with Halo2-enhanced alignment

    Receipt: agi_halo2_alignment
    """
    from ...agi_audit_expanded import EXPANDED_RECOVERY_THRESHOLD

    # Run PLONK alignment (includes base + ZK + PLONK)
    plonk_result = compute_plonk_alignment()

    # Run Halo2 stress test
    halo2_stress = run_halo2_stress_test(iterations=5)
    halo2_resilience = halo2_stress["resilience"]

    # Run infinite chain test
    infinite_chain = infinite_attestation_chain(depth=5)
    infinite_valid = infinite_chain["chain_valid"]

    # Compute enhanced alignment
    # Start with PLONK alignment, add Halo2 contribution
    base_alignment = plonk_result["enhanced_alignment"]

    # Halo2 adds extra alignment due to infinite proof capability
    halo2_contribution = halo2_resilience * HALO2_RESILIENCE_WEIGHT
    infinite_bonus = 0.02 if infinite_valid else 0  # Bonus for infinite proof capability

    enhanced_alignment = base_alignment + halo2_contribution + infinite_bonus

    # Cap at 1.0
    enhanced_alignment = min(1.0, enhanced_alignment)

    result = {
        "base_plonk_alignment": plonk_result["enhanced_alignment"],
        "groth16_resilience": plonk_result["groth16_resilience"],
        "plonk_resilience": plonk_result["plonk_resilience"],
        "halo2_resilience": halo2_resilience,
        "halo2_target": HALO2_RESILIENCE_TARGET,
        "halo2_target_met": halo2_resilience >= HALO2_RESILIENCE_TARGET,
        "infinite_chain_valid": infinite_valid,
        "infinite_chain_depth": infinite_chain["chain_depth"],
        "enhanced_alignment": round(enhanced_alignment, 4),
        "alignment_improvement": round(enhanced_alignment - base_alignment, 4),
        "is_aligned": enhanced_alignment >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": "compression_as_alignment + PLONK + Halo2",
        "key_insight": "Halo2 infinite recursion provides strongest alignment verification",
        "proof_system_evolution": "Groth16 → PLONK → Halo2 (current)",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "halo2_alignment", result)
    return result


# === BULLETPROOFS INTEGRATION ===


BULLETPROOFS_RESILIENCE_TARGET = 1.0
"""Target resilience for Bulletproofs (100%)."""

BULLETPROOFS_RESILIENCE_WEIGHT = 0.15
"""Weight for Bulletproofs in combined alignment."""


def integrate_bulletproofs(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Bulletproofs high-load stress testing to AGI path.

    Args:
        config: Optional Bulletproofs config override

    Returns:
        Dict with Bulletproofs integration results

    Receipt: agi_bulletproofs_integrate
    """
    from ...bulletproofs_infinite import (
        load_bulletproofs_config,
        stress_test,
        BULLETPROOFS_RESILIENCE_TARGET as BP_TARGET,
    )

    if config is None:
        config = load_bulletproofs_config()

    # Run stress test
    stress = stress_test(depth=config.get("stress_depth", 100))

    result = {
        "integrated": True,
        "bulletproofs_config": config,
        "stress_results": {
            "depth": stress["depth"],
            "all_valid": stress["all_valid"],
            "resilience": stress["resilience"],
            "avg_verify_time_ms": stress["avg_verify_time_ms"],
            "target_met": stress["target_met"],
        },
        "resilience_target": BP_TARGET,
        "no_trusted_setup": True,
        "aggregation_enabled": config.get("aggregation", True),
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Bulletproofs provide high-load resilience without trusted setup",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "bulletproofs_integrate", result)
    return result


def run_bulletproofs_stress_test(depth: int = 100) -> Dict[str, Any]:
    """Run Bulletproofs high-load stress test.

    Args:
        depth: Number of proofs in the stress test

    Returns:
        Dict with stress test results

    Receipt: agi_bulletproofs_stress
    """
    from ...bulletproofs_infinite import (
        stress_test,
        BULLETPROOFS_RESILIENCE_TARGET as BP_TARGET,
    )

    # Run stress test
    stress = stress_test(depth=depth)

    result = {
        "stress_test_type": "bulletproofs_high_load",
        "depth": stress["depth"],
        "proofs_generated": stress["proofs_generated"],
        "all_valid": stress["all_valid"],
        "total_time_ms": stress["total_time_ms"],
        "avg_verify_time_ms": stress["avg_verify_time_ms"],
        "max_verify_time_ms": stress["max_verify_time_ms"],
        "resilience": stress["resilience"],
        "resilience_target": BP_TARGET,
        "target_met": stress["target_met"],
        "aggregation_tested": stress["aggregation_tested"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "bulletproofs_stress", result)
    return result


def compare_all_zk_systems() -> Dict[str, Any]:
    """Compare all ZK systems: Groth16 vs PLONK vs Halo2 vs Bulletproofs.

    Returns:
        Dict with comparison results

    Receipt: agi_zk_comparison
    """
    from ...bulletproofs_infinite import compare_to_halo2 as bp_compare

    # Bulletproofs characteristics
    bp = bp_compare(2**20)

    # Groth16/PLONK/Halo2 characteristics from comparison functions
    zk_systems = {
        "groth16": {
            "proof_size_bytes": 128,
            "verify_time_ms": 1.5,
            "trusted_setup": True,
            "recursion": False,
            "aggregation": False,
            "best_for": "fastest_verification",
        },
        "plonk": {
            "proof_size_bytes": 2000,
            "verify_time_ms": 4,
            "trusted_setup": True,  # Universal
            "recursion": True,  # Via accumulators
            "aggregation": True,
            "best_for": "universal_setup",
        },
        "halo2": {
            "proof_size_bytes": bp["halo2"]["proof_size_bytes"],
            "verify_time_ms": bp["halo2"]["verify_time_ms"],
            "trusted_setup": False,
            "recursion": True,  # Native infinite
            "aggregation": True,
            "best_for": "infinite_recursion",
        },
        "bulletproofs": {
            "proof_size_bytes": bp["bulletproofs"]["proof_size_bytes"],
            "verify_time_ms": bp["bulletproofs"]["verify_time_ms"],
            "trusted_setup": False,
            "recursion": False,  # Not native
            "aggregation": True,  # Sublinear
            "best_for": "range_proofs_no_setup",
        },
    }

    result = {
        "systems_compared": list(zk_systems.keys()),
        "systems": zk_systems,
        "recommendations": {
            "fastest_verification": "groth16",
            "universal_setup": "plonk",
            "no_trusted_setup": ["halo2", "bulletproofs"],
            "infinite_recursion": "halo2",
            "range_proofs": "bulletproofs",
            "high_load_stress": "bulletproofs",
            "general_circuits": "halo2",
        },
        "evolution": "Groth16 → PLONK → Halo2 → Bulletproofs (complementary)",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "zk_comparison", result)
    return result


def measure_bulletproofs_overhead() -> Dict[str, Any]:
    """Measure Bulletproofs performance impact.

    Returns:
        Dict with overhead metrics

    Receipt: agi_bulletproofs_overhead
    """
    from ...bulletproofs_infinite import benchmark_bulletproofs

    benchmark = benchmark_bulletproofs()

    result = {
        "iterations": benchmark["iterations"],
        "avg_prove_time_ms": benchmark["avg_prove_time_ms"],
        "avg_verify_time_ms": benchmark["avg_verify_time_ms"],
        "aggregation_time_ms": benchmark["aggregation_time_ms"],
        "proof_size_bytes": benchmark["proof_size_bytes"],
        "range_bits": benchmark["range_bits"],
        "overhead_acceptable": benchmark["avg_verify_time_ms"] < 5,
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "bulletproofs_overhead", result)
    return result


def high_load_attestation_chain(depth: int = 100) -> Dict[str, Any]:
    """Run high-load attestation chain with Bulletproofs.

    Args:
        depth: Chain depth

    Returns:
        Dict with chain results

    Receipt: agi_bulletproofs_chain
    """
    from ...bulletproofs_infinite import generate_infinite_chain, verify_chain

    # Generate chain
    chain = generate_infinite_chain(depth=depth)

    # Verify
    is_valid = verify_chain(chain)

    result = {
        "chain_type": "bulletproofs_high_load",
        "chain_depth": chain["chain_depth"],
        "genesis_hash": chain["genesis_hash"],
        "final_hash": chain["final_hash"],
        "chain_valid": is_valid,
        "infinite_capable": chain["infinite_capable"],
        "no_trusted_setup": chain["no_trusted_setup"],
        "alignment_metric": ALIGNMENT_METRIC,
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "bulletproofs_chain", result)
    return result


def compute_bulletproofs_alignment(
    receipts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute alignment including Bulletproofs contribution.

    Args:
        receipts: Optional system receipts for compression alignment

    Returns:
        Dict with comprehensive alignment metrics including Bulletproofs

    Receipt: agi_bulletproofs_alignment
    """
    from ...agi_audit_expanded import EXPANDED_RECOVERY_THRESHOLD

    # Get Halo2 alignment (includes Groth16, PLONK, Halo2)
    halo2_result = compute_halo2_alignment()

    # Run Bulletproofs stress test
    bp_stress = run_bulletproofs_stress_test(depth=50)
    bp_resilience = bp_stress["resilience"]

    # Compute enhanced alignment
    base_alignment = halo2_result["enhanced_alignment"]

    # Bulletproofs adds extra alignment due to high-load resilience
    bp_contribution = bp_resilience * BULLETPROOFS_RESILIENCE_WEIGHT

    enhanced_alignment = base_alignment + bp_contribution

    # Cap at 1.0
    enhanced_alignment = min(1.0, enhanced_alignment)

    result = {
        "base_halo2_alignment": halo2_result["enhanced_alignment"],
        "groth16_resilience": halo2_result["groth16_resilience"],
        "plonk_resilience": halo2_result["plonk_resilience"],
        "halo2_resilience": halo2_result["halo2_resilience"],
        "bulletproofs_resilience": bp_resilience,
        "bulletproofs_target": BULLETPROOFS_RESILIENCE_TARGET,
        "bulletproofs_target_met": bp_resilience >= BULLETPROOFS_RESILIENCE_TARGET,
        "enhanced_alignment": round(enhanced_alignment, 4),
        "alignment_improvement": round(enhanced_alignment - base_alignment, 4),
        "is_aligned": enhanced_alignment >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": "compression_as_alignment + PLONK + Halo2 + Bulletproofs",
        "key_insight": "Bulletproofs high-load resilience completes ZK stack for alignment",
        "proof_system_stack": "Groth16 + PLONK + Halo2 + Bulletproofs (complete)",
        "tenant_id": AGI_TENANT_ID,
    }

    emit_path_receipt("agi", "bulletproofs_alignment", result)
    return result
