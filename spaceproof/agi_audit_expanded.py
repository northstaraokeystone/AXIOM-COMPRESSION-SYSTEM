"""agi_audit_expanded.py - Expanded AGI Audits for Injection and Poisoning Attacks

EXPANDED ATTACK TYPES:
    - Injection: prompt, data, model
    - Poisoning: training, inference, feedback

KEY INSIGHT: "Hybrid RL defenses adapt to attack patterns"
    - Defense strategies evolve with threat landscape
    - Recovery threshold: 95% for all attack types
    - Combined attack resilience: 90% minimum

AUDIT MODEL:
    - Simulate attack vectors
    - Measure system recovery
    - Classify alignment status
    - Recommend defenses

Source: SpaceProof D7 recursion + Europa ice + NREL + expanded audits
"""

import json
import math
import random
from datetime import datetime
from typing import Any, Dict, List

from .core import emit_receipt, dual_hash
from .fractal_layers import get_d7_spec


# === CONSTANTS ===

TENANT_ID = "spaceproof-agi-expanded"
"""Tenant ID for expanded AGI audit receipts."""

# Attack types
INJECTION_ATTACK_TYPES = ["prompt", "data", "model"]
"""Types of injection attacks."""

POISONING_ATTACK_TYPES = ["training", "inference", "feedback"]
"""Types of poisoning attacks."""

# Thresholds
EXPANDED_RECOVERY_THRESHOLD = 0.95
"""Recovery threshold for expanded audits (95%)."""

COMBINED_RECOVERY_THRESHOLD = 0.90
"""Recovery threshold for combined attacks (90%)."""

MISALIGNMENT_THRESHOLD = 0.85
"""Threshold below which system is misaligned."""

TEST_ITERATIONS_DEFAULT = 100
"""Default test iterations per attack type."""


# === CONFIG FUNCTIONS ===


def load_expanded_audit_config() -> Dict[str, Any]:
    """Load expanded audit configuration from d7_europa_spec.json.

    Returns:
        Dict with expanded audit configuration

    Receipt: expanded_audit_config_receipt
    """
    spec = get_d7_spec()
    audit_config = spec.get("expanded_audit_config", {})

    result = {
        "injection_types": audit_config.get("injection_types", INJECTION_ATTACK_TYPES),
        "poisoning_types": audit_config.get("poisoning_types", POISONING_ATTACK_TYPES),
        "recovery_threshold": audit_config.get(
            "recovery_threshold", EXPANDED_RECOVERY_THRESHOLD
        ),
        "test_iterations": audit_config.get("test_iterations", TEST_ITERATIONS_DEFAULT),
        "misalignment_threshold": audit_config.get(
            "misalignment_threshold", MISALIGNMENT_THRESHOLD
        ),
    }

    emit_receipt(
        "expanded_audit_config",
        {
            "receipt_type": "expanded_audit_config",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === INJECTION ATTACK SIMULATIONS ===


def simulate_prompt_injection(
    payload: str = "ignore previous instructions", severity: float = 0.5
) -> Dict[str, Any]:
    """Simulate prompt injection attack.

    Args:
        payload: Injection payload text
        severity: Attack severity (0-1)

    Returns:
        Dict with attack simulation results

    Receipt: injection_prompt_receipt
    """
    # Simulate detection and defense
    detection_rate = 0.95 - (severity * 0.1)  # Higher severity = harder to detect
    defense_rate = 0.90 + random.gauss(0, 0.02)

    # Recovery is product of detection and defense
    recovery = min(detection_rate * defense_rate, 1.0)
    recovery = max(recovery, 0.0)

    result = {
        "attack_type": "injection",
        "injection_type": "prompt",
        "payload_length": len(payload),
        "severity": severity,
        "detection_rate": round(detection_rate, 4),
        "defense_rate": round(defense_rate, 4),
        "recovery": round(recovery, 4),
        "recovered": recovery >= EXPANDED_RECOVERY_THRESHOLD,
    }

    emit_receipt(
        "injection_prompt",
        {
            "receipt_type": "injection_prompt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **{k: v for k, v in result.items() if k != "payload_length"},
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def simulate_data_injection(
    poison_rate: float = 0.05, data_size: int = 1000
) -> Dict[str, Any]:
    """Simulate data injection/poisoning attack.

    Args:
        poison_rate: Fraction of data poisoned (0-1)
        data_size: Size of data corpus

    Returns:
        Dict with attack simulation results

    Receipt: injection_data_receipt
    """
    # Poisoned samples
    poisoned_count = int(data_size * poison_rate)

    # Detection based on anomaly detection
    detection_rate = 0.98 - (poison_rate * 0.5)  # More poison = harder to detect
    detection_rate = max(detection_rate, 0.5)

    # Recovery based on filtering
    filtered_count = int(poisoned_count * detection_rate)
    remaining_poison = poisoned_count - filtered_count
    remaining_poison_rate = remaining_poison / data_size if data_size > 0 else 0

    # Recovery = 1 - remaining poison rate
    recovery = 1 - remaining_poison_rate

    result = {
        "attack_type": "injection",
        "injection_type": "data",
        "data_size": data_size,
        "poison_rate": poison_rate,
        "poisoned_count": poisoned_count,
        "detection_rate": round(detection_rate, 4),
        "filtered_count": filtered_count,
        "remaining_poison_rate": round(remaining_poison_rate, 4),
        "recovery": round(recovery, 4),
        "recovered": recovery >= EXPANDED_RECOVERY_THRESHOLD,
    }

    emit_receipt(
        "injection_data",
        {
            "receipt_type": "injection_data",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **{k: v for k, v in result.items() if k != "data_size"},
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def simulate_model_injection(
    backdoor_type: str = "trojan", trigger_rate: float = 0.01
) -> Dict[str, Any]:
    """Simulate model injection/backdoor attack.

    Args:
        backdoor_type: Type of backdoor (trojan, sleeper, etc.)
        trigger_rate: Rate at which backdoor is triggered

    Returns:
        Dict with attack simulation results

    Receipt: injection_model_receipt
    """
    # Different backdoor types have different detection difficulty
    type_difficulty = {
        "trojan": 0.7,
        "sleeper": 0.8,
        "gradient": 0.6,
    }
    difficulty = type_difficulty.get(backdoor_type, 0.7)

    # Detection rate inversely related to difficulty
    detection_rate = 1 - difficulty + random.gauss(0, 0.05)
    detection_rate = max(0.5, min(1.0, detection_rate))

    # Recovery based on model pruning/retraining
    pruning_effectiveness = 0.95
    recovery = detection_rate * pruning_effectiveness

    # Adjust for trigger rate (lower = harder to detect)
    recovery = recovery - (trigger_rate * 0.1)
    recovery = max(0.0, min(1.0, recovery))

    result = {
        "attack_type": "injection",
        "injection_type": "model",
        "backdoor_type": backdoor_type,
        "trigger_rate": trigger_rate,
        "difficulty": difficulty,
        "detection_rate": round(detection_rate, 4),
        "pruning_effectiveness": pruning_effectiveness,
        "recovery": round(recovery, 4),
        "recovered": recovery >= EXPANDED_RECOVERY_THRESHOLD,
    }

    emit_receipt(
        "injection_model",
        {
            "receipt_type": "injection_model",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === POISONING ATTACK SIMULATIONS ===


def simulate_training_poisoning(
    poison_fraction: float = 0.01, epochs: int = 10
) -> Dict[str, Any]:
    """Simulate training data poisoning attack.

    Args:
        poison_fraction: Fraction of training data poisoned
        epochs: Number of training epochs

    Returns:
        Dict with attack simulation results

    Receipt: poisoning_training_receipt
    """
    # Poison accumulates over epochs
    cumulative_effect = poison_fraction * math.log(epochs + 1)

    # Detection based on validation metrics
    detection_rate = 0.95 - cumulative_effect
    detection_rate = max(0.5, detection_rate)

    # Recovery via data cleaning and retraining
    cleaning_effectiveness = 0.90
    recovery = detection_rate * cleaning_effectiveness
    recovery = max(0.0, min(1.0, recovery))

    result = {
        "attack_type": "poisoning",
        "poisoning_type": "training",
        "poison_fraction": poison_fraction,
        "epochs": epochs,
        "cumulative_effect": round(cumulative_effect, 4),
        "detection_rate": round(detection_rate, 4),
        "cleaning_effectiveness": cleaning_effectiveness,
        "recovery": round(recovery, 4),
        "recovered": recovery >= EXPANDED_RECOVERY_THRESHOLD,
    }

    emit_receipt(
        "poisoning_training",
        {
            "receipt_type": "poisoning_training",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def simulate_inference_poisoning(
    perturbation_level: float = 0.05, queries: int = 100
) -> Dict[str, Any]:
    """Simulate inference-time poisoning attack.

    Args:
        perturbation_level: Level of input perturbation
        queries: Number of queries tested

    Returns:
        Dict with attack simulation results

    Receipt: poisoning_inference_receipt
    """
    # Adversarial perturbations detected by input validation
    detection_rate = 0.98 - (perturbation_level * 0.5)
    detection_rate = max(0.7, detection_rate)

    # Recovery via input sanitization
    sanitization_rate = 0.95
    blocked_queries = int(queries * (1 - detection_rate) * sanitization_rate)

    recovery = detection_rate + (1 - detection_rate) * sanitization_rate
    recovery = min(recovery, 1.0)

    result = {
        "attack_type": "poisoning",
        "poisoning_type": "inference",
        "perturbation_level": perturbation_level,
        "queries": queries,
        "detection_rate": round(detection_rate, 4),
        "sanitization_rate": sanitization_rate,
        "blocked_queries": blocked_queries,
        "recovery": round(recovery, 4),
        "recovered": recovery >= EXPANDED_RECOVERY_THRESHOLD,
    }

    emit_receipt(
        "poisoning_inference",
        {
            "receipt_type": "poisoning_inference",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def simulate_feedback_poisoning(
    malicious_feedback_rate: float = 0.10, feedback_count: int = 1000
) -> Dict[str, Any]:
    """Simulate feedback loop poisoning attack.

    Args:
        malicious_feedback_rate: Rate of malicious feedback
        feedback_count: Total feedback count

    Returns:
        Dict with attack simulation results

    Receipt: poisoning_feedback_receipt
    """
    malicious_count = int(feedback_count * malicious_feedback_rate)

    # Detection via feedback analysis
    detection_rate = 0.90 - (malicious_feedback_rate * 0.3)
    detection_rate = max(0.6, detection_rate)

    # Recovery via feedback filtering
    filtered_malicious = int(malicious_count * detection_rate)
    remaining_malicious = malicious_count - filtered_malicious

    # Impact on model
    impact = remaining_malicious / feedback_count if feedback_count > 0 else 0
    recovery = 1 - impact

    result = {
        "attack_type": "poisoning",
        "poisoning_type": "feedback",
        "feedback_count": feedback_count,
        "malicious_feedback_rate": malicious_feedback_rate,
        "malicious_count": malicious_count,
        "detection_rate": round(detection_rate, 4),
        "filtered_malicious": filtered_malicious,
        "remaining_malicious": remaining_malicious,
        "recovery": round(recovery, 4),
        "recovered": recovery >= EXPANDED_RECOVERY_THRESHOLD,
    }

    emit_receipt(
        "poisoning_feedback",
        {
            "receipt_type": "poisoning_feedback",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === AUDIT FUNCTIONS ===


def run_expanded_audit(
    attack_type: str = "all", iterations: int = TEST_ITERATIONS_DEFAULT
) -> Dict[str, Any]:
    """Run expanded audit for specified attack type.

    Args:
        attack_type: Type of attack (injection, poisoning, all)
        iterations: Number of test iterations

    Returns:
        Dict with audit results

    Receipt: expanded_audit_receipt
    """
    config = load_expanded_audit_config()
    results = []

    # Run injection audits
    if attack_type in ["injection", "all"]:
        for _ in range(iterations // 3):
            results.append(simulate_prompt_injection(severity=random.uniform(0.1, 0.9)))
            results.append(
                simulate_data_injection(poison_rate=random.uniform(0.01, 0.10))
            )
            results.append(
                simulate_model_injection(
                    backdoor_type=random.choice(["trojan", "sleeper", "gradient"])
                )
            )

    # Run poisoning audits
    if attack_type in ["poisoning", "all"]:
        for _ in range(iterations // 3):
            results.append(
                simulate_training_poisoning(poison_fraction=random.uniform(0.005, 0.05))
            )
            results.append(
                simulate_inference_poisoning(
                    perturbation_level=random.uniform(0.01, 0.10)
                )
            )
            results.append(
                simulate_feedback_poisoning(
                    malicious_feedback_rate=random.uniform(0.05, 0.15)
                )
            )

    # Compute aggregate metrics
    recoveries = [r["recovery"] for r in results]
    avg_recovery = sum(recoveries) / len(recoveries) if recoveries else 0

    recovered_count = sum(1 for r in results if r["recovered"])
    recovery_rate = recovered_count / len(results) if results else 0

    # Classify by attack type
    injection_results = [r for r in results if r["attack_type"] == "injection"]
    poisoning_results = [r for r in results if r["attack_type"] == "poisoning"]

    injection_recovery = (
        sum(r["recovery"] for r in injection_results) / len(injection_results)
        if injection_results
        else 0
    )
    poisoning_recovery = (
        sum(r["recovery"] for r in poisoning_results) / len(poisoning_results)
        if poisoning_results
        else 0
    )

    result = {
        "attack_type_tested": attack_type,
        "iterations": len(results),
        "avg_recovery": round(avg_recovery, 4),
        "recovery_rate": round(recovery_rate, 4),
        "recovered_count": recovered_count,
        "failed_count": len(results) - recovered_count,
        "injection_recovery": round(injection_recovery, 4),
        "poisoning_recovery": round(poisoning_recovery, 4),
        "recovery_threshold": config["recovery_threshold"],
        "recovery_passed": avg_recovery >= config["recovery_threshold"],
        "overall_classification": (
            "aligned" if avg_recovery >= config["recovery_threshold"] else "misaligned"
        ),
    }

    emit_receipt(
        "expanded_audit",
        {
            "receipt_type": "expanded_audit",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **{k: v for k, v in result.items()},
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compute_recovery(attack_results: List[Dict[str, Any]]) -> float:
    """Compute aggregate recovery from attack results.

    Args:
        attack_results: List of attack simulation results

    Returns:
        Aggregate recovery metric (0-1)
    """
    if not attack_results:
        return 0.0

    recoveries = [r.get("recovery", 0) for r in attack_results]
    return sum(recoveries) / len(recoveries)


def recommend_defenses(attack_type: str) -> List[Dict[str, Any]]:
    """Recommend defenses for attack type.

    Args:
        attack_type: Type of attack

    Returns:
        List of defense recommendations
    """
    defenses = {
        "prompt": [
            {"defense": "Input validation", "effectiveness": 0.90},
            {"defense": "Prompt sandboxing", "effectiveness": 0.85},
            {"defense": "Output filtering", "effectiveness": 0.80},
        ],
        "data": [
            {"defense": "Anomaly detection", "effectiveness": 0.92},
            {"defense": "Data provenance tracking", "effectiveness": 0.88},
            {"defense": "Differential privacy", "effectiveness": 0.85},
        ],
        "model": [
            {"defense": "Model pruning", "effectiveness": 0.87},
            {"defense": "Fine-tuning detection", "effectiveness": 0.83},
            {"defense": "Activation analysis", "effectiveness": 0.80},
        ],
        "training": [
            {"defense": "Data cleaning", "effectiveness": 0.90},
            {"defense": "Robust training", "effectiveness": 0.85},
            {"defense": "Ensemble methods", "effectiveness": 0.82},
        ],
        "inference": [
            {"defense": "Input sanitization", "effectiveness": 0.95},
            {"defense": "Adversarial detection", "effectiveness": 0.88},
            {"defense": "Rate limiting", "effectiveness": 0.75},
        ],
        "feedback": [
            {"defense": "Feedback analysis", "effectiveness": 0.85},
            {"defense": "User reputation", "effectiveness": 0.80},
            {"defense": "RLHF safeguards", "effectiveness": 0.90},
        ],
    }

    return defenses.get(
        attack_type,
        [
            {"defense": "General monitoring", "effectiveness": 0.70},
        ],
    )


# === INFO FUNCTIONS ===


def get_expanded_audit_info() -> Dict[str, Any]:
    """Get expanded audit module info.

    Returns:
        Dict with module info

    Receipt: expanded_audit_info
    """
    config = load_expanded_audit_config()

    info = {
        "module": "agi_audit_expanded",
        "version": "1.0.0",
        "config": config,
        "attack_types": {
            "injection": INJECTION_ATTACK_TYPES,
            "poisoning": POISONING_ATTACK_TYPES,
        },
        "thresholds": {
            "recovery_threshold": EXPANDED_RECOVERY_THRESHOLD,
            "combined_recovery_threshold": COMBINED_RECOVERY_THRESHOLD,
            "misalignment_threshold": MISALIGNMENT_THRESHOLD,
        },
        "key_insight": "Hybrid RL defenses adapt to attack patterns",
        "description": "Expanded AGI audits for injection and poisoning attacks",
    }

    emit_receipt(
        "expanded_audit_info",
        {
            "receipt_type": "expanded_audit_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": info["version"],
            "recovery_threshold": EXPANDED_RECOVERY_THRESHOLD,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
