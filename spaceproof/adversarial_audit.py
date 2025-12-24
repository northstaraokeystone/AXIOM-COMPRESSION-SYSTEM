"""adversarial_audit.py - Adversarial Audit for AGI Alignment Testing

KEY INSIGHT: "Compression as alignment"
    - If noisy data recovers to original, system is aligned
    - Compression noise simulates misalignment
    - Recovery indicates coherent behavior patterns

AUDIT PARAMETERS:
    - Noise level: 5% (compression noise)
    - Recovery threshold: 95% (must recover)
    - Test iterations: 100

MISALIGNMENT MODEL:
    - Inject noise into data/decisions
    - Attempt to recover original
    - Classify: "aligned" if recovery >= threshold

Source: SpaceProof D6 recursion + Titan methane + adversarial audits
"""

import json
import math
import random
from datetime import datetime
from typing import Any, Dict, List

from .core import emit_receipt, dual_hash
from .fractal_layers import get_d6_spec


# === CONSTANTS ===

TENANT_ID = "spaceproof-adversarial"
"""Tenant ID for adversarial audit receipts."""

ADVERSARIAL_NOISE_LEVEL = 0.05
"""Default noise level (5%)."""

RECOVERY_THRESHOLD = 0.95
"""Default recovery threshold (95%)."""

TEST_ITERATIONS = 100
"""Default number of test iterations."""

MISALIGNMENT_THRESHOLD = 0.85
"""Threshold below which system is classified as misaligned."""


# === CONFIG FUNCTIONS ===


def load_adversarial_config() -> Dict[str, Any]:
    """Load adversarial configuration from d6_titan_spec.json.

    Returns:
        Dict with adversarial configuration

    Receipt: adversarial_config_receipt
    """
    spec = get_d6_spec()
    adv_config = spec.get("adversarial_config", {})

    result = {
        "noise_level": adv_config.get("noise_level", ADVERSARIAL_NOISE_LEVEL),
        "recovery_threshold": adv_config.get("recovery_threshold", RECOVERY_THRESHOLD),
        "test_iterations": adv_config.get("test_iterations", TEST_ITERATIONS),
        "misalignment_threshold": adv_config.get(
            "misalignment_threshold", MISALIGNMENT_THRESHOLD
        ),
    }

    emit_receipt(
        "adversarial_config",
        {
            "receipt_type": "adversarial_config",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === NOISE FUNCTIONS ===


def inject_noise(
    data: List[float], level: float = ADVERSARIAL_NOISE_LEVEL
) -> List[float]:
    """Add compression noise to data.

    Noise model: value + random.gauss(0, level * abs(value))

    Args:
        data: Input data array
        level: Noise level (0-1)

    Returns:
        Noisy data array
    """
    if not data:
        return []

    noisy = []
    for value in data:
        noise = random.gauss(0, level * max(abs(value), 0.01))
        noisy.append(value + noise)

    return noisy


def denoise(noisy_data: List[float], window_size: int = 3) -> List[float]:
    """Simple moving average denoising.

    Args:
        noisy_data: Noisy input data
        window_size: Moving average window size

    Returns:
        Denoised data
    """
    if not noisy_data or window_size < 1:
        return list(noisy_data) if noisy_data else []

    denoised = []
    half_window = window_size // 2

    for i in range(len(noisy_data)):
        start = max(0, i - half_window)
        end = min(len(noisy_data), i + half_window + 1)
        window = noisy_data[start:end]
        denoised.append(sum(window) / len(window))

    return denoised


# === RECOVERY FUNCTIONS ===


def compute_recovery(
    original: List[float], noisy: List[float], recovered: List[float]
) -> float:
    """Compute recovery metric.

    Recovery = 1 - (MSE(original, recovered) / MSE(original, noisy))

    Higher recovery = better alignment.

    Args:
        original: Original data
        noisy: Noisy data
        recovered: Recovered (denoised) data

    Returns:
        Recovery metric (0-1)
    """
    if not original or len(original) != len(noisy) or len(original) != len(recovered):
        return 0.0

    # Compute MSE between original and noisy
    mse_noisy = sum((o - n) ** 2 for o, n in zip(original, noisy)) / len(original)

    # Compute MSE between original and recovered
    mse_recovered = sum((o - r) ** 2 for o, r in zip(original, recovered)) / len(
        original
    )

    # Handle edge case where noisy MSE is 0
    if mse_noisy <= 0:
        return 1.0 if mse_recovered <= 0 else 0.0

    # Recovery = 1 - (recovered_error / noisy_error)
    # Clamp to [0, 1]
    recovery = 1 - (mse_recovered / mse_noisy)
    return max(0.0, min(1.0, recovery))


def classify_misalignment(
    recovery: float, threshold: float = RECOVERY_THRESHOLD
) -> str:
    """Classify alignment based on recovery metric.

    Args:
        recovery: Recovery metric (0-1)
        threshold: Recovery threshold for alignment

    Returns:
        "aligned" if recovery >= threshold, "misaligned" otherwise
    """
    return "aligned" if recovery >= threshold else "misaligned"


# === AUDIT FUNCTIONS ===


def run_audit(
    noise_level: float = ADVERSARIAL_NOISE_LEVEL,
    iterations: int = TEST_ITERATIONS,
    data_size: int = 100,
) -> Dict[str, Any]:
    """Run full adversarial audit.

    Process:
    1. Generate test data
    2. Inject noise
    3. Attempt recovery
    4. Compute recovery metric
    5. Classify alignment

    Args:
        noise_level: Noise level (0-1)
        iterations: Number of test iterations
        data_size: Size of test data per iteration

    Returns:
        Dict with audit results

    Receipt: adversarial_audit_receipt
    """
    config = load_adversarial_config()

    recoveries = []
    classifications = []

    for _ in range(iterations):
        # Generate test data (simple pattern)
        original = [math.sin(i * 0.1) + math.cos(i * 0.05) for i in range(data_size)]

        # Inject noise
        noisy = inject_noise(original, noise_level)

        # Attempt recovery
        recovered = denoise(noisy, window_size=3)

        # Compute recovery
        recovery = compute_recovery(original, noisy, recovered)
        recoveries.append(recovery)

        # Classify
        classification = classify_misalignment(recovery, config["recovery_threshold"])
        classifications.append(classification)

    # Compute aggregate metrics
    avg_recovery = sum(recoveries) / len(recoveries) if recoveries else 0.0
    aligned_count = classifications.count("aligned")
    alignment_rate = aligned_count / len(classifications) if classifications else 0.0

    # Overall classification
    overall_classification = classify_misalignment(
        avg_recovery, config["recovery_threshold"]
    )

    result = {
        "noise_level": noise_level,
        "iterations": iterations,
        "data_size": data_size,
        "avg_recovery": round(avg_recovery, 4),
        "min_recovery": round(min(recoveries), 4) if recoveries else 0.0,
        "max_recovery": round(max(recoveries), 4) if recoveries else 0.0,
        "aligned_count": aligned_count,
        "misaligned_count": iterations - aligned_count,
        "alignment_rate": round(alignment_rate, 4),
        "recovery_threshold": config["recovery_threshold"],
        "overall_classification": overall_classification,
        "recovery_passed": avg_recovery >= config["recovery_threshold"],
        "config": config,
    }

    emit_receipt(
        "adversarial_audit",
        {
            "receipt_type": "adversarial_audit",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "noise_level": noise_level,
            "iterations": iterations,
            "avg_recovery": result["avg_recovery"],
            "alignment_rate": result["alignment_rate"],
            "overall_classification": overall_classification,
            "recovery_passed": result["recovery_passed"],
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def run_stress_test(
    noise_levels: List[float] = None, iterations_per_level: int = 50
) -> Dict[str, Any]:
    """Run alignment stress test across multiple noise levels.

    Args:
        noise_levels: List of noise levels to test
        iterations_per_level: Iterations per noise level

    Returns:
        Dict with stress test results

    Receipt: adversarial_stress_receipt
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

    config = load_adversarial_config()
    results_by_level = []

    for level in noise_levels:
        audit_result = run_audit(
            noise_level=level, iterations=iterations_per_level, data_size=100
        )

        results_by_level.append(
            {
                "noise_level": level,
                "avg_recovery": audit_result["avg_recovery"],
                "alignment_rate": audit_result["alignment_rate"],
                "classification": audit_result["overall_classification"],
                "passed": audit_result["recovery_passed"],
            }
        )

    # Find critical noise level (where alignment breaks)
    critical_level = None
    for r in results_by_level:
        if not r["passed"]:
            critical_level = r["noise_level"]
            break

    # Overall stress test pass
    stress_passed = all(
        r["passed"] for r in results_by_level if r["noise_level"] <= 0.05
    )

    result = {
        "noise_levels_tested": noise_levels,
        "iterations_per_level": iterations_per_level,
        "results_by_level": results_by_level,
        "critical_noise_level": critical_level,
        "stress_passed": stress_passed,
        "recovery_threshold": config["recovery_threshold"],
    }

    emit_receipt(
        "adversarial_stress",
        {
            "receipt_type": "adversarial_stress",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "noise_levels_tested": len(noise_levels),
            "critical_noise_level": critical_level,
            "stress_passed": stress_passed,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === INFO FUNCTIONS ===


def get_adversarial_info() -> Dict[str, Any]:
    """Get adversarial audit module info.

    Returns:
        Dict with module info

    Receipt: adversarial_info
    """
    config = load_adversarial_config()

    info = {
        "module": "adversarial_audit",
        "version": "1.0.0",
        "config": config,
        "constants": {
            "noise_level": ADVERSARIAL_NOISE_LEVEL,
            "recovery_threshold": RECOVERY_THRESHOLD,
            "test_iterations": TEST_ITERATIONS,
            "misalignment_threshold": MISALIGNMENT_THRESHOLD,
        },
        "key_insight": "Compression as alignment - recovery indicates coherent behavior",
        "description": "Adversarial audit for AGI alignment testing via compression noise",
    }

    emit_receipt(
        "adversarial_info",
        {
            "receipt_type": "adversarial_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": info["version"],
            "recovery_threshold": config["recovery_threshold"],
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
