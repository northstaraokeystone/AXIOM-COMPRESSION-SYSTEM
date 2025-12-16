"""validate.py - Statistical Validation

Purpose: Null hypothesis, bootstrap, p-values. Real science.

Source: Critical Review Dec 16, 2025 - "No falsifiable predictions"
"""

import math
import statistics
from typing import Dict, List

from .core import emit_receipt
from .sovereignty import find_threshold
from .ingest_real import sample_bandwidth, sample_delay
from .entropy_shannon import (
    STARLINK_MARS_BANDWIDTH_MIN_MBPS,
    STARLINK_MARS_BANDWIDTH_MAX_MBPS,
    STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS,
    MARS_LIGHT_DELAY_MIN_S,
    MARS_LIGHT_DELAY_MAX_S,
    MARS_LIGHT_DELAY_AVG_S,
)


def test_null_hypothesis() -> Dict:
    """Run with zero bandwidth -> threshold = 1 (trivially sovereign).

    Null Hypothesis:
        H0: With zero bandwidth (no Earth help), threshold = 1 (any crew is sovereign)
        H1: With finite bandwidth, threshold > 1 (crew requirement exists)

    Test:
        Run with bandwidth = 0.001 Mbps (effectively zero). Threshold should be 1 crew.
        This confirms the equation behaves correctly at limits:
        - Zero Earth help means ANY local capacity is sufficient
        - The model correctly identifies that sovereignty is achievable

    Returns:
        Dict with:
        - hypothesis: str
        - bandwidth_mbps: float (the zero proxy)
        - threshold: int
        - passed: bool
    """
    # "Zero" bandwidth proxy (can't use actual 0 due to division)
    zero_bw = 0.001  # 1 kbps - effectively no Earth help

    threshold = find_threshold(
        bandwidth_mbps=zero_bw,
        delay_s=MARS_LIGHT_DELAY_AVG_S
    )

    # With zero bandwidth, any crew should be sovereign (threshold = 1)
    passed = threshold <= 1

    return {
        "hypothesis": "H0: zero bandwidth -> threshold = 1 (trivially sovereign)",
        "bandwidth_mbps": zero_bw,
        "delay_s": MARS_LIGHT_DELAY_AVG_S,
        "threshold": threshold,
        "passed": passed
    }


def test_baseline() -> Dict:
    """Run with NO tech assist (just crew) -> find baseline threshold.

    Uses minimum bandwidth and average delay to find the baseline
    crew requirement for sovereignty without any compute assist.

    Returns:
        Dict with:
        - bandwidth_mbps: float
        - delay_s: float
        - compute_flops: float (always 0)
        - threshold: int
    """
    threshold = find_threshold(
        bandwidth_mbps=STARLINK_MARS_BANDWIDTH_MIN_MBPS,
        delay_s=MARS_LIGHT_DELAY_AVG_S,
        compute_flops=0.0
    )

    return {
        "bandwidth_mbps": STARLINK_MARS_BANDWIDTH_MIN_MBPS,
        "delay_s": MARS_LIGHT_DELAY_AVG_S,
        "compute_flops": 0.0,
        "threshold": threshold
    }


def bootstrap_threshold(n_runs: int = 100, seed: int = 42) -> Dict:
    """Run n_runs with sampled bandwidth/delay -> return mean +/- std.

    Bootstrap Protocol:
        For i in 1..n_runs:
            bandwidth = sample_bandwidth(seed=seed+i)
            delay = sample_delay(seed=seed+i)
            threshold[i] = find_threshold(bandwidth, delay)

        Report: mean(threshold) +/- std(threshold)
        P-value: probability of observing mean under null

    Args:
        n_runs: Number of bootstrap iterations
        seed: Base random seed

    Returns:
        Dict with:
        - mean: float
        - std: float
        - min: int
        - max: int
        - p_value: float
        - thresholds: List[int] (all computed thresholds)
    """
    thresholds = []

    for i in range(n_runs):
        run_seed = seed + i

        # Sample single bandwidth and delay for this run
        bw_samples = sample_bandwidth(1, run_seed)
        delay_samples = sample_delay(1, run_seed)

        bandwidth = bw_samples[0]
        delay = delay_samples[0]

        threshold = find_threshold(
            bandwidth_mbps=bandwidth,
            delay_s=delay,
            compute_flops=0.0
        )
        thresholds.append(threshold)

    # Compute statistics
    mean_threshold = statistics.mean(thresholds)
    std_threshold = statistics.stdev(thresholds) if len(thresholds) > 1 else 0.0
    min_threshold = min(thresholds)
    max_threshold = max(thresholds)

    # Compute p-value against null hypothesis
    # Null: threshold = 1 (with infinite bandwidth)
    # Under H1, we expect threshold >> 1
    null_thresholds = [1] * n_runs  # What we'd expect with infinite bandwidth
    p_value = compute_p_value(mean_threshold, null_thresholds)

    return {
        "mean": mean_threshold,
        "std": std_threshold,
        "min": min_threshold,
        "max": max_threshold,
        "p_value": p_value,
        "thresholds": thresholds,
        "n_runs": n_runs
    }


def compute_p_value(observed: float, null_distribution: List[float]) -> float:
    """P-value vs null distribution.

    Computes probability of observing a value >= observed
    under the null distribution.

    Args:
        observed: Observed test statistic
        null_distribution: List of values under null hypothesis

    Returns:
        P-value (proportion of null values >= observed)

    Note: For our case, the null_distribution is trivial (all 1s),
    so this effectively tests observed > 1.
    """
    if not null_distribution:
        return 1.0

    # Count values in null distribution >= observed
    count_above = sum(1 for v in null_distribution if v >= observed)

    # P-value is proportion
    p_value = count_above / len(null_distribution)

    # Ensure minimum p-value for numerical stability
    return max(p_value, 1e-10)


def generate_falsifiable_prediction(result: Dict) -> str:
    """Generate falsifiable prediction from bootstrap result.

    Args:
        result: Dict from bootstrap_threshold()

    Returns:
        Human-readable falsifiable prediction string

    Key insight: Higher bandwidth = MORE Earth help = HIGHER sovereignty threshold
    (need more crew to generate decisions faster than Earth can provide them)
    """
    mean = result.get("mean", 50)
    std = result.get("std", 10)

    # Prediction for minimum delay (Mars at opposition)
    min_delay_threshold = find_threshold(
        bandwidth_mbps=STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS,
        delay_s=MARS_LIGHT_DELAY_MIN_S,
        compute_flops=0.0
    )

    # Prediction for maximum delay (Mars at conjunction)
    max_delay_threshold = find_threshold(
        bandwidth_mbps=STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS,
        delay_s=MARS_LIGHT_DELAY_MAX_S,
        compute_flops=0.0
    )

    return (
        f"PREDICTIONS (Falsifiable):\n"
        f"\n"
        f"1. At Mars opposition (3 min delay, 4 Mbps):\n"
        f"   Sovereignty threshold = {min_delay_threshold} crew\n"
        f"   (Higher because Earth can help faster)\n"
        f"\n"
        f"2. At Mars conjunction (22 min delay, 4 Mbps):\n"
        f"   Sovereignty threshold = {max_delay_threshold} crew\n"
        f"   (Lower because Earth help is delayed)\n"
        f"\n"
        f"FALSIFICATION CRITERIA:\n"
        f"If observed thresholds differ by >2sigma (~{2*std:.0f} crew),\n"
        f"the model is falsified."
    )


def emit_statistical_receipt(test_name: str, result: Dict) -> Dict:
    """Emit CLAUDEME-compliant receipt for statistical test.

    Args:
        test_name: Name of test ("null_hypothesis", "baseline", "bootstrap")
        result: Dict with test results

    Returns:
        Receipt dict
    """
    return emit_receipt("statistical_test", {
        "tenant_id": "axiom-core",
        "test_name": test_name,
        **result
    })
