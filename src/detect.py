"""detect.py - Entropy-Based Anomaly Detection

D20 Production Evolution: Stakeholder-intuitive name for entropy analysis.

THE DETECTION INSIGHT:
    Entropy is the universal accounting system.
    Fraud = entropy anomaly.
    Systems in distress leak information.

Source: AXIOM D20 Production Evolution

Stakeholder Value:
    - DOGE: "$31-162B improper payments detectable via entropy"
    - DOT: "Infrastructure fraud detection"

SLOs:
    - False positive rate: < 0.01
    - Detection latency: < 1 second
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .core import emit_receipt

# === CONSTANTS ===

TENANT_ID = "axiom-detect"

# Detection thresholds
FRAUD_THRESHOLD_SIGMA = 3.0  # 3 standard deviations
DRIFT_THRESHOLD_SIGMA = 1.5
DEGRADATION_THRESHOLD_SIGMA = 2.0

# DOGE-specific constants (from GAO reports)
IMPROPER_PAYMENTS_TOTAL_B = 162  # GAO FY2024
MEDICAID_IMPROPER_B = 31.1  # CMS FY2024
MEDICARE_FFS_IMPROPER_B = 31.7  # CMS FY2024


@dataclass
class BaselineStats:
    """Baseline statistics for anomaly detection."""

    mean: float
    std: float
    entropy: float
    n_samples: int


@dataclass
class DetectionResult:
    """Result of anomaly detection."""

    classification: str  # "normal", "drift", "degradation", "violation", "fraud"
    entropy_before: float
    entropy_after: float
    delta: float
    delta_sigma: float
    severity: str  # "low", "medium", "high", "critical"
    confidence: float


def shannon_entropy(distribution: np.ndarray) -> float:
    """Compute Shannon entropy.

    H = -sum(p(x) * log2(p(x)))

    Args:
        distribution: Probability distribution array (must sum to 1)

    Returns:
        Entropy in bits
    """
    # Remove zeros to avoid log(0)
    p = distribution[distribution > 0]

    if len(p) == 0:
        return 0.0

    # Normalize if needed
    p = p / np.sum(p)

    return -np.sum(p * np.log2(p))


def entropy_delta(before: np.ndarray, after: np.ndarray) -> float:
    """Compute change in entropy.

    Positive = gaining disorder (potential issue)
    Negative = gaining order (could be normal or suspicious)

    Args:
        before: Previous distribution
        after: Current distribution

    Returns:
        Change in entropy (after - before)
    """
    h_before = shannon_entropy(before)
    h_after = shannon_entropy(after)
    return h_after - h_before


def detect_anomaly(
    stream: np.ndarray, baseline: Optional[BaselineStats] = None
) -> DetectionResult:
    """Compare stream to baseline, return anomaly classification.

    Args:
        stream: Current data stream
        baseline: Optional baseline stats (computed if not provided)

    Returns:
        DetectionResult with classification and metrics
    """
    # Compute distribution from stream
    if len(stream) == 0:
        return DetectionResult(
            classification="normal",
            entropy_before=0.0,
            entropy_after=0.0,
            delta=0.0,
            delta_sigma=0.0,
            severity="low",
            confidence=1.0,
        )

    # Histogram-based distribution
    hist, _ = np.histogram(stream, bins=50, density=True)
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

    h_stream = shannon_entropy(hist)

    if baseline is None:
        # No baseline, can only report entropy
        return DetectionResult(
            classification="normal",
            entropy_before=0.0,
            entropy_after=h_stream,
            delta=0.0,
            delta_sigma=0.0,
            severity="low",
            confidence=0.5,  # Low confidence without baseline
        )

    # Compute delta in standard deviations
    delta = h_stream - baseline.entropy
    delta_sigma = abs(delta / baseline.std) if baseline.std > 0 else 0

    # Classify
    classification = classify_anomaly(delta_sigma, FRAUD_THRESHOLD_SIGMA)

    # Severity
    if delta_sigma >= FRAUD_THRESHOLD_SIGMA:
        severity = "critical"
    elif delta_sigma >= DEGRADATION_THRESHOLD_SIGMA:
        severity = "high"
    elif delta_sigma >= DRIFT_THRESHOLD_SIGMA:
        severity = "medium"
    else:
        severity = "low"

    # Confidence based on sample size
    confidence = min(1.0, baseline.n_samples / 1000)

    result = DetectionResult(
        classification=classification,
        entropy_before=baseline.entropy,
        entropy_after=h_stream,
        delta=delta,
        delta_sigma=delta_sigma,
        severity=severity,
        confidence=confidence,
    )

    # Emit detect receipt
    emit_receipt(
        "detect_receipt",
        {
            "tenant_id": TENANT_ID,
            "entropy_before": baseline.entropy,
            "entropy_after": h_stream,
            "delta": delta,
            "classification": classification,
            "severity": severity,
        },
    )

    return result


def classify_anomaly(
    delta_sigma: float, threshold: float = FRAUD_THRESHOLD_SIGMA
) -> str:
    """Classify anomaly based on delta in standard deviations.

    Args:
        delta_sigma: Absolute delta in standard deviations
        threshold: Fraud threshold (default 3 sigma)

    Returns:
        Classification: "drift", "degradation", "violation", "fraud", or "normal"
    """
    if delta_sigma >= threshold:
        return "fraud"
    elif delta_sigma >= DEGRADATION_THRESHOLD_SIGMA:
        return "violation"
    elif delta_sigma >= DRIFT_THRESHOLD_SIGMA:
        return "degradation"
    elif delta_sigma >= 1.0:
        return "drift"
    else:
        return "normal"


def build_baseline(samples: List[np.ndarray]) -> BaselineStats:
    """Build baseline statistics from historical samples.

    Args:
        samples: List of historical data arrays

    Returns:
        BaselineStats for anomaly detection
    """
    entropies = []

    for sample in samples:
        if len(sample) == 0:
            continue
        hist, _ = np.histogram(sample, bins=50, density=True)
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        entropies.append(shannon_entropy(hist))

    if len(entropies) == 0:
        return BaselineStats(mean=0.0, std=1.0, entropy=0.0, n_samples=0)

    return BaselineStats(
        mean=float(np.mean(entropies)),
        std=float(np.std(entropies)) if len(entropies) > 1 else 1.0,
        entropy=float(np.mean(entropies)),
        n_samples=len(entropies),
    )


def detect_fraud_pattern(
    transactions: List[Dict], threshold_sigma: float = FRAUD_THRESHOLD_SIGMA
) -> List[Dict]:
    """Detect fraud patterns in transaction data.

    DOGE use case: Identify improper payments via entropy analysis.

    Args:
        transactions: List of transaction dicts with 'amount' and 'category'
        threshold_sigma: Standard deviation threshold for fraud

    Returns:
        List of flagged transactions with fraud scores
    """
    if len(transactions) == 0:
        return []

    # Extract amounts by category
    categories: Dict[str, List[float]] = {}
    for tx in transactions:
        cat = tx.get("category", "unknown")
        amount = tx.get("amount", 0)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(amount)

    # Build baseline per category
    baselines: Dict[str, BaselineStats] = {}
    for cat, amounts in categories.items():
        if len(amounts) < 10:
            continue
        baselines[cat] = BaselineStats(
            mean=float(np.mean(amounts)),
            std=float(np.std(amounts)) if len(amounts) > 1 else 1.0,
            entropy=shannon_entropy(np.histogram(amounts, bins=20, density=True)[0]),
            n_samples=len(amounts),
        )

    # Flag anomalies
    flagged = []
    for tx in transactions:
        cat = tx.get("category", "unknown")
        amount = tx.get("amount", 0)

        if cat not in baselines:
            continue

        baseline = baselines[cat]
        z_score = abs(amount - baseline.mean) / baseline.std if baseline.std > 0 else 0

        if z_score >= threshold_sigma:
            flagged.append(
                {
                    **tx,
                    "fraud_score": float(z_score),
                    "classification": classify_anomaly(z_score, threshold_sigma),
                    "baseline_mean": baseline.mean,
                    "baseline_std": baseline.std,
                }
            )

    # Emit summary receipt
    if flagged:
        emit_receipt(
            "fraud_detection",
            {
                "tenant_id": TENANT_ID,
                "total_transactions": len(transactions),
                "flagged_count": len(flagged),
                "flagged_pct": len(flagged) / len(transactions) * 100,
                "threshold_sigma": threshold_sigma,
            },
        )

    return flagged


# === DOGE-SPECIFIC FUNCTIONS ===


def estimate_improper_payments(flagged: List[Dict]) -> Dict:
    """Estimate improper payment amounts from flagged transactions.

    Args:
        flagged: List of flagged transaction dicts

    Returns:
        Dict with estimated improper amounts
    """
    total_flagged = sum(tx.get("amount", 0) for tx in flagged)
    fraud_only = sum(
        tx.get("amount", 0) for tx in flagged if tx.get("classification") == "fraud"
    )

    # Apply confidence adjustment
    confidence = min(1.0, len(flagged) / 100)

    return {
        "total_flagged_amount": total_flagged,
        "fraud_amount": fraud_only,
        "confidence": confidence,
        "gao_baseline_b": IMPROPER_PAYMENTS_TOTAL_B,
        "potential_savings_pct": (fraud_only / total_flagged * 100)
        if total_flagged > 0
        else 0,
    }
