"""Self-referencing pattern detection for D19.

A pattern is "alive" when it emits receipts that reference itself.
This is the core of autocatalysis in the swarm.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ..core import emit_receipt, dual_hash, TENANT_ID

# === D19 AUTOCATALYSIS CONSTANTS ===

SELF_REFERENCE_THRESHOLD = 0.70
"""Minimum self-reference score for autocatalysis."""

COHERENCE_THRESHOLD = 0.60
"""Minimum coherence for pattern to be considered alive."""

MIN_CYCLES_FOR_AUTOCATALYSIS = 10
"""Minimum cycles for pattern to achieve autocatalysis."""


@dataclass
class DetectedPattern:
    """Pattern detected in receipt stream."""

    pattern_id: str
    receipt_types: List[str] = field(default_factory=list)
    self_references: int = 0
    total_receipts: int = 0
    coherence: float = 0.0
    is_autocatalytic: bool = False
    cycles_active: int = 0
    first_detected: str = ""
    last_seen: str = ""


@dataclass
class PatternDetector:
    """Detector for self-referencing patterns."""

    detector_id: str
    patterns: Dict[str, DetectedPattern] = field(default_factory=dict)
    scan_count: int = 0


def init_detector(config: Dict = None) -> PatternDetector:
    """Initialize pattern detector.

    Args:
        config: Optional configuration dict

    Returns:
        PatternDetector instance
    """
    detector_id = str(uuid.uuid4())[:8]
    return PatternDetector(detector_id=detector_id)


def scan_receipt_stream(detector: PatternDetector, receipts: List[Dict]) -> List[Dict]:
    """Scan receipt stream for self-referencing patterns.

    Args:
        detector: PatternDetector instance
        receipts: List of receipts to scan

    Returns:
        List of detected patterns

    Receipt: pattern_scan_receipt
    """
    detector.scan_count += 1
    now = datetime.utcnow().isoformat() + "Z"

    # Group receipts by pattern (using receipt_type as pattern identifier)
    patterns_found = {}
    for receipt in receipts:
        rtype = receipt.get("receipt_type", "unknown")
        if rtype not in patterns_found:
            patterns_found[rtype] = {
                "receipts": [],
                "self_references": 0,
            }
        patterns_found[rtype]["receipts"].append(receipt)

        # Check for self-references
        payload = str(receipt.get("payload_hash", ""))
        for other_type in patterns_found:
            if other_type in payload or rtype in payload:
                patterns_found[rtype]["self_references"] += 1

    # Update detector patterns
    detected = []
    for pattern_type, data in patterns_found.items():
        if pattern_type not in detector.patterns:
            detector.patterns[pattern_type] = DetectedPattern(
                pattern_id=pattern_type, first_detected=now
            )

        pattern = detector.patterns[pattern_type]
        pattern.receipt_types = [pattern_type]
        pattern.total_receipts += len(data["receipts"])
        pattern.self_references += data["self_references"]
        pattern.last_seen = now
        pattern.cycles_active += 1

        detected.append(
            {
                "pattern_id": pattern_type,
                "receipts_in_scan": len(data["receipts"]),
                "self_references": data["self_references"],
            }
        )

    emit_receipt(
        "pattern_scan",
        {
            "receipt_type": "pattern_scan",
            "tenant_id": TENANT_ID,
            "ts": now,
            "detector_id": detector.detector_id,
            "scan_count": detector.scan_count,
            "receipts_scanned": len(receipts),
            "patterns_found": len(detected),
            "payload_hash": dual_hash(
                json.dumps({"scan": detector.scan_count}, sort_keys=True)
            ),
        },
    )

    return detected


def measure_self_reference(detector: PatternDetector, pattern: Dict) -> float:
    """Measure self-reference score for pattern.

    Self-reference = ratio of receipts that reference the pattern itself.

    Args:
        detector: PatternDetector instance
        pattern: Pattern dict

    Returns:
        Self-reference score 0-1
    """
    pattern_id = pattern.get("pattern_id", "")
    if pattern_id not in detector.patterns:
        return 0.0

    p = detector.patterns[pattern_id]
    if p.total_receipts == 0:
        return 0.0

    score = p.self_references / p.total_receipts
    return round(min(1.0, score), 4)


def detect_autocatalysis(detector: PatternDetector, pattern: Dict) -> bool:
    """Detect if pattern is autocatalytic.

    Autocatalysis requires:
    - Self-reference >= 0.70
    - Coherence >= 0.60
    - Active for >= 10 cycles

    Args:
        detector: PatternDetector instance
        pattern: Pattern dict

    Returns:
        True if pattern is autocatalytic

    Receipt: autocatalysis_detection_receipt
    """
    pattern_id = pattern.get("pattern_id", "")
    if pattern_id not in detector.patterns:
        return False

    p = detector.patterns[pattern_id]

    self_ref = measure_self_reference(detector, pattern)
    coherence = p.coherence
    cycles = p.cycles_active

    is_autocatalytic = (
        self_ref >= SELF_REFERENCE_THRESHOLD
        and coherence >= COHERENCE_THRESHOLD
        and cycles >= MIN_CYCLES_FOR_AUTOCATALYSIS
    )

    p.is_autocatalytic = is_autocatalytic

    emit_receipt(
        "autocatalysis_detection",
        {
            "receipt_type": "autocatalysis_detection",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "detector_id": detector.detector_id,
            "pattern_id": pattern_id,
            "self_reference": self_ref,
            "coherence": coherence,
            "cycles": cycles,
            "is_autocatalytic": is_autocatalytic,
            "payload_hash": dual_hash(
                json.dumps({"pattern_id": pattern_id}, sort_keys=True)
            ),
        },
    )

    return is_autocatalytic


def track_pattern_coherence(detector: PatternDetector, pattern: Dict) -> float:
    """Track pattern coherence over time.

    Coherence = consistency of pattern occurrence.

    Args:
        detector: PatternDetector instance
        pattern: Pattern dict

    Returns:
        Coherence score 0-1

    Receipt: coherence_measurement_receipt
    """
    pattern_id = pattern.get("pattern_id", "")
    if pattern_id not in detector.patterns:
        return 0.0

    p = detector.patterns[pattern_id]

    # Coherence based on receipt rate consistency
    if p.cycles_active == 0:
        coherence = 0.0
    else:
        avg_receipts_per_cycle = p.total_receipts / p.cycles_active
        # Higher consistency = higher coherence
        coherence = min(1.0, avg_receipts_per_cycle / 10)

    p.coherence = coherence

    emit_receipt(
        "coherence_measurement",
        {
            "receipt_type": "coherence_measurement",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "detector_id": detector.detector_id,
            "pattern_id": pattern_id,
            "coherence": round(coherence, 4),
            "cycles_active": p.cycles_active,
            "payload_hash": dual_hash(
                json.dumps({"pattern_id": pattern_id}, sort_keys=True)
            ),
        },
    )

    return coherence


def predict_pattern_fate(detector: PatternDetector, pattern: Dict) -> str:
    """Predict pattern's future: growing, stable, or dying.

    Args:
        detector: PatternDetector instance
        pattern: Pattern dict

    Returns:
        Fate prediction: "growing" | "stable" | "dying"

    Receipt: fate_prediction_receipt
    """
    pattern_id = pattern.get("pattern_id", "")
    if pattern_id not in detector.patterns:
        return "unknown"

    p = detector.patterns[pattern_id]

    # Predict based on coherence trend and autocatalysis
    if p.is_autocatalytic and p.coherence > 0.8:
        fate = "growing"
    elif p.is_autocatalytic or p.coherence > 0.5:
        fate = "stable"
    else:
        fate = "dying"

    emit_receipt(
        "fate_prediction",
        {
            "receipt_type": "fate_prediction",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "detector_id": detector.detector_id,
            "pattern_id": pattern_id,
            "fate": fate,
            "coherence": p.coherence,
            "is_autocatalytic": p.is_autocatalytic,
            "payload_hash": dual_hash(
                json.dumps({"pattern_id": pattern_id, "fate": fate}, sort_keys=True)
            ),
        },
    )

    return fate


def get_detector_status() -> Dict[str, Any]:
    """Get current detector status.

    Returns:
        Detector status dict
    """
    return {
        "module": "autocatalytic.pattern_detector",
        "version": "19.0.0",
        "self_reference_threshold": SELF_REFERENCE_THRESHOLD,
        "coherence_threshold": COHERENCE_THRESHOLD,
        "min_cycles": MIN_CYCLES_FOR_AUTOCATALYSIS,
    }
