"""orbital_compute.py - Orbital Compute Provenance (Starcloud Target).

D20 Production Evolution: Provenance receipts for in-space AI computation.

THE ORBITAL COMPUTE PARADIGM:
    GPU satellites process TB/day of sensor data.
    Radiation causes bit flips.
    Entropy accounting detects tampering.

    If H(output) >> H(expected), flag radiation event.

Target: Starcloud GPU satellites
Receipts: compute_provenance_receipt

SLOs:
    - 100% inferences have provenance receipts
    - 100% radiation events detected (entropy > threshold)
    - Entropy conservation |ΔS| < 0.01
    - Topology classification accuracy >= 95%

Source: Grok Research Starcloud pain points
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from spaceproof.core import dual_hash, emit_receipt, merkle

# === CONSTANTS ===

ORBITAL_COMPUTE_TENANT = "spaceproof-orbital-compute"

# Radiation detection thresholds
RADIATION_ENTROPY_THRESHOLD = 0.15  # 15% deviation flags radiation event
ENTROPY_CONSERVATION_LIMIT = 0.01  # |ΔS| must be < 0.01

# Starcloud specifications
STARCLOUD_BANDWIDTH_GBPS = 10.0  # TB/day processing
STARCLOUD_GPU_COUNT = 8  # GPUs per satellite


@dataclass
class IngestResult:
    """Result of raw data ingestion."""

    satellite_id: str
    input_hash: str
    data_size_bytes: int
    ingestion_timestamp: str
    receipt: Dict[str, Any]


@dataclass
class InferenceResult:
    """Result of AI inference execution."""

    satellite_id: str
    input_hash: str
    model_id: str
    model_version: str
    output_hash: str
    entropy_input: float
    entropy_output: float
    entropy_delta: float
    inference_time_ms: float
    receipt: Dict[str, Any]


@dataclass
class RadiationDetectionResult:
    """Result of radiation anomaly detection."""

    detected: bool
    expected_entropy: float
    actual_entropy: float
    deviation: float
    threshold: float
    receipt: Dict[str, Any]


@dataclass
class ProvenanceChain:
    """Complete compute provenance chain."""

    satellite_id: str
    chain_id: str
    receipts: List[Dict[str, Any]]
    merkle_anchor: str
    total_entropy_delta: float
    radiation_events: int
    provenance_receipt: Dict[str, Any]

    @property
    def receipt_count(self) -> int:
        """Return count of receipts in chain."""
        return len(self.receipts)


def _compute_entropy(data: bytes) -> float:
    """Compute normalized Shannon entropy of data.

    Args:
        data: Input bytes

    Returns:
        Normalized entropy (0.0 - 1.0)
    """
    if len(data) == 0:
        return 0.0

    # Byte frequency distribution
    from collections import Counter
    import math

    freq = Counter(data)
    total = len(data)

    # Shannon entropy
    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    # Normalize to 0-1 (max entropy for bytes is 8 bits)
    return entropy / 8.0


def ingest_raw_data(sensor_data: bytes, satellite_id: str) -> IngestResult:
    """Emit ingest receipt for raw sensor data.

    Args:
        sensor_data: Raw sensor data bytes
        satellite_id: Satellite identifier

    Returns:
        IngestResult with receipt
    """
    input_hash = dual_hash(sensor_data)
    timestamp = datetime.utcnow().isoformat() + "Z"

    receipt = emit_receipt(
        "data_ingest",
        {
            "tenant_id": ORBITAL_COMPUTE_TENANT,
            "satellite_id": satellite_id,
            "input_hash": input_hash,
            "data_size_bytes": len(sensor_data),
            "ingestion_timestamp": timestamp,
            "entropy": _compute_entropy(sensor_data),
        },
    )

    return IngestResult(
        satellite_id=satellite_id,
        input_hash=input_hash,
        data_size_bytes=len(sensor_data),
        ingestion_timestamp=timestamp,
        receipt=receipt,
    )


def execute_inference(
    input_hash: str,
    model_id: str,
    inference_result: Dict[str, Any],
    satellite_id: str = "unknown",
    model_version: str = "1.0.0",
    input_entropy: Optional[float] = None,
) -> InferenceResult:
    """Emit compute receipt for AI inference step.

    Args:
        input_hash: Dual-hash of input data
        model_id: Model identifier
        inference_result: Dictionary with inference outputs
        satellite_id: Satellite identifier
        model_version: Model version string
        input_entropy: Pre-computed input entropy (optional)

    Returns:
        InferenceResult with receipt
    """
    # Serialize result for hashing
    result_bytes = json.dumps(inference_result, sort_keys=True).encode()
    output_hash = dual_hash(result_bytes)

    # Compute output entropy
    entropy_output = _compute_entropy(result_bytes)
    entropy_input = input_entropy if input_entropy is not None else 0.5
    entropy_delta = entropy_output - entropy_input

    receipt = emit_receipt(
        "compute_inference",
        {
            "tenant_id": ORBITAL_COMPUTE_TENANT,
            "satellite_id": satellite_id,
            "input_hash": input_hash,
            "model_id": model_id,
            "model_version": model_version,
            "inference_result_hash": output_hash,
            "entropy_input": entropy_input,
            "entropy_output": entropy_output,
            "entropy_delta": entropy_delta,
        },
    )

    return InferenceResult(
        satellite_id=satellite_id,
        input_hash=input_hash,
        model_id=model_id,
        model_version=model_version,
        output_hash=output_hash,
        entropy_input=entropy_input,
        entropy_output=entropy_output,
        entropy_delta=entropy_delta,
        inference_time_ms=0.0,
        receipt=receipt,
    )


def detect_radiation_anomaly(
    expected_entropy: float,
    actual_entropy: float,
    threshold: float = RADIATION_ENTROPY_THRESHOLD,
) -> RadiationDetectionResult:
    """Flag radiation event if entropy spike exceeds threshold.

    Radiation-induced bit flips cause entropy anomalies.
    If |actual - expected| / expected > threshold, flag event.

    Args:
        expected_entropy: Expected output entropy
        actual_entropy: Actual measured entropy
        threshold: Deviation threshold (default 15%)

    Returns:
        RadiationDetectionResult with detection status
    """
    # Avoid division by zero
    if expected_entropy < 0.001:
        deviation = abs(actual_entropy) if actual_entropy > 0.001 else 0.0
    else:
        deviation = abs(actual_entropy - expected_entropy) / expected_entropy

    detected = deviation > threshold

    receipt = emit_receipt(
        "radiation_detection",
        {
            "tenant_id": ORBITAL_COMPUTE_TENANT,
            "expected_entropy": expected_entropy,
            "actual_entropy": actual_entropy,
            "deviation": deviation,
            "threshold": threshold,
            "radiation_detected": detected,
            "classification": "radiation_anomaly" if detected else "normal",
            "action": "flag_for_review" if detected else "continue",
        },
    )

    return RadiationDetectionResult(
        detected=detected,
        expected_entropy=expected_entropy,
        actual_entropy=actual_entropy,
        deviation=deviation,
        threshold=threshold,
        receipt=receipt,
    )


def emit_provenance_chain(
    receipts: List[Dict[str, Any]],
    satellite_id: str = "unknown",
) -> ProvenanceChain:
    """Merkle-anchor full compute pipeline into provenance chain.

    Args:
        receipts: List of receipt dictionaries to chain
        satellite_id: Satellite identifier

    Returns:
        ProvenanceChain with Merkle anchor
    """
    if not receipts:
        receipts = []

    # Compute Merkle anchor
    merkle_anchor = merkle(receipts)

    # Calculate total entropy delta
    total_entropy_delta = 0.0
    radiation_events = 0

    for r in receipts:
        if "entropy_delta" in r:
            total_entropy_delta += r.get("entropy_delta", 0.0)
        if r.get("radiation_detected", False):
            radiation_events += 1

    # Generate chain ID
    chain_id = dual_hash(merkle_anchor)

    provenance_receipt = emit_receipt(
        "compute_provenance",
        {
            "tenant_id": ORBITAL_COMPUTE_TENANT,
            "satellite_id": satellite_id,
            "chain_id": chain_id,
            "merkle_anchor": merkle_anchor,
            "receipt_count": len(receipts),
            "total_entropy_delta": total_entropy_delta,
            "radiation_events_detected": radiation_events,
            "entropy_conservation_valid": abs(total_entropy_delta) < ENTROPY_CONSERVATION_LIMIT,
        },
    )

    return ProvenanceChain(
        satellite_id=satellite_id,
        chain_id=chain_id,
        receipts=receipts,
        merkle_anchor=merkle_anchor,
        total_entropy_delta=total_entropy_delta,
        radiation_events=radiation_events,
        provenance_receipt=provenance_receipt,
    )


def verify_provenance(chain: ProvenanceChain) -> bool:
    """Verify provenance chain integrity.

    Args:
        chain: ProvenanceChain to verify

    Returns:
        True if chain is valid
    """
    # Recompute Merkle anchor
    recomputed = merkle(chain.receipts)
    return recomputed == chain.merkle_anchor


def compute_effectiveness(receipts: List[Dict[str, Any]]) -> float:
    """Compute effectiveness score for Meta-Loop topology classification.

    E = (H_before - H_after) / n_receipts

    High effectiveness (>= 0.90) indicates "open" topology → graduate.

    Args:
        receipts: List of receipts with entropy data

    Returns:
        Effectiveness score (0.0 - 1.0)
    """
    if not receipts:
        return 0.0

    total_reduction = 0.0
    count = 0

    for r in receipts:
        if "entropy_input" in r and "entropy_output" in r:
            reduction = r["entropy_input"] - r["entropy_output"]
            total_reduction += max(0, reduction)  # Only count positive reductions
            count += 1

    if count == 0:
        return 0.0

    # Normalize to 0-1 range (assume max reduction is 1.0 per receipt)
    effectiveness = total_reduction / count
    return min(1.0, max(0.0, effectiveness))
