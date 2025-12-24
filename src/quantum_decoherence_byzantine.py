"""Decoherence-based Byzantine fault detection for D19.

Maps quantum decoherence to Byzantine behavior.

Key insight: In classical systems, Byzantine detection requires 3f+1 nodes.
In quantum systems, DECOHERENCE ITSELF is the signal.
If a node's entanglement decoheres faster than physics predicts,
it's behaving non-deterministically (Byzantine).
"""

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from .core import emit_receipt, dual_hash, TENANT_ID
from .quantum_entangled_consensus import QuantumConsensus

# === D19 BYZANTINE DETECTION CONSTANTS ===

NATURAL_DECOHERENCE_RATE = 0.0001
"""Natural decoherence rate per second (physics model)."""

ANOMALY_THRESHOLD = 2.0
"""Multiplier above natural rate to flag as anomalous."""

BYZANTINE_TYPES = ["crash", "omission", "commission", "arbitrary"]
"""Types of Byzantine behavior."""


@dataclass
class ByzantineNode:
    """Node flagged as potentially Byzantine."""

    node_id: str
    behavior_type: str
    decoherence_rate: float
    anomaly_score: float
    flagged_at: str
    quarantined: bool = False
    recovery_attempts: int = 0


@dataclass
class ByzantineDetector:
    """Byzantine fault detector using decoherence."""

    detector_id: str
    qc: QuantumConsensus
    flagged_nodes: Dict[str, ByzantineNode] = field(default_factory=dict)
    quarantined_nodes: Dict[str, ByzantineNode] = field(default_factory=dict)
    recovered_nodes: List[str] = field(default_factory=list)


def init_byzantine_detector(qc: QuantumConsensus) -> ByzantineDetector:
    """Initialize Byzantine detector.

    Args:
        qc: QuantumConsensus instance

    Returns:
        ByzantineDetector instance
    """
    detector_id = str(uuid.uuid4())[:8]
    return ByzantineDetector(detector_id=detector_id, qc=qc)


def measure_decoherence_rate(bd: ByzantineDetector, node_id: str) -> float:
    """Measure decoherence rate for node.

    Args:
        bd: ByzantineDetector instance
        node_id: Node identifier

    Returns:
        Measured decoherence rate

    Receipt: decoherence_rate_receipt
    """
    # Find pairs involving this node
    rates = []
    for pair_id in bd.qc.node_pairs.get(node_id, []):
        pair = bd.qc.pairs.get(pair_id)
        if pair:
            rates.append(pair.decoherence_rate)

    # Average decoherence rate
    rate = sum(rates) / len(rates) if rates else NATURAL_DECOHERENCE_RATE

    emit_receipt(
        "decoherence_rate",
        {
            "receipt_type": "decoherence_rate",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "detector_id": bd.detector_id,
            "node_id": node_id,
            "rate": round(rate, 6),
            "pairs_measured": len(rates),
            "payload_hash": dual_hash(
                json.dumps({"node_id": node_id, "rate": rate}, sort_keys=True)
            ),
        },
    )

    return rate


def predict_natural_decoherence(bd: ByzantineDetector, node_id: str) -> float:
    """Predict natural decoherence rate based on physics model.

    Args:
        bd: ByzantineDetector instance
        node_id: Node identifier

    Returns:
        Predicted natural decoherence rate
    """
    # Physics model: natural decoherence based on environment
    # In production, would consider temperature, isolation, etc.
    base_rate = NATURAL_DECOHERENCE_RATE

    # Slight variation per node
    node_factor = hash(node_id) % 100 / 10000
    predicted = base_rate + node_factor

    return predicted


def detect_anomalous_decoherence(bd: ByzantineDetector, node_id: str) -> bool:
    """Detect if node shows anomalous decoherence.

    Anomaly = decoherence faster than physics predicts.

    Args:
        bd: ByzantineDetector instance
        node_id: Node identifier

    Returns:
        True if anomalous decoherence detected

    Receipt: anomaly_detection_receipt
    """
    measured_rate = measure_decoherence_rate(bd, node_id)
    predicted_rate = predict_natural_decoherence(bd, node_id)

    # Anomaly score: how many times faster than predicted
    anomaly_score = measured_rate / predicted_rate if predicted_rate > 0 else 1.0

    is_anomalous = anomaly_score > ANOMALY_THRESHOLD

    if is_anomalous:
        # Flag node as potentially Byzantine
        now = datetime.utcnow().isoformat() + "Z"
        bd.flagged_nodes[node_id] = ByzantineNode(
            node_id=node_id,
            behavior_type="unknown",
            decoherence_rate=measured_rate,
            anomaly_score=anomaly_score,
            flagged_at=now,
        )

    emit_receipt(
        "anomaly_detection",
        {
            "receipt_type": "anomaly_detection",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "detector_id": bd.detector_id,
            "node_id": node_id,
            "measured_rate": round(measured_rate, 6),
            "predicted_rate": round(predicted_rate, 6),
            "anomaly_score": round(anomaly_score, 4),
            "is_anomalous": is_anomalous,
            "payload_hash": dual_hash(
                json.dumps(
                    {"node_id": node_id, "anomalous": is_anomalous}, sort_keys=True
                )
            ),
        },
    )

    return is_anomalous


def classify_byzantine_behavior(bd: ByzantineDetector, node_id: str) -> str:
    """Classify type of Byzantine behavior.

    Types:
    - crash: Node stops responding
    - omission: Node drops messages
    - commission: Node sends incorrect messages
    - arbitrary: Node behaves unpredictably

    Args:
        bd: ByzantineDetector instance
        node_id: Node identifier

    Returns:
        Classification: "crash" | "omission" | "commission" | "arbitrary"

    Receipt: byzantine_classification_receipt
    """
    flagged = bd.flagged_nodes.get(node_id)
    if not flagged:
        return "none"

    # Classify based on anomaly score pattern
    score = flagged.anomaly_score

    if score > 10:
        classification = "crash"  # Complete failure
    elif score > 5:
        classification = "arbitrary"  # Highly unpredictable
    elif score > 3:
        classification = "commission"  # Active misbehavior
    else:
        classification = "omission"  # Passive misbehavior

    flagged.behavior_type = classification

    emit_receipt(
        "byzantine_classification",
        {
            "receipt_type": "byzantine_classification",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "detector_id": bd.detector_id,
            "node_id": node_id,
            "classification": classification,
            "anomaly_score": round(score, 4),
            "payload_hash": dual_hash(
                json.dumps({"node_id": node_id, "type": classification}, sort_keys=True)
            ),
        },
    )

    return classification


def quarantine_byzantine_node(bd: ByzantineDetector, node_id: str) -> Dict[str, Any]:
    """Quarantine Byzantine node from consensus.

    Args:
        bd: ByzantineDetector instance
        node_id: Node identifier

    Returns:
        Quarantine result

    Receipt: quarantine_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    flagged = bd.flagged_nodes.get(node_id)
    if flagged:
        flagged.quarantined = True
        bd.quarantined_nodes[node_id] = flagged

    # Remove node from consensus pairs
    removed_pairs = 0
    for pair_id in list(bd.qc.node_pairs.get(node_id, [])):
        if pair_id in bd.qc.pairs:
            del bd.qc.pairs[pair_id]
            removed_pairs += 1

    # Clear node's pair list
    if node_id in bd.qc.node_pairs:
        del bd.qc.node_pairs[node_id]

    result = {
        "node_id": node_id,
        "quarantined": True,
        "pairs_removed": removed_pairs,
        "quarantined_at": now,
        "total_quarantined": len(bd.quarantined_nodes),
    }

    emit_receipt(
        "quarantine",
        {
            "receipt_type": "quarantine",
            "tenant_id": TENANT_ID,
            "ts": now,
            "detector_id": bd.detector_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def recover_byzantine_node(bd: ByzantineDetector, node_id: str) -> Dict[str, Any]:
    """Attempt to recover Byzantine node.

    Args:
        bd: ByzantineDetector instance
        node_id: Node identifier

    Returns:
        Recovery result

    Receipt: recovery_receipt
    """
    now = datetime.utcnow().isoformat() + "Z"

    quarantined = bd.quarantined_nodes.get(node_id)
    if not quarantined:
        return {"error": "node_not_quarantined", "node_id": node_id}

    quarantined.recovery_attempts += 1

    # Simulate recovery attempt
    # Success based on behavior type and attempts
    success_prob = {
        "crash": 0.3,
        "omission": 0.5,
        "commission": 0.4,
        "arbitrary": 0.2,
    }.get(quarantined.behavior_type, 0.3)

    # More attempts = lower success
    success_prob *= 1 / (quarantined.recovery_attempts + 1)

    recovered = random.random() < success_prob

    if recovered:
        # Remove from quarantine
        del bd.quarantined_nodes[node_id]
        bd.recovered_nodes.append(node_id)

        # Remove from flagged
        if node_id in bd.flagged_nodes:
            del bd.flagged_nodes[node_id]

    result = {
        "node_id": node_id,
        "recovered": recovered,
        "recovery_attempts": quarantined.recovery_attempts,
        "behavior_type": quarantined.behavior_type,
        "total_recovered": len(bd.recovered_nodes),
    }

    emit_receipt(
        "recovery",
        {
            "receipt_type": "recovery",
            "tenant_id": TENANT_ID,
            "ts": now,
            "detector_id": bd.detector_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_byzantine_status() -> Dict[str, Any]:
    """Get current Byzantine detection status.

    Returns:
        Byzantine detection status dict
    """
    return {
        "module": "quantum_decoherence_byzantine",
        "version": "19.0.0",
        "natural_decoherence_rate": NATURAL_DECOHERENCE_RATE,
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "byzantine_types": BYZANTINE_TYPES,
        "detection_mode": "decoherence_based",
    }
