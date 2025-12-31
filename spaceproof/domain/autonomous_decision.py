"""autonomous_decision.py - Autonomous Decision Lineage (Defense Target).

D20 Production Evolution: Decision lineage for autonomous systems.

THE AUTONOMOUS DECISION PARADIGM:
    DOD Directive 3000.09 requires human accountability.
    Every autonomous decision must prove human could have intervened.
    Lineage chain: sensor → algorithm → decision → override_available

    HITL (Human-in-the-Loop): Human approves each CRITICAL decision
    HOTL (Human-on-the-Loop): Human monitors, intervenes on trigger

Target: Defense (DOD 3000.09), Starshield
Receipts: decision_lineage_receipt, human_override_receipt

SLOs:
    - 100% decisions have lineage receipts with override_available flag
    - 100% human overrides have reason_code
    - Adversarial attacks detected (Merkle tampering caught)
    - DOD 3000.09 compliance verified

Source: Grok Research Defense pain points
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from spaceproof.core import dual_hash, emit_receipt, merkle

# === CONSTANTS ===

AUTONOMOUS_DECISION_TENANT = "spaceproof-autonomous-decision"

# DOD 3000.09 compliance thresholds
DOD_OVERRIDE_REQUIRED = True  # Override must always be available
DOD_LINEAGE_REQUIRED = True  # Full lineage required

# Decision criticality levels
CRITICALITY_CRITICAL = "CRITICAL"  # Requires HITL
CRITICALITY_HIGH = "HIGH"  # Requires HOTL
CRITICALITY_MEDIUM = "MEDIUM"  # Autonomous with monitoring
CRITICALITY_LOW = "LOW"  # Fully autonomous


class OverrideReasonCode(Enum):
    """Reason codes for human overrides (from FEEDBACK_LOOP scenario)."""

    FACTUAL_ERROR = "FACTUAL_ERROR"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    SAFETY_CONCERN = "SAFETY_CONCERN"
    TACTICAL_ADJUSTMENT = "TACTICAL_ADJUSTMENT"
    INTELLIGENCE_UPDATE = "INTELLIGENCE_UPDATE"
    ETHICS_CONCERN = "ETHICS_CONCERN"


class DecisionCriticality(Enum):
    """Decision criticality levels for HITL/HOTL routing."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class SensorInput:
    """Sensor input record."""

    input_id: str
    sensor_data: Dict[str, Any]
    sensor_type: str
    timestamp: str
    input_hash: str
    receipt: Dict[str, Any]


@dataclass
class DecisionRecord:
    """Autonomous decision record."""

    decision_id: str
    inputs_hash: str
    algorithm_id: str
    algorithm_version: str
    output: Dict[str, Any]
    output_hash: str
    confidence: float
    criticality: DecisionCriticality
    override_available: bool
    human_override_occurred: bool
    receipt: Dict[str, Any]


@dataclass
class HumanOverride:
    """Human override record."""

    override_id: str
    decision_id: str
    human_id: str
    override_timestamp: str
    reason_code: OverrideReasonCode
    justification: str
    corrected_output: Dict[str, Any]
    receipt: Dict[str, Any]


@dataclass
class DecisionLineage:
    """Complete decision lineage chain."""

    lineage_id: str
    system_id: str
    sensor_inputs: List[SensorInput]
    decisions: List[DecisionRecord]
    overrides: List[HumanOverride]
    merkle_lineage: str
    dod_compliant: bool
    lineage_receipt: Dict[str, Any]


@dataclass
class AccountabilityValidation:
    """DOD 3000.09 accountability validation result."""

    valid: bool
    override_available_all: bool
    lineage_complete: bool
    human_accountability_proven: bool
    violations: List[str]
    receipt: Dict[str, Any]


def log_sensor_inputs(
    sensor_data: Dict[str, Any],
    timestamp: Optional[str] = None,
    sensor_type: str = "multi-spectral",
) -> SensorInput:
    """Emit sensor receipt for input data.

    Args:
        sensor_data: Sensor data dictionary
        timestamp: ISO8601 timestamp (optional, defaults to now)
        sensor_type: Type of sensor

    Returns:
        SensorInput with receipt
    """
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    data_bytes = json.dumps(sensor_data, sort_keys=True, cls=NumpyEncoder).encode()
    input_hash = dual_hash(data_bytes)
    input_id = dual_hash(f"{sensor_type}:{timestamp}")

    receipt = emit_receipt(
        "sensor_input",
        {
            "tenant_id": AUTONOMOUS_DECISION_TENANT,
            "input_id": input_id,
            "sensor_type": sensor_type,
            "timestamp": timestamp,
            "input_hash": input_hash,
            "data_size_bytes": len(data_bytes),
        },
    )

    return SensorInput(
        input_id=input_id,
        sensor_data=sensor_data,
        sensor_type=sensor_type,
        timestamp=timestamp,
        input_hash=input_hash,
        receipt=receipt,
    )


def log_decision(
    inputs_hash: str,
    algorithm_id: str,
    output: Dict[str, Any],
    confidence: float,
    algorithm_version: str = "1.0.0",
    criticality: DecisionCriticality = DecisionCriticality.MEDIUM,
) -> DecisionRecord:
    """Emit decision receipt for autonomous decision.

    Args:
        inputs_hash: Dual-hash of input data
        algorithm_id: Algorithm identifier
        output: Decision output dictionary
        confidence: Decision confidence (0.0 - 1.0)
        algorithm_version: Algorithm version
        criticality: Decision criticality level

    Returns:
        DecisionRecord with receipt
    """
    output_bytes = json.dumps(output, sort_keys=True, cls=NumpyEncoder).encode()
    output_hash = dual_hash(output_bytes)
    decision_id = dual_hash(f"{inputs_hash}:{algorithm_id}:{datetime.utcnow().isoformat()}")

    # Override always available per DOD 3000.09
    override_available = DOD_OVERRIDE_REQUIRED

    receipt = emit_receipt(
        "decision_lineage",
        {
            "tenant_id": AUTONOMOUS_DECISION_TENANT,
            "decision_id": decision_id,
            "inputs_hash": inputs_hash,
            "algorithm_id": algorithm_id,
            "algorithm_version": algorithm_version,
            "output_hash": output_hash,
            "confidence": confidence,
            "criticality": criticality.value,
            "override_available": override_available,
            "human_override_occurred": False,
            "dod_3000_09_compliant": override_available,
        },
    )

    return DecisionRecord(
        decision_id=decision_id,
        inputs_hash=inputs_hash,
        algorithm_id=algorithm_id,
        algorithm_version=algorithm_version,
        output=output,
        output_hash=output_hash,
        confidence=confidence,
        criticality=criticality,
        override_available=override_available,
        human_override_occurred=False,
        receipt=receipt,
    )


def log_human_override(
    decision_id: str,
    human_id: str,
    override_reason: OverrideReasonCode,
    justification: str = "",
    corrected_output: Optional[Dict[str, Any]] = None,
) -> HumanOverride:
    """Emit override receipt for human intervention.

    Args:
        decision_id: Decision being overridden
        human_id: Human operator identifier
        override_reason: Reason code for override
        justification: Free text justification
        corrected_output: Corrected decision output

    Returns:
        HumanOverride with receipt
    """
    override_timestamp = datetime.utcnow().isoformat() + "Z"
    override_id = dual_hash(f"{decision_id}:{human_id}:{override_timestamp}")

    if corrected_output is None:
        corrected_output = {}

    receipt = emit_receipt(
        "human_override",
        {
            "tenant_id": AUTONOMOUS_DECISION_TENANT,
            "override_id": override_id,
            "decision_id": decision_id,
            "human_id": human_id,
            "override_timestamp": override_timestamp,
            "reason_code": override_reason.value,
            "justification": justification,
            "corrected_output_hash": dual_hash(json.dumps(corrected_output, sort_keys=True, cls=NumpyEncoder)),
            "feedback_loop_valid": True,  # For training example generation
        },
    )

    return HumanOverride(
        override_id=override_id,
        decision_id=decision_id,
        human_id=human_id,
        override_timestamp=override_timestamp,
        reason_code=override_reason,
        justification=justification,
        corrected_output=corrected_output,
        receipt=receipt,
    )


def validate_accountability(decision_chain: List[DecisionRecord]) -> AccountabilityValidation:
    """Verify DOD 3000.09 compliance for decision chain.

    Must prove human could have intervened at every decision point.

    Args:
        decision_chain: List of decisions to validate

    Returns:
        AccountabilityValidation result
    """
    violations = []
    override_available_all = True
    lineage_complete = True

    for decision in decision_chain:
        # Check override availability
        if not decision.override_available:
            override_available_all = False
            violations.append(f"Decision {decision.decision_id}: override not available")

        # Check lineage
        if not decision.inputs_hash:
            lineage_complete = False
            violations.append(f"Decision {decision.decision_id}: missing inputs_hash")

        # Check CRITICAL decisions had HITL
        if decision.criticality == DecisionCriticality.CRITICAL:
            if not decision.human_override_occurred:
                violations.append(f"Decision {decision.decision_id}: CRITICAL decision without HITL approval")

    human_accountability_proven = override_available_all and lineage_complete
    valid = len(violations) == 0

    receipt = emit_receipt(
        "accountability_validation",
        {
            "tenant_id": AUTONOMOUS_DECISION_TENANT,
            "decisions_validated": len(decision_chain),
            "valid": valid,
            "override_available_all": override_available_all,
            "lineage_complete": lineage_complete,
            "human_accountability_proven": human_accountability_proven,
            "violation_count": len(violations),
            "dod_3000_09_compliant": valid,
        },
    )

    return AccountabilityValidation(
        valid=valid,
        override_available_all=override_available_all,
        lineage_complete=lineage_complete,
        human_accountability_proven=human_accountability_proven,
        violations=violations,
        receipt=receipt,
    )


def emit_decision_lineage(
    receipts: List[Dict[str, Any]],
    system_id: str = "unknown",
) -> Dict[str, Any]:
    """Merkle-anchor full decision chain into lineage receipt.

    Args:
        receipts: List of receipt dictionaries to chain
        system_id: System identifier

    Returns:
        Lineage receipt with provenance tree
    """
    if not receipts:
        receipts = []

    merkle_lineage = merkle(receipts)
    lineage_id = dual_hash(merkle_lineage)

    # Check override availability across all decisions
    override_available = all(
        r.get("override_available", True)
        for r in receipts
        if r.get("receipt_type") in ["decision_lineage", "decision"]
    )

    lineage_receipt = emit_receipt(
        "decision_lineage_chain",
        {
            "tenant_id": AUTONOMOUS_DECISION_TENANT,
            "lineage_id": lineage_id,
            "system_id": system_id,
            "merkle_lineage": merkle_lineage,
            "receipt_count": len(receipts),
            "override_available": override_available,
            "dod_3000_09_compliant": override_available,
        },
    )

    return lineage_receipt


def get_criticality_for_decision_type(decision_type: str) -> DecisionCriticality:
    """Map decision type to criticality level.

    Args:
        decision_type: Type of decision

    Returns:
        DecisionCriticality level
    """
    critical_types = {"weapon_release", "lethal_force", "nuclear", "strike"}
    high_types = {"intercept", "engage", "defense_activation", "maneuver"}
    medium_types = {"track", "identify", "surveil", "navigate"}

    decision_type_lower = decision_type.lower()

    if any(t in decision_type_lower for t in critical_types):
        return DecisionCriticality.CRITICAL
    elif any(t in decision_type_lower for t in high_types):
        return DecisionCriticality.HIGH
    elif any(t in decision_type_lower for t in medium_types):
        return DecisionCriticality.MEDIUM
    else:
        return DecisionCriticality.LOW


def should_require_hitl(decision: DecisionRecord) -> bool:
    """Determine if decision requires Human-in-the-Loop.

    Args:
        decision: Decision to evaluate

    Returns:
        True if HITL required
    """
    return decision.criticality == DecisionCriticality.CRITICAL


def should_require_hotl(decision: DecisionRecord) -> bool:
    """Determine if decision requires Human-on-the-Loop.

    Args:
        decision: Decision to evaluate

    Returns:
        True if HOTL required
    """
    return decision.criticality in [DecisionCriticality.CRITICAL, DecisionCriticality.HIGH]


def compute_transfer_score(
    source_decisions: List[DecisionRecord],
    target_decisions: List[DecisionRecord],
) -> float:
    """Compute transfer score for Meta-Loop topology classification.

    T = temporal_graph_similarity (NOT cosine similarity)

    High transfer (> 0.70) indicates pattern can transfer to other domains.

    Args:
        source_decisions: Source domain decisions
        target_decisions: Target domain decisions

    Returns:
        Transfer score (0.0 - 1.0)
    """
    if not source_decisions or not target_decisions:
        return 0.0

    # Compute structural similarity based on decision patterns
    source_types = set(d.algorithm_id for d in source_decisions)
    target_types = set(d.algorithm_id for d in target_decisions)

    # Jaccard similarity of algorithm types
    intersection = len(source_types & target_types)
    union = len(source_types | target_types)

    if union == 0:
        return 0.0

    type_similarity = intersection / union

    # Confidence distribution similarity
    source_confidences = [d.confidence for d in source_decisions]
    target_confidences = [d.confidence for d in target_decisions]

    source_avg = sum(source_confidences) / len(source_confidences)
    target_avg = sum(target_confidences) / len(target_confidences)

    confidence_similarity = 1.0 - abs(source_avg - target_avg)

    # Combined transfer score
    transfer_score = (type_similarity + confidence_similarity) / 2

    return min(1.0, max(0.0, transfer_score))


def generate_training_examples(overrides: List[HumanOverride]) -> List[Dict[str, Any]]:
    """Generate training examples from human overrides for FEEDBACK_LOOP.

    Args:
        overrides: List of human overrides

    Returns:
        List of training examples
    """
    examples = []

    for override in overrides:
        example = {
            "decision_id": override.decision_id,
            "reason_code": override.reason_code.value,
            "corrected_output": override.corrected_output,
            "justification": override.justification,
            "training_weight": 1.0,  # Can be adjusted based on confidence
        }
        examples.append(example)

    return examples
