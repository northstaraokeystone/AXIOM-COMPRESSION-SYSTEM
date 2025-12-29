"""extractor.py - Transform human interventions into training examples.

Every human correction becomes a labeled training example.
Captures context, actions, and justification for model improvement.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from spaceproof.core import dual_hash, emit_receipt

# === CONSTANTS ===

TRAINING_TENANT = "spaceproof-training"


@dataclass
class InputContext:
    """Input context for training example."""

    telemetry: Dict[str, Any]
    perception: Dict[str, Any]
    mission_context: Dict[str, Any]
    model_provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "telemetry": self.telemetry,
            "perception": self.perception,
            "mission_context": self.mission_context,
            "model_provenance": self.model_provenance,
        }


@dataclass
class BadOutput:
    """Incorrect agent output for training."""

    action: Dict[str, Any]
    reasoning: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


@dataclass
class GoodOutput:
    """Correct output for training."""

    action: Dict[str, Any]
    reasoning: str
    source: str  # "human" for corrections

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "source": self.source,
        }


@dataclass
class TrainingExample:
    """Complete training example from intervention."""

    example_id: str
    source_decision_id: str
    source_intervention_id: str
    timestamp: str
    input_context: InputContext
    bad_output: BadOutput
    good_output: GoodOutput
    label: Dict[str, Any]
    quality_score: float
    retraining_priority: str
    exported_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "example_id": self.example_id,
            "source_decision_id": self.source_decision_id,
            "source_intervention_id": self.source_intervention_id,
            "timestamp": self.timestamp,
            "input": self.input_context.to_dict(),
            "bad_output": self.bad_output.to_dict(),
            "good_output": self.good_output.to_dict(),
            "label": self.label,
            "quality_score": self.quality_score,
            "retraining_priority": self.retraining_priority,
            "exported_to": self.exported_to,
        }


# Storage for training examples
_training_examples: Dict[str, TrainingExample] = {}


def gather_input_context(
    decision_id: str,
    telemetry: Optional[Dict[str, Any]] = None,
    perception: Optional[Dict[str, Any]] = None,
    mission_context: Optional[Dict[str, Any]] = None,
    model_provenance: Optional[Dict[str, Any]] = None,
) -> InputContext:
    """Gather input context for training example.

    Args:
        decision_id: Decision identifier
        telemetry: Telemetry data (optional)
        perception: Perception data (optional)
        mission_context: Mission context (optional)
        model_provenance: Model provenance (optional)

    Returns:
        InputContext with all available data
    """
    # In production, would query ledger for prior receipts
    return InputContext(
        telemetry=telemetry or {"decision_id": decision_id},
        perception=perception or {},
        mission_context=mission_context or {},
        model_provenance=model_provenance or {"model_id": "unknown", "version": "unknown"},
    )


def format_bad_output(
    original_action: Dict[str, Any],
    reasoning: str = "",
    confidence: float = 0.0,
) -> BadOutput:
    """Format incorrect agent output.

    Args:
        original_action: Original agent action
        reasoning: Agent's reasoning (if available)
        confidence: Agent's confidence score

    Returns:
        Formatted BadOutput
    """
    return BadOutput(
        action=original_action,
        reasoning=reasoning or "No reasoning captured",
        confidence=confidence,
    )


def format_good_output(
    corrected_action: Dict[str, Any],
    reasoning: str = "",
    source: str = "human",
) -> GoodOutput:
    """Format correct output from correction.

    Args:
        corrected_action: Corrected action
        reasoning: Human's reasoning (if available)
        source: Source of correction (default "human")

    Returns:
        Formatted GoodOutput
    """
    return GoodOutput(
        action=corrected_action,
        reasoning=reasoning or "Human correction",
        source=source,
    )


def compute_initial_quality(
    intervention: Dict[str, Any],
    input_context: InputContext,
) -> float:
    """Compute initial quality score for training example.

    Args:
        intervention: Intervention data
        input_context: Input context

    Returns:
        Quality score (0.0 - 1.0)
    """
    score = 0.5  # Base score

    # Bonus for complete context
    if input_context.telemetry:
        score += 0.1
    if input_context.perception:
        score += 0.1
    if input_context.mission_context:
        score += 0.1
    if input_context.model_provenance.get("model_id") != "unknown":
        score += 0.1

    # Bonus for justification
    if intervention.get("justification"):
        score += 0.1

    return min(1.0, score)


def get_retraining_priority(reason_code: str, severity: str) -> str:
    """Determine retraining priority from reason code.

    Args:
        reason_code: Reason code string
        severity: Severity level

    Returns:
        Priority (IMMEDIATE, HIGH, MEDIUM, LOW)
    """
    if severity == "CRITICAL":
        return "IMMEDIATE"
    elif severity == "HIGH":
        return "HIGH"
    elif severity == "MEDIUM":
        return "MEDIUM"
    else:
        return "LOW"


def extract_training_example(
    intervention_receipt: Dict[str, Any],
    input_context: Optional[InputContext] = None,
) -> TrainingExample:
    """Transform intervention into training example.

    Args:
        intervention_receipt: Intervention receipt from governance
        input_context: Optional pre-gathered context

    Returns:
        TrainingExample ready for labeling/export
    """
    example_id = str(uuid.uuid4())
    decision_id = intervention_receipt.get("target_decision_id", "unknown")
    intervention_id = intervention_receipt.get("intervention_id", "unknown")

    # Gather context if not provided
    if input_context is None:
        input_context = gather_input_context(decision_id)

    # Format outputs
    bad_output = format_bad_output(
        original_action=intervention_receipt.get("original_action", {}),
        reasoning=intervention_receipt.get("original_reasoning", ""),
        confidence=intervention_receipt.get("original_confidence", 0.0),
    )

    good_output = format_good_output(
        corrected_action=intervention_receipt.get("corrected_action", {}),
        reasoning=intervention_receipt.get("justification", ""),
    )

    # Build label
    reason_code = intervention_receipt.get("reason_code", "UNKNOWN")
    severity = intervention_receipt.get("reason_severity", "MEDIUM")

    label = {
        "reason_code": reason_code,
        "severity": severity,
        "category": intervention_receipt.get("category", "other"),
        "intervention_type": intervention_receipt.get("intervention_type", "CORRECTION"),
    }

    # Compute quality
    quality_score = compute_initial_quality(intervention_receipt, input_context)

    # Get priority
    retraining_priority = get_retraining_priority(reason_code, severity)

    example = TrainingExample(
        example_id=example_id,
        source_decision_id=decision_id,
        source_intervention_id=intervention_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        input_context=input_context,
        bad_output=bad_output,
        good_output=good_output,
        label=label,
        quality_score=quality_score,
        retraining_priority=retraining_priority,
    )

    # Store example
    _training_examples[example_id] = example

    return example


def emit_training_example_receipt(
    example: TrainingExample,
) -> Dict[str, Any]:
    """Emit training example receipt.

    Args:
        example: TrainingExample to emit

    Returns:
        Receipt dict with dual-hash
    """
    receipt_data = {
        "tenant_id": TRAINING_TENANT,
        "example_id": example.example_id,
        "source_intervention_id": example.source_intervention_id,
        "reason_code": example.label.get("reason_code", "UNKNOWN"),
        "quality_score": example.quality_score,
        "retraining_priority": example.retraining_priority,
        "exported_to": example.exported_to or "pending",
    }

    return emit_receipt("training_example", receipt_data)


def get_training_example(example_id: str) -> Optional[TrainingExample]:
    """Get training example by ID.

    Args:
        example_id: Example identifier

    Returns:
        TrainingExample or None
    """
    return _training_examples.get(example_id)


def get_all_examples() -> List[TrainingExample]:
    """Get all training examples.

    Returns:
        List of all TrainingExample objects
    """
    return list(_training_examples.values())


def clear_examples() -> None:
    """Clear all training examples (for testing)."""
    global _training_examples
    _training_examples = {}


# Alias for API compatibility
def emit_extraction_receipt(example: TrainingExample) -> Dict[str, Any]:
    """Alias for emit_training_example_receipt."""
    return emit_training_example_receipt(example)
