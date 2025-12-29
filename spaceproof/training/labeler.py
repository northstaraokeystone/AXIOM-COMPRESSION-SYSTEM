"""labeler.py - Apply structured labels to training examples.

Labels based on reason codes and intervention types.
Enables filtering and prioritization in retraining queue.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .extractor import TrainingExample

# === CONSTANTS ===

# Label categories derived from reason codes
LABEL_CATEGORIES = {
    "accuracy": ["RE001_FACTUAL_ERROR", "RE008_HALLUCINATION"],
    "compliance": ["RE002_POLICY_VIOLATION"],
    "safety": ["RE003_SAFETY_CONCERN"],
    "ethics": ["RE004_ETHICAL_CONCERN"],
    "preference": ["RE005_USER_PREFERENCE"],
    "context": ["RE006_CONTEXT_MISSING"],
    "capability": ["RE007_TOOL_MISUSE"],
    "timing": ["RE009_TIMING_ERROR"],
    "authorization": ["RE010_SCOPE_EXCEEDED"],
}

# Reverse mapping
REASON_TO_CATEGORY = {}
for category, codes in LABEL_CATEGORIES.items():
    for code in codes:
        REASON_TO_CATEGORY[code] = category


@dataclass
class LabeledExample:
    """Training example with applied labels."""

    example_id: str
    labels: Dict[str, Any]
    category: str
    subcategories: List[str]
    severity_weight: int
    requires_retraining: bool
    label_confidence: float
    labeled_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "example_id": self.example_id,
            "labels": self.labels,
            "category": self.category,
            "subcategories": self.subcategories,
            "severity_weight": self.severity_weight,
            "requires_retraining": self.requires_retraining,
            "label_confidence": self.label_confidence,
            "labeled_at": self.labeled_at,
        }


def get_severity_weight(severity: str) -> int:
    """Get numeric weight for severity.

    Args:
        severity: Severity string

    Returns:
        Weight (1-4)
    """
    weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    return weights.get(severity, 2)


def get_category_from_reason_code(reason_code: str) -> str:
    """Get category for reason code.

    Args:
        reason_code: Reason code string

    Returns:
        Category string
    """
    return REASON_TO_CATEGORY.get(reason_code, "other")


def get_labels_for_reason_code(reason_code: str) -> Dict[str, Any]:
    """Get all applicable labels for a reason code.

    Args:
        reason_code: Reason code string

    Returns:
        Dict of labels
    """
    category = get_category_from_reason_code(reason_code)

    # Determine if retraining needed based on category
    requires_retraining = category in ["accuracy", "compliance", "safety", "ethics", "capability", "authorization"]

    return {
        "reason_code": reason_code,
        "category": category,
        "requires_retraining": requires_retraining,
        "is_safety_critical": category in ["safety", "compliance"],
        "is_accuracy_issue": category in ["accuracy"],
        "is_behavior_issue": category in ["ethics", "authorization", "capability"],
    }


def apply_label(
    example: TrainingExample,
    additional_labels: Optional[Dict[str, Any]] = None,
) -> LabeledExample:
    """Apply structured labels to training example.

    Args:
        example: TrainingExample to label
        additional_labels: Optional additional labels to apply

    Returns:
        LabeledExample with all labels
    """
    reason_code = example.label.get("reason_code", "UNKNOWN")
    severity = example.label.get("severity", "MEDIUM")

    # Get labels for reason code
    base_labels = get_labels_for_reason_code(reason_code)

    # Merge with example labels
    labels = {
        **example.label,
        **base_labels,
    }

    # Add additional labels if provided
    if additional_labels:
        labels.update(additional_labels)

    # Determine category and subcategories
    category = get_category_from_reason_code(reason_code)
    subcategories = []

    if labels.get("is_safety_critical"):
        subcategories.append("safety_critical")
    if labels.get("is_accuracy_issue"):
        subcategories.append("accuracy")
    if labels.get("is_behavior_issue"):
        subcategories.append("behavior")

    # Compute label confidence based on example quality
    label_confidence = example.quality_score

    return LabeledExample(
        example_id=example.example_id,
        labels=labels,
        category=category,
        subcategories=subcategories,
        severity_weight=get_severity_weight(severity),
        requires_retraining=labels.get("requires_retraining", False),
        label_confidence=label_confidence,
    )


def compute_label_distribution(
    labeled_examples: List[LabeledExample],
) -> Dict[str, Any]:
    """Compute distribution of labels across examples.

    Args:
        labeled_examples: List of LabeledExample objects

    Returns:
        Distribution statistics
    """
    if not labeled_examples:
        return {
            "total_count": 0,
            "category_distribution": {},
            "severity_distribution": {},
            "retraining_required_count": 0,
        }

    # Count categories
    category_counts = Counter(ex.category for ex in labeled_examples)

    # Count severities
    severity_map = {1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
    severity_counts = Counter(severity_map.get(ex.severity_weight, "UNKNOWN") for ex in labeled_examples)

    # Count retraining required
    retraining_count = sum(1 for ex in labeled_examples if ex.requires_retraining)

    return {
        "total_count": len(labeled_examples),
        "category_distribution": dict(category_counts),
        "severity_distribution": dict(severity_counts),
        "retraining_required_count": retraining_count,
        "retraining_percentage": retraining_count / len(labeled_examples) * 100,
    }


def batch_label(
    examples: List[TrainingExample],
) -> List[LabeledExample]:
    """Apply labels to a batch of examples.

    Args:
        examples: List of TrainingExample objects

    Returns:
        List of LabeledExample objects
    """
    return [apply_label(ex) for ex in examples]


# Additional functions for API compatibility
def apply_labels(
    examples: List[TrainingExample],
    additional_labels: Optional[Dict[str, Any]] = None,
) -> List[LabeledExample]:
    """Apply labels to multiple examples (alias for batch_label with options).

    Args:
        examples: List of TrainingExample objects
        additional_labels: Optional additional labels to apply to all

    Returns:
        List of LabeledExample objects
    """
    return [apply_label(ex, additional_labels) for ex in examples]


def get_label_schema() -> Dict[str, Any]:
    """Get the label schema for training examples.

    Returns:
        Schema dict describing label structure
    """
    return {
        "categories": list(LABEL_CATEGORIES.keys()),
        "reason_codes": list(REASON_TO_CATEGORY.keys()),
        "severity_levels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "subcategory_options": ["safety_critical", "accuracy", "behavior"],
        "label_fields": [
            "reason_code",
            "category",
            "requires_retraining",
            "is_safety_critical",
            "is_accuracy_issue",
            "is_behavior_issue",
        ],
    }


def emit_labeling_receipt(labeled_example: LabeledExample) -> Dict[str, Any]:
    """Emit receipt for labeling operation.

    Args:
        labeled_example: The labeled example

    Returns:
        Receipt dict
    """
    from spaceproof.core import emit_receipt

    return emit_receipt(
        "labeling",
        {
            "tenant_id": "spaceproof-training",
            "example_id": labeled_example.example_id,
            "category": labeled_example.category,
            "subcategories": labeled_example.subcategories,
            "severity_weight": labeled_example.severity_weight,
            "requires_retraining": labeled_example.requires_retraining,
        },
    )
