"""quality.py - Score training example quality.

Quality gates ensure only high-quality examples enter training set.
Scores completeness, clarity, and relevance of training data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .extractor import TrainingExample
from .labeler import LabeledExample

# === CONSTANTS ===

# Quality thresholds
DEFAULT_QUALITY_THRESHOLD = 0.8
MIN_QUALITY_THRESHOLD = 0.5


@dataclass
class QualityScore:
    """Quality score breakdown for a training example."""

    example_id: str
    overall_score: float
    component_scores: Dict[str, float]
    passes_threshold: bool
    threshold_used: float
    issues: List[str]
    scored_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "example_id": self.example_id,
            "overall_score": self.overall_score,
            "component_scores": self.component_scores,
            "passes_threshold": self.passes_threshold,
            "threshold_used": self.threshold_used,
            "issues": self.issues,
            "scored_at": self.scored_at,
        }


def score_context_completeness(example: TrainingExample) -> Tuple[float, List[str]]:
    """Score input context completeness.

    Args:
        example: TrainingExample to score

    Returns:
        Tuple of (score, issues)
    """
    score = 0.0
    issues = []

    context = example.input_context

    # Check telemetry
    if context.telemetry and len(context.telemetry) > 1:
        score += 0.25
    else:
        issues.append("Missing or minimal telemetry data")

    # Check perception
    if context.perception and len(context.perception) > 0:
        score += 0.25
    else:
        issues.append("Missing perception data")

    # Check mission context
    if context.mission_context and len(context.mission_context) > 0:
        score += 0.25
    else:
        issues.append("Missing mission context")

    # Check model provenance
    if context.model_provenance.get("model_id") not in [None, "unknown"]:
        score += 0.25
    else:
        issues.append("Unknown model provenance")

    return score, issues


def score_output_quality(example: TrainingExample) -> Tuple[float, List[str]]:
    """Score output quality (bad and good outputs).

    Args:
        example: TrainingExample to score

    Returns:
        Tuple of (score, issues)
    """
    score = 0.0
    issues = []

    # Check bad output
    if example.bad_output.action:
        score += 0.25
    else:
        issues.append("Bad output action is empty")

    if example.bad_output.reasoning and len(example.bad_output.reasoning) > 10:
        score += 0.25
    else:
        issues.append("Bad output reasoning is minimal")

    # Check good output
    if example.good_output.action:
        score += 0.25
    else:
        issues.append("Good output action is empty")

    if example.good_output.reasoning and len(example.good_output.reasoning) > 10:
        score += 0.25
    else:
        issues.append("Good output reasoning is minimal")

    return score, issues


def score_label_quality(
    example: TrainingExample,
    labeled: Optional[LabeledExample] = None,
) -> Tuple[float, List[str]]:
    """Score label quality.

    Args:
        example: TrainingExample to score
        labeled: Optional LabeledExample

    Returns:
        Tuple of (score, issues)
    """
    score = 0.0
    issues = []

    # Check reason code
    reason_code = example.label.get("reason_code", "UNKNOWN")
    if reason_code != "UNKNOWN" and reason_code.startswith("RE"):
        score += 0.5
    else:
        issues.append("Invalid or missing reason code")

    # Check severity
    if example.label.get("severity") in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        score += 0.25
    else:
        issues.append("Invalid severity level")

    # Check category
    if example.label.get("category") or (labeled and labeled.category):
        score += 0.25
    else:
        issues.append("Missing category")

    return score, issues


def score_contrast_quality(example: TrainingExample) -> Tuple[float, List[str]]:
    """Score the contrast between bad and good outputs.

    Args:
        example: TrainingExample to score

    Returns:
        Tuple of (score, issues)
    """
    issues = []

    bad_action = example.bad_output.action
    good_action = example.good_output.action

    # Check if outputs are different
    if bad_action == good_action:
        issues.append("Bad and good outputs are identical")
        return 0.0, issues

    # Check if both have content
    if not bad_action or not good_action:
        issues.append("One or both outputs are empty")
        return 0.25, issues

    # Good contrast
    return 1.0, issues


def score_example_quality(
    example: TrainingExample,
    labeled: Optional[LabeledExample] = None,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> QualityScore:
    """Compute overall quality score for training example.

    Args:
        example: TrainingExample to score
        labeled: Optional LabeledExample
        threshold: Quality threshold (default 0.8)

    Returns:
        QualityScore with breakdown
    """
    all_issues = []
    component_scores = {}

    # Score each component
    context_score, context_issues = score_context_completeness(example)
    component_scores["context_completeness"] = context_score
    all_issues.extend(context_issues)

    output_score, output_issues = score_output_quality(example)
    component_scores["output_quality"] = output_score
    all_issues.extend(output_issues)

    label_score, label_issues = score_label_quality(example, labeled)
    component_scores["label_quality"] = label_score
    all_issues.extend(label_issues)

    contrast_score, contrast_issues = score_contrast_quality(example)
    component_scores["contrast_quality"] = contrast_score
    all_issues.extend(contrast_issues)

    # Compute weighted average
    weights = {
        "context_completeness": 0.25,
        "output_quality": 0.30,
        "label_quality": 0.20,
        "contrast_quality": 0.25,
    }

    overall_score = sum(component_scores[k] * weights[k] for k in component_scores)

    return QualityScore(
        example_id=example.example_id,
        overall_score=overall_score,
        component_scores=component_scores,
        passes_threshold=overall_score >= threshold,
        threshold_used=threshold,
        issues=all_issues,
    )


def filter_by_quality(
    examples,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
    labeled_map: Optional[Dict[str, LabeledExample]] = None,
) -> List:
    """Filter examples by quality threshold.

    Args:
        examples: List of TrainingExample or LabeledExample objects
        threshold: Quality threshold
        labeled_map: Optional mapping of example_id to LabeledExample

    Returns:
        List of examples that pass the threshold
    """
    passed = []

    for example in examples:
        # Handle LabeledExample - use label_confidence as quality
        if isinstance(example, LabeledExample):
            quality_score = example.label_confidence
            if quality_score >= threshold:
                passed.append(example)
        else:
            # TrainingExample - use full quality scoring
            labeled = labeled_map.get(example.example_id) if labeled_map else None
            quality = score_example_quality(example, labeled, threshold)

            if quality.passes_threshold:
                passed.append(example)

    return passed


def compute_quality_distribution(
    examples: List[TrainingExample],
    labeled_map: Optional[Dict[str, LabeledExample]] = None,
) -> Dict[str, Any]:
    """Compute quality score distribution for examples.

    Args:
        examples: List of TrainingExample objects
        labeled_map: Optional mapping of example_id to LabeledExample

    Returns:
        Distribution statistics
    """
    if not examples:
        return {
            "count": 0,
            "mean_score": 0.0,
            "median_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "above_threshold_count": 0,
            "above_threshold_percentage": 0.0,
        }

    scores = []
    for example in examples:
        labeled = labeled_map.get(example.example_id) if labeled_map else None
        quality = score_example_quality(example, labeled)
        scores.append(quality.overall_score)

    scores.sort()
    above_threshold = sum(1 for s in scores if s >= DEFAULT_QUALITY_THRESHOLD)

    return {
        "count": len(scores),
        "mean_score": sum(scores) / len(scores),
        "median_score": scores[len(scores) // 2],
        "min_score": min(scores),
        "max_score": max(scores),
        "above_threshold_count": above_threshold,
        "above_threshold_percentage": above_threshold / len(scores) * 100,
    }
