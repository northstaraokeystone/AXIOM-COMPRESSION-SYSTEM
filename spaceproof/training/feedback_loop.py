"""feedback_loop.py - Fine-tuning integration.

Manage retraining queue and validate learning improvements.
Prioritizes CRITICAL and HIGH severity interventions for immediate training.
"""

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

from spaceproof.core import emit_receipt

from .extractor import TrainingExample

# === CONSTANTS ===

TRAINING_TENANT = "spaceproof-training"

# Batch sizes
DEFAULT_BATCH_SIZE = 32
MAX_QUEUE_SIZE = 10000


@dataclass
class RetrainingBatch:
    """Batch of examples ready for retraining."""

    batch_id: str
    examples: List[TrainingExample]
    priority: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    processed: bool = False
    processed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "example_count": len(self.examples),
            "example_ids": [e.example_id for e in self.examples],
            "priority": self.priority,
            "created_at": self.created_at,
            "processed": self.processed,
            "processed_at": self.processed_at,
        }


@dataclass
class FeedbackLoopState:
    """State of the feedback loop system."""

    queue_size: int
    batches_pending: int
    batches_processed: int
    total_examples_queued: int
    total_examples_processed: int
    correction_rate: float  # Current correction rate
    baseline_correction_rate: float  # Initial correction rate
    correction_rate_improvement: float  # Percentage improvement
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_size": self.queue_size,
            "batches_pending": self.batches_pending,
            "batches_processed": self.batches_processed,
            "total_examples_queued": self.total_examples_queued,
            "total_examples_processed": self.total_examples_processed,
            "correction_rate": self.correction_rate,
            "baseline_correction_rate": self.baseline_correction_rate,
            "correction_rate_improvement": self.correction_rate_improvement,
            "last_updated": self.last_updated,
        }


# Global state
_retraining_queue: Dict[str, Deque[TrainingExample]] = {
    "IMMEDIATE": deque(maxlen=MAX_QUEUE_SIZE),
    "HIGH": deque(maxlen=MAX_QUEUE_SIZE),
    "MEDIUM": deque(maxlen=MAX_QUEUE_SIZE),
    "LOW": deque(maxlen=MAX_QUEUE_SIZE),
}

_processed_batches: List[RetrainingBatch] = []
_total_queued: int = 0
_total_processed: int = 0
_baseline_correction_rate: float = 0.1  # 10% baseline
_current_correction_rate: float = 0.1


def add_to_retraining_queue(
    example,
    priority: Optional[str] = None,
) -> "FeedbackLoopState":
    """Add example to retraining queue.

    Args:
        example: TrainingExample or LabeledExample to queue
        priority: Optional priority override

    Returns:
        Current FeedbackLoopState
    """
    global _total_queued

    # Determine priority - handle both TrainingExample and LabeledExample
    if priority is None:
        if hasattr(example, 'retraining_priority'):
            priority = example.retraining_priority
        elif hasattr(example, 'severity_weight'):
            # Convert severity_weight to priority
            weight = example.severity_weight
            if weight >= 4:
                priority = "IMMEDIATE"
            elif weight >= 3:
                priority = "HIGH"
            elif weight >= 2:
                priority = "MEDIUM"
            else:
                priority = "LOW"
        else:
            priority = "MEDIUM"

    if priority not in _retraining_queue:
        priority = "MEDIUM"

    # Add to queue
    _retraining_queue[priority].append(example)
    _total_queued += 1

    # Return current state
    return get_feedback_loop_state()


def get_retraining_queue(priority: Optional[str] = None) -> List[TrainingExample]:
    """Get examples from retraining queue.

    Args:
        priority: Optional priority filter

    Returns:
        List of TrainingExample objects
    """
    if priority:
        return list(_retraining_queue.get(priority, []))

    # Return all, ordered by priority
    result = []
    for p in ["IMMEDIATE", "HIGH", "MEDIUM", "LOW"]:
        result.extend(_retraining_queue.get(p, []))
    return result


def create_retraining_batch(
    batch_size: int = DEFAULT_BATCH_SIZE,
    priority_filter: Optional[str] = None,
) -> Optional[RetrainingBatch]:
    """Create a batch from queued examples.

    Args:
        batch_size: Maximum batch size
        priority_filter: Optional priority filter

    Returns:
        RetrainingBatch or None if queue empty
    """
    batch_examples = []
    batch_priority = "LOW"

    # Pull from queues in priority order
    for priority in ["IMMEDIATE", "HIGH", "MEDIUM", "LOW"]:
        if priority_filter and priority != priority_filter:
            continue

        queue = _retraining_queue[priority]
        while queue and len(batch_examples) < batch_size:
            example = queue.popleft()
            batch_examples.append(example)
            batch_priority = priority  # Set to highest priority in batch

    if not batch_examples:
        return None

    return RetrainingBatch(
        batch_id=str(uuid.uuid4()),
        examples=batch_examples,
        priority=batch_priority,
    )


def process_retraining_batch(
    batch_or_list,
    simulate_training: bool = True,
) -> Dict[str, Any]:
    """Process a retraining batch.

    Args:
        batch_or_list: RetrainingBatch or list of examples to process
        simulate_training: If True, simulate training (for testing)

    Returns:
        Processing result
    """
    global _total_processed

    # Handle list input - convert to RetrainingBatch
    if isinstance(batch_or_list, list):
        batch = RetrainingBatch(
            batch_id=str(uuid.uuid4()),
            examples=batch_or_list,
            priority="MEDIUM",
        )
    else:
        batch = batch_or_list

    # In production, this would trigger actual model fine-tuning
    # For simulation, just mark as processed
    batch.processed = True
    batch.processed_at = datetime.utcnow().isoformat() + "Z"

    _processed_batches.append(batch)
    _total_processed += len(batch.examples)

    # Emit receipt
    emit_receipt(
        "retraining_batch",
        {
            "tenant_id": TRAINING_TENANT,
            **batch.to_dict(),
            "simulated": simulate_training,
        },
    )

    return {
        "batch_id": batch.batch_id,
        "examples_processed": len(batch.examples),
        "simulated": simulate_training,
    }


def update_correction_rate(new_rate: float) -> None:
    """Update current correction rate.

    Args:
        new_rate: New correction rate (0.0 - 1.0)
    """
    global _current_correction_rate
    _current_correction_rate = max(0.0, min(1.0, new_rate))


def validate_learning(target_improvement: float = 0.5) -> Tuple[bool, Dict[str, Any]]:
    """Validate learning improvement from feedback loop.

    FEEDBACK_LOOP scenario requirement: correction rate -50%

    Args:
        target_improvement: Target improvement percentage (default 50%)

    Returns:
        Tuple of (passed, metrics)
    """
    if _baseline_correction_rate == 0:
        improvement = 0.0
    else:
        improvement = (_baseline_correction_rate - _current_correction_rate) / _baseline_correction_rate

    passed = improvement >= target_improvement

    metrics = {
        "baseline_rate": _baseline_correction_rate,
        "current_rate": _current_correction_rate,
        "improvement_percentage": improvement * 100,
        "target_improvement": target_improvement * 100,
        "passed": passed,
        "total_examples_processed": _total_processed,
    }

    # Emit validation receipt
    emit_receipt(
        "learning_validation",
        {
            "tenant_id": TRAINING_TENANT,
            **metrics,
        },
    )

    return passed, metrics


def get_feedback_loop_state() -> FeedbackLoopState:
    """Get current feedback loop state.

    Returns:
        FeedbackLoopState with all metrics
    """
    queue_size = sum(len(q) for q in _retraining_queue.values())
    batches_pending = sum(1 for b in _processed_batches if not b.processed)
    batches_processed = sum(1 for b in _processed_batches if b.processed)

    if _baseline_correction_rate == 0:
        improvement = 0.0
    else:
        improvement = (_baseline_correction_rate - _current_correction_rate) / _baseline_correction_rate

    return FeedbackLoopState(
        queue_size=queue_size,
        batches_pending=batches_pending,
        batches_processed=batches_processed,
        total_examples_queued=_total_queued,
        total_examples_processed=_total_processed,
        correction_rate=_current_correction_rate,
        baseline_correction_rate=_baseline_correction_rate,
        correction_rate_improvement=improvement * 100,
    )


def set_baseline_correction_rate(rate: float) -> None:
    """Set baseline correction rate for improvement tracking.

    Args:
        rate: Baseline correction rate (0.0 - 1.0)
    """
    global _baseline_correction_rate
    _baseline_correction_rate = max(0.0, min(1.0, rate))


def reset_feedback_loop() -> None:
    """Reset feedback loop state (for testing)."""
    global _retraining_queue, _processed_batches, _total_queued, _total_processed
    global _baseline_correction_rate, _current_correction_rate

    _retraining_queue = {
        "IMMEDIATE": deque(maxlen=MAX_QUEUE_SIZE),
        "HIGH": deque(maxlen=MAX_QUEUE_SIZE),
        "MEDIUM": deque(maxlen=MAX_QUEUE_SIZE),
        "LOW": deque(maxlen=MAX_QUEUE_SIZE),
    }
    _processed_batches = []
    _total_queued = 0
    _total_processed = 0
    _baseline_correction_rate = 0.1
    _current_correction_rate = 0.1
