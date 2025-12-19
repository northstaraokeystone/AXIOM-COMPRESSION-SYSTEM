"""Pattern lifecycle management for D19.

Manage pattern birth, survival, and death.
"""

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, dual_hash, TENANT_ID
from .pattern_detector import PatternDetector, DetectedPattern

# === D19 LIFECYCLE CONSTANTS ===

PATTERN_BIRTH_FITNESS = 0.60
"""Minimum fitness for pattern birth."""

PATTERN_DEATH_FITNESS = 0.20
"""Fitness below which pattern dies."""


@dataclass
class LivePattern:
    """Active pattern in lifecycle."""

    pattern_id: str
    born_at: str
    fitness: float = 0.0
    entropy_reduction: float = 0.0
    coordination_success: float = 0.0
    stability: float = 0.0
    diversity_contribution: float = 0.0
    recency_bonus: float = 0.0
    status: str = "alive"


@dataclass
class PatternLifecycle:
    """Pattern lifecycle manager."""

    lifecycle_id: str
    detector: PatternDetector
    active_patterns: Dict[str, LivePattern] = field(default_factory=dict)
    superposition_patterns: Dict[str, LivePattern] = field(default_factory=dict)
    dead_patterns: Dict[str, LivePattern] = field(default_factory=dict)


def init_lifecycle(detector: PatternDetector) -> PatternLifecycle:
    """Initialize lifecycle manager.

    Args:
        detector: PatternDetector instance

    Returns:
        PatternLifecycle instance
    """
    lifecycle_id = str(uuid.uuid4())[:8]
    return PatternLifecycle(lifecycle_id=lifecycle_id, detector=detector)


def birth_pattern(lifecycle: PatternLifecycle, pattern: Dict) -> Dict[str, Any]:
    """Register new pattern (autocatalysis achieved).

    Args:
        lifecycle: PatternLifecycle instance
        pattern: Pattern dict

    Returns:
        Birth result

    Receipt: pattern_birth_receipt
    """
    pattern_id = pattern.get("pattern_id", str(uuid.uuid4())[:8])
    now = datetime.utcnow().isoformat() + "Z"

    # Create live pattern
    live = LivePattern(
        pattern_id=pattern_id,
        born_at=now,
        fitness=pattern.get("fitness", 0.0),
        entropy_reduction=random.uniform(0.5, 0.9),
        coordination_success=random.uniform(0.6, 0.95),
        stability=random.uniform(0.6, 0.9),
        diversity_contribution=random.uniform(0.1, 0.3),
        recency_bonus=1.0,  # Max bonus at birth
        status="alive",
    )

    lifecycle.active_patterns[pattern_id] = live

    result = {
        "pattern_id": pattern_id,
        "born_at": now,
        "initial_fitness": live.fitness,
        "status": "alive",
        "active_patterns": len(lifecycle.active_patterns),
    }

    emit_receipt(
        "pattern_birth",
        {
            "receipt_type": "pattern_birth",
            "tenant_id": TENANT_ID,
            "ts": now,
            "lifecycle_id": lifecycle.lifecycle_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def evaluate_fitness(lifecycle: PatternLifecycle, pattern_id: str) -> Dict[str, Any]:
    """Evaluate multi-dimensional fitness for pattern.

    Fitness = weighted sum of:
    - 0.35 * entropy_reduction
    - 0.25 * coordination_success
    - 0.20 * stability
    - 0.10 * diversity_contribution
    - 0.10 * recency_bonus

    Args:
        lifecycle: PatternLifecycle instance
        pattern_id: Pattern identifier

    Returns:
        Fitness evaluation result

    Receipt: fitness_evaluation_receipt
    """
    if pattern_id not in lifecycle.active_patterns:
        return {"error": "pattern_not_found", "pattern_id": pattern_id}

    pattern = lifecycle.active_patterns[pattern_id]

    # Compute weighted fitness
    fitness = (
        0.35 * pattern.entropy_reduction
        + 0.25 * pattern.coordination_success
        + 0.20 * pattern.stability
        + 0.10 * pattern.diversity_contribution
        + 0.10 * pattern.recency_bonus
    )

    pattern.fitness = round(fitness, 4)

    # Decay recency bonus
    pattern.recency_bonus = max(0.0, pattern.recency_bonus - 0.01)

    result = {
        "pattern_id": pattern_id,
        "fitness": pattern.fitness,
        "entropy_reduction": pattern.entropy_reduction,
        "coordination_success": pattern.coordination_success,
        "stability": pattern.stability,
        "diversity_contribution": pattern.diversity_contribution,
        "recency_bonus": pattern.recency_bonus,
    }

    emit_receipt(
        "fitness_evaluation",
        {
            "receipt_type": "fitness_evaluation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "lifecycle_id": lifecycle.lifecycle_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def apply_selection_pressure(lifecycle: PatternLifecycle) -> List[Dict]:
    """Apply Thompson sampling selection over patterns.

    Args:
        lifecycle: PatternLifecycle instance

    Returns:
        List of selected patterns

    Receipt: selection_pressure_receipt
    """
    selected = []

    for pattern_id, pattern in list(lifecycle.active_patterns.items()):
        # Thompson sampling: sample from Beta distribution based on fitness
        alpha = max(1, int(pattern.fitness * 10))
        beta = max(1, int((1 - pattern.fitness) * 10))
        sample = random.betavariate(alpha, beta)

        if sample >= 0.5:
            selected.append({"pattern_id": pattern_id, "sample": round(sample, 4)})

        # Kill low-fitness patterns
        if pattern.fitness < PATTERN_DEATH_FITNESS:
            kill_pattern(lifecycle, pattern_id, "low_fitness")

    emit_receipt(
        "selection_pressure",
        {
            "receipt_type": "selection_pressure",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "lifecycle_id": lifecycle.lifecycle_id,
            "patterns_evaluated": len(lifecycle.active_patterns) + len(selected),
            "patterns_selected": len(selected),
            "payload_hash": dual_hash(json.dumps({"selected": len(selected)}, sort_keys=True)),
        },
    )

    return selected


def kill_pattern(lifecycle: PatternLifecycle, pattern_id: str, reason: str) -> Dict[str, Any]:
    """Remove pattern (fitness too low).

    Args:
        lifecycle: PatternLifecycle instance
        pattern_id: Pattern identifier
        reason: Reason for death

    Returns:
        Death result

    Receipt: pattern_death_receipt
    """
    if pattern_id not in lifecycle.active_patterns:
        # Check superposition
        if pattern_id in lifecycle.superposition_patterns:
            pattern = lifecycle.superposition_patterns.pop(pattern_id)
        else:
            return {"error": "pattern_not_found", "pattern_id": pattern_id}
    else:
        pattern = lifecycle.active_patterns.pop(pattern_id)

    pattern.status = "dead"
    lifecycle.dead_patterns[pattern_id] = pattern

    result = {
        "pattern_id": pattern_id,
        "reason": reason,
        "final_fitness": pattern.fitness,
        "status": "dead",
        "active_patterns": len(lifecycle.active_patterns),
    }

    emit_receipt(
        "pattern_death",
        {
            "receipt_type": "pattern_death",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "lifecycle_id": lifecycle.lifecycle_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def resurrect_pattern(lifecycle: PatternLifecycle, pattern_id: str) -> Dict[str, Any]:
    """Resurrect pattern from superposition if conditions improve.

    Args:
        lifecycle: PatternLifecycle instance
        pattern_id: Pattern identifier

    Returns:
        Resurrection result

    Receipt: pattern_resurrection_receipt
    """
    if pattern_id not in lifecycle.superposition_patterns:
        return {"error": "pattern_not_in_superposition", "pattern_id": pattern_id}

    pattern = lifecycle.superposition_patterns.pop(pattern_id)
    pattern.status = "alive"
    pattern.recency_bonus = 0.5  # Partial bonus on resurrection
    lifecycle.active_patterns[pattern_id] = pattern

    result = {
        "pattern_id": pattern_id,
        "status": "resurrected",
        "fitness": pattern.fitness,
        "active_patterns": len(lifecycle.active_patterns),
    }

    emit_receipt(
        "pattern_resurrection",
        {
            "receipt_type": "pattern_resurrection",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "lifecycle_id": lifecycle.lifecycle_id,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_active_patterns(lifecycle: PatternLifecycle) -> List[Dict]:
    """Get currently alive patterns.

    Args:
        lifecycle: PatternLifecycle instance

    Returns:
        List of active pattern dicts
    """
    return [
        {
            "pattern_id": p.pattern_id,
            "fitness": p.fitness,
            "status": p.status,
            "born_at": p.born_at,
        }
        for p in lifecycle.active_patterns.values()
    ]


def get_superposition_patterns(lifecycle: PatternLifecycle) -> List[Dict]:
    """Get patterns in superposition (dormant).

    Args:
        lifecycle: PatternLifecycle instance

    Returns:
        List of superposition pattern dicts
    """
    return [
        {
            "pattern_id": p.pattern_id,
            "fitness": p.fitness,
            "status": p.status,
        }
        for p in lifecycle.superposition_patterns.values()
    ]
