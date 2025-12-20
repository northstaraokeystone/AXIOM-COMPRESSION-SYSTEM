"""Multi-dimensional fitness evaluation for D19.

Implements QED v12 fitness model:
- Entropy reduction
- Coordination success
- Stability
- Diversity contribution
- Recency bonus

D19.2 UPDATE - PROJECTED FUTURE FITNESS:
  OLD: "Darwinian selection on OBSERVED fitness"
  NEW: "Selection on PROJECTED future fitness"

  Grok's Core Insight:
    "High-future-compression paths are pre-amplified in today's selection.
     Low-future paths pre-starved before they waste cycles."

  Selection is on PROJECTED future, not observed past.
"""

import math
import random
from typing import Any, Dict, List

# === D19.2 PROJECTED FITNESS CONSTANTS ===

PROJECTED_FITNESS_WEIGHT = 0.60
"""Weight for projected future fitness (vs current observed)."""

OBSERVED_FITNESS_WEIGHT = 0.40
"""Weight for current observed fitness."""

SELECTION_ON_PAST = False
"""Selection on past KILLED - projected future only."""

HIGH_FUTURE_FITNESS_THRESHOLD = 0.85
"""Threshold for pre-amplification."""

LOW_FUTURE_FITNESS_THRESHOLD = 0.50
"""Threshold for pre-starvation."""


def compute_pattern_fitness(
    entropy_reduction: float,
    coordination_success: float,
    stability: float,
    diversity_contribution: float,
    recency_bonus: float,
) -> float:
    """Compute weighted pattern fitness.

    Weights from QED v12:
    - 0.35 * entropy_reduction
    - 0.25 * coordination_success
    - 0.20 * stability
    - 0.10 * diversity_contribution
    - 0.10 * recency_bonus

    Args:
        entropy_reduction: How much entropy pattern reduces (0-1)
        coordination_success: Success rate of coordination (0-1)
        stability: Pattern stability over time (0-1)
        diversity_contribution: Contribution to swarm diversity (0-1)
        recency_bonus: Bonus for recently created patterns (0-1)

    Returns:
        Fitness score 0-1
    """
    fitness = (
        0.35 * entropy_reduction
        + 0.25 * coordination_success
        + 0.20 * stability
        + 0.10 * diversity_contribution
        + 0.10 * recency_bonus
    )

    return round(max(0.0, min(1.0, fitness)), 4)


def compute_multi_dimensional_fitness(pattern: Dict) -> Dict[str, Any]:
    """Compute all fitness dimensions for pattern.

    Args:
        pattern: Pattern dict with metrics

    Returns:
        Dict with all fitness dimensions and total
    """
    entropy_reduction = pattern.get("entropy_reduction", 0.0)
    coordination_success = pattern.get("coordination_success", 0.0)
    stability = pattern.get("stability", 0.0)
    diversity_contribution = pattern.get("diversity_contribution", 0.0)
    recency_bonus = pattern.get("recency_bonus", 0.0)

    total = compute_pattern_fitness(
        entropy_reduction,
        coordination_success,
        stability,
        diversity_contribution,
        recency_bonus,
    )

    return {
        "entropy_reduction": entropy_reduction,
        "coordination_success": coordination_success,
        "stability": stability,
        "diversity_contribution": diversity_contribution,
        "recency_bonus": recency_bonus,
        "total_fitness": total,
        "weights": {
            "entropy_reduction": 0.35,
            "coordination_success": 0.25,
            "stability": 0.20,
            "diversity_contribution": 0.10,
            "recency_bonus": 0.10,
        },
    }


def thompson_sampling_select(patterns: List[Dict], k: int = 1) -> List[Dict]:
    """Select patterns using Thompson sampling.

    Args:
        patterns: List of patterns with fitness scores
        k: Number of patterns to select

    Returns:
        Selected patterns
    """
    if not patterns:
        return []

    # Sample from Beta distribution for each pattern
    samples = []
    for pattern in patterns:
        fitness = pattern.get("fitness", 0.5)
        alpha = max(1, int(fitness * 10))
        beta = max(1, int((1 - fitness) * 10))
        sample = random.betavariate(alpha, beta)
        samples.append((sample, pattern))

    # Sort by sample value and select top k
    samples.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in samples[:k]]


def compute_diversity_contribution(pattern: Dict, all_patterns: List[Dict]) -> float:
    """Compute how much pattern contributes to swarm diversity.

    Args:
        pattern: Pattern to evaluate
        all_patterns: All patterns in swarm

    Returns:
        Diversity contribution 0-1
    """
    if not all_patterns or len(all_patterns) <= 1:
        return 1.0

    pattern_fitness = pattern.get("fitness", 0.5)

    # Compute distance to other patterns (fitness diversity)
    distances = []
    for other in all_patterns:
        if other.get("pattern_id") != pattern.get("pattern_id"):
            other_fitness = other.get("fitness", 0.5)
            distance = abs(pattern_fitness - other_fitness)
            distances.append(distance)

    if not distances:
        return 1.0

    # Average distance = diversity contribution
    avg_distance = sum(distances) / len(distances)

    return round(min(1.0, avg_distance * 5), 4)  # Scale up


def compute_recency_bonus(cycles_since_birth: int, decay_rate: float = 0.01) -> float:
    """Compute recency bonus that decays over time.

    Args:
        cycles_since_birth: Number of cycles since pattern was born
        decay_rate: Decay rate per cycle

    Returns:
        Recency bonus 0-1
    """
    bonus = max(0.0, 1.0 - (cycles_since_birth * decay_rate))
    return round(bonus, 4)


# === D19.2 PROJECTED FUTURE FITNESS ===


def compute_projected_fitness(
    pattern: Dict,
    projection_years: float = 10.0,
    entropy_growth_rate: float = 0.05,
) -> Dict[str, Any]:
    """Compute PROJECTED future fitness for a pattern.

    D19.2: Selection on PROJECTED future, not observed past.

    Args:
        pattern: Pattern dict with current fitness metrics
        projection_years: Years into future to project
        entropy_growth_rate: Annual entropy growth rate

    Returns:
        Dict with projected fitness metrics
    """
    # Get current observed fitness
    current_fitness = pattern.get("fitness", 0.5)
    current_entropy = pattern.get("entropy", 1.0)
    stability = pattern.get("stability", 0.5)

    # Project entropy into future
    entropy_growth_factor = 1 + (projection_years * entropy_growth_rate)
    projected_entropy = current_entropy * entropy_growth_factor

    # Projected entropy reduction (inverse of entropy growth)
    projected_entropy_reduction = max(0.0, 1.0 - (projected_entropy / 10.0))

    # Stability decay over projection window
    stability_decay = math.exp(-projection_years * 0.1)
    projected_stability = stability * stability_decay

    # Projected fitness combines current and projected
    projected_fitness = (
        OBSERVED_FITNESS_WEIGHT * current_fitness +
        PROJECTED_FITNESS_WEIGHT * (
            0.50 * projected_entropy_reduction +
            0.50 * projected_stability
        )
    )
    projected_fitness = max(0.0, min(1.0, projected_fitness))

    # Classify for preemptive selection
    if projected_fitness >= HIGH_FUTURE_FITNESS_THRESHOLD:
        classification = "high_future"
        recommendation = "amplify"
    elif projected_fitness <= LOW_FUTURE_FITNESS_THRESHOLD:
        classification = "low_future"
        recommendation = "starve"
    else:
        classification = "medium_future"
        recommendation = "neutral"

    return {
        "current_fitness": round(current_fitness, 4),
        "projected_fitness": round(projected_fitness, 4),
        "projection_years": projection_years,
        "projected_entropy": round(projected_entropy, 6),
        "projected_entropy_reduction": round(projected_entropy_reduction, 4),
        "projected_stability": round(projected_stability, 4),
        "classification": classification,
        "recommendation": recommendation,
        "selection_on_past": SELECTION_ON_PAST,
    }


def select_by_projected_fitness(
    patterns: List[Dict],
    k: int = 1,
    projection_years: float = 10.0,
) -> List[Dict]:
    """Select patterns based on PROJECTED future fitness.

    D19.2: Pre-amplify high-future-compression, pre-starve low-future.

    Args:
        patterns: List of patterns to select from
        k: Number of patterns to select
        projection_years: Years into future to project

    Returns:
        Selected patterns (highest projected fitness)
    """
    if not patterns:
        return []

    # Compute projected fitness for all patterns
    projections = []
    for pattern in patterns:
        projected = compute_projected_fitness(pattern, projection_years)
        projections.append({
            **pattern,
            "projected_fitness": projected["projected_fitness"],
            "classification": projected["classification"],
            "recommendation": projected["recommendation"],
        })

    # Sort by projected fitness descending
    projections.sort(key=lambda x: x.get("projected_fitness", 0), reverse=True)

    return projections[:k]


def get_fitness_status() -> Dict[str, Any]:
    """Get fitness evaluator status.

    Returns:
        Status dict
    """
    return {
        "module": "autocatalytic.fitness_evaluator",
        "version": "19.2.0",
        "projected_fitness_weight": PROJECTED_FITNESS_WEIGHT,
        "observed_fitness_weight": OBSERVED_FITNESS_WEIGHT,
        "selection_on_past": SELECTION_ON_PAST,
        "high_future_threshold": HIGH_FUTURE_FITNESS_THRESHOLD,
        "low_future_threshold": LOW_FUTURE_FITNESS_THRESHOLD,
        "insight": "Selection on PROJECTED future, not observed past",
    }
