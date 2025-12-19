"""Multi-dimensional fitness evaluation for D19.

Implements QED v12 fitness model:
- Entropy reduction
- Coordination success
- Stability
- Diversity contribution
- Recency bonus
"""

import random
from typing import Any, Dict, List


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
