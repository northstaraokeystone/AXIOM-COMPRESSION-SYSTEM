"""optimize.py - Thompson Sampling Optimization Agent

THE OPTIMIZATION INSIGHT:
    "Quantum" = superposition patterns, not literal QM.
    Thompson sampling replaces Boltzmann selection.
    High-variance patterns explored, high-mean patterns exploited.

Source Pattern: QED v12 §3.7 - Thompson sampling selection
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .core import emit_receipt


# === CONSTANTS ===

TENANT_ID = "spaceproof-autonomy"
"""Tenant for optimization receipts."""


# === DATACLASSES ===


@dataclass
class OptimizationConfig:
    """Configuration for optimization agent.

    Attributes:
        sample_count: Number of Thompson samples per selection (default 100)
        exploration_bonus: Bonus for high-variance patterns (default 0.1)
        fitness_decay: Decay factor for old fitness values (default 0.95)
    """

    sample_count: int = 100
    exploration_bonus: float = 0.1
    fitness_decay: float = 0.95


@dataclass
class OptimizationState:
    """State of optimization agent.

    Attributes:
        pattern_fitness: Dict mapping pattern_id to (mean, variance) tuple
        selection_history: List of selected pattern IDs
        improvement_trace: List of improvement values over time
    """

    pattern_fitness: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    selection_history: List[str] = field(default_factory=list)
    improvement_trace: List[float] = field(default_factory=list)


# === FUNCTIONS ===


def sample_thompson(mean: float, variance: float, n_samples: int = 100) -> float:
    """Draw from beta distribution approximation, return expected value.

    Uses beta distribution parameterized by mean and variance.
    For exploration: high variance = wider distribution = more random samples.

    Args:
        mean: Mean fitness value (0-1 range recommended)
        variance: Variance in fitness estimates
        n_samples: Number of samples to draw

    Returns:
        Expected value from Thompson sampling
    """
    # Clamp mean to valid range
    mean = max(0.001, min(0.999, mean))
    variance = max(0.0001, min(mean * (1 - mean), variance))

    # Compute beta parameters from mean and variance
    # mean = alpha / (alpha + beta)
    # variance = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
    common = mean * (1 - mean) / variance - 1
    alpha = max(0.1, mean * common)
    beta = max(0.1, (1 - mean) * common)

    # Draw samples and return mean of samples
    samples = [random.betavariate(alpha, beta) for _ in range(n_samples)]
    return sum(samples) / len(samples)


def selection_pressure(
    patterns: List[str],
    fitness_scores: Dict[str, Tuple[float, float]],
    config: OptimizationConfig = None,
) -> List[str]:
    """Select patterns via Thompson sampling.

    High-variance patterns explored, high-mean patterns exploited.

    Args:
        patterns: List of pattern IDs to select from
        fitness_scores: Dict mapping pattern_id to (mean, variance)
        config: OptimizationConfig (uses defaults if None)

    Returns:
        List of selected pattern IDs, sorted by Thompson sample value

    Emits: optimization_receipt
    """
    if config is None:
        config = OptimizationConfig()

    if not patterns:
        return []

    # Thompson sample each pattern
    sampled_values = {}
    exploration_count = 0

    for pattern_id in patterns:
        if pattern_id in fitness_scores:
            mean, variance = fitness_scores[pattern_id]
        else:
            # Unknown pattern: high variance for exploration
            mean, variance = 0.5, 0.25

        # Add exploration bonus for high-variance patterns
        effective_variance = variance
        if variance > 0.1:
            exploration_count += 1

        sampled_value = sample_thompson(mean, effective_variance, config.sample_count)
        sampled_value += config.exploration_bonus * math.sqrt(variance)
        sampled_values[pattern_id] = sampled_value

    # Sort by sampled value (descending)
    selected = sorted(patterns, key=lambda p: sampled_values.get(p, 0), reverse=True)

    # Find top pattern
    top_pattern = selected[0] if selected else ""
    exploration_ratio = exploration_count / len(patterns) if patterns else 0.0

    # Compute improvement vs random baseline
    random_expected = (
        sum(fitness_scores.get(p, (0.5, 0.25))[0] for p in patterns) / len(patterns)
        if patterns
        else 0.5
    )
    top_mean = fitness_scores.get(top_pattern, (0.5, 0.25))[0] if top_pattern else 0.5
    improvement = top_mean / random_expected if random_expected > 0 else 1.0

    # Emit receipt
    emit_receipt(
        "optimization",
        {
            "tenant_id": TENANT_ID,
            "cycle": len(sampled_values),
            "patterns_evaluated": len(patterns),
            "patterns_selected": len(selected),
            "top_pattern": top_pattern,
            "improvement_vs_random": round(improvement, 4),
            "exploration_ratio": round(exploration_ratio, 4),
        },
    )

    return selected


def update_fitness(
    pattern_id: str,
    outcome: float,
    state: OptimizationState,
    config: OptimizationConfig = None,
) -> OptimizationState:
    """Update mean/variance for pattern using Bayesian update.

    Args:
        pattern_id: Pattern to update
        outcome: Observed outcome (0-1)
        state: Current OptimizationState
        config: OptimizationConfig for decay factor

    Returns:
        Updated OptimizationState
    """
    if config is None:
        config = OptimizationConfig()

    # Get current estimates
    if pattern_id in state.pattern_fitness:
        old_mean, old_variance = state.pattern_fitness[pattern_id]
    else:
        old_mean, old_variance = 0.5, 0.25

    # Apply decay
    decayed_mean = old_mean * config.fitness_decay
    decayed_variance = old_variance * config.fitness_decay

    # Bayesian update (simplified: weighted average)
    n_observations = 1.0 / (old_variance + 0.01)  # Pseudo-count from variance
    new_n = n_observations + 1

    new_mean = (decayed_mean * n_observations + outcome) / new_n
    new_variance = max(
        0.01, (decayed_variance * n_observations + (outcome - new_mean) ** 2) / new_n
    )

    # Update state
    new_pattern_fitness = dict(state.pattern_fitness)
    new_pattern_fitness[pattern_id] = (new_mean, new_variance)

    new_selection_history = list(state.selection_history)
    new_selection_history.append(pattern_id)

    return OptimizationState(
        pattern_fitness=new_pattern_fitness,
        selection_history=new_selection_history,
        improvement_trace=list(state.improvement_trace),
    )


def measure_improvement(state: OptimizationState) -> float:
    """Return improvement rate over baseline random selection.

    Args:
        state: Current OptimizationState

    Returns:
        Improvement multiplier (>1.0 means better than random)
    """
    if not state.pattern_fitness:
        return 1.0

    # Calculate average fitness of patterns we've selected
    selected_fitness = []
    for pattern_id in state.selection_history:
        if pattern_id in state.pattern_fitness:
            mean, _ = state.pattern_fitness[pattern_id]
            selected_fitness.append(mean)

    if not selected_fitness:
        return 1.0

    avg_selected = sum(selected_fitness) / len(selected_fitness)

    # Compare to overall average (random baseline)
    all_means = [m for m, _ in state.pattern_fitness.values()]
    avg_all = sum(all_means) / len(all_means) if all_means else 0.5

    improvement = avg_selected / avg_all if avg_all > 0 else 1.0

    # Track improvement
    state.improvement_trace.append(improvement)

    return improvement


def initialize_state() -> OptimizationState:
    """Create fresh optimization state.

    Returns:
        New OptimizationState with empty fitness dict
    """
    return OptimizationState(
        pattern_fitness={}, selection_history=[], improvement_trace=[]
    )


def get_exploration_exploitation_ratio(state: OptimizationState) -> Tuple[float, float]:
    """Calculate exploration vs exploitation balance.

    Returns:
        Tuple of (exploration_ratio, exploitation_ratio)
        Exploration = selections of high-variance patterns
        Exploitation = selections of high-mean patterns
    """
    if not state.selection_history:
        return 0.5, 0.5

    exploration_count = 0
    exploitation_count = 0

    for pattern_id in state.selection_history:
        if pattern_id in state.pattern_fitness:
            mean, variance = state.pattern_fitness[pattern_id]
            if variance > 0.1:
                exploration_count += 1
            if mean > 0.7:
                exploitation_count += 1

    total = len(state.selection_history)
    return exploration_count / total, exploitation_count / total


def integrate_roi(
    pattern_fitness: Dict[str, Tuple[float, float]], roi_scores: Dict[str, float]
) -> Dict[str, Tuple[float, float]]:
    """Weight pattern fitness by ROI scores.

    High-ROI patterns get exploration bonus via increased variance.
    This encourages the optimizer to explore high-ROI patterns more.

    Args:
        pattern_fitness: Dict mapping pattern_id to (mean, variance)
        roi_scores: Dict mapping pattern_id to ROI score

    Returns:
        Modified fitness dict with ROI-weighted values

    Receipt: roi_integration_receipt
    """
    if not roi_scores:
        return pattern_fitness

    weighted = {}

    for pattern_id, (mean, variance) in pattern_fitness.items():
        roi = roi_scores.get(pattern_id, 0.0)

        # ROI bonus: increase mean for high-ROI patterns
        # Scale: ROI of 1.0 adds 0.1 to mean
        roi_bonus = min(0.3, roi * 0.1)  # Cap at 0.3 bonus
        weighted_mean = min(0.99, mean + roi_bonus)

        # Also increase variance for high-ROI to encourage exploration
        # This makes the optimizer more likely to try high-ROI patterns
        roi_variance_boost = 1.0 + (roi * 0.2)  # Up to +20% variance per ROI point
        weighted_variance = min(0.25, variance * roi_variance_boost)

        weighted[pattern_id] = (weighted_mean, weighted_variance)

    emit_receipt(
        "roi_integration",
        {
            "tenant_id": TENANT_ID,
            "patterns_weighted": len(weighted),
            "avg_roi_bonus": sum(roi_scores.values()) / len(roi_scores)
            if roi_scores
            else 0,
            "patterns_with_roi": len(roi_scores),
        },
    )

    return weighted
