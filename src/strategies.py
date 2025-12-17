"""strategies.py - τ Reduction Strategy Comparator

THE STRATEGY INSIGHT:
    τ penalty isn't a constraint to accept. It's a problem with multiple solutions.
    The right solution depends on ROI.

THE THREE STRATEGIES:
    | Strategy     | τ Reduction       | Trade-off          | Best For              |
    |--------------|-------------------|--------------------|-----------------------|
    | Onboard AI   | Effective α → 1.2+| Compute cost       | High-frequency decisions|
    | Predictive   | 30% delay cut     | c×0.8 overhead     | Predictable sequences |
    | Relay Swarm  | Physical τ ÷ 2    | P cost per sat     | Critical real-time ops|

Source: Grok - "L4 agents predict & pre-actuate 80% of decisions locally (boost eff α to 1.2+)"
Source: Grok - "Quantum-inspired error correction: Cut effective delay 30% via predictive sims (c=0.8)"
Source: Grok - "Relay swarms: Midpoint satellites halve τ to 10min"
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional

from .core import emit_receipt
from .relay import (
    RelayConfig,
    compute_relay_tau,
    compute_relay_p_cost,
    RELAY_TAU_FACTOR,
    RELAY_SWARM_OPTIMAL,
)
from .latency import tau_penalty, TAU_MARS_MAX


# === CONSTANTS ===

ONBOARD_AI_LOCAL_PCT = 0.80
"""Onboard AI handles 80% of decisions locally."""

ONBOARD_AI_EFF_ALPHA_FLOOR = 1.2
"""Minimum effective α with onboard AI. Source: Grok 'boost eff α to 1.2+'"""

PREDICTIVE_TAU_REDUCTION = 0.30
"""Predictive sims reduce effective delay by 30%."""

PREDICTIVE_C_FACTOR = 0.8
"""c factor overhead for predictive computation. Source: Grok 'c=0.8'"""


# === ENUMS ===

class Strategy(Enum):
    """τ reduction strategies."""
    BASELINE = "baseline"
    ONBOARD_AI = "onboard_ai"
    PREDICTIVE = "predictive"
    RELAY_SWARM = "relay_swarm"
    COMBINED = "combined"


# === DATACLASSES ===

@dataclass
class StrategyConfig:
    """Configuration for τ reduction strategy.

    Attributes:
        strategy: Strategy enum value
        relay_swarm_size: Number of relay satellites (for RELAY_SWARM/COMBINED)
        onboard_ai_coverage: Fraction of decisions handled locally (default 0.8)
        predictive_enabled: Whether predictive sims are active
    """
    strategy: Strategy = Strategy.BASELINE
    relay_swarm_size: int = 0
    onboard_ai_coverage: float = ONBOARD_AI_LOCAL_PCT
    predictive_enabled: bool = False

    def __post_init__(self):
        """Set defaults based on strategy."""
        if self.strategy == Strategy.RELAY_SWARM and self.relay_swarm_size == 0:
            self.relay_swarm_size = RELAY_SWARM_OPTIMAL
        elif self.strategy == Strategy.PREDICTIVE:
            self.predictive_enabled = True
        elif self.strategy == Strategy.COMBINED:
            if self.relay_swarm_size == 0:
                self.relay_swarm_size = RELAY_SWARM_OPTIMAL
            self.predictive_enabled = True


@dataclass
class StrategyResult:
    """Result of applying a τ reduction strategy.

    Attributes:
        strategy: Strategy that was applied
        effective_tau: τ after all reductions
        effective_alpha: α after latency penalty and floors
        c_factor: Compute overhead factor (1.0 = no overhead, 0.8 = 20% overhead)
        p_cost: P factor cost (relay infrastructure)
        cycles_to_10k: Cycles to reach 10³ person-eq milestone
        roi_score: Computed ROI (0 = baseline, higher = better)
    """
    strategy: Strategy
    effective_tau: float
    effective_alpha: float
    c_factor: float
    p_cost: float
    cycles_to_10k: int
    roi_score: float = 0.0


# === CORE FUNCTIONS ===

def compute_effective_tau(base_tau: float, config: StrategyConfig) -> float:
    """Apply τ reductions from strategy.

    PREDICTIVE: τ × 0.70 (30% reduction)
    RELAY_SWARM: τ × 0.50 (halved)
    COMBINED: τ × 0.50 × 0.70 = τ × 0.35

    Args:
        base_tau: Base latency in seconds
        config: StrategyConfig with active strategies

    Returns:
        Effective τ after reductions
    """
    effective_tau = base_tau

    # Apply relay reduction (physical)
    if config.strategy in (Strategy.RELAY_SWARM, Strategy.COMBINED):
        if config.relay_swarm_size > 0:
            effective_tau *= RELAY_TAU_FACTOR

    # Apply predictive reduction (computational)
    if config.strategy == Strategy.PREDICTIVE or config.predictive_enabled:
        effective_tau *= (1.0 - PREDICTIVE_TAU_REDUCTION)

    return effective_tau


def compute_effective_alpha(
    base_alpha: float,
    config: StrategyConfig,
    effective_tau: float
) -> float:
    """Apply α modifications from strategy.

    ONBOARD_AI/COMBINED: α floor of 1.2 regardless of τ penalty
    Otherwise: standard τ penalty applied

    Args:
        base_alpha: Base compounding exponent (e.g., 1.69)
        config: StrategyConfig with active strategies
        effective_tau: τ after reductions (for penalty calc)

    Returns:
        Effective α after penalties and floors
    """
    # Compute standard α with τ penalty
    penalty = tau_penalty(effective_tau)
    computed_alpha = base_alpha * penalty

    # Apply onboard AI floor if applicable
    if config.strategy in (Strategy.ONBOARD_AI, Strategy.COMBINED):
        return max(computed_alpha, ONBOARD_AI_EFF_ALPHA_FLOOR)

    return computed_alpha


def compute_c_factor(config: StrategyConfig) -> float:
    """Get c factor overhead from strategy.

    PREDICTIVE/COMBINED: c = 0.8 (20% overhead)
    Otherwise: c = 1.0 (no overhead)

    Args:
        config: StrategyConfig

    Returns:
        c factor (1.0 = no overhead)
    """
    if config.strategy == Strategy.PREDICTIVE or config.predictive_enabled:
        return PREDICTIVE_C_FACTOR
    return 1.0


def compute_p_cost(config: StrategyConfig) -> float:
    """Get P factor cost from strategy.

    RELAY_SWARM/COMBINED: swarm_size × 0.05
    Otherwise: 0

    Args:
        config: StrategyConfig

    Returns:
        P factor cost
    """
    if config.strategy in (Strategy.RELAY_SWARM, Strategy.COMBINED):
        if config.relay_swarm_size > 0:
            relay_config = RelayConfig(swarm_size=config.relay_swarm_size)
            return compute_relay_p_cost(relay_config)
    return 0.0


def estimate_cycles_to_10k(
    effective_alpha: float,
    c_factor: float = 1.0,
    c_base: float = 50.0,
    p_factor: float = 1.8
) -> int:
    """Estimate cycles to reach 10³ person-eq milestone.

    Uses simplified sovereignty timeline model.
    Growth per cycle depends on effective α.

    Args:
        effective_alpha: α after penalties/floors
        c_factor: Compute overhead (reduces build rate)
        c_base: Initial capacity
        p_factor: Propulsion growth factor

    Returns:
        Estimated cycles to reach 1000 person-eq
    """
    # Alpha ratio determines growth rate
    alpha_ratio = effective_alpha / 1.69
    base_multiplier = 1.0 + (2.75 - 1.0) * alpha_ratio

    # Apply c_factor as growth rate modifier
    effective_multiplier = 1.0 + (base_multiplier - 1.0) * c_factor

    person_eq = c_base
    for cycle in range(1, 50):
        propulsion_factor = 1.0 + (p_factor - 1.0) * (0.9 ** (cycle - 1))
        cycle_multiplier = effective_multiplier * propulsion_factor
        person_eq *= cycle_multiplier

        if person_eq >= 1000:
            return cycle

    return 50  # Max cycles


def apply_strategy(
    base_tau: float,
    base_alpha: float,
    config: StrategyConfig
) -> StrategyResult:
    """Apply strategy and compute all metrics.

    Main entry point for strategy evaluation.

    Args:
        base_tau: Base τ (e.g., 1200 for Mars max)
        base_alpha: Base α (e.g., 1.69)
        config: StrategyConfig to apply

    Returns:
        StrategyResult with all computed metrics

    Receipt: strategy_application_receipt
    """
    # Compute metrics
    effective_tau = compute_effective_tau(base_tau, config)
    effective_alpha = compute_effective_alpha(base_alpha, config, effective_tau)
    c_factor = compute_c_factor(config)
    p_cost = compute_p_cost(config)
    cycles = estimate_cycles_to_10k(effective_alpha, c_factor)

    result = StrategyResult(
        strategy=config.strategy,
        effective_tau=effective_tau,
        effective_alpha=effective_alpha,
        c_factor=c_factor,
        p_cost=p_cost,
        cycles_to_10k=cycles,
        roi_score=0.0  # Computed separately by ROI module
    )

    emit_receipt("strategy_application", {
        "tenant_id": "axiom-autonomy",
        "strategy": config.strategy.value,
        "base_tau": base_tau,
        "base_alpha": base_alpha,
        "effective_tau": effective_tau,
        "effective_alpha": effective_alpha,
        "c_factor": c_factor,
        "p_cost": p_cost,
        "cycles_to_10k": cycles,
        "relay_swarm_size": config.relay_swarm_size,
        "predictive_enabled": config.predictive_enabled,
        "onboard_ai_coverage": config.onboard_ai_coverage,
    })

    return result


def compare_strategies(
    strategies: List[StrategyConfig],
    baseline: Dict[str, float]
) -> List[StrategyResult]:
    """Compare multiple strategies and rank by ROI.

    Args:
        strategies: List of StrategyConfig to compare
        baseline: Dict with 'tau' and 'alpha' baseline values

    Returns:
        List of StrategyResult, sorted by cycles_to_10k (ascending)

    Receipt: strategy_comparison_receipt
    """
    base_tau = baseline.get('tau', TAU_MARS_MAX)
    base_alpha = baseline.get('alpha', 1.69)

    results = []
    for config in strategies:
        result = apply_strategy(base_tau, base_alpha, config)
        results.append(result)

    # Sort by cycles (fewer = better)
    results.sort(key=lambda r: r.cycles_to_10k)

    # Find best
    best_result = results[0] if results else None

    emit_receipt("strategy_comparison", {
        "tenant_id": "axiom-autonomy",
        "strategies_evaluated": len(strategies),
        "best_roi": best_result.strategy.value if best_result else None,
        "best_cycles": best_result.cycles_to_10k if best_result else None,
        "roi_ranking": [
            {
                "strategy": r.strategy.value,
                "cycles": r.cycles_to_10k,
                "effective_tau": r.effective_tau,
                "effective_alpha": r.effective_alpha,
                "p_cost": r.p_cost,
            }
            for r in results
        ],
    })

    return results


def recommend_strategy(
    results: List[StrategyResult],
    constraints: Dict[str, float] = None
) -> Optional[StrategyResult]:
    """Recommend highest ROI strategy within constraints.

    Args:
        results: List of StrategyResult from compare_strategies
        constraints: Optional dict with max_p_cost, min_c_factor limits

    Returns:
        Best StrategyResult within constraints, or None

    Receipt: strategy_recommendation_receipt
    """
    if constraints is None:
        constraints = {}

    max_p_cost = constraints.get('max_p_cost', float('inf'))
    min_c_factor = constraints.get('min_c_factor', 0.0)

    # Filter by constraints
    valid = [
        r for r in results
        if r.p_cost <= max_p_cost and r.c_factor >= min_c_factor
    ]

    if not valid:
        emit_receipt("strategy_recommendation", {
            "tenant_id": "axiom-autonomy",
            "recommended": None,
            "reason": "no_valid_strategies",
            "constraints": constraints,
        })
        return None

    # Best by cycles (already sorted)
    best = valid[0]

    emit_receipt("strategy_recommendation", {
        "tenant_id": "axiom-autonomy",
        "recommended": best.strategy.value,
        "effective_tau": best.effective_tau,
        "effective_alpha": best.effective_alpha,
        "cycles_to_10k": best.cycles_to_10k,
        "p_cost": best.p_cost,
        "c_factor": best.c_factor,
        "constraints_applied": constraints,
    })

    return best


def get_all_strategy_configs() -> List[StrategyConfig]:
    """Get configurations for all strategies.

    Convenience function for comparing all options.

    Returns:
        List of StrategyConfig for each Strategy enum value
    """
    return [
        StrategyConfig(strategy=Strategy.BASELINE),
        StrategyConfig(strategy=Strategy.ONBOARD_AI),
        StrategyConfig(strategy=Strategy.PREDICTIVE),
        StrategyConfig(strategy=Strategy.RELAY_SWARM, relay_swarm_size=RELAY_SWARM_OPTIMAL),
        StrategyConfig(strategy=Strategy.COMBINED, relay_swarm_size=RELAY_SWARM_OPTIMAL),
    ]
