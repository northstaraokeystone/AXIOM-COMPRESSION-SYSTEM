"""roi.py - ROI Reward/Penalty System

THE ROI INSIGHT:
    Optimize if delivering highest ROI.
    Every strategy has costs. Every strategy has benefits.
    ROI = (cycles_saved × reward) - (p_cost + c_reduction) × penalty

Source Pattern: QED v9 §3.5 — "ROI change or kill: if dollar_value > 1M and FP can tune to < 0.01, keep live"
Source: Grok - "Build out reward/penalty system to optimize if delivering highest roi."
"""

from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

from .core import emit_receipt

if TYPE_CHECKING:
    from .strategies import StrategyResult


# === CONSTANTS ===

DEFAULT_REWARD_PER_CYCLE = 1.0
"""Reward per cycle saved vs baseline."""

DEFAULT_PENALTY_PER_P_COST = 0.5
"""Penalty multiplier for P factor cost."""

DEFAULT_PENALTY_PER_C_REDUCTION = 0.3
"""Penalty multiplier for c factor reduction."""

DEFAULT_MIN_VIABLE_ROI = 0.1
"""Minimum ROI to consider strategy viable."""

ROI_GATE_DEPLOY = 0.5
"""ROI threshold for 'deploy' decision."""

ROI_GATE_SHADOW = 0.1
"""ROI threshold for 'shadow' (vs 'kill') decision."""


# === DATACLASS ===


@dataclass
class ROIConfig:
    """Configuration for ROI computation.

    Attributes:
        reward_per_cycle_saved: Reward for each cycle saved vs baseline
        penalty_per_p_cost: Penalty weight for P factor cost
        penalty_per_c_reduction: Penalty weight for c factor reduction
        min_viable_roi: Minimum ROI to keep strategy alive
    """

    reward_per_cycle_saved: float = DEFAULT_REWARD_PER_CYCLE
    penalty_per_p_cost: float = DEFAULT_PENALTY_PER_P_COST
    penalty_per_c_reduction: float = DEFAULT_PENALTY_PER_C_REDUCTION
    min_viable_roi: float = DEFAULT_MIN_VIABLE_ROI


# === FUNCTIONS ===


def reward(cycles_saved: int, config: ROIConfig = None) -> float:
    """Compute reward for cycles saved.

    Args:
        cycles_saved: Number of cycles saved vs baseline (positive = good)
        config: ROIConfig (uses defaults if None)

    Returns:
        Reward value

    Receipt: roi_reward_receipt
    """
    if config is None:
        config = ROIConfig()

    reward_value = cycles_saved * config.reward_per_cycle_saved

    emit_receipt(
        "roi_reward",
        {
            "tenant_id": "spaceproof-autonomy",
            "cycles_saved": cycles_saved,
            "reward_per_cycle": config.reward_per_cycle_saved,
            "reward_value": reward_value,
        },
    )

    return reward_value


def penalty(p_cost: float, c_reduction: float, config: ROIConfig = None) -> float:
    """Compute penalty for costs.

    penalty = (p_cost × penalty_per_p_cost) + (c_reduction × penalty_per_c_reduction)

    Args:
        p_cost: P factor cost (relay infrastructure)
        c_reduction: c factor reduction (1.0 - c_factor)
        config: ROIConfig (uses defaults if None)

    Returns:
        Penalty value

    Receipt: roi_penalty_receipt
    """
    if config is None:
        config = ROIConfig()

    p_penalty = p_cost * config.penalty_per_p_cost
    c_penalty = c_reduction * config.penalty_per_c_reduction
    penalty_value = p_penalty + c_penalty

    emit_receipt(
        "roi_penalty",
        {
            "tenant_id": "spaceproof-autonomy",
            "p_cost": p_cost,
            "c_reduction": c_reduction,
            "p_penalty": p_penalty,
            "c_penalty": c_penalty,
            "penalty_value": penalty_value,
        },
    )

    return penalty_value


def compute_roi(
    result: "StrategyResult", baseline: "StrategyResult", config: ROIConfig = None
) -> float:
    """Compute ROI for a strategy vs baseline.

    ROI = reward - penalty
    Where:
        reward = cycles_saved × reward_per_cycle
        penalty = (p_cost × penalty_p) + (c_reduction × penalty_c)

    Args:
        result: StrategyResult to evaluate
        baseline: Baseline StrategyResult (usually BASELINE strategy)
        config: ROIConfig (uses defaults if None)

    Returns:
        ROI score (higher = better)

    Receipt: roi_computation_receipt
    """
    if config is None:
        config = ROIConfig()

    # Compute components
    cycles_saved = baseline.cycles_to_10k - result.cycles_to_10k
    p_cost = result.p_cost - baseline.p_cost
    c_reduction = 1.0 - result.c_factor  # c=0.8 means 0.2 reduction

    reward_value = reward(cycles_saved, config)
    penalty_value = penalty(max(0, p_cost), max(0, c_reduction), config)

    roi_score = reward_value - penalty_value

    emit_receipt(
        "roi_computation",
        {
            "tenant_id": "spaceproof-autonomy",
            "strategy": result.strategy.value,
            "baseline_cycles": baseline.cycles_to_10k,
            "result_cycles": result.cycles_to_10k,
            "cycles_saved": cycles_saved,
            "p_cost": p_cost,
            "c_reduction": c_reduction,
            "reward_value": reward_value,
            "penalty_value": penalty_value,
            "roi_score": roi_score,
        },
    )

    return roi_score


def roi_gate(roi_score: float, config: ROIConfig = None) -> str:
    """Gate decision based on ROI.

    Gate logic:
        roi ≥ 0.5 → "deploy"
        0.1 ≤ roi < 0.5 → "shadow"
        roi < 0.1 → "kill"

    Args:
        roi_score: Computed ROI
        config: ROIConfig (uses min_viable_roi)

    Returns:
        Decision string: "deploy", "shadow", or "kill"

    Receipt: roi_gate_receipt
    """
    if config is None:
        config = ROIConfig()

    if roi_score >= ROI_GATE_DEPLOY:
        decision = "deploy"
    elif roi_score >= ROI_GATE_SHADOW:
        decision = "shadow"
    else:
        decision = "kill"

    emit_receipt(
        "roi_gate",
        {
            "tenant_id": "spaceproof-autonomy",
            "roi_score": roi_score,
            "decision": decision,
            "threshold_deploy": ROI_GATE_DEPLOY,
            "threshold_shadow": ROI_GATE_SHADOW,
        },
    )

    return decision


def rank_by_roi(
    results: List["StrategyResult"],
    baseline: "StrategyResult",
    config: ROIConfig = None,
) -> List[Tuple["StrategyResult", float]]:
    """Rank strategies by ROI.

    Args:
        results: List of StrategyResult to rank
        baseline: Baseline for ROI computation
        config: ROIConfig

    Returns:
        List of (StrategyResult, roi_score) tuples, sorted descending by ROI

    Receipt: roi_ranking_receipt
    """
    if config is None:
        config = ROIConfig()

    # Compute ROI for each
    ranked = []
    for result in results:
        roi = compute_roi(result, baseline, config)
        ranked.append((result, roi))

    # Sort by ROI descending
    ranked.sort(key=lambda x: x[1], reverse=True)

    emit_receipt(
        "roi_ranking",
        {
            "tenant_id": "spaceproof-autonomy",
            "strategies_ranked": len(ranked),
            "top_strategy": ranked[0][0].strategy.value if ranked else None,
            "top_roi": ranked[0][1] if ranked else None,
            "ranking": [{"strategy": r[0].strategy.value, "roi": r[1]} for r in ranked],
        },
    )

    return ranked


def update_result_roi(
    result: "StrategyResult", baseline: "StrategyResult", config: ROIConfig = None
) -> "StrategyResult":
    """Update StrategyResult with computed ROI.

    Args:
        result: StrategyResult to update
        baseline: Baseline for ROI computation
        config: ROIConfig

    Returns:
        New StrategyResult with roi_score populated
    """
    from .strategies import StrategyResult

    roi = compute_roi(result, baseline, config)

    return StrategyResult(
        strategy=result.strategy,
        effective_tau=result.effective_tau,
        effective_alpha=result.effective_alpha,
        c_factor=result.c_factor,
        p_cost=result.p_cost,
        cycles_to_10k=result.cycles_to_10k,
        roi_score=roi,
    )


def evaluate_strategy_roi(
    result: "StrategyResult", baseline: "StrategyResult", config: ROIConfig = None
) -> dict:
    """Full ROI evaluation with gate decision.

    Args:
        result: Strategy to evaluate
        baseline: Baseline strategy
        config: ROIConfig

    Returns:
        Dict with roi_score, decision, and breakdown
    """
    if config is None:
        config = ROIConfig()

    roi = compute_roi(result, baseline, config)
    decision = roi_gate(roi, config)

    cycles_saved = baseline.cycles_to_10k - result.cycles_to_10k
    p_cost = result.p_cost - baseline.p_cost
    c_reduction = 1.0 - result.c_factor

    return {
        "strategy": result.strategy.value,
        "roi_score": roi,
        "decision": decision,
        "cycles_saved": cycles_saved,
        "p_cost": p_cost,
        "c_reduction": c_reduction,
        "effective_tau": result.effective_tau,
        "effective_alpha": result.effective_alpha,
    }
