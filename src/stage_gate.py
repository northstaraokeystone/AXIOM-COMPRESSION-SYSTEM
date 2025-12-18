"""stage_gate.py - Dynamic Allocation with Alpha-Triggered Escalation

THE HEDGE (v2.0 - Grok Integration):
    "Stage-gate the pivot - 30% now, trigger +10% if alpha > 1.9 confirmed in next 12 months"

Static allocation is wrong. The stage-gate mechanism provides:
1. Conservative initial allocation (30%)
2. Observable trigger condition (alpha > 1.9)
3. Bounded escalation (+10%, max 40%)
4. Time-boxed evaluation window (12 months)

This is not indecision - it's calibrated risk management.

Source: Grok - "Stage-gate the pivot"
"""

from dataclasses import dataclass
from typing import Tuple

from .core import emit_receipt


# === CONSTANTS (from Grok's hedge recommendation) ===

STAGE_GATE_INITIAL = 0.30
"""Initial autonomy allocation fraction.
Source: Grok - '30% now'"""

STAGE_GATE_TRIGGER_ALPHA = 1.9
"""Alpha threshold that triggers escalation.
Source: Grok - '+10% if alpha > 1.9'"""

STAGE_GATE_ESCALATION = 0.10
"""Increment to add when trigger is met.
Source: Grok - '+10%'"""

STAGE_GATE_WINDOW_MONTHS = 12
"""Evaluation window in months.
Source: Grok - 'in next 12 months'"""

STAGE_GATE_MAX_AUTONOMY = 0.40
"""Maximum autonomy fraction (ceiling).
Source: Grok - '40% (recommended)'"""

STAGE_GATE_MIN_PROPULSION = 0.60
"""Minimum propulsion fraction (viability floor).
Inverse of max autonomy."""

ALPHA_CONFIDENCE_THRESHOLD = 0.70
"""Minimum confidence level required to trigger escalation.
Engineering judgment: don't escalate on noisy signal."""


@dataclass
class StageGateConfig:
    """Configuration for stage-gate mechanism.

    Attributes:
        initial_autonomy_fraction: Starting allocation (default 0.30)
        trigger_alpha: Alpha threshold for escalation (default 1.9)
        escalation_increment: Fraction to add on trigger (default 0.10)
        window_months: Evaluation window in months (default 12)
        max_autonomy_fraction: Ceiling for autonomy (default 0.40)
    """

    initial_autonomy_fraction: float = STAGE_GATE_INITIAL
    trigger_alpha: float = STAGE_GATE_TRIGGER_ALPHA
    escalation_increment: float = STAGE_GATE_ESCALATION
    window_months: int = STAGE_GATE_WINDOW_MONTHS
    max_autonomy_fraction: float = STAGE_GATE_MAX_AUTONOMY


@dataclass
class StageGateState:
    """State of the stage-gate mechanism.

    Attributes:
        current_autonomy_fraction: Current allocation fraction
        alpha_measured: Most recent alpha measurement
        alpha_confidence: Confidence in alpha measurement (0-1)
        trigger_met: True if alpha > trigger within window
        months_elapsed: Months since window started
        escalation_applied: True if escalation has been applied
    """

    current_autonomy_fraction: float
    alpha_measured: float
    alpha_confidence: float
    trigger_met: bool
    months_elapsed: int
    escalation_applied: bool


def evaluate_gate(
    alpha_measured: float,
    alpha_confidence: float,
    months_elapsed: int,
    config: StageGateConfig = None,
) -> StageGateState:
    """Evaluate stage gate and determine if escalation triggers.

    The gate evaluation logic:
    1. If alpha > trigger AND confidence >= threshold -> trigger_met = True
    2. If trigger_met AND within window AND not already escalated -> escalate
    3. If window expires without trigger -> stay at current allocation

    Args:
        alpha_measured: Measured alpha value from fleet data
        alpha_confidence: Confidence in measurement (0-1)
        months_elapsed: Months since evaluation window started
        config: StageGateConfig (uses defaults if None)

    Returns:
        StageGateState with updated values including escalation decision
    """
    if config is None:
        config = StageGateConfig()

    # Check trigger conditions
    alpha_high_enough = alpha_measured > config.trigger_alpha
    confidence_sufficient = alpha_confidence >= ALPHA_CONFIDENCE_THRESHOLD
    within_window = months_elapsed <= config.window_months

    trigger_met = alpha_high_enough and confidence_sufficient and within_window

    # Determine current allocation
    if trigger_met:
        new_fraction = apply_escalation(
            config.initial_autonomy_fraction,
            config.escalation_increment,
            config.max_autonomy_fraction,
        )
        escalation_applied = True
    else:
        new_fraction = config.initial_autonomy_fraction
        escalation_applied = False

    state = StageGateState(
        current_autonomy_fraction=new_fraction,
        alpha_measured=alpha_measured,
        alpha_confidence=alpha_confidence,
        trigger_met=trigger_met,
        months_elapsed=months_elapsed,
        escalation_applied=escalation_applied,
    )

    # Emit receipt
    emit_receipt(
        "stage_gate",
        {
            "tenant_id": "axiom-autonomy",
            "current_autonomy_fraction": new_fraction,
            "alpha_measured": alpha_measured,
            "alpha_confidence": alpha_confidence,
            "trigger_alpha": config.trigger_alpha,
            "trigger_met": trigger_met,
            "escalation_applied": escalation_applied,
            "new_autonomy_fraction": new_fraction,
            "months_elapsed": months_elapsed,
            "window_months": config.window_months,
        },
    )

    return state


def apply_escalation(
    current_fraction: float, increment: float, max_fraction: float
) -> float:
    """Apply escalation increment, respecting ceiling.

    Args:
        current_fraction: Current autonomy fraction
        increment: Fraction to add
        max_fraction: Maximum allowed fraction

    Returns:
        New fraction = min(current + increment, max)
    """
    return min(current_fraction + increment, max_fraction)


def get_allocation(
    state: StageGateState, config: StageGateConfig = None
) -> Tuple[float, float]:
    """Get current propulsion/autonomy allocation from gate state.

    Args:
        state: Current StageGateState
        config: StageGateConfig (for reference)

    Returns:
        Tuple of (propulsion_fraction, autonomy_fraction)
        Always sums to 1.0

    SLO: propulsion_fraction >= 0.60 (viability floor)
    """
    autonomy = state.current_autonomy_fraction
    propulsion = 1.0 - autonomy

    # Enforce viability floor
    if propulsion < STAGE_GATE_MIN_PROPULSION:
        propulsion = STAGE_GATE_MIN_PROPULSION
        autonomy = 1.0 - propulsion

    return (propulsion, autonomy)


def reset_window(state: StageGateState) -> StageGateState:
    """Reset elapsed months for next evaluation window.

    Called after window expires or escalation is applied.

    Args:
        state: Current StageGateState

    Returns:
        New state with months_elapsed = 0
    """
    return StageGateState(
        current_autonomy_fraction=state.current_autonomy_fraction,
        alpha_measured=state.alpha_measured,
        alpha_confidence=state.alpha_confidence,
        trigger_met=False,  # Reset trigger for new window
        months_elapsed=0,
        escalation_applied=state.escalation_applied,  # Keep history
    )


def check_gate_slos(state: StageGateState, config: StageGateConfig = None) -> dict:
    """Check stage gate SLOs.

    SLOs:
    1. Gate must evaluate every 12 months minimum
    2. Escalation must not exceed max_autonomy_fraction (0.40)
    3. propulsion_fraction must remain >= 0.60 (viability floor)

    Args:
        state: Current StageGateState
        config: StageGateConfig (uses defaults if None)

    Returns:
        Dict with SLO check results
    """
    if config is None:
        config = StageGateConfig()

    propulsion, autonomy = get_allocation(state, config)

    slo_checks = {
        "evaluation_frequency": state.months_elapsed <= config.window_months,
        "autonomy_ceiling": state.current_autonomy_fraction
        <= config.max_autonomy_fraction,
        "propulsion_floor": propulsion >= STAGE_GATE_MIN_PROPULSION,
        "allocations_sum_to_one": abs(propulsion + autonomy - 1.0) < 0.001,
    }

    slo_checks["all_passed"] = all(slo_checks.values())

    return slo_checks


def simulate_gate_progression(
    alpha_trajectory: list,
    confidence_trajectory: list = None,
    config: StageGateConfig = None,
) -> list:
    """Simulate gate evaluations over multiple months.

    Args:
        alpha_trajectory: List of alpha measurements by month
        confidence_trajectory: List of confidence values (default all 0.85)
        config: StageGateConfig (uses defaults if None)

    Returns:
        List of StageGateState for each month
    """
    if config is None:
        config = StageGateConfig()

    if confidence_trajectory is None:
        confidence_trajectory = [0.85] * len(alpha_trajectory)

    states = []
    current_fraction = config.initial_autonomy_fraction
    escalated = False

    for month, (alpha, conf) in enumerate(zip(alpha_trajectory, confidence_trajectory)):
        if escalated:
            # Already escalated, maintain new allocation
            state = StageGateState(
                current_autonomy_fraction=min(
                    config.initial_autonomy_fraction + config.escalation_increment,
                    config.max_autonomy_fraction,
                ),
                alpha_measured=alpha,
                alpha_confidence=conf,
                trigger_met=True,
                months_elapsed=month,
                escalation_applied=True,
            )
        else:
            # Evaluate gate
            state = evaluate_gate(alpha, conf, month, config)
            if state.trigger_met:
                escalated = True

        states.append(state)

    return states


def get_gate_recommendation(
    alpha_current: float, alpha_confidence: float, months_remaining: int
) -> str:
    """Get human-readable recommendation based on gate status.

    Args:
        alpha_current: Current measured alpha
        alpha_confidence: Confidence in measurement
        months_remaining: Months left in evaluation window

    Returns:
        Recommendation string
    """
    if (
        alpha_current > STAGE_GATE_TRIGGER_ALPHA
        and alpha_confidence >= ALPHA_CONFIDENCE_THRESHOLD
    ):
        return f"ESCALATE: alpha={alpha_current:.2f} > {STAGE_GATE_TRIGGER_ALPHA} with {alpha_confidence:.0%} confidence. Increase autonomy to 40%."

    if alpha_confidence < ALPHA_CONFIDENCE_THRESHOLD:
        return f"HOLD: Confidence {alpha_confidence:.0%} below threshold {ALPHA_CONFIDENCE_THRESHOLD:.0%}. Need more data."

    if months_remaining <= 0:
        return f"HOLD: Window expired. alpha={alpha_current:.2f} did not exceed {STAGE_GATE_TRIGGER_ALPHA}. Maintain 30% allocation."

    gap = STAGE_GATE_TRIGGER_ALPHA - alpha_current
    return f"MONITOR: alpha={alpha_current:.2f}, gap to trigger={gap:.2f}. {months_remaining} months remaining."


def emit_stage_gate_receipt(
    state: StageGateState, config: StageGateConfig = None, recommendation: str = None
) -> dict:
    """Emit detailed stage gate receipt per CLAUDEME.

    Args:
        state: Current StageGateState
        config: StageGateConfig used
        recommendation: Optional recommendation string

    Returns:
        Receipt dict
    """
    if config is None:
        config = StageGateConfig()

    propulsion, autonomy = get_allocation(state, config)

    return emit_receipt(
        "stage_gate",
        {
            "tenant_id": "axiom-autonomy",
            "current_autonomy_fraction": state.current_autonomy_fraction,
            "current_propulsion_fraction": propulsion,
            "alpha_measured": state.alpha_measured,
            "alpha_confidence": state.alpha_confidence,
            "trigger_alpha": config.trigger_alpha,
            "trigger_met": state.trigger_met,
            "escalation_applied": state.escalation_applied,
            "new_autonomy_fraction": state.current_autonomy_fraction,
            "months_elapsed": state.months_elapsed,
            "window_months": config.window_months,
            "months_remaining": max(0, config.window_months - state.months_elapsed),
            "recommendation": recommendation
            or get_gate_recommendation(
                state.alpha_measured,
                state.alpha_confidence,
                config.window_months - state.months_elapsed,
            ),
        },
    )
