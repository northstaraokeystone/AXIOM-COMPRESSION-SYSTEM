"""leading_indicators.py - Observable Proxy Monitoring

THE INDICATORS (v2.0 - Grok Integration):

1. SIM_FIDELITY: Autonomy sim vs real Starship telemetry gap
   - Target: >= 95% correlation with real telemetry
   - Grok: "Autonomy sim fidelity on real Starship telemetry (current gap?)"

2. FLEET_LEARNING_RATE: Measured alpha from FSD/Optimus trajectories
   - Target: alpha >= 1.8 with confidence >= 80%
   - Grok: "What's your current best calibration of alpha?"

3. TAU_VELOCITY: d(tau)/dt in high-latency tests
   - Target: d(tau)/dt < -0.05 (improving by 5% per cycle)
   - Grok: "tau (decision loop time) reduction velocity"

These are the de-risking metrics. Observable proxies that validate
or invalidate the model before betting the farm.

Source: Grok - "Leading indicators to monitor"
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

from .core import emit_receipt
from .calibration import CalibrationOutput, ALPHA_BASELINE


# === CONSTANTS (target values for gap calculation) ===

SIM_FIDELITY_TARGET = 0.95
"""Target simulation fidelity vs real telemetry.
Source: Engineering judgment - 95% correlation."""

FLEET_LEARNING_ALPHA_TARGET = 1.8
"""Target alpha value from fleet learning.
Source: Grok - 'calibrated to alpha=1.8 baseline'"""

FLEET_LEARNING_CONFIDENCE_TARGET = 0.80
"""Target confidence for alpha estimate.
Source: Grok - high confidence required for stage gate."""

TAU_VELOCITY_TARGET = -0.05
"""Target tau velocity: 5% improvement per cycle.
Negative = improving (tau decreasing)."""


class LeadingIndicator(Enum):
    """Leading indicator types."""
    SIM_FIDELITY = "sim_fidelity"
    FLEET_LEARNING_RATE = "fleet_learning_rate"
    TAU_VELOCITY = "tau_velocity"


@dataclass
class IndicatorMeasurement:
    """Measurement of a leading indicator.

    Attributes:
        indicator_type: LeadingIndicator enum value
        current_value: Current measured value
        target_value: Target value to achieve
        gap: current - target (negative = below target)
        trend: "improving", "stable", or "degrading"
        confidence: Confidence in measurement (0-1)
    """
    indicator_type: LeadingIndicator
    current_value: float
    target_value: float
    gap: float
    trend: str
    confidence: float


def measure_sim_fidelity(
    sim_predictions: List[float],
    actual_telemetry: List[float]
) -> IndicatorMeasurement:
    """Compare sim predictions to real Starship telemetry.

    Measures correlation between simulation predictions and actual
    flight telemetry data. High fidelity = reliable model.

    Args:
        sim_predictions: List of predicted values from simulation
        actual_telemetry: List of actual measured values

    Returns:
        IndicatorMeasurement for sim fidelity
    """
    if not sim_predictions or not actual_telemetry:
        return IndicatorMeasurement(
            indicator_type=LeadingIndicator.SIM_FIDELITY,
            current_value=0.0,
            target_value=SIM_FIDELITY_TARGET,
            gap=-SIM_FIDELITY_TARGET,
            trend="unknown",
            confidence=0.0
        )

    # Ensure same length
    n = min(len(sim_predictions), len(actual_telemetry))
    sim = sim_predictions[:n]
    actual = actual_telemetry[:n]

    # Calculate Pearson correlation
    mean_sim = sum(sim) / n
    mean_actual = sum(actual) / n

    numerator = sum((sim[i] - mean_sim) * (actual[i] - mean_actual) for i in range(n))
    denom_sim = sum((sim[i] - mean_sim) ** 2 for i in range(n))
    denom_actual = sum((actual[i] - mean_actual) ** 2 for i in range(n))

    if denom_sim == 0 or denom_actual == 0:
        correlation = 0.0
    else:
        correlation = numerator / ((denom_sim * denom_actual) ** 0.5)

    # Correlation can be negative; we want absolute value for fidelity
    fidelity = abs(correlation)

    # Gap from target
    gap = fidelity - SIM_FIDELITY_TARGET

    # Trend based on recent improvement (would need historical data)
    # For now, estimate from gap
    if gap >= 0:
        trend = "stable"  # At or above target
    elif gap >= -0.10:
        trend = "improving"  # Close to target
    else:
        trend = "degrading"  # Far from target

    # Confidence based on data quantity
    confidence = min(1.0, n / 100.0)  # Full confidence at 100+ points

    measurement = IndicatorMeasurement(
        indicator_type=LeadingIndicator.SIM_FIDELITY,
        current_value=fidelity,
        target_value=SIM_FIDELITY_TARGET,
        gap=gap,
        trend=trend,
        confidence=confidence
    )

    # Emit receipt
    emit_receipt("leading_indicator", {
        "tenant_id": "axiom-autonomy",
        "indicator_type": LeadingIndicator.SIM_FIDELITY.value,
        "current_value": fidelity,
        "target_value": SIM_FIDELITY_TARGET,
        "gap": gap,
        "trend": trend,
        "confidence": confidence,
        "data_points": n,
    })

    return measurement


def measure_fleet_learning_rate(
    calibration_output: CalibrationOutput
) -> IndicatorMeasurement:
    """Extract alpha estimate as learning rate indicator.

    Uses calibration module output to measure fleet learning rate.
    High alpha with high confidence = validated compounding model.

    Args:
        calibration_output: Output from calibration.estimate_alpha()

    Returns:
        IndicatorMeasurement for fleet learning rate
    """
    current_alpha = calibration_output.alpha_estimate
    current_confidence = calibration_output.confidence_level

    # Combined target: alpha >= 1.8 AND confidence >= 0.80
    # Score as weighted combination
    alpha_score = min(1.0, current_alpha / FLEET_LEARNING_ALPHA_TARGET)
    conf_score = min(1.0, current_confidence / FLEET_LEARNING_CONFIDENCE_TARGET)

    combined_score = (alpha_score + conf_score) / 2

    # Gap from combined target (1.0 = both targets met)
    gap = combined_score - 1.0

    # Trend from confidence (proxy for data accumulation)
    if current_confidence >= FLEET_LEARNING_CONFIDENCE_TARGET:
        trend = "stable" if gap >= 0 else "improving"
    elif current_confidence >= 0.5:
        trend = "improving"
    else:
        trend = "degrading"

    measurement = IndicatorMeasurement(
        indicator_type=LeadingIndicator.FLEET_LEARNING_RATE,
        current_value=current_alpha,
        target_value=FLEET_LEARNING_ALPHA_TARGET,
        gap=current_alpha - FLEET_LEARNING_ALPHA_TARGET,
        trend=trend,
        confidence=current_confidence
    )

    # Emit receipt
    emit_receipt("leading_indicator", {
        "tenant_id": "axiom-autonomy",
        "indicator_type": LeadingIndicator.FLEET_LEARNING_RATE.value,
        "current_value": current_alpha,
        "target_value": FLEET_LEARNING_ALPHA_TARGET,
        "gap": current_alpha - FLEET_LEARNING_ALPHA_TARGET,
        "trend": trend,
        "confidence": current_confidence,
        "alpha_confidence_interval": calibration_output.confidence_interval,
        "dominant_signal": calibration_output.dominant_signal,
    })

    return measurement


def measure_tau_velocity(
    tau_history: List[float],
    time_history: List[float] = None
) -> IndicatorMeasurement:
    """Compute d(tau)/dt from historical tau measurements.

    Tau velocity measures the rate of improvement in decision latency.
    Negative velocity = improving (tau decreasing over time).

    Args:
        tau_history: List of tau values over time (seconds)
        time_history: Optional list of timestamps (default: sequential)

    Returns:
        IndicatorMeasurement for tau velocity
    """
    if not tau_history or len(tau_history) < 2:
        return IndicatorMeasurement(
            indicator_type=LeadingIndicator.TAU_VELOCITY,
            current_value=0.0,
            target_value=TAU_VELOCITY_TARGET,
            gap=abs(TAU_VELOCITY_TARGET),
            trend="unknown",
            confidence=0.0
        )

    n = len(tau_history)

    if time_history is None:
        time_history = list(range(n))

    # Linear regression to find slope (velocity)
    mean_t = sum(time_history) / n
    mean_tau = sum(tau_history) / n

    numerator = sum((time_history[i] - mean_t) * (tau_history[i] - mean_tau) for i in range(n))
    denominator = sum((time_history[i] - mean_t) ** 2 for i in range(n))

    if denominator == 0:
        velocity = 0.0
    else:
        velocity = numerator / denominator

    # Normalize to percentage change per cycle
    if mean_tau > 0:
        velocity_pct = velocity / mean_tau
    else:
        velocity_pct = 0.0

    # Gap from target (negative velocity = improving)
    gap = velocity_pct - TAU_VELOCITY_TARGET

    # Trend classification based on improvement rate
    # "stable" means tau isn't changing, not that we're meeting target
    if velocity_pct < -0.20:
        trend = "rapid_improvement"  # >20% improvement per cycle
    elif velocity_pct < -0.10:
        trend = "good_improvement"  # 10-20% improvement per cycle
    elif velocity_pct < -0.01:
        trend = "improving"  # Any meaningful improvement
    elif velocity_pct > 0.01:
        trend = "degrading"  # Tau increasing (bad)
    else:
        trend = "stable"  # Tau not changing meaningfully

    # Confidence from data quantity and fit quality
    ss_tot = sum((tau_history[i] - mean_tau) ** 2 for i in range(n))
    predicted = [mean_tau + velocity * (time_history[i] - mean_t) for i in range(n)]
    ss_res = sum((tau_history[i] - predicted[i]) ** 2 for i in range(n))

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    confidence = min(1.0, max(0.0, r_squared * n / 10.0))

    measurement = IndicatorMeasurement(
        indicator_type=LeadingIndicator.TAU_VELOCITY,
        current_value=velocity_pct,
        target_value=TAU_VELOCITY_TARGET,
        gap=gap,
        trend=trend,
        confidence=confidence
    )

    # Emit receipt
    emit_receipt("leading_indicator", {
        "tenant_id": "axiom-autonomy",
        "indicator_type": LeadingIndicator.TAU_VELOCITY.value,
        "current_value": velocity_pct,
        "target_value": TAU_VELOCITY_TARGET,
        "gap": gap,
        "trend": trend,
        "confidence": confidence,
        "tau_start": tau_history[0],
        "tau_end": tau_history[-1],
        "observations": n,
    })

    return measurement


def assess_all_indicators(
    sim_predictions: List[float] = None,
    actual_telemetry: List[float] = None,
    calibration_output: CalibrationOutput = None,
    tau_history: List[float] = None
) -> List[IndicatorMeasurement]:
    """Run all three indicators and emit receipts.

    Convenience function to assess all leading indicators at once.

    Args:
        sim_predictions: Simulation predictions for fidelity check
        actual_telemetry: Actual telemetry for fidelity check
        calibration_output: Calibration output for learning rate
        tau_history: Historical tau values for velocity

    Returns:
        List of IndicatorMeasurement for all indicators
    """
    measurements = []

    # Sim fidelity
    if sim_predictions and actual_telemetry:
        measurements.append(measure_sim_fidelity(sim_predictions, actual_telemetry))
    else:
        # Default empty measurement
        measurements.append(IndicatorMeasurement(
            indicator_type=LeadingIndicator.SIM_FIDELITY,
            current_value=0.0,
            target_value=SIM_FIDELITY_TARGET,
            gap=-SIM_FIDELITY_TARGET,
            trend="unknown",
            confidence=0.0
        ))

    # Fleet learning rate
    if calibration_output:
        measurements.append(measure_fleet_learning_rate(calibration_output))
    else:
        measurements.append(IndicatorMeasurement(
            indicator_type=LeadingIndicator.FLEET_LEARNING_RATE,
            current_value=ALPHA_BASELINE,
            target_value=FLEET_LEARNING_ALPHA_TARGET,
            gap=0.0,
            trend="unknown",
            confidence=0.0
        ))

    # Tau velocity
    if tau_history and len(tau_history) >= 2:
        measurements.append(measure_tau_velocity(tau_history))
    else:
        measurements.append(IndicatorMeasurement(
            indicator_type=LeadingIndicator.TAU_VELOCITY,
            current_value=0.0,
            target_value=TAU_VELOCITY_TARGET,
            gap=abs(TAU_VELOCITY_TARGET),
            trend="unknown",
            confidence=0.0
        ))

    return measurements


def indicators_to_confidence(measurements: List[IndicatorMeasurement]) -> float:
    """Aggregate indicator quality into overall model confidence.

    Combines all indicators into single confidence score for the
    autonomy compounding model.

    Args:
        measurements: List of IndicatorMeasurement

    Returns:
        Overall confidence (0-1)
    """
    if not measurements:
        return 0.0

    # Weight by indicator importance
    weights = {
        LeadingIndicator.FLEET_LEARNING_RATE: 0.5,  # Most important
        LeadingIndicator.TAU_VELOCITY: 0.3,
        LeadingIndicator.SIM_FIDELITY: 0.2,
    }

    total_weight = 0.0
    weighted_confidence = 0.0

    for m in measurements:
        w = weights.get(m.indicator_type, 0.33)

        # Score based on gap and confidence
        gap_score = max(0, 1.0 + m.gap) if m.gap < 0 else 1.0
        combined = m.confidence * gap_score

        weighted_confidence += w * combined
        total_weight += w

    if total_weight == 0:
        return 0.0

    return weighted_confidence / total_weight


def get_indicator_status(measurements: List[IndicatorMeasurement]) -> dict:
    """Get status summary for all indicators.

    Args:
        measurements: List of IndicatorMeasurement

    Returns:
        Dict with status summary for each indicator
    """
    status = {}

    for m in measurements:
        indicator_name = m.indicator_type.value

        if m.gap >= 0:
            status_str = "ON_TARGET"
        elif m.trend == "improving":
            status_str = "IMPROVING"
        elif m.trend == "degrading":
            status_str = "AT_RISK"
        else:
            status_str = "UNKNOWN"

        status[indicator_name] = {
            "status": status_str,
            "current": m.current_value,
            "target": m.target_value,
            "gap": m.gap,
            "trend": m.trend,
            "confidence": m.confidence,
        }

    # Overall status
    overall_confidence = indicators_to_confidence(measurements)
    if overall_confidence >= 0.8:
        overall_status = "GREEN"
    elif overall_confidence >= 0.5:
        overall_status = "YELLOW"
    else:
        overall_status = "RED"

    status["overall"] = {
        "status": overall_status,
        "confidence": overall_confidence,
    }

    return status


def format_indicator_report(measurements: List[IndicatorMeasurement]) -> str:
    """Format indicators as human-readable report.

    Args:
        measurements: List of IndicatorMeasurement

    Returns:
        Formatted report string
    """
    status = get_indicator_status(measurements)
    overall = status.pop("overall")

    lines = [
        "=" * 60,
        f"LEADING INDICATORS - Status: {overall['status']}",
        f"Overall Confidence: {overall['confidence']:.0%}",
        "=" * 60,
        "",
    ]

    for name, info in status.items():
        lines.append(f"{name.upper()}")
        lines.append(f"  Status: {info['status']}")
        lines.append(f"  Current: {info['current']:.4f}")
        lines.append(f"  Target:  {info['target']:.4f}")
        lines.append(f"  Gap:     {info['gap']:+.4f}")
        lines.append(f"  Trend:   {info['trend']}")
        lines.append(f"  Confidence: {info['confidence']:.0%}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def emit_leading_indicator_receipt(measurement: IndicatorMeasurement) -> dict:
    """Emit detailed leading indicator receipt per CLAUDEME.

    Args:
        measurement: IndicatorMeasurement to emit

    Returns:
        Receipt dict
    """
    return emit_receipt("leading_indicator", {
        "tenant_id": "axiom-autonomy",
        "indicator_type": measurement.indicator_type.value,
        "current_value": measurement.current_value,
        "target_value": measurement.target_value,
        "gap": measurement.gap,
        "trend": measurement.trend,
        "confidence": measurement.confidence,
    })
