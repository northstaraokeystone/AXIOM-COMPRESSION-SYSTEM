"""test_leading_indicators.py - Tests for observable proxy monitoring

Validates leading indicators:
    SIM_FIDELITY: Target >= 95%
    FLEET_LEARNING_RATE: alpha >= 1.8, confidence >= 80%
    TAU_VELOCITY: d(tau)/dt < -5%
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.leading_indicators import (
    measure_sim_fidelity,
    measure_fleet_learning_rate,
    measure_tau_velocity,
    assess_all_indicators,
    indicators_to_confidence,
    get_indicator_status,
    format_indicator_report,
    LeadingIndicator,
    IndicatorMeasurement,
    SIM_FIDELITY_TARGET,
    FLEET_LEARNING_ALPHA_TARGET,
    TAU_VELOCITY_TARGET,
)
from src.calibration import CalibrationOutput


class TestMeasureSimFidelity:
    """Tests for measure_sim_fidelity function."""

    def test_high_correlation(self):
        """Highly correlated predictions should give high fidelity."""
        sim = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        actual = [1.1, 2.1, 2.9, 4.2, 4.8, 6.1, 7.0, 8.2, 8.9, 10.1]
        measurement = measure_sim_fidelity(sim, actual)
        assert measurement.current_value > 0.9, (
            f"Expected high fidelity, got {measurement.current_value}"
        )

    def test_low_correlation(self):
        """Uncorrelated data should give low fidelity."""
        sim = [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = [5.0, 3.0, 1.0, 4.0, 2.0]  # Random order
        measurement = measure_sim_fidelity(sim, actual)
        assert measurement.current_value <= 0.6, (
            f"Expected low fidelity, got {measurement.current_value}"
        )

    def test_empty_data(self):
        """Empty data should give zero fidelity."""
        measurement = measure_sim_fidelity([], [])
        assert measurement.current_value == 0.0
        assert measurement.confidence == 0.0

    def test_gap_calculation(self):
        """Gap should be current - target."""
        sim = [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]  # Perfect correlation
        measurement = measure_sim_fidelity(sim, actual)
        expected_gap = measurement.current_value - SIM_FIDELITY_TARGET
        assert abs(measurement.gap - expected_gap) < 0.001


class TestMeasureFleetLearningRate:
    """Tests for measure_fleet_learning_rate function."""

    def test_high_alpha_high_confidence(self):
        """High alpha with high confidence should meet target."""
        calibration = CalibrationOutput(
            alpha_estimate=2.0,
            confidence_interval=(1.8, 2.2),
            confidence_level=0.90,
            dominant_signal="fsd",
            data_quality_score=0.85,
        )
        measurement = measure_fleet_learning_rate(calibration)
        assert measurement.current_value >= FLEET_LEARNING_ALPHA_TARGET
        assert measurement.confidence >= 0.80

    def test_low_alpha(self):
        """Low alpha should show negative gap."""
        calibration = CalibrationOutput(
            alpha_estimate=1.5,
            confidence_interval=(1.3, 1.7),
            confidence_level=0.80,
            dominant_signal="fsd",
            data_quality_score=0.85,
        )
        measurement = measure_fleet_learning_rate(calibration)
        assert measurement.gap < 0, "Low alpha should have negative gap"


class TestMeasureTauVelocity:
    """Tests for measure_tau_velocity function."""

    def test_improving_tau(self):
        """Decreasing tau should give negative velocity."""
        tau_history = [300, 250, 200, 150, 100]  # Improving
        measurement = measure_tau_velocity(tau_history)
        assert measurement.current_value < 0, (
            f"Improving tau should have negative velocity, got {measurement.current_value}"
        )

    def test_stable_tau(self):
        """Stable tau should give near-zero velocity."""
        tau_history = [200, 200, 200, 200, 200]
        measurement = measure_tau_velocity(tau_history)
        assert abs(measurement.current_value) < 0.01

    def test_worsening_tau(self):
        """Increasing tau should give positive velocity."""
        tau_history = [100, 150, 200, 250, 300]  # Getting worse
        measurement = measure_tau_velocity(tau_history)
        assert measurement.current_value > 0

    def test_insufficient_data(self):
        """Single point should return zero velocity."""
        measurement = measure_tau_velocity([200])
        assert measurement.current_value == 0.0
        assert measurement.confidence == 0.0

    def test_trend_classification(self):
        """Should classify trend correctly."""
        # Fast improvement
        measurement = measure_tau_velocity([300, 200, 100])
        assert measurement.trend in [
            "rapid_improvement",
            "good_improvement",
            "improving",
        ]


class TestAssessAllIndicators:
    """Tests for assess_all_indicators function."""

    def test_returns_three_measurements(self):
        """Should return measurements for all three indicators."""
        measurements = assess_all_indicators()
        assert len(measurements) == 3

    def test_with_data(self):
        """Should process provided data."""
        calibration = CalibrationOutput(
            alpha_estimate=1.9,
            confidence_interval=(1.7, 2.1),
            confidence_level=0.85,
            dominant_signal="fsd",
            data_quality_score=0.8,
        )
        measurements = assess_all_indicators(
            sim_predictions=[1, 2, 3, 4, 5],
            actual_telemetry=[1.1, 2.0, 3.1, 3.9, 5.1],
            calibration_output=calibration,
            tau_history=[300, 250, 200, 150],
        )
        assert len(measurements) == 3
        # All should have non-zero values
        assert all(m.current_value != 0 or m.confidence == 0 for m in measurements)


class TestIndicatorsToConfidence:
    """Tests for indicators_to_confidence function."""

    def test_high_quality_indicators(self):
        """Good indicators should give high confidence."""
        measurements = [
            IndicatorMeasurement(
                indicator_type=LeadingIndicator.SIM_FIDELITY,
                current_value=0.98,
                target_value=0.95,
                gap=0.03,
                trend="stable",
                confidence=0.9,
            ),
            IndicatorMeasurement(
                indicator_type=LeadingIndicator.FLEET_LEARNING_RATE,
                current_value=2.0,
                target_value=1.8,
                gap=0.2,
                trend="improving",
                confidence=0.85,
            ),
            IndicatorMeasurement(
                indicator_type=LeadingIndicator.TAU_VELOCITY,
                current_value=-0.08,
                target_value=-0.05,
                gap=-0.03,
                trend="good_improvement",
                confidence=0.8,
            ),
        ]
        confidence = indicators_to_confidence(measurements)
        assert confidence > 0.7, (
            f"Good indicators should give high confidence, got {confidence}"
        )

    def test_poor_indicators(self):
        """Poor indicators should give low confidence."""
        measurements = [
            IndicatorMeasurement(
                indicator_type=LeadingIndicator.SIM_FIDELITY,
                current_value=0.5,
                target_value=0.95,
                gap=-0.45,
                trend="degrading",
                confidence=0.3,
            ),
        ]
        confidence = indicators_to_confidence(measurements)
        assert confidence < 0.5

    def test_empty_measurements(self):
        """Empty measurements should give zero confidence."""
        confidence = indicators_to_confidence([])
        assert confidence == 0.0


class TestGetIndicatorStatus:
    """Tests for get_indicator_status function."""

    def test_status_structure(self):
        """Should return status for each indicator."""
        measurements = assess_all_indicators()
        status = get_indicator_status(measurements)

        assert "sim_fidelity" in status
        assert "fleet_learning_rate" in status
        assert "tau_velocity" in status
        assert "overall" in status

    def test_overall_status(self):
        """Should have overall status classification."""
        measurements = assess_all_indicators()
        status = get_indicator_status(measurements)

        assert status["overall"]["status"] in ["GREEN", "YELLOW", "RED"]
        assert "confidence" in status["overall"]


class TestFormatIndicatorReport:
    """Tests for format_indicator_report function."""

    def test_report_structure(self):
        """Should produce readable report."""
        measurements = assess_all_indicators()
        report = format_indicator_report(measurements)

        assert "LEADING INDICATORS" in report
        assert "SIM_FIDELITY" in report
        assert "FLEET_LEARNING_RATE" in report
        assert "TAU_VELOCITY" in report


class TestIndicatorMeasurement:
    """Tests for IndicatorMeasurement dataclass."""

    def test_measurement_fields(self):
        """Should have all required fields."""
        measurement = IndicatorMeasurement(
            indicator_type=LeadingIndicator.SIM_FIDELITY,
            current_value=0.92,
            target_value=0.95,
            gap=-0.03,
            trend="improving",
            confidence=0.8,
        )
        assert measurement.indicator_type == LeadingIndicator.SIM_FIDELITY
        assert measurement.current_value == 0.92
        assert measurement.target_value == 0.95
        assert measurement.gap == -0.03
        assert measurement.trend == "improving"
        assert measurement.confidence == 0.8


class TestConstants:
    """Tests for leading indicator constants."""

    def test_sim_fidelity_target(self):
        """Sim fidelity target should be 95%."""
        assert SIM_FIDELITY_TARGET == 0.95

    def test_fleet_learning_target(self):
        """Fleet learning target should be 1.8."""
        assert FLEET_LEARNING_ALPHA_TARGET == 1.8

    def test_tau_velocity_target(self):
        """Tau velocity target should be -5%."""
        assert TAU_VELOCITY_TARGET == -0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
