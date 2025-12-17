"""test_calibration.py - Tests for alpha estimation from fleet proxies

Validates alpha calibration from FSD/Optimus/Starship data.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import (
    estimate_alpha,
    estimate_alpha_from_lists,
    fsd_to_alpha_proxy,
    optimus_to_alpha_proxy,
    starship_to_alpha_proxy,
    combine_proxies,
    validate_alpha_range,
    compute_data_quality,
    CalibrationConfig,
    CalibrationInput,
    CalibrationOutput,
    ALPHA_MIN_PLAUSIBLE,
    ALPHA_MAX_PLAUSIBLE,
    ALPHA_BASELINE,
    MIN_DATA_POINTS,
    CONFIDENCE_THRESHOLD,
)


class TestFsdToAlphaProxy:
    """Tests for fsd_to_alpha_proxy function."""

    def test_accelerating_rates_high_alpha(self):
        """Accelerating improvement rates should give high alpha."""
        rates = [5.0, 8.0, 12.0, 18.0, 27.0, 40.0]  # ~1.5x each
        alpha, conf = fsd_to_alpha_proxy(rates)
        assert alpha > 1.5, f"Accelerating rates should give alpha > 1.5, got {alpha}"

    def test_constant_rates_low_alpha(self):
        """Constant improvement rates should give alpha ~1.0."""
        rates = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        alpha, conf = fsd_to_alpha_proxy(rates)
        assert 0.8 <= alpha <= 1.5, f"Constant rates should give alpha ~1.0, got {alpha}"

    def test_insufficient_data(self):
        """Too few data points should return baseline with low confidence."""
        rates = [5.0, 8.0]  # Only 2 points
        alpha, conf = fsd_to_alpha_proxy(rates)
        assert conf < 0.5, "Low data should give low confidence"

    def test_alpha_in_plausible_range(self):
        """Alpha should always be in plausible range."""
        rates = [1.0, 10.0, 100.0, 1000.0]  # Extreme acceleration
        alpha, conf = fsd_to_alpha_proxy(rates)
        assert ALPHA_MIN_PLAUSIBLE <= alpha <= ALPHA_MAX_PLAUSIBLE


class TestOptimusToAlphaProxy:
    """Tests for optimus_to_alpha_proxy function."""

    def test_capability_growth(self):
        """Growing capabilities should indicate positive alpha."""
        caps = [10, 18, 30, 50, 80]
        alpha, conf = optimus_to_alpha_proxy(caps)
        assert alpha > 1.0, "Growing capabilities should give alpha > 1.0"


class TestStarshipToAlphaProxy:
    """Tests for starship_to_alpha_proxy function."""

    def test_decreasing_resolution_times(self):
        """Decreasing times should indicate improving learning."""
        times = [30, 20, 14, 10, 7, 5]  # Improving
        alpha, conf = starship_to_alpha_proxy(times)
        assert alpha > 1.0, "Improving resolution times should give alpha > 1.0"


class TestCombineProxies:
    """Tests for combine_proxies function."""

    def test_weighted_average(self):
        """Should combine proxies with confidence weighting."""
        proxies = [
            (1.8, 0.9),  # High confidence
            (2.0, 0.5),  # Low confidence
            (1.6, 0.7),  # Medium confidence
        ]
        alpha, ci, conf = combine_proxies(proxies)
        # Should be closer to 1.8 (highest confidence)
        assert 1.6 <= alpha <= 2.0

    def test_confidence_interval(self):
        """Should produce valid confidence interval."""
        proxies = [(1.8, 0.9), (1.9, 0.8)]
        alpha, (ci_low, ci_high), conf = combine_proxies(proxies)
        assert ci_low < alpha < ci_high

    def test_empty_proxies(self):
        """Empty proxies should return baseline."""
        alpha, ci, conf = combine_proxies([])
        assert alpha == ALPHA_BASELINE
        assert conf == 0.0


class TestValidateAlphaRange:
    """Tests for validate_alpha_range function."""

    def test_valid_alpha(self):
        """Alpha in [1.0, 3.0] should be valid."""
        assert validate_alpha_range(1.8) is True
        assert validate_alpha_range(1.0) is True
        assert validate_alpha_range(3.0) is True

    def test_invalid_alpha(self):
        """Alpha outside [1.0, 3.0] should be invalid."""
        assert validate_alpha_range(0.5) is False
        assert validate_alpha_range(3.5) is False


class TestComputeDataQuality:
    """Tests for compute_data_quality function."""

    def test_high_quality_data(self):
        """Good data should give high quality score."""
        inputs = CalibrationInput(
            fsd_improvement_rate=10.0,
            optimus_capability_growth=5.0,
            starship_anomaly_resolution_time=20.0,
            observation_count=30,
            observation_window_months=18
        )
        quality = compute_data_quality(inputs)
        assert quality >= 0.7, f"Good data should give quality >= 0.7, got {quality}"

    def test_low_quality_data(self):
        """Poor data should give low quality score."""
        inputs = CalibrationInput(
            fsd_improvement_rate=-10.0,  # Invalid
            optimus_capability_growth=-5.0,  # Invalid
            starship_anomaly_resolution_time=-20.0,  # Invalid
            observation_count=3,  # Too few
            observation_window_months=2  # Too short
        )
        quality = compute_data_quality(inputs)
        assert quality < 0.5, "Poor data should give low quality score"


class TestEstimateAlpha:
    """Tests for estimate_alpha main function."""

    def test_sufficient_data(self):
        """With sufficient data, should produce valid estimate."""
        inputs = CalibrationInput(
            fsd_improvement_rate=15.0,
            optimus_capability_growth=8.0,
            starship_anomaly_resolution_time=25.0,
            observation_count=20,
            observation_window_months=12
        )
        output = estimate_alpha(inputs)
        assert validate_alpha_range(output.alpha_estimate)
        assert output.confidence_level > 0

    def test_insufficient_data(self):
        """With insufficient data, should return baseline with zero confidence."""
        inputs = CalibrationInput(
            fsd_improvement_rate=15.0,
            optimus_capability_growth=8.0,
            starship_anomaly_resolution_time=25.0,
            observation_count=3,  # Below MIN_DATA_POINTS
            observation_window_months=12
        )
        output = estimate_alpha(inputs)
        assert output.alpha_estimate == ALPHA_BASELINE
        assert output.confidence_level == 0.0
        assert output.dominant_signal == "insufficient_data"


class TestEstimateAlphaFromLists:
    """Tests for estimate_alpha_from_lists function."""

    def test_with_explicit_data(self):
        """Should estimate from explicit data lists."""
        fsd_rates = [5, 8, 12, 18, 27]
        optimus_caps = [10, 18, 30, 50, 80]
        starship_times = [30, 20, 14, 10, 7]

        output = estimate_alpha_from_lists(fsd_rates, optimus_caps, starship_times)
        assert validate_alpha_range(output.alpha_estimate)
        assert output.confidence_level > 0

    def test_dominant_signal_identified(self):
        """Should identify which proxy contributed most."""
        output = estimate_alpha_from_lists(
            [5, 8, 12, 18, 27],
            [10, 18, 30, 50, 80],
            [30, 20, 14, 10, 7]
        )
        assert output.dominant_signal in ["fsd", "optimus", "starship"]


class TestCalibrationOutput:
    """Tests for CalibrationOutput dataclass."""

    def test_confidence_interval_bounds(self):
        """Confidence interval should be within plausible range."""
        inputs = CalibrationInput(
            fsd_improvement_rate=15.0,
            optimus_capability_growth=8.0,
            starship_anomaly_resolution_time=25.0,
            observation_count=20,
            observation_window_months=12
        )
        output = estimate_alpha(inputs)
        ci_low, ci_high = output.confidence_interval
        assert ci_low >= ALPHA_MIN_PLAUSIBLE
        assert ci_high <= ALPHA_MAX_PLAUSIBLE
        assert ci_low <= output.alpha_estimate <= ci_high


class TestConstants:
    """Tests for calibration constants."""

    def test_alpha_range(self):
        """Plausible alpha range should be reasonable."""
        assert ALPHA_MIN_PLAUSIBLE == 1.0
        assert ALPHA_MAX_PLAUSIBLE == 3.0
        assert ALPHA_MIN_PLAUSIBLE < ALPHA_BASELINE < ALPHA_MAX_PLAUSIBLE

    def test_thresholds(self):
        """Thresholds should be reasonable."""
        assert MIN_DATA_POINTS == 6
        assert CONFIDENCE_THRESHOLD == 0.70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
