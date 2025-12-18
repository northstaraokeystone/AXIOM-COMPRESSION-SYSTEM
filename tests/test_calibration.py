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
    CalibrationInput,
    ALPHA_MIN_PLAUSIBLE,
    ALPHA_MAX_PLAUSIBLE,
    ALPHA_BASELINE,
    MIN_DATA_POINTS,
    CONFIDENCE_THRESHOLD,
    # Empirical FSD functions and constants
    load_fsd_empirical,
    fit_alpha_empirical,
    compute_gain_factor,
    compute_safety_ratio,
    MPI_V13,
    MPI_V14,
    GAIN_FACTOR_EMPIRICAL,
    SAFETY_AP_MPCM,
    SAFETY_HUMAN_MPCM,
    ALPHA_EMPIRICAL_LOW,
    ALPHA_EMPIRICAL_HIGH,
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


class TestComputeGainFactor:
    """Tests for compute_gain_factor function."""

    def test_gain_factor_v13_v14(self):
        """Gain factor from v13 to v14 should be ~20.86."""
        gain = compute_gain_factor(MPI_V13, MPI_V14)
        assert abs(gain - 20.86) < 0.01, f"Expected ~20.86, got {gain}"

    def test_gain_factor_basic(self):
        """Basic gain factor computation."""
        assert compute_gain_factor(100, 200) == 2.0
        assert compute_gain_factor(50, 150) == 3.0

    def test_gain_factor_zero_before_raises(self):
        """Zero mpi_before should raise ValueError."""
        with pytest.raises(ValueError):
            compute_gain_factor(0, 100)


class TestComputeSafetyRatio:
    """Tests for compute_safety_ratio function."""

    def test_safety_ratio_computed(self):
        """Safety ratio should be ~9.09."""
        ratio = compute_safety_ratio(SAFETY_AP_MPCM, SAFETY_HUMAN_MPCM)
        assert abs(ratio - 9.09) < 0.01, f"Expected ~9.09, got {ratio}"

    def test_safety_ratio_zero_human_raises(self):
        """Zero human_mpcm should raise ValueError."""
        with pytest.raises(ValueError):
            compute_safety_ratio(100, 0)


class TestLoadFsdEmpirical:
    """Tests for load_fsd_empirical function."""

    def test_load_fsd_empirical_hash_valid(self, capsys):
        """Payload hash should match computed hash."""
        try:
            data = load_fsd_empirical()
            assert 'payload_hash' in data
            # If we get here without StopRule, hash was valid
        except Exception as e:
            # Print debug info if test fails
            import json
            import os
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(repo_root, "data/verified/fsd_empirical.json")
            with open(path, 'r') as f:
                raw_data = json.load(f)
            print(f"DEBUG: Exception={e}")
            print(f"DEBUG: File path={path}")
            print(f"DEBUG: File exists={os.path.exists(path)}")
            print(f"DEBUG: Raw keys={list(raw_data.keys())}")
            raise

    def test_load_fsd_empirical_emits_receipt(self, capsys):
        """Should emit fsd_empirical_ingest receipt."""
        load_fsd_empirical()
        captured = capsys.readouterr()
        assert 'fsd_empirical_ingest' in captured.out

    def test_load_fsd_empirical_data_structure(self):
        """Should return correct data structure."""
        data = load_fsd_empirical()
        assert data['versions'] == ['v12', 'v12.3', 'v13', 'v14']
        assert data['mpi_values'][2] == 441
        assert data['mpi_values'][3] == 9200
        assert data['safety_ap_mpcm'] == 6360000
        assert data['safety_human_mpcm'] == 700000


class TestFitAlphaEmpirical:
    """Tests for fit_alpha_empirical function."""

    def test_fit_alpha_empirical_in_range(self, capsys):
        """Alpha estimate should be in validated range [1.5, 2.1]."""
        data = load_fsd_empirical()
        result = fit_alpha_empirical(data)
        assert ALPHA_EMPIRICAL_LOW <= result['alpha_estimate'] <= ALPHA_EMPIRICAL_HIGH, \
            f"Alpha {result['alpha_estimate']} not in range [{ALPHA_EMPIRICAL_LOW}, {ALPHA_EMPIRICAL_HIGH}]"

    def test_fit_alpha_empirical_gain_factor(self, capsys):
        """Gain factor should be ~20.86."""
        data = load_fsd_empirical()
        result = fit_alpha_empirical(data)
        assert abs(result['gain_factor'] - 20.86) < 0.01, \
            f"Expected gain_factor ~20.86, got {result['gain_factor']}"

    def test_fit_alpha_empirical_emits_receipt(self, capsys):
        """Should emit alpha_calibration receipt with method=empirical."""
        data = load_fsd_empirical()
        fit_alpha_empirical(data)
        captured = capsys.readouterr()
        assert 'alpha_calibration' in captured.out
        assert '"method": "empirical"' in captured.out

    def test_fit_alpha_empirical_method_field(self, capsys):
        """Result should have method='empirical'."""
        data = load_fsd_empirical()
        result = fit_alpha_empirical(data)
        assert result['method'] == 'empirical'

    def test_fit_alpha_empirical_range_bounds(self, capsys):
        """Result should include range bounds."""
        data = load_fsd_empirical()
        result = fit_alpha_empirical(data)
        assert result['range_low'] == ALPHA_EMPIRICAL_LOW
        assert result['range_high'] == ALPHA_EMPIRICAL_HIGH


class TestEmpiricalConstants:
    """Tests for empirical FSD constants."""

    def test_mpi_constants(self):
        """MPI constants should match spec."""
        assert MPI_V13 == 441
        assert MPI_V14 == 9200

    def test_gain_factor_constant(self):
        """Gain factor constant should be computed correctly."""
        computed = MPI_V14 / MPI_V13
        assert abs(GAIN_FACTOR_EMPIRICAL - computed) < 0.01

    def test_safety_constants(self):
        """Safety constants should match spec."""
        assert SAFETY_AP_MPCM == 6_360_000
        assert SAFETY_HUMAN_MPCM == 700_000

    def test_alpha_empirical_range(self):
        """Alpha empirical range should be [1.6, 1.8]."""
        assert ALPHA_EMPIRICAL_LOW == 1.6
        assert ALPHA_EMPIRICAL_HIGH == 1.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
