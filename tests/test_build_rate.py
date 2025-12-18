"""test_build_rate.py - Tests for multiplicative build rate model

Validates Grok's core equation: B = c x A^alpha x P
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build_rate import (
    compute_build_rate,
    autonomy_to_level,
    propulsion_to_level,
    annual_multiplier,
    allocation_to_multiplier,
    validate_grok_multipliers,
    compute_build_rate_state,
    MULTIPLIER_40PCT,
)


class TestComputeBuildRate:
    """Tests for compute_build_rate function."""

    def test_basic_computation(self):
        """B = c x A^alpha x P should compute correctly."""
        # At A=0.40, P=1.0, alpha=1.8, c=1.0
        # B = 1.0 x 0.40^1.8 x 1.0 = 0.192...
        result = compute_build_rate(0.40, 1.0, 1.8, 1.0)
        assert 0.18 < result < 0.22, f"Expected ~0.192, got {result}"

    def test_zero_autonomy_is_zero(self):
        """Zero autonomy = zero build rate (existential stall)."""
        result = compute_build_rate(0.0, 1.0, 1.8, 1.0)
        assert result == 0.0, "Zero autonomy should yield zero build rate"

    def test_autonomy_bounds(self):
        """Autonomy must be in [0, 1]."""
        with pytest.raises(ValueError):
            compute_build_rate(-0.1, 1.0)
        with pytest.raises(ValueError):
            compute_build_rate(1.1, 1.0)

    def test_propulsion_positive(self):
        """Propulsion must be non-negative."""
        with pytest.raises(ValueError):
            compute_build_rate(0.5, -1.0)

    def test_multiplicative_relationship(self):
        """40% vs 25% allocation should show ~2.2x ratio."""
        b_40 = compute_build_rate(0.40, 1.0, 1.8, 1.0)
        b_25 = compute_build_rate(0.25, 1.0, 1.8, 1.0)
        ratio = b_40 / b_25
        # Grok table shows 2.5-3.0x vs 1.6-2.0x, ratio ~1.5-1.9x
        assert 1.5 < ratio < 2.5, f"Expected ratio ~2.0, got {ratio}"


class TestAutonomyToLevel:
    """Tests for autonomy_to_level normalization."""

    def test_baseline_values(self):
        """Baseline tau, expertise, capacity should give reasonable level."""
        level = autonomy_to_level(tau=300.0, expertise=0.8, decision_capacity=1000.0)
        assert 0 < level <= 1.0, f"Level should be in (0, 1], got {level}"

    def test_lower_tau_higher_level(self):
        """Lower tau should give higher autonomy level."""
        level_high_tau = autonomy_to_level(300.0, 0.8, 1000.0)
        level_low_tau = autonomy_to_level(150.0, 0.8, 1000.0)
        assert level_low_tau > level_high_tau, "Lower tau should yield higher level"

    def test_tau_must_be_positive(self):
        """Tau must be positive."""
        with pytest.raises(ValueError):
            autonomy_to_level(0.0, 0.8, 1000.0)
        with pytest.raises(ValueError):
            autonomy_to_level(-10.0, 0.8, 1000.0)


class TestPropulsionToLevel:
    """Tests for propulsion_to_level normalization."""

    def test_baseline_propulsion(self):
        """Baseline values should give level ~1.0."""
        level = propulsion_to_level(
            launches_per_year=10.0, payload_tons=100.0, reliability=0.95
        )
        assert 0.9 < level < 1.1, f"Baseline should give ~1.0, got {level}"

    def test_higher_launches_higher_level(self):
        """More launches = higher level."""
        level_low = propulsion_to_level(5.0, 100.0, 0.95)
        level_high = propulsion_to_level(20.0, 100.0, 0.95)
        assert level_high > level_low

    def test_reliability_bounds(self):
        """Reliability must be in [0, 1]."""
        with pytest.raises(ValueError):
            propulsion_to_level(10.0, 100.0, 1.5)
        with pytest.raises(ValueError):
            propulsion_to_level(10.0, 100.0, -0.1)


class TestAnnualMultiplier:
    """Tests for annual_multiplier calculation."""

    def test_positive_growth(self):
        """Growing build rate should give multiplier > 1."""
        mult = annual_multiplier(2.0, 1.0)
        assert mult == 2.0, f"Expected 2.0, got {mult}"

    def test_zero_prior_rate(self):
        """Zero prior rate with positive current should give inf."""
        mult = annual_multiplier(1.0, 0.0)
        assert mult == float("inf")

    def test_both_zero(self):
        """Both zero should give 1.0."""
        mult = annual_multiplier(0.0, 0.0)
        assert mult == 1.0


class TestAllocationToMultiplier:
    """Tests for allocation_to_multiplier mapping."""

    def test_40pct_multiplier(self):
        """40% allocation should give 2.5-3.0x multiplier."""
        mult = allocation_to_multiplier(0.40)
        assert MULTIPLIER_40PCT[0] <= mult <= MULTIPLIER_40PCT[1], (
            f"Expected {MULTIPLIER_40PCT}, got {mult}"
        )

    def test_25pct_multiplier(self):
        """25% allocation should give 1.6-2.0x multiplier."""
        mult = allocation_to_multiplier(0.25)
        # Allow some tolerance for interpolation
        assert 1.5 <= mult <= 2.2, f"Expected ~1.8, got {mult}"

    def test_zero_allocation(self):
        """0% allocation should give ~1.1x multiplier."""
        mult = allocation_to_multiplier(0.0)
        assert 1.0 <= mult <= 1.2, f"Expected ~1.1, got {mult}"

    def test_monotonic_increasing(self):
        """Higher allocation should give higher multiplier."""
        m_15 = allocation_to_multiplier(0.15)
        m_25 = allocation_to_multiplier(0.25)
        m_40 = allocation_to_multiplier(0.40)
        assert m_15 < m_25 < m_40, "Multiplier should increase with allocation"


class TestValidateGrokMultipliers:
    """Tests for validate_grok_multipliers."""

    def test_valid_results(self):
        """Valid results matching Grok table should pass."""
        results = {
            0.40: 2.75,
            0.25: 1.80,
            0.15: 1.30,
            0.00: 1.10,
        }
        assert validate_grok_multipliers(results) is True

    def test_invalid_results(self):
        """Results outside Grok table should fail."""
        results = {
            0.40: 1.0,  # Too low
            0.25: 5.0,  # Too high
        }
        assert validate_grok_multipliers(results) is False


class TestBuildRateState:
    """Tests for compute_build_rate_state."""

    def test_state_computation(self):
        """Should compute full state correctly."""
        state = compute_build_rate_state(
            autonomy=0.40, propulsion=1.0, prior_build_rate=0.1
        )
        assert state.autonomy_level == 0.40
        assert state.propulsion_level == 1.0
        assert state.build_rate > 0
        assert state.annual_multiplier > 1.0

    def test_first_cycle_multiplier(self):
        """First cycle (no prior) should give multiplier 1.0."""
        state = compute_build_rate_state(
            autonomy=0.40,
            propulsion=1.0,
            prior_build_rate=0.0,  # No prior
        )
        assert state.annual_multiplier == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
