"""Tests for BUILD C2: entropy.py module.

Verifies all entropy functions meet specifications per CLAUDEME §8.
"""
import numpy as np
import pytest

from src.entropy import (
    # Constants - verified
    HUMAN_METABOLIC_W,
    MOXIE_O2_G_PER_HR,
    ISS_WATER_RECOVERY,
    ISS_O2_CLOSURE,
    MARS_RELAY_MBPS,
    LIGHT_DELAY_MIN,
    LIGHT_DELAY_MAX,
    SOLAR_FLUX_MAX,
    SOLAR_FLUX_DUST,
    KILOPOWER_KW,
    # Constants - placeholder
    DECISION_BITS_PER_PERSON_PER_SEC,
    EXPERTISE_MULTIPLIER,
    LATENCY_COST_FACTOR,
    SUBSYSTEM_WEIGHTS,
    # Functions
    shannon_entropy,
    subsystem_entropy,
    total_colony_entropy,
    entropy_rate,
    decision_capacity,
    earth_input_rate,
    sovereignty_threshold,
    survival_bound,
    entropy_status,
    emit_entropy_receipt,
)


class TestShannonEntropy:
    """Tests for shannon_entropy function."""

    def test_shannon_entropy_uniform(self):
        """Uniform distribution [0.25, 0.25, 0.25, 0.25] -> 2.0 bits."""
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        h = shannon_entropy(dist)
        assert abs(h - 2.0) < 0.01, f"Expected ~2.0, got {h}"

    def test_shannon_entropy_certain(self):
        """Single 1.0 -> 0.0 bits (no uncertainty)."""
        dist = np.array([1.0])
        h = shannon_entropy(dist)
        assert abs(h - 0.0) < 0.01, f"Expected ~0.0, got {h}"

    def test_shannon_entropy_empty(self):
        """Empty array -> 0.0."""
        dist = np.array([])
        h = shannon_entropy(dist)
        assert h == 0.0, f"Expected 0.0, got {h}"

    def test_shannon_entropy_zeros(self):
        """All zeros -> 0.0."""
        dist = np.array([0, 0, 0, 0])
        h = shannon_entropy(dist)
        assert h == 0.0, f"Expected 0.0, got {h}"

    def test_shannon_entropy_unnormalized(self):
        """[1, 1, 1, 1] normalizes to uniform -> 2.0 bits."""
        dist = np.array([1, 1, 1, 1])
        h = shannon_entropy(dist)
        assert abs(h - 2.0) < 0.01, f"Expected ~2.0, got {h}"


class TestSubsystemEntropy:
    """Tests for subsystem_entropy function."""

    def test_subsystem_entropy_returns_float(self):
        """Any valid state + subsystem -> float."""
        state = {
            "O2_pct": 0.21,
            "CO2_pct": 0.0004,
            "N2_pct": 0.78,
            "temperature_C": 20.0,
            "water_ratio": 1.0,
            "food_ratio": 1.0,
            "power_ratio": 1.0,
        }
        for subsystem in ["atmosphere", "thermal", "resource", "decision"]:
            h = subsystem_entropy(state, subsystem)
            assert isinstance(h, float), f"Expected float for {subsystem}, got {type(h)}"


class TestTotalColonyEntropy:
    """Tests for total_colony_entropy function."""

    def test_total_colony_entropy_sums_subsystems(self):
        """Sum matches weighted individual entropies."""
        state = {
            "O2_pct": 0.21,
            "CO2_pct": 0.0004,
            "N2_pct": 0.78,
            "temperature_C": 20.0,
            "water_ratio": 1.0,
            "food_ratio": 1.0,
            "power_ratio": 1.0,
        }
        total = total_colony_entropy(state)

        # Compute manually
        expected = 0.0
        for subsystem, weight in SUBSYSTEM_WEIGHTS.items():
            expected += weight * subsystem_entropy(state, subsystem)

        assert abs(total - expected) < 0.001, f"Expected {expected}, got {total}"


class TestEntropyRate:
    """Tests for entropy_rate function."""

    def test_entropy_rate_positive_accumulating(self):
        """Increasing entropy series -> positive rate."""
        # Create states with increasing entropy
        states = [
            {"O2_pct": 0.21, "CO2_pct": 0.0004, "N2_pct": 0.78, "water_ratio": 1.0, "food_ratio": 1.0, "power_ratio": 1.0},
            {"O2_pct": 0.18, "CO2_pct": 0.03, "N2_pct": 0.78, "water_ratio": 0.8, "food_ratio": 0.9, "power_ratio": 0.7},
            {"O2_pct": 0.15, "CO2_pct": 0.05, "N2_pct": 0.78, "water_ratio": 0.5, "food_ratio": 0.6, "power_ratio": 0.4},
        ]
        rate = entropy_rate(states)
        assert rate > 0, f"Expected positive rate, got {rate}"

    def test_entropy_rate_negative_shedding(self):
        """Decreasing entropy series -> negative rate."""
        # Create states with decreasing entropy (improving conditions)
        states = [
            {"O2_pct": 0.15, "CO2_pct": 0.05, "N2_pct": 0.78, "water_ratio": 0.5, "food_ratio": 0.6, "power_ratio": 0.4},
            {"O2_pct": 0.18, "CO2_pct": 0.03, "N2_pct": 0.78, "water_ratio": 0.8, "food_ratio": 0.9, "power_ratio": 0.7},
            {"O2_pct": 0.21, "CO2_pct": 0.0004, "N2_pct": 0.78, "water_ratio": 1.0, "food_ratio": 1.0, "power_ratio": 1.0},
        ]
        rate = entropy_rate(states)
        assert rate < 0, f"Expected negative rate, got {rate}"

    def test_entropy_rate_short_list(self):
        """Single state -> 0.0."""
        states = [{"O2_pct": 0.21}]
        rate = entropy_rate(states)
        assert rate == 0.0, f"Expected 0.0, got {rate}"


class TestDecisionCapacity:
    """Tests for decision_capacity function."""

    def test_decision_capacity_scales_with_crew(self):
        """More crew -> higher capacity."""
        dc_5 = decision_capacity(5, {}, 2.0, 180)
        dc_10 = decision_capacity(10, {}, 2.0, 180)
        assert dc_10 > dc_5, f"Expected dc_10 ({dc_10}) > dc_5 ({dc_5})"

    def test_decision_capacity_zero_crew(self):
        """crew=0 -> 0.0."""
        dc = decision_capacity(0, {}, 2.0, 180)
        assert dc == 0.0, f"Expected 0.0, got {dc}"

    def test_decision_capacity_expertise_boost(self):
        """Higher expertise -> higher capacity."""
        dc_low = decision_capacity(10, {"piloting": 0.5}, 2.0, 180)
        dc_high = decision_capacity(10, {"piloting": 1.0}, 2.0, 180)
        assert dc_high > dc_low, f"Expected dc_high ({dc_high}) > dc_low ({dc_low})"


class TestEarthInputRate:
    """Tests for earth_input_rate function."""

    def test_earth_input_rate_degrades_with_latency(self):
        """Higher latency -> lower rate."""
        er_low = earth_input_rate(2.0, 60)   # 1 min latency
        er_high = earth_input_rate(2.0, 600)  # 10 min latency
        assert er_low > er_high, f"Expected er_low ({er_low}) > er_high ({er_high})"

    def test_earth_input_rate_max_latency(self):
        """latency=22 min -> severely degraded."""
        er_min = earth_input_rate(2.0, LIGHT_DELAY_MIN * 60)  # Opposition
        er_max = earth_input_rate(2.0, LIGHT_DELAY_MAX * 60)  # Conjunction
        # Max should be severely degraded (10x penalty + latency factor)
        ratio = er_min / er_max
        assert ratio > 50, f"Expected severe degradation, ratio: {ratio}"


class TestSovereigntyThreshold:
    """Tests for sovereignty_threshold function."""

    def test_sovereignty_threshold_true(self):
        """internal=1.0, external=0.5 -> True."""
        sov = sovereignty_threshold(1.0, 0.5)
        assert sov is True, f"Expected True, got {sov}"

    def test_sovereignty_threshold_false(self):
        """internal=0.5, external=1.0 -> False."""
        sov = sovereignty_threshold(0.5, 1.0)
        assert sov is False, f"Expected False, got {sov}"

    def test_sovereignty_threshold_equal(self):
        """internal=1.0, external=1.0 -> False (not >)."""
        sov = sovereignty_threshold(1.0, 1.0)
        assert sov is False, f"Expected False (not >), got {sov}"


class TestSurvivalBound:
    """Tests for survival_bound function."""

    def test_survival_bound_positive(self):
        """Any valid inputs -> positive bound."""
        bound = survival_bound(10, 500, 100)
        assert bound > 0, f"Expected positive bound, got {bound}"


class TestEntropyStatus:
    """Tests for entropy_status function."""

    def test_entropy_status_stable(self):
        """rate=-0.1 -> 'stable'."""
        status = entropy_status(-0.1, 10.0, 5.0)
        assert status == "stable", f"Expected 'stable', got {status}"

    def test_entropy_status_accumulating(self):
        """rate=0.1, current < bound -> 'accumulating'."""
        status = entropy_status(0.1, 10.0, 5.0)
        assert status == "accumulating", f"Expected 'accumulating', got {status}"

    def test_entropy_status_critical(self):
        """current >= bound -> 'critical'."""
        status = entropy_status(0.1, 10.0, 10.0)
        assert status == "critical", f"Expected 'critical', got {status}"


class TestEmitEntropyReceipt:
    """Tests for emit_entropy_receipt function."""

    def test_emit_entropy_receipt_has_all_fields(self, capsys):
        """All required fields present."""
        state = {
            "crew": 10,
            "expertise": {"piloting": 0.8, "engineering": 0.9},
            "bandwidth": 2.0,
            "latency_sec": 180,
            "volume_m3": 500,
            "power_kw": 40,
            "O2_pct": 0.21,
            "CO2_pct": 0.0004,
            "N2_pct": 0.78,
            "temperature_C": 20.0,
            "water_ratio": 1.0,
            "food_ratio": 1.0,
            "power_ratio": 1.0,
        }
        states = [state]

        receipt = emit_entropy_receipt("mars-alpha", state, states)

        # Check all required fields
        required_fields = [
            "receipt_type",
            "ts",
            "tenant_id",
            "colony_id",
            "H_atmosphere",
            "H_thermal",
            "H_resource",
            "H_decision",
            "H_total",
            "entropy_rate",
            "decision_capacity_bps",
            "earth_input_bps",
            "sovereignty",
            "survival_bound",
            "status",
            "payload_hash",
        ]

        for field in required_fields:
            assert field in receipt, f"Missing field: {field}"

        assert receipt["receipt_type"] == "entropy"
        assert receipt["tenant_id"] == "axiom-colony"
        assert receipt["colony_id"] == "mars-alpha"
        assert isinstance(receipt["sovereignty"], bool)


class TestConstants:
    """Tests for constant definitions."""

    def test_constants_exist(self):
        """All 10 verified constants defined."""
        # Verified constants
        assert HUMAN_METABOLIC_W == 100
        assert MOXIE_O2_G_PER_HR == 5.5
        assert ISS_WATER_RECOVERY == 0.98
        assert ISS_O2_CLOSURE == 0.875
        assert MARS_RELAY_MBPS == 2.0
        assert LIGHT_DELAY_MIN == 3
        assert LIGHT_DELAY_MAX == 22
        assert SOLAR_FLUX_MAX == 590
        assert SOLAR_FLUX_DUST == 6
        assert KILOPOWER_KW == 10

        # Placeholder constants (verify they exist)
        assert DECISION_BITS_PER_PERSON_PER_SEC == 0.1
        assert EXPERTISE_MULTIPLIER == 2.0
        assert LATENCY_COST_FACTOR == 0.8
        assert "decision" in SUBSYSTEM_WEIGHTS
        assert SUBSYSTEM_WEIGHTS["decision"] == 2.0
