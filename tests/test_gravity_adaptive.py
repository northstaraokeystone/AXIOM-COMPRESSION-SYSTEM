"""Tests for gravity adaptive module."""

import pytest
from src.gravity_adaptive import (
    get_gravity_status,
    get_planet_gravity,
    adjust_for_gravity,
    calculate_timing_adjustment,
    adjust_consensus_timing,
    adjust_packet_timing,
    validate_gravity_adjustment,
    get_all_planet_adjustments,
    EARTH_GRAVITY_G,
    MARS_GRAVITY_G,
)


class TestGravityConstants:
    """Tests for gravity constants."""

    def test_earth_gravity(self):
        """Earth gravity is 1.0g."""
        assert EARTH_GRAVITY_G == 1.0

    def test_mars_gravity(self):
        """Mars gravity is 0.38g."""
        assert MARS_GRAVITY_G == 0.38


class TestGravityStatus:
    """Tests for gravity status."""

    def test_gravity_status(self):
        """Status query works."""
        status = get_gravity_status()
        assert "adaptive_enabled" in status
        assert "planets_configured" in status
        assert "default_gravity_map" in status


class TestPlanetGravity:
    """Tests for planet gravity lookup."""

    def test_mars_gravity_lookup(self):
        """Mars gravity returned."""
        gravity = get_planet_gravity("mars")
        assert gravity == 0.38

    def test_earth_gravity_lookup(self):
        """Earth gravity returned."""
        gravity = get_planet_gravity("earth")
        assert gravity == 1.0

    def test_unknown_planet_default(self):
        """Unknown planet returns 1.0g."""
        gravity = get_planet_gravity("unknown_planet")
        assert gravity == 1.0


class TestGravityAdjustment:
    """Tests for gravity adjustment calculation."""

    def test_mars_adjustment(self):
        """Mars adjustment calculated."""
        result = adjust_for_gravity(0.38)
        assert "timing_factor" in result
        assert "consensus_multiplier" in result
        assert result["gravity_ratio"] < 1.0

    def test_earth_adjustment(self):
        """Earth has no adjustment."""
        result = adjust_for_gravity(1.0)
        assert result["timing_factor"] == 1.0
        assert result["consensus_multiplier"] == 1.0


class TestTimingAdjustment:
    """Tests for timing adjustment formula."""

    def test_timing_factor_formula(self):
        """Formula: 1.0 / (gravity_ratio ^ 0.5)."""
        factor = calculate_timing_adjustment(0.38)
        expected = 1.0 / (0.38 ** 0.5)
        assert abs(factor - expected) < 0.001


class TestConsensusAdjustment:
    """Tests for consensus timing adjustment."""

    def test_consensus_timing_mars(self):
        """Mars consensus timing adjusted."""
        result = adjust_consensus_timing(0.38)
        assert result["adjusted_heartbeat_ms"] > result["base_heartbeat_ms"]
        assert result["adjusted_election_timeout_ms"] > result["base_election_timeout_ms"]


class TestPacketAdjustment:
    """Tests for packet timing adjustment."""

    def test_packet_timing_mars(self):
        """Mars packet timing adjusted."""
        result = adjust_packet_timing(0.38)
        assert result["adjusted_timeout_ms"] > result["base_timeout_ms"]
        assert result["adjusted_retry_delay_ms"] > result["base_retry_delay_ms"]


class TestGravityValidation:
    """Tests for gravity validation."""

    def test_mars_validation_passes(self):
        """Mars validation passes."""
        result = validate_gravity_adjustment("mars")
        assert result["valid"] is True
        assert result["gravity_g"] == 0.38

    def test_earth_validation_passes(self):
        """Earth validation passes."""
        result = validate_gravity_adjustment("earth")
        assert result["valid"] is True


class TestAllPlanetAdjustments:
    """Tests for all planet adjustments."""

    def test_all_planets_returned(self):
        """All planets returned."""
        adjustments = get_all_planet_adjustments()
        assert "mars" in adjustments
        assert "venus" in adjustments
        assert "europa" in adjustments

    def test_adjustment_structure(self):
        """Each planet has expected fields."""
        adjustments = get_all_planet_adjustments()
        for planet, adj in adjustments.items():
            assert "gravity_g" in adj
            assert "timing_factor" in adj
            assert "consensus_multiplier" in adj


class TestGravityReceipts:
    """Tests for receipt emission."""

    def test_gravity_receipt(self, capsys):
        """Receipt emitted."""
        adjust_for_gravity(0.38)
        captured = capsys.readouterr()
        assert "gravity_adjustment_receipt" in captured.out
