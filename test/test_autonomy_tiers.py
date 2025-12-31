"""Tests for autonomy_tiers.py - Multi-Tier Autonomy Framework.

Tests LEO → Mars → Deep-space tier transitions.
"""


from spaceproof.tiers.autonomy_tiers import (
    AutonomyTier,
    LIGHT_DELAY_LEO_SEC,
    LIGHT_DELAY_MARS_SEC,
    LIGHT_DELAY_DEEP_SPACE_SEC,
    LOOP_FREQUENCY_LEO_SEC,
    LOOP_FREQUENCY_MARS_SEC,
    LOOP_FREQUENCY_DEEP_SPACE_SEC,
    get_tier_from_delay,
    calculate_tier_decision_capacity,
    earth_input_by_tier,
    tier_transition,
    calculate_evolution_rate,
    multi_tier_loop_config,
    validate_tier_readiness,
)


class TestConstants:
    """Test tier constants."""

    def test_leo_instant(self):
        """LEO should have instant communication."""
        assert LIGHT_DELAY_LEO_SEC == 0.0

    def test_mars_delay(self):
        """Mars should have 3 min delay."""
        assert LIGHT_DELAY_MARS_SEC == 180.0

    def test_deep_space_years(self):
        """Deep space should have years of delay."""
        assert LIGHT_DELAY_DEEP_SPACE_SEC > 1e8  # More than 3 years in seconds

    def test_loop_frequencies(self):
        """Loop frequencies should increase with isolation."""
        assert LOOP_FREQUENCY_LEO_SEC < LOOP_FREQUENCY_MARS_SEC
        assert LOOP_FREQUENCY_MARS_SEC < LOOP_FREQUENCY_DEEP_SPACE_SEC


class TestTierEnum:
    """Test AutonomyTier enum."""

    def test_tier_values(self):
        """Tiers should have correct values."""
        assert AutonomyTier.LEO.tier_name == "leo"
        assert AutonomyTier.MARS.tier_name == "mars"
        assert AutonomyTier.DEEP_SPACE.tier_name == "deep_space"

    def test_tier_light_delay(self):
        """Tiers should have correct light delays."""
        assert AutonomyTier.LEO.light_delay_sec == 0.0
        assert AutonomyTier.MARS.light_delay_sec == 180.0


class TestGetTierFromDelay:
    """Test tier determination from delay."""

    def test_leo_from_zero_delay(self):
        """Zero delay should map to LEO."""
        tier = get_tier_from_delay(0.0)
        assert tier == AutonomyTier.LEO

    def test_leo_from_small_delay(self):
        """Small delay should map to LEO."""
        tier = get_tier_from_delay(30.0)
        assert tier == AutonomyTier.LEO

    def test_mars_from_mars_delay(self):
        """Mars delay should map to MARS."""
        tier = get_tier_from_delay(180.0)
        assert tier == AutonomyTier.MARS

    def test_deep_space_from_large_delay(self):
        """Large delay should map to DEEP_SPACE."""
        tier = get_tier_from_delay(1e7)
        assert tier == AutonomyTier.DEEP_SPACE


class TestTierDecisionCapacity:
    """Test tier decision capacity calculations."""

    def test_capacity_positive(self):
        """Capacity should be positive."""
        cap = calculate_tier_decision_capacity(
            AutonomyTier.MARS, crew=10, bandwidth_mbps=2.0
        )
        assert cap > 0

    def test_leo_highest_external(self):
        """LEO should have highest external rate."""
        leo = earth_input_by_tier(AutonomyTier.LEO, 100.0)
        mars = earth_input_by_tier(AutonomyTier.MARS, 100.0)
        assert leo > mars

    def test_deep_space_zero_external(self):
        """Deep space should have zero external rate."""
        rate = earth_input_by_tier(AutonomyTier.DEEP_SPACE, 100.0)
        assert rate == 0.0

    def test_augmentation_increases_capacity(self):
        """Augmentation should increase capacity."""
        base = calculate_tier_decision_capacity(
            AutonomyTier.MARS, crew=10, bandwidth_mbps=2.0, augmentation_factor=1.0
        )
        aug = calculate_tier_decision_capacity(
            AutonomyTier.MARS, crew=10, bandwidth_mbps=2.0, augmentation_factor=5.0
        )
        assert aug > base


class TestTierTransition:
    """Test tier transitions."""

    def test_leo_to_mars(self, capsys):
        """Should handle LEO to Mars transition."""
        result = tier_transition(
            AutonomyTier.LEO,
            AutonomyTier.MARS,
            crew=10,
            bandwidth_mbps=2.0,
        )

        assert result.from_tier == AutonomyTier.LEO
        assert result.to_tier == AutonomyTier.MARS
        assert result.light_delay_change > 0

    def test_mars_to_deep_space(self, capsys):
        """Should handle Mars to Deep Space transition."""
        result = tier_transition(
            AutonomyTier.MARS,
            AutonomyTier.DEEP_SPACE,
            crew=10,
            bandwidth_mbps=2.0,
        )

        assert result.to_tier == AutonomyTier.DEEP_SPACE
        assert result.light_delay_change > 0

    def test_transition_adjustments(self, capsys):
        """Transition should identify required adjustments."""
        result = tier_transition(
            AutonomyTier.LEO,
            AutonomyTier.MARS,
            crew=10,
        )

        assert "loop_frequency" in result.adjustment_needed

    def test_emits_receipt(self, capsys):
        """Should emit tier transition receipt."""
        tier_transition(
            AutonomyTier.LEO,
            AutonomyTier.MARS,
            crew=10,
        )
        output = capsys.readouterr().out
        assert "tier_transition_receipt" in output


class TestEvolutionRate:
    """Test evolution rate calculations."""

    def test_higher_isolation_faster_evolution(self):
        """Higher isolation should mean faster evolution."""
        leo_rate = calculate_evolution_rate(AutonomyTier.LEO)
        mars_rate = calculate_evolution_rate(AutonomyTier.MARS)
        deep_rate = calculate_evolution_rate(AutonomyTier.DEEP_SPACE)

        # More isolated = higher autonomy requirement = faster evolution
        assert deep_rate > mars_rate
        assert mars_rate > leo_rate


class TestMultiTierLoopConfig:
    """Test multi-tier loop configuration."""

    def test_config_has_required_fields(self):
        """Config should have required fields."""
        config = multi_tier_loop_config(AutonomyTier.MARS)

        assert "tier" in config
        assert "cycle_time_sec" in config
        assert "compression_target" in config

    def test_tier_specific_values(self):
        """Different tiers should have different values."""
        leo = multi_tier_loop_config(AutonomyTier.LEO)
        mars = multi_tier_loop_config(AutonomyTier.MARS)

        assert leo["cycle_time_sec"] < mars["cycle_time_sec"]


class TestTierReadiness:
    """Test tier readiness validation."""

    def test_readiness_check(self, capsys):
        """Should check tier readiness."""
        result = validate_tier_readiness(
            tier=AutonomyTier.MARS,
            crew=10,
            compute_mass_kg=100,
            power_available_w=5000,
            current_compression_ratio=0.9,
        )

        assert "overall_ready" in result
        assert "compression_ready" in result
        assert "power_ready" in result

    def test_emits_receipt(self, capsys):
        """Should emit readiness receipt."""
        validate_tier_readiness(
            tier=AutonomyTier.MARS,
            crew=10,
            compute_mass_kg=100,
            power_available_w=5000,
            current_compression_ratio=0.9,
        )
        output = capsys.readouterr().out
        assert "tier_readiness_receipt" in output
