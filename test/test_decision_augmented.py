"""Tests for decision_augmented.py - AI/Neuralink Augmentation.

Tests augmentation factors and sovereignty equivalence.
"""

import pytest

from spaceproof.decision_augmented import (
    AI_AUGMENTATION_FACTOR,
    NEURALINK_AUGMENTATION_FACTOR,
    HUMAN_ONLY_FACTOR,
    calculate_augmentation_factor,
    effective_crew_size,
    augmentation_energy_cost,
    augmentation_mass_cost,
    validate_augmentation,
    calculate_sovereignty_with_augmentation,
    optimal_augmentation_mix,
)


class TestConstants:
    """Test augmentation constants."""

    def test_ai_factor_5x(self):
        """AI augmentation should be 5x."""
        assert AI_AUGMENTATION_FACTOR == 5.0

    def test_neuralink_factor_20x(self):
        """Neuralink augmentation should be 20x."""
        assert NEURALINK_AUGMENTATION_FACTOR == 20.0

    def test_human_only_1x(self):
        """Human-only should be 1x."""
        assert HUMAN_ONLY_FACTOR == 1.0


class TestAugmentationFactor:
    """Test augmentation factor calculations."""

    def test_human_only_factor(self):
        """Human-only should always return 1.0."""
        factor = calculate_augmentation_factor("human_only", 100)
        assert factor == 1.0

    def test_ai_factor_scales_with_mass(self):
        """AI factor should scale with compute mass."""
        f1 = calculate_augmentation_factor("ai_assisted", 10)
        f2 = calculate_augmentation_factor("ai_assisted", 100)
        assert f2 > f1

    def test_neuralink_factor(self):
        """Neuralink factor should be higher than AI."""
        ai = calculate_augmentation_factor("ai_assisted", 100)
        neuralink = calculate_augmentation_factor("neuralink_assisted", 100)
        assert neuralink > ai

    def test_zero_compute_mass(self):
        """Zero compute mass should return human-only factor."""
        factor = calculate_augmentation_factor("ai_assisted", 0)
        assert factor == HUMAN_ONLY_FACTOR


class TestEffectiveCrew:
    """Test effective crew size calculations."""

    def test_effective_crew_with_ai(self):
        """4 crew with 5x AI = 20 effective."""
        factor = 5.0
        effective = effective_crew_size(4, factor)
        assert effective == 20

    def test_effective_crew_human_only(self):
        """Human-only: effective = physical."""
        effective = effective_crew_size(10, 1.0)
        assert effective == 10


class TestEnergyCost:
    """Test augmentation energy cost calculations."""

    def test_human_only_no_cost(self):
        """Human-only should have zero energy cost."""
        cost = augmentation_energy_cost("human_only", 1.0)
        assert cost == 0.0

    def test_ai_energy_cost(self):
        """AI should have energy cost."""
        cost = augmentation_energy_cost("ai_assisted", 5.0)
        assert cost > 0

    def test_neuralink_lower_cost(self):
        """Neuralink should have lower cost per factor."""
        ai_cost = augmentation_energy_cost("ai_assisted", 5.0)
        # Neuralink at same factor point should cost less
        neuralink_cost = augmentation_energy_cost("neuralink_assisted", 5.0)
        assert neuralink_cost < ai_cost


class TestMassCost:
    """Test augmentation mass cost calculations."""

    def test_human_only_no_mass(self):
        """Human-only should have zero mass cost."""
        mass = augmentation_mass_cost("human_only", 1.0)
        assert mass == 0.0

    def test_ai_mass_cost(self):
        """AI should have mass cost."""
        mass = augmentation_mass_cost("ai_assisted", 5.0)
        assert mass > 0


class TestValidateAugmentation:
    """Test augmentation validation."""

    def test_feasible_augmentation(self, capsys):
        """Should validate feasible configuration."""
        result = validate_augmentation(
            crew=4,
            augmentation_type="ai_assisted",
            compute_mass_kg=100,
            power_available=10000,
        )
        assert result["feasible"]
        assert result["factor"] > 1.0

    def test_insufficient_power(self, capsys):
        """Should detect insufficient power."""
        result = validate_augmentation(
            crew=4,
            augmentation_type="ai_assisted",
            compute_mass_kg=100,
            power_available=10,  # Very low
        )
        assert result["bottleneck"] == "power"

    def test_emits_receipt(self, capsys):
        """Should emit augmentation receipt."""
        validate_augmentation(
            crew=4,
            augmentation_type="ai_assisted",
            compute_mass_kg=100,
            power_available=10000,
        )
        output = capsys.readouterr().out
        assert "augmentation_receipt" in output


class TestSovereigntyWithAugmentation:
    """Test sovereignty calculation with augmentation."""

    def test_ai_augmentation_threshold(self, capsys):
        """4 crew + AI should approach 20 crew human-only."""
        result = calculate_sovereignty_with_augmentation(
            crew=4,
            augmentation_type="ai_assisted",
            compute_mass_kg=100,
            bandwidth_mbps=2.0,
            delay_s=180,
        )
        assert result["effective_crew"] == pytest.approx(4 * result["factor"], rel=0.1)

    def test_emits_receipt(self, capsys):
        """Should emit augmented sovereignty receipt."""
        calculate_sovereignty_with_augmentation(
            crew=4,
            augmentation_type="ai_assisted",
            compute_mass_kg=100,
            bandwidth_mbps=2.0,
            delay_s=180,
        )
        output = capsys.readouterr().out
        assert "augmented_sovereignty_receipt" in output


class TestOptimalMix:
    """Test optimal augmentation mix calculation."""

    def test_finds_optimal_config(self, capsys):
        """Should find optimal crew + augmentation mix."""
        result = optimal_augmentation_mix(
            target_effective_crew=20,
            power_budget_w=10000,
            mass_budget_kg=200,
        )
        assert result["effective_crew"] >= 20
        assert result["physical_crew"] < 20  # Should use augmentation

    def test_emits_receipt(self, capsys):
        """Should emit optimal augmentation receipt."""
        optimal_augmentation_mix(
            target_effective_crew=20,
            power_budget_w=10000,
            mass_budget_kg=200,
        )
        output = capsys.readouterr().out
        assert "optimal_augmentation_receipt" in output
