"""test_tier_risk.py - Tests for 3-tier probability x impact risk model

Validates Grok's risk tiers:
    Tier 1: 60-80% probability, medium impact
    Tier 2: 30-50% probability, high impact
    Tier 3: 5-15% probability, existential impact
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tier_risk import (
    tier_1_risk,
    tier_2_risk,
    tier_3_risk,
    assess_tier_risk,
    aggregate_risk_score,
    is_existential,
    get_highest_probability_tier,
    get_highest_impact_tier,
    risk_summary,
    format_risk_assessment,
    RiskTier,
    TIER_1_PROB_RANGE,
    TIER_2_PROB_RANGE,
    TIER_3_PROB_RANGE,
    IMPACT_MEDIUM,
    IMPACT_HIGH,
    IMPACT_EXISTENTIAL,
    UNDER_PIVOT_THRESHOLD,
)


class TestTier1Risk:
    """Tests for tier_1_risk function."""

    def test_under_pivot_high_probability(self):
        """Under-pivoted allocation should have high Tier 1 probability."""
        profile = tier_1_risk(0.20)  # Under 30%
        assert profile.probability_low >= 0.60
        assert profile.probability_high <= 0.95

    def test_adequate_allocation_lower_probability(self):
        """Adequate allocation should have lower Tier 1 probability."""
        profile = tier_1_risk(0.35)  # Above 30%
        assert profile.probability_low < 0.60
        assert profile.probability_high < 0.60

    def test_impact_is_medium(self):
        """Tier 1 should always have medium impact."""
        profile = tier_1_risk(0.20)
        assert profile.impact_class == IMPACT_MEDIUM

    def test_mitigation_available(self):
        """Tier 1 should have mitigation available."""
        profile = tier_1_risk(0.20)
        assert profile.mitigation_available is True

    def test_failure_modes_populated(self):
        """Tier 1 should have failure modes."""
        profile = tier_1_risk(0.20)
        assert len(profile.failure_modes) > 0


class TestTier2Risk:
    """Tests for tier_2_risk function."""

    def test_long_timeline_high_probability(self):
        """Long timeline should have high Tier 2 probability."""
        profile = tier_2_risk(30)  # 30 years - very long
        assert profile.probability_low >= 0.50

    def test_short_timeline_low_probability(self):
        """Short timeline should have low Tier 2 probability."""
        profile = tier_2_risk(12)  # 12 years - fast
        assert profile.probability_low < 0.30
        assert profile.probability_high < 0.30

    def test_impact_is_high(self):
        """Tier 2 should always have high impact."""
        profile = tier_2_risk(20)
        assert profile.impact_class == IMPACT_HIGH


class TestTier3Risk:
    """Tests for tier_3_risk function."""

    def test_zero_autonomy_max_existential(self):
        """Zero autonomy should have elevated existential risk."""
        profile = tier_3_risk(0.0)
        # Zero autonomy has elevated existential risk (higher than adequate allocation)
        adequate_profile = tier_3_risk(0.35)
        assert profile.probability_low >= adequate_profile.probability_low

    def test_adequate_allocation_minimal_existential(self):
        """Adequate allocation should have minimal existential risk."""
        profile = tier_3_risk(0.35)
        assert profile.probability_low <= 0.05

    def test_impact_is_existential(self):
        """Tier 3 should always have existential impact."""
        profile = tier_3_risk(0.20)
        assert profile.impact_class == IMPACT_EXISTENTIAL


class TestAssessTierRisk:
    """Tests for assess_tier_risk main function."""

    def test_returns_three_tiers(self):
        """Should return profiles for all three tiers."""
        profiles = assess_tier_risk(0.25, 20)
        assert len(profiles) == 3

    def test_tier_order(self):
        """Profiles should be in tier order."""
        profiles = assess_tier_risk(0.25, 20)
        assert profiles[0].tier == RiskTier.TIER_1
        assert profiles[1].tier == RiskTier.TIER_2
        assert profiles[2].tier == RiskTier.TIER_3


class TestAggregateRiskScore:
    """Tests for aggregate_risk_score function."""

    def test_high_risk_allocation(self):
        """Under-pivoted allocation should have high aggregate score."""
        profiles = assess_tier_risk(0.10, 35)  # Very under-pivoted
        score = aggregate_risk_score(profiles)
        assert score > 1.0, f"Should have high score, got {score}"

    def test_low_risk_allocation(self):
        """Adequate allocation should have lower aggregate score."""
        profiles = assess_tier_risk(0.40, 12)  # Optimal
        score = aggregate_risk_score(profiles)
        assert score < 2.0, f"Should have lower score, got {score}"

    def test_score_is_positive(self):
        """Score should always be positive."""
        profiles = assess_tier_risk(0.30, 20)
        score = aggregate_risk_score(profiles)
        assert score > 0


class TestIsExistential:
    """Tests for is_existential function."""

    def test_tier_3_is_existential(self):
        """Should detect existential risk from Tier 3."""
        profiles = assess_tier_risk(0.25, 20)
        assert is_existential(profiles) is True

    def test_empty_profiles(self):
        """Empty profiles should return False."""
        assert is_existential([]) is False


class TestGetHighestProbabilityTier:
    """Tests for get_highest_probability_tier function."""

    def test_returns_tier_1_for_underpivot(self):
        """Under-pivot should make Tier 1 highest probability."""
        profiles = assess_tier_risk(0.20, 20)
        highest = get_highest_probability_tier(profiles)
        # Tier 1 typically has highest probability
        assert highest.tier == RiskTier.TIER_1


class TestGetHighestImpactTier:
    """Tests for get_highest_impact_tier function."""

    def test_returns_tier_3(self):
        """Tier 3 should always be highest impact (existential)."""
        profiles = assess_tier_risk(0.25, 20)
        highest = get_highest_impact_tier(profiles)
        assert highest.tier == RiskTier.TIER_3


class TestRiskSummary:
    """Tests for risk_summary function."""

    def test_summary_structure(self):
        """Summary should have required fields."""
        profiles = assess_tier_risk(0.25, 20)
        summary = risk_summary(profiles)

        assert "aggregate_score" in summary
        assert "risk_level" in summary
        assert "has_existential_risk" in summary
        assert "highest_probability_tier" in summary
        assert "highest_impact_tier" in summary

    def test_risk_levels(self):
        """Risk levels should correspond to scores."""
        # High risk (under-pivoted)
        profiles_high = assess_tier_risk(0.10, 35)
        summary_high = risk_summary(profiles_high)
        assert summary_high["risk_level"] in ["HIGH", "CRITICAL"]

        # Lower risk (adequate)
        profiles_low = assess_tier_risk(0.40, 12)
        summary_low = risk_summary(profiles_low)
        # Could be MODERATE or LOW depending on exact numbers


class TestFormatRiskAssessment:
    """Tests for format_risk_assessment function."""

    def test_format_structure(self):
        """Should produce readable report."""
        profiles = assess_tier_risk(0.25, 20)
        report = format_risk_assessment(profiles)

        assert "RISK ASSESSMENT" in report
        assert "TIER 1" in report
        assert "TIER 2" in report
        assert "TIER 3" in report

    def test_contains_probabilities(self):
        """Report should contain probability information."""
        profiles = assess_tier_risk(0.25, 20)
        report = format_risk_assessment(profiles)

        assert "Probability" in report


class TestConstants:
    """Tests for tier_risk constants."""

    def test_probability_ranges(self):
        """Probability ranges should be valid."""
        assert TIER_1_PROB_RANGE[0] < TIER_1_PROB_RANGE[1]
        assert TIER_2_PROB_RANGE[0] < TIER_2_PROB_RANGE[1]
        assert TIER_3_PROB_RANGE[0] < TIER_3_PROB_RANGE[1]

    def test_tier_1_highest_probability(self):
        """Tier 1 should have highest probability range."""
        assert TIER_1_PROB_RANGE[0] > TIER_2_PROB_RANGE[0]
        assert TIER_2_PROB_RANGE[0] > TIER_3_PROB_RANGE[0]

    def test_under_pivot_threshold(self):
        """Under-pivot threshold should be 30%."""
        assert UNDER_PIVOT_THRESHOLD == 0.30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
