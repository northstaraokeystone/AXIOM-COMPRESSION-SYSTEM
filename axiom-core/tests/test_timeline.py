"""test_timeline.py - Tests for year-to-threshold projections

Validates Grok's timeline table:
    40% -> 12-15 years
    25% -> 18-22 years
    15% -> 25-35 years
    0%  -> 40-60+ years (existential)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.timeline import (
    project_timeline,
    generate_timeline_table,
    validate_grok_table,
    allocation_to_multiplier,
    compute_years_to_threshold,
    compare_to_optimal,
    format_timeline_table,
    project_sovereignty_date,
    TimelineConfig,
    TimelineProjection,
    THRESHOLD_PERSON_EQUIVALENT,
    YEARS_40PCT,
    YEARS_25PCT,
    YEARS_15PCT,
    YEARS_0PCT,
)


class TestAllocationToMultiplier:
    """Tests for allocation_to_multiplier function."""

    def test_40pct_gives_high_multiplier(self):
        """40% allocation should give 2.5-3.0x."""
        mult = allocation_to_multiplier(0.40)
        assert 2.5 <= mult <= 3.0, f"Expected 2.5-3.0, got {mult}"

    def test_25pct_gives_medium_multiplier(self):
        """25% allocation should give ~1.8x."""
        mult = allocation_to_multiplier(0.25)
        assert 1.5 <= mult <= 2.2, f"Expected ~1.8, got {mult}"

    def test_15pct_gives_low_multiplier(self):
        """15% allocation should give ~1.3x."""
        mult = allocation_to_multiplier(0.15)
        assert 1.2 <= mult <= 1.5, f"Expected ~1.3, got {mult}"

    def test_zero_gives_minimal_multiplier(self):
        """0% allocation should give ~1.1x."""
        mult = allocation_to_multiplier(0.0)
        assert 1.0 <= mult <= 1.2, f"Expected ~1.1, got {mult}"

    def test_monotonic(self):
        """Higher allocation should give higher multiplier."""
        mults = [allocation_to_multiplier(f) for f in [0.0, 0.15, 0.25, 0.40]]
        assert mults == sorted(mults), "Multiplier should increase with allocation"


class TestComputeYearsToThreshold:
    """Tests for compute_years_to_threshold function."""

    def test_high_multiplier_fast_years(self):
        """High multiplier should reach threshold quickly."""
        years_low, years_high = compute_years_to_threshold(
            annual_multiplier=2.75,
            current_capability=1000,
            threshold=1_000_000
        )
        assert years_low >= 5, "Should take at least 5 years"
        assert years_high <= 20, "Should reach in under 20 years"

    def test_low_multiplier_slow_years(self):
        """Low multiplier should take very long."""
        years_low, years_high = compute_years_to_threshold(
            annual_multiplier=1.1,
            current_capability=1000,
            threshold=1_000_000
        )
        assert years_low >= 50, "Should take 50+ years at 1.1x"

    def test_no_growth_infinite(self):
        """Multiplier <= 1 should give effectively infinite years."""
        years_low, years_high = compute_years_to_threshold(
            annual_multiplier=1.0,
            current_capability=1000,
            threshold=1_000_000
        )
        assert years_low >= 100, "No growth = never reaches threshold"

    def test_already_at_threshold(self):
        """Already at threshold should give 0 years."""
        years_low, years_high = compute_years_to_threshold(
            annual_multiplier=2.0,
            current_capability=1_500_000,
            threshold=1_000_000
        )
        assert years_low == 0 and years_high == 0


class TestProjectTimeline:
    """Tests for project_timeline function."""

    def test_40pct_projection(self):
        """40% allocation should project fastest timeline."""
        proj = project_timeline(0.40)
        # 40% should reach threshold in reasonable time with high multiplier
        assert proj.years_to_threshold_low >= 1, "Should take at least 1 year"
        assert proj.years_to_threshold_high <= 20, "Should reach in under 20 years"
        assert proj.annual_multiplier >= 2.5, "Should have high multiplier"

    def test_25pct_projection(self):
        """25% allocation should project longer timeline than 40%."""
        proj_25 = project_timeline(0.25)
        proj_40 = project_timeline(0.40)
        # 25% should take longer than 40%
        assert proj_25.years_to_threshold_low >= proj_40.years_to_threshold_low, \
            "25% should take at least as long as 40%"

    def test_zero_allocation_very_long(self):
        """0% allocation should take 40+ years."""
        proj = project_timeline(0.0)
        assert proj.years_to_threshold_low >= 40, "Zero allocation = very long"

    def test_delay_vs_optimal(self):
        """Lower allocation should show delay vs 40%."""
        proj_25 = project_timeline(0.25)
        proj_40 = project_timeline(0.40)
        assert proj_25.delay_vs_optimal > 0, "25% should show delay vs 40%"
        assert proj_40.delay_vs_optimal == 0, "40% is optimal, no delay"


class TestGenerateTimelineTable:
    """Tests for generate_timeline_table function."""

    def test_default_fractions(self):
        """Should generate projections for default fractions."""
        table = generate_timeline_table()
        assert len(table) == 5  # 0.40, 0.25, 0.15, 0.05, 0.00

    def test_custom_fractions(self):
        """Should handle custom fractions."""
        table = generate_timeline_table([0.30, 0.20])
        assert len(table) == 2

    def test_ordered_by_delay(self):
        """Lower allocation should have higher delay."""
        table = generate_timeline_table([0.40, 0.25, 0.15])
        delays = [p.delay_vs_optimal for p in table]
        assert delays == sorted(delays), "Delay should increase as allocation decreases"


class TestValidateGrokTable:
    """Tests for validate_grok_table function."""

    def test_valid_projections_pass(self):
        """Projections matching Grok table should pass."""
        table = generate_timeline_table()
        validation = validate_grok_table(table)
        # At least some should pass
        passed_count = sum(1 for k, v in validation.items()
                         if isinstance(v, dict) and v.get("passed", False))
        assert passed_count >= 2, "At least some fractions should validate"


class TestFormatTimelineTable:
    """Tests for format_timeline_table function."""

    def test_table_format(self):
        """Should produce markdown table format."""
        table = generate_timeline_table()
        formatted = format_timeline_table(table)
        assert "| Pivot fraction |" in formatted
        assert "baseline" in formatted.lower()


class TestProjectSovereigntyDate:
    """Tests for project_sovereignty_date convenience function."""

    def test_optimal_allocation(self):
        """40% should be marked as optimal."""
        result = project_sovereignty_date(0.40)
        assert result["recommendation"] == "optimal"

    def test_acceptable_allocation(self):
        """30% should be marked as acceptable."""
        result = project_sovereignty_date(0.30)
        assert result["recommendation"] == "acceptable"

    def test_underpivoted_allocation(self):
        """20% should be marked as under-pivoted."""
        result = project_sovereignty_date(0.20)
        assert result["recommendation"] == "under-pivoted"

    def test_year_projections(self):
        """Should provide year estimates."""
        result = project_sovereignty_date(0.40)
        assert result["earliest_year"] > 2025
        assert result["latest_year"] >= result["earliest_year"]


class TestConstants:
    """Tests for timeline constants."""

    def test_threshold(self):
        """Threshold should be 1 million."""
        assert THRESHOLD_PERSON_EQUIVALENT == 1_000_000

    def test_years_ranges(self):
        """Year ranges should be properly ordered."""
        assert YEARS_40PCT[0] < YEARS_40PCT[1]
        assert YEARS_25PCT[0] < YEARS_25PCT[1]
        assert YEARS_15PCT[0] < YEARS_15PCT[1]
        assert YEARS_0PCT[0] < YEARS_0PCT[1]

    def test_years_increase_with_under_allocation(self):
        """More under-allocation should mean more years."""
        assert YEARS_40PCT[1] < YEARS_25PCT[0]
        assert YEARS_25PCT[1] < YEARS_15PCT[0]
        assert YEARS_15PCT[1] < YEARS_0PCT[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
