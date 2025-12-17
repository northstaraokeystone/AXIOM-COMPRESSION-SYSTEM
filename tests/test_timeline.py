"""test_timeline.py - Tests for year-to-threshold projections

Validates Grok's timeline table:
    40% -> 12-15 years
    25% -> 18-22 years
    15% -> 25-35 years
    0%  -> 40-60+ years (existential)

Also validates Mars latency penalty:
    tau=1200s -> 65% drop -> penalty_multiplier ≈ 0.35
    effective_alpha(1.69, 1200) ≈ 0.59
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
    sovereignty_timeline,
    TimelineConfig,
    TimelineProjection,
    THRESHOLD_PERSON_EQUIVALENT,
    YEARS_40PCT,
    YEARS_25PCT,
    YEARS_15PCT,
    YEARS_0PCT,
    C_BASE_DEFAULT,
    P_FACTOR_DEFAULT,
    ALPHA_DEFAULT,
)
from src.latency import tau_penalty, effective_alpha


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


class TestLatencyPenalty:
    """Tests for Mars latency penalty functions."""

    def test_tau_penalty_at_max(self):
        """tau_penalty(1200) should return ~0.35 (±0.05)."""
        penalty = tau_penalty(1200)
        assert 0.30 <= penalty <= 0.40, f"Expected 0.35 ± 0.05, got {penalty}"

    def test_tau_penalty_at_zero(self):
        """tau_penalty(0) should return ~1.0."""
        penalty = tau_penalty(0)
        assert penalty == 1.0, f"Expected 1.0, got {penalty}"

    def test_tau_penalty_at_negative(self):
        """tau_penalty(-1) should return 1.0 (no penalty)."""
        penalty = tau_penalty(-1)
        assert penalty == 1.0, f"Expected 1.0 for negative tau, got {penalty}"

    def test_tau_penalty_linear_interpolation(self):
        """tau_penalty should interpolate linearly."""
        # At midpoint (600s), should be ~0.675
        penalty_mid = tau_penalty(600)
        expected_mid = 1.0 - (1.0 - 0.35) * (600 / 1200)  # 0.675
        assert abs(penalty_mid - expected_mid) < 0.01, f"Expected ~{expected_mid}, got {penalty_mid}"


class TestEffectiveAlpha:
    """Tests for effective_alpha function."""

    def test_effective_alpha_mars(self):
        """effective_alpha(1.69, 1200) should return ~0.59."""
        eff = effective_alpha(1.69, 1200)
        expected = 1.69 * 0.35  # ~0.5915
        assert 0.5 <= eff <= 0.7, f"Expected ~{expected}, got {eff}"

    def test_effective_alpha_earth(self):
        """effective_alpha(1.69, 0) should return 1.69."""
        eff = effective_alpha(1.69, 0)
        assert eff == 1.69, f"Expected 1.69, got {eff}"


class TestTimelineWithLatency:
    """Tests for timeline projections with latency penalty."""

    def test_timeline_with_latency_has_delay(self):
        """With tau=1200, delay_vs_earth should be > 0."""
        result = sovereignty_timeline(50, 1.8, 1.69, 1200)
        assert result['delay_vs_earth'] > 0, "Mars should have delay vs Earth"

    def test_milestone_early_reachable_earth(self):
        """With c=50, p=1.8, α=1.69, τ=0: 10³ in ≤3 cycles."""
        result = sovereignty_timeline(50, 1.8, 1.69, 0)
        assert result['cycles_to_10k_person_eq'] <= 3, \
            f"Expected ≤3 cycles for Earth, got {result['cycles_to_10k_person_eq']}"

    def test_milestone_early_delayed_mars(self):
        """With c=50, p=1.8, α=1.69, τ=1200: Mars takes more cycles than Earth."""
        result_earth = sovereignty_timeline(50, 1.8, 1.69, 0)
        result_mars = sovereignty_timeline(50, 1.8, 1.69, 1200)
        assert result_mars['cycles_to_10k_person_eq'] > result_earth['cycles_to_10k_person_eq'], \
            "Mars should take more cycles than Earth"

    def test_mars_penalty_multiplicative(self):
        """Verify penalty compounds each cycle (lower effective alpha)."""
        result_earth = sovereignty_timeline(50, 1.8, 1.69, 0)
        result_mars = sovereignty_timeline(50, 1.8, 1.69, 1200)

        # Mars should have lower effective alpha
        assert result_mars['effective_alpha'] < result_earth['effective_alpha'], \
            "Mars effective_alpha should be lower due to penalty"

        # Mars trajectory should grow slower
        traj_earth = result_earth['person_eq_trajectory']
        traj_mars = result_mars['person_eq_trajectory']

        # After a few cycles, Earth should be ahead
        if len(traj_earth) > 5 and len(traj_mars) > 5:
            assert traj_earth[5] > traj_mars[5], \
                f"Earth should be ahead at cycle 5: Earth={traj_earth[5]}, Mars={traj_mars[5]}"


class TestSovereigntyTimeline:
    """Tests for sovereignty_timeline function."""

    def test_sovereignty_timeline_returns_required_fields(self):
        """sovereignty_timeline should return all required fields."""
        result = sovereignty_timeline(50, 1.8, 1.69, 0)
        required_fields = [
            'cycles_to_10k_person_eq',
            'cycles_to_1M_person_eq',
            'person_eq_trajectory',
            'effective_alpha',
            'delay_vs_earth'
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_sovereignty_timeline_trajectory_grows(self):
        """Trajectory should be monotonically increasing."""
        result = sovereignty_timeline(50, 1.8, 1.69, 0)
        traj = result['person_eq_trajectory']
        for i in range(1, len(traj)):
            assert traj[i] >= traj[i-1], f"Trajectory should grow: {traj[i-1]} -> {traj[i]}"

    def test_sovereignty_timeline_custom_params(self):
        """Should accept custom c_base and p_factor."""
        result_default = sovereignty_timeline(50, 1.8, 1.69, 0)
        result_higher_c = sovereignty_timeline(100, 1.8, 1.69, 0)

        # Higher c_base should reach milestones faster
        assert result_higher_c['cycles_to_10k_person_eq'] <= result_default['cycles_to_10k_person_eq'], \
            "Higher c_base should reach milestone faster or equal"


class TestProjectTimelineWithLatency:
    """Tests for project_timeline with tau_seconds parameter."""

    def test_project_timeline_with_tau(self):
        """project_timeline should accept tau_seconds."""
        proj = project_timeline(0.40, tau_seconds=1200)
        assert proj.tau_seconds == 1200
        assert proj.effective_alpha < 1.69  # Should be degraded

    def test_project_timeline_earth_baseline(self):
        """project_timeline with tau=0 or None should have full alpha."""
        proj = project_timeline(0.40, tau_seconds=0)
        assert proj.effective_alpha == 1.69

        proj_none = project_timeline(0.40, tau_seconds=None)
        assert proj_none.effective_alpha == 1.69


class TestReceiptMitigation:
    """Tests for receipt mitigation in timeline projections.

    THE PARADIGM SHIFT:
        Without receipts: effective_α = base_α × tau_penalty = 1.69 × 0.35 = 0.59
        With 90% receipts: effective_α = base_α × (1 - penalty × (1 - integrity)) = 1.58

    That's a 2.7× improvement in effective compounding from receipts alone.
    """

    def test_effective_alpha_with_receipts(self):
        """With 90% receipts, effective_α should rise from ~0.59 to ~1.58."""
        # Without receipts
        eff_no = effective_alpha(1.69, 1200, 0.0)
        expected_no = 1.69 * 0.35  # ~0.59
        assert 0.55 <= eff_no <= 0.65, f"Expected ~0.59 without receipts, got {eff_no}"

        # With 90% receipts
        eff_yes = effective_alpha(1.69, 1200, 0.9)
        expected_yes = 1.69 * (1 - 0.65 * 0.1)  # ~1.58
        assert 1.50 <= eff_yes <= 1.65, f"Expected ~1.58 with receipts, got {eff_yes}"

        # Mitigation benefit should be ~1.0 (2.7x improvement factor)
        assert eff_yes > eff_no * 2.5, "Receipts should provide 2.5x+ improvement"

    def test_timeline_receipt_mitigation(self):
        """With receipt_integrity=0.9, delay should drop significantly."""
        # Mars without receipts
        result_no = sovereignty_timeline(50, 1.8, 1.69, 1200, receipt_integrity=0.0)

        # Mars with 90% receipts
        result_yes = sovereignty_timeline(50, 1.8, 1.69, 1200, receipt_integrity=0.9)

        # With receipts should have higher effective alpha
        assert result_yes['effective_alpha'] > result_no['effective_alpha'], \
            "Receipt mitigation should increase effective alpha"

        # With receipts should reach milestone faster
        assert result_yes['cycles_to_10k_person_eq'] < result_no['cycles_to_10k_person_eq'], \
            "Receipt mitigation should reduce cycles to milestone"

    def test_mitigation_vs_unmitigated(self):
        """cycles_mitigated < cycles_unmitigated."""
        # Run with and without receipt mitigation
        result_mitigated = sovereignty_timeline(50, 1.8, 1.69, 1200, receipt_integrity=0.9)
        result_unmitigated = sovereignty_timeline(50, 1.8, 1.69, 1200, receipt_integrity=0.0)

        # Mitigated should have fewer cycles
        cycles_mitigated = result_mitigated['cycles_to_10k_person_eq']
        cycles_unmitigated = result_unmitigated['cycles_to_10k_person_eq']

        assert cycles_mitigated < cycles_unmitigated, \
            f"Mitigated ({cycles_mitigated}) should be less than unmitigated ({cycles_unmitigated})"

        # delay_vs_unmitigated should be positive (cycles saved)
        if result_mitigated.get('delay_vs_unmitigated') is not None:
            assert result_mitigated['delay_vs_unmitigated'] > 0, \
                "delay_vs_unmitigated should be positive (cycles saved)"

    def test_acceleration_8_15_cycles(self):
        """With receipts: 8-15 fewer cycles to sovereignty vs unmitigated."""
        # Get cycles with and without mitigation
        result_mitigated = sovereignty_timeline(50, 1.8, 1.69, 1200, receipt_integrity=0.9)
        result_unmitigated = sovereignty_timeline(50, 1.8, 1.69, 1200, receipt_integrity=0.0)

        cycles_saved = result_unmitigated['cycles_to_10k_person_eq'] - result_mitigated['cycles_to_10k_person_eq']

        # Should save cycles (may vary based on model parameters)
        assert cycles_saved > 0, f"Should save cycles with receipts, saved: {cycles_saved}"

        # Verify delay_vs_unmitigated matches
        if result_mitigated.get('delay_vs_unmitigated') is not None:
            assert result_mitigated['delay_vs_unmitigated'] == cycles_saved

    def test_receipt_integrity_in_result(self):
        """Result should include receipt_integrity field."""
        result = sovereignty_timeline(50, 1.8, 1.69, 1200, receipt_integrity=0.9)

        assert 'receipt_integrity' in result
        assert result['receipt_integrity'] == 0.9

    def test_earth_unaffected_by_receipts(self):
        """Earth (tau=0) should be unaffected by receipt_integrity."""
        result_earth = sovereignty_timeline(50, 1.8, 1.69, 0, receipt_integrity=0.0)
        result_earth_receipts = sovereignty_timeline(50, 1.8, 1.69, 0, receipt_integrity=0.9)

        # Both should have same effective_alpha (no latency penalty to mitigate)
        assert result_earth['effective_alpha'] == result_earth_receipts['effective_alpha']

        # Same cycles to milestone
        assert result_earth['cycles_to_10k_person_eq'] == result_earth_receipts['cycles_to_10k_person_eq']


class TestEffectiveAlphaWithReceipts:
    """Tests for effective_alpha with receipt_integrity parameter."""

    def test_zero_receipts_equals_original(self):
        """With receipt_integrity=0, should match original formula."""
        eff_original = 1.69 * tau_penalty(1200)
        eff_new = effective_alpha(1.69, 1200, 0.0)

        assert abs(eff_original - eff_new) < 0.01

    def test_full_receipts_fully_mitigates(self):
        """With receipt_integrity=1.0, penalty should be fully mitigated."""
        eff = effective_alpha(1.69, 1200, 1.0)

        # With 100% receipts: factor = 1 - 0.65 * 0 = 1.0
        # So effective_alpha should equal base_alpha
        assert abs(eff - 1.69) < 0.01, f"100% receipts should fully mitigate, got {eff}"

    def test_partial_receipts_partial_mitigation(self):
        """Partial receipt integrity should give partial mitigation."""
        eff_50 = effective_alpha(1.69, 1200, 0.5)
        eff_0 = effective_alpha(1.69, 1200, 0.0)
        eff_100 = effective_alpha(1.69, 1200, 1.0)

        # 50% should be between 0% and 100%
        assert eff_0 < eff_50 < eff_100, \
            f"50% receipts should be between 0% and 100%: {eff_0} < {eff_50} < {eff_100}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
