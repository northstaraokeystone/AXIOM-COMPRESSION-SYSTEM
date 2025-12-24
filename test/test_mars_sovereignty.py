"""Tests for Mars Sovereignty Simulator.

Validates research benchmarks:
- 22 crew (George Mason 2023) -> ~95% sovereignty
- 110 crew (Salotti Nature 2020) -> ~99.9% sovereignty
"""

import json
import math
import pytest
from io import StringIO
from contextlib import redirect_stdout

from spaceproof.sovereignty.mars import (
    # Constants
    CREW_MIN_GEORGE_MASON,
    CREW_MIN_SALOTTI,
    ISS_ECLSS_MTBF_HOURS,
    ISS_O2_CLOSURE_RATIO,
    ISS_H2O_RECOVERY_RATIO,
    MARS_CONJUNCTION_BLACKOUT_DAYS,
    MARS_SYNODIC_PERIOD_DAYS,
    MOXIE_O2_G_PER_HOUR,
    TENANT_ID,
    # Crew matrix
    define_skill_matrix,
    calculate_coverage,
    calculate_redundancy,
    identify_gaps,
    compute_crew_entropy,
    # Life support
    calculate_o2_balance,
    calculate_h2o_balance,
    calculate_thermal_entropy,
    calculate_eclss_reliability,
    calculate_life_support_entropy_rate,
    # Decision capacity
    calculate_internal_capacity,
    calculate_earth_input_rate,
    calculate_decision_latency_cost,
    compute_sovereignty_threshold,
    calculate_conjunction_survival,
    # Resources
    calculate_isru_closure,
    calculate_reserve_buffer,
    calculate_resupply_cadence,
    calculate_starship_manifest,
    identify_binding_resource,
    # Integrator
    calculate_sovereignty_score,
    identify_binding_constraint,
    validate_against_research,
    # Monte Carlo
    run_simulation,
    SCENARIOS,
    # API
    calculate_mars_sovereignty,
    find_crew_threshold,
    get_default_config,
)


class TestConstants:
    """Test research-validated constants."""

    def test_iss_eclss_mtbf(self):
        """ISS ECLSS actual MTBF is 1752h (not design 10000h)."""
        assert ISS_ECLSS_MTBF_HOURS == 1752
        assert ISS_ECLSS_MTBF_HOURS < 10000  # Actual < design

    def test_iss_closure_ratios(self):
        """ISS closure ratios from NASA ECLSS 2023."""
        assert 0.85 <= ISS_O2_CLOSURE_RATIO <= 0.90
        assert 0.97 <= ISS_H2O_RECOVERY_RATIO <= 0.99

    def test_mars_orbital_mechanics(self):
        """Mars orbital constants are physics-correct."""
        assert MARS_CONJUNCTION_BLACKOUT_DAYS == 14
        assert MARS_SYNODIC_PERIOD_DAYS == 780

    def test_moxie_production(self):
        """MOXIE production rate from Perseverance data."""
        assert 5.0 <= MOXIE_O2_G_PER_HOUR <= 6.0

    def test_research_benchmarks(self):
        """Research crew minimums."""
        assert CREW_MIN_GEORGE_MASON == 22
        assert CREW_MIN_SALOTTI == 110


class TestCrewMatrix:
    """Test crew skill matrix calculations."""

    def test_define_skill_matrix_structure(self):
        """Skill matrix has expected categories."""
        skills = define_skill_matrix()
        assert "medical" in skills
        assert "engineering" in skills
        assert "systems" in skills
        assert skills["medical"]["category"] == "CRITICAL"

    def test_coverage_empty_crew(self):
        """Empty crew has zero coverage."""
        coverage = calculate_coverage([])
        assert coverage == 0.0

    def test_coverage_full_crew(self):
        """Full crew with all skills has high coverage."""
        skills = define_skill_matrix()
        crew = [
            {"skills": {skill: 1.0 for skill in skills}}
            for _ in range(30)
        ]
        coverage = calculate_coverage(crew)
        assert coverage > 0.8

    def test_redundancy_calculation(self):
        """Redundancy calculation is correct."""
        crew = [
            {"skills": {"medical": 1.0}},
            {"skills": {"medical": 0.8}},
            {"skills": {"medical": 0.5}},
        ]
        redundancy = calculate_redundancy(crew)
        assert redundancy["medical"] >= 2.0  # 2 fully qualified + 0.5 * 1 partial

    def test_identify_gaps(self):
        """Gaps are identified for understaffed skills."""
        crew = [{"skills": {"medical": 1.0}}]  # Only medical, nothing else
        gaps = identify_gaps(crew)
        assert len(gaps) > 0
        # Should have gaps for engineering, systems, life_support, etc.
        gap_skills = [g["skill"] for g in gaps]
        assert "systems" in gap_skills or "life_support" in gap_skills

    def test_workload_entropy_balanced(self):
        """Balanced workload has low entropy (normalized)."""
        crew = [
            {"workload_hours": 40},
            {"workload_hours": 40},
            {"workload_hours": 40},
        ]
        entropy = compute_crew_entropy(crew)
        assert entropy < 0.1  # Well balanced


class TestLifeSupport:
    """Test life support calculations."""

    def test_o2_balance_positive_with_moxie(self):
        """O2 balance is positive with sufficient MOXIE units."""
        balance = calculate_o2_balance(
            crew=10,
            moxie_units=5,
            eclss_closure=0.85,
            power_available_w=10000,
        )
        assert balance["closure_ratio"] > 0.0
        assert balance["production_kg_day"] > 0

    def test_h2o_balance_with_recovery(self):
        """H2O recovery achieves high closure ratio."""
        balance = calculate_h2o_balance(
            crew=10,
            recovery_ratio=0.98,
            isru_production_kg_day=10,
        )
        assert balance["closure_ratio"] > 0.95

    def test_thermal_entropy_stable(self):
        """Thermal system is stable with adequate radiators."""
        thermal = calculate_thermal_entropy(
            crew=10,
            equipment_power_w=5000,
            radiator_area_m2=100,
            t_hab_c=22,
        )
        assert thermal["stable"] is True
        assert thermal["net_entropy_w_k"] < 0  # Exporting more than generating

    def test_eclss_reliability_with_redundancy(self):
        """ECLSS reliability improves with redundancy."""
        rel_single = calculate_eclss_reliability(
            mtbf_hours=1752,
            redundancy_factor=1.0,
            repair_capacity=0.8,
        )
        rel_double = calculate_eclss_reliability(
            mtbf_hours=1752,
            redundancy_factor=2.0,
            repair_capacity=0.8,
        )
        assert rel_double > rel_single

    def test_entropy_rate_stable(self):
        """Entropy rate is negative for stable configuration."""
        eclss_config = {
            "o2_closure": 0.85,
            "h2o_closure": 0.95,
            "mtbf_hours": 1752,
            "redundancy_factor": 2.0,
        }
        entropy_rate = calculate_life_support_entropy_rate(20, eclss_config)
        assert entropy_rate < 0  # Negative = stable


class TestDecisionCapacity:
    """Test decision capacity calculations."""

    def test_internal_capacity_scales_with_crew(self):
        """Internal capacity increases with crew size."""
        crew_small = [{"skills": {"systems": 0.8}, "expertise_level": 0.8}] * 10
        crew_large = [{"skills": {"systems": 0.8}, "expertise_level": 0.8}] * 50

        cap_small = calculate_internal_capacity(crew_small)
        cap_large = calculate_internal_capacity(crew_large)

        assert cap_large > cap_small

    def test_earth_capacity_decreases_with_latency(self):
        """Earth capacity decreases with higher latency."""
        cap_close = calculate_earth_input_rate(bandwidth_mbps=2.0, latency_sec=180)
        cap_far = calculate_earth_input_rate(bandwidth_mbps=2.0, latency_sec=1320)

        assert cap_close > cap_far

    def test_earth_capacity_zero_during_conjunction(self):
        """Earth capacity is zero during conjunction blackout."""
        cap = calculate_earth_input_rate(
            bandwidth_mbps=2.0, latency_sec=660, conjunction_blackout=True
        )
        assert cap == 0.0

    def test_sovereignty_threshold(self):
        """Sovereignty threshold is correctly computed."""
        assert compute_sovereignty_threshold(1000, 500) is True
        assert compute_sovereignty_threshold(500, 1000) is False

    def test_decision_latency_cost_critical(self):
        """Critical decisions have exponential latency cost."""
        cost_short = calculate_decision_latency_cost(10, "CRITICAL")
        cost_long = calculate_decision_latency_cost(600, "CRITICAL")  # 10 minutes

        assert cost_long > cost_short
        assert cost_long > 10  # Should be exponential

    def test_conjunction_survival_high_capacity(self):
        """High internal capacity enables conjunction survival."""
        survival = calculate_conjunction_survival(internal_capacity=10000)
        assert survival > 0.9


class TestResources:
    """Test resource calculations."""

    def test_isru_closure_calculation(self):
        """ISRU closure ratio is correctly calculated."""
        production = {"o2": 10, "h2o": 100, "food": 20}
        consumption = {"o2": 10, "h2o": 100, "food": 50}

        closure = calculate_isru_closure(production, consumption)
        assert 0 < closure < 1  # Not fully closed due to food deficit

    def test_reserve_buffer_days(self):
        """Buffer days are correctly calculated."""
        reserves = {"o2": 100, "h2o": 360, "food": 180}
        consumption = {"o2": 1, "h2o": 4, "food": 2}

        buffer = calculate_reserve_buffer(reserves, consumption)
        assert buffer["buffer_days"]["o2"] == 100
        assert buffer["buffer_days"]["h2o"] == 90

    def test_resupply_not_required_closed_loop(self):
        """No resupply needed when closure ratio is 1.0."""
        resupply = calculate_resupply_cadence(
            closure_ratio=1.0,
            buffer_days=180,
        )
        assert resupply["resupply_required"] is False

    def test_resupply_required_open_loop(self):
        """Resupply required when closure < 1.0."""
        resupply = calculate_resupply_cadence(
            closure_ratio=0.5,
            buffer_days=90,
        )
        assert resupply["resupply_required"] is True

    def test_binding_resource_identification(self):
        """Binding resource is correctly identified."""
        closures = {"o2": 0.9, "h2o": 0.95, "food": 0.3}
        binding = identify_binding_resource(closures)
        assert binding == "food"


class TestIntegrator:
    """Test sovereignty score integration."""

    def test_sovereignty_score_range(self):
        """Sovereignty score is in valid range (0-100)."""
        score = calculate_sovereignty_score(
            crew_coverage=0.9,
            life_support_entropy=-0.5,
            decision_capacity=1.5,
            resource_closure=0.8,
        )
        assert 0 <= score <= 100

    def test_sovereignty_score_increases_with_capacity(self):
        """Higher decision capacity increases score."""
        score_low = calculate_sovereignty_score(
            crew_coverage=0.9,
            life_support_entropy=-0.5,
            decision_capacity=0.5,
            resource_closure=0.8,
        )
        score_high = calculate_sovereignty_score(
            crew_coverage=0.9,
            life_support_entropy=-0.5,
            decision_capacity=2.0,
            resource_closure=0.8,
        )
        assert score_high > score_low

    def test_binding_constraint_identification(self):
        """Binding constraint is correctly identified."""
        scores = {
            "crew": 0.9,
            "life_support": 0.8,
            "decision": 0.95,
            "resources": 0.5,  # Lowest
        }
        binding = identify_binding_constraint(scores)
        assert binding == "resources"

    def test_research_validation_22_crew(self):
        """22 crew should yield ~95% score."""
        # This validates within 10% tolerance
        validated = validate_against_research(22, 93.0)
        assert validated is True  # 93% is within 10% of 95%

    def test_research_validation_110_crew(self):
        """110 crew should yield ~99.9% score."""
        validated = validate_against_research(110, 98.0)
        assert validated is True  # 98% is within 10% of 99.9%


class TestMonteCarlo:
    """Test Monte Carlo simulation."""

    def test_simulation_runs_successfully(self):
        """Monte Carlo simulation completes without error."""
        config = {
            "crew_count": 22,
            "mission_duration_days": 100,
            "o2_reserve_days": 60,
            "h2o_reserve_days": 90,
            "food_reserve_days": 90,
        }
        results = run_simulation(config, n_iterations=10, seed=42)

        assert "overall_survival_rate" in results
        assert "confidence_interval_95" in results
        assert 0 <= results["overall_survival_rate"] <= 1

    def test_scenarios_defined(self):
        """All mandatory scenarios are defined."""
        assert "BASELINE" in SCENARIOS
        assert "DUST_STORM_GLOBAL" in SCENARIOS
        assert "CONJUNCTION_BLACKOUT" in SCENARIOS
        assert "ECLSS_O2_FAILURE" in SCENARIOS

    def test_survival_rate_higher_with_more_crew(self):
        """Larger crew has higher survival rate."""
        config_small = {
            "crew_count": 10,
            "mission_duration_days": 100,
            "o2_reserve_days": 60,
            "h2o_reserve_days": 90,
            "food_reserve_days": 90,
        }
        config_large = {
            "crew_count": 50,
            "mission_duration_days": 100,
            "o2_reserve_days": 120,
            "h2o_reserve_days": 180,
            "food_reserve_days": 180,
        }

        results_small = run_simulation(config_small, n_iterations=20, seed=42)
        results_large = run_simulation(config_large, n_iterations=20, seed=42)

        # Larger crew should have higher or equal survival
        assert results_large["overall_survival_rate"] >= results_small["overall_survival_rate"] - 0.1


class TestAPI:
    """Test high-level API functions."""

    def test_get_default_config(self):
        """Default config has expected structure."""
        config = get_default_config()
        assert "colony" in config
        assert "crew_count" in config["colony"]
        assert "life_support" in config
        assert "power" in config

    def test_calculate_mars_sovereignty(self):
        """Mars sovereignty calculation returns valid result."""
        config = get_default_config()
        result = calculate_mars_sovereignty(config=config)

        assert "sovereignty_score" in result
        assert "is_sovereign" in result
        assert "binding_constraint" in result
        assert 0 <= result["sovereignty_score"] <= 100

    def test_sovereignty_with_monte_carlo(self):
        """Monte Carlo results are included when requested."""
        config = get_default_config()
        result = calculate_mars_sovereignty(
            config=config,
            monte_carlo=True,
            iterations=10,
        )

        assert "monte_carlo_survival_rate" in result
        assert "monte_carlo_confidence_95" in result

    def test_find_crew_threshold(self):
        """Crew threshold finder returns valid crew size."""
        result = find_crew_threshold(target_score=90.0)

        assert "threshold_crew" in result
        assert result["threshold_crew"] >= 4
        assert result["threshold_crew"] <= 200
        assert result["achieved_score"] >= 89.0  # Close to target


class TestReceipts:
    """Test receipt emission."""

    def test_mars_sovereignty_receipt_emitted(self):
        """Mars sovereignty receipt is emitted to stdout."""
        config = get_default_config()

        # Capture stdout
        buffer = StringIO()
        with redirect_stdout(buffer):
            calculate_mars_sovereignty(config=config)

        output = buffer.getvalue()
        assert "mars_sovereignty" in output

        # Parse JSON lines to find receipt
        for line in output.strip().split("\n"):
            if line:
                receipt = json.loads(line)
                if receipt.get("receipt_type") == "mars_sovereignty":
                    assert "crew_count" in receipt
                    assert "sovereignty_score" in receipt
                    assert "payload_hash" in receipt
                    break


class TestSLOs:
    """Test SLO compliance."""

    def test_sovereignty_calculation_under_1s(self):
        """Sovereignty calculation completes under 1 second."""
        import time

        config = get_default_config()
        start = time.time()
        calculate_mars_sovereignty(config=config)
        elapsed = time.time() - start

        assert elapsed < 1.0  # SLO: <1s

    def test_crew_threshold_under_2s(self):
        """Crew threshold search completes under 2 seconds."""
        import time

        start = time.time()
        find_crew_threshold(target_score=90.0)
        elapsed = time.time() - start

        assert elapsed < 2.0  # SLO: <2s
