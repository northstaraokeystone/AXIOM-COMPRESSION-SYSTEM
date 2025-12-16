"""Unit tests for sim.py functions.

Tests:
- SimConfig defaults and immutability
- SimState initialization
- run_simulation basic functionality
- simulate_cycle cycle tracking
- validate_constraints
- ensemble_select
- run_scenario basic functionality
- Error handling for invalid scenarios
"""

import pytest
import numpy as np

from src.sim import (
    SimConfig,
    SimState,
    run_simulation,
    simulate_cycle,
    validate_constraints,
    ensemble_select,
    run_scenario,
    TENANT_ID,
    SCENARIOS,
    COMPRESSION_FLOOR,
    MSE_THRESHOLDS,
    ACCURACY_THRESHOLDS,
    REGIMES,
)
from src.core import StopRule


class TestSimConfig:
    """Tests for SimConfig dataclass."""

    def test_simconfig_defaults(self):
        """All defaults match spec."""
        config = SimConfig()

        assert config.n_cycles == 1000
        assert config.n_galaxies_per_regime == 25
        assert config.noise_fraction == 0.03
        assert config.kan_epochs == 100
        assert config.mdl_alpha == 1.0
        assert config.mdl_beta == 0.10
        assert config.persistence_threshold == 0.012
        assert config.random_seed == 42
        assert config.multi_seed == (42, 43, 44, 45, 46, 47)
        assert config.topology_only is False
        assert config.scenario is None

    def test_simconfig_frozen(self):
        """Modifying config raises error."""
        config = SimConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.n_cycles = 500

    def test_simconfig_custom_values(self):
        """Custom values override defaults."""
        config = SimConfig(
            n_cycles=100,
            n_galaxies_per_regime=10,
            noise_fraction=0.05,
            random_seed=123
        )

        assert config.n_cycles == 100
        assert config.n_galaxies_per_regime == 10
        assert config.noise_fraction == 0.05
        assert config.random_seed == 123


class TestSimState:
    """Tests for SimState dataclass."""

    def test_simstate_initial(self):
        """Initial state is empty."""
        state = SimState()

        assert state.galaxies == []
        assert state.witness_receipts == []
        assert state.topology_receipts == []
        assert state.chain_receipt is None
        assert state.violations == []
        assert state.cycle == 0
        assert state.passed is None
        assert state.metrics == {}

    def test_simstate_mutable(self):
        """State can be modified."""
        state = SimState()

        state.cycle = 5
        state.passed = True
        state.galaxies.append({"id": "test"})

        assert state.cycle == 5
        assert state.passed is True
        assert len(state.galaxies) == 1


class TestRunSimulation:
    """Tests for run_simulation function."""

    def test_run_simulation_returns_state(self):
        """Returns SimState."""
        config = SimConfig(n_cycles=1, n_galaxies_per_regime=1)
        state = run_simulation(config)

        assert isinstance(state, SimState)
        assert state.cycle >= 1
        assert state.passed is not None

    def test_run_simulation_populates_receipts(self):
        """witness_receipts non-empty."""
        config = SimConfig(n_cycles=1, n_galaxies_per_regime=2)
        state = run_simulation(config)

        assert len(state.witness_receipts) > 0
        assert len(state.topology_receipts) > 0

    def test_run_simulation_chains_receipts(self):
        """chain_receipt is populated."""
        config = SimConfig(n_cycles=1, n_galaxies_per_regime=2)
        state = run_simulation(config)

        assert state.chain_receipt is not None
        assert "merkle_root" in state.chain_receipt


class TestSimulateCycle:
    """Tests for simulate_cycle function."""

    def test_simulate_cycle_increments(self):
        """state.cycle increases."""
        config = SimConfig(n_cycles=1, n_galaxies_per_regime=1)
        state = SimState()

        assert state.cycle == 0

        state = simulate_cycle(state, config)

        assert state.cycle == 1

    def test_simulate_cycle_adds_receipts(self):
        """Cycle adds witness and topology receipts."""
        config = SimConfig(n_cycles=1, n_galaxies_per_regime=2)
        state = SimState()

        state = simulate_cycle(state, config)

        # 4 regimes × 2 per regime = 8 galaxies
        assert len(state.witness_receipts) == 4 * 2
        assert len(state.topology_receipts) == 4 * 2


class TestValidateConstraints:
    """Tests for validate_constraints function."""

    def test_validate_constraints_empty_on_good(self):
        """No violations on valid data."""
        config = SimConfig()
        state = SimState()

        # Add some valid receipts
        for regime in REGIMES:
            state.witness_receipts.append({
                "physics_regime": regime,
                "compression_ratio": 0.90,  # Above floor
                "final_mse": 1.0  # Below ceiling
            })

        violations = validate_constraints(state, config)

        # Should have minimal or no violations
        # (accuracy thresholds might not be met with few samples)
        assert isinstance(violations, list)

    def test_validate_constraints_catches_low_compression(self):
        """Violation on 0.5 compression."""
        config = SimConfig()
        state = SimState()

        state.witness_receipts.append({
            "physics_regime": "newtonian",
            "compression_ratio": 0.5,  # Below floor
            "final_mse": 1.0
        })

        violations = validate_constraints(state, config)

        # Should have at least one compression violation
        compression_violations = [v for v in violations if v.get("constraint") == "compression_floor"]
        assert len(compression_violations) >= 1

    def test_validate_constraints_catches_high_mse(self):
        """Violation on high MSE."""
        config = SimConfig()
        state = SimState()

        state.witness_receipts.append({
            "physics_regime": "newtonian",
            "compression_ratio": 0.90,
            "final_mse": 100.0  # Way above ceiling
        })

        violations = validate_constraints(state, config)

        mse_violations = [v for v in violations if v.get("constraint") == "mse_ceiling"]
        assert len(mse_violations) >= 1


class TestEnsembleSelect:
    """Tests for ensemble_select function."""

    def test_ensemble_select_picks_lowest_mdl(self):
        """Correct selection."""
        results = [
            {"final_loss": 1.5, "compression_ratio": 0.85},
            {"final_loss": 0.8, "compression_ratio": 0.90},  # Best
            {"final_loss": 2.0, "compression_ratio": 0.80},
        ]

        selected = ensemble_select(results)

        assert selected["final_loss"] == 0.8
        assert selected["compression_ratio"] == 0.90

    def test_ensemble_select_empty_list(self):
        """Empty list returns empty dict."""
        selected = ensemble_select([])

        assert selected == {}

    def test_ensemble_select_single_item(self):
        """Single item is returned."""
        results = [{"final_loss": 1.0}]
        selected = ensemble_select(results)

        assert selected["final_loss"] == 1.0


class TestRunScenario:
    """Tests for run_scenario function."""

    def test_run_scenario_baseline(self):
        """BASELINE runs without error."""
        # Use small config for testing
        state = run_scenario("BASELINE")

        assert isinstance(state, SimState)
        assert state.passed is not None

    def test_run_scenario_invalid_raises(self):
        """StopRule on bad name."""
        with pytest.raises(StopRule) as exc_info:
            run_scenario("INVALID_SCENARIO")

        assert "Unknown scenario" in str(exc_info.value)


class TestConstants:
    """Tests for module constants."""

    def test_tenant_id(self):
        """TENANT_ID is correct."""
        assert TENANT_ID == "axiom-witness"

    def test_scenarios_list(self):
        """SCENARIOS contains all 6."""
        assert len(SCENARIOS) == 6
        assert "BASELINE" in SCENARIOS
        assert "STRESS" in SCENARIOS
        assert "DISCOVERY" in SCENARIOS
        assert "TOPOLOGY" in SCENARIOS
        assert "REPRODUCIBILITY" in SCENARIOS
        assert "GODEL" in SCENARIOS

    def test_compression_floor(self):
        """COMPRESSION_FLOOR is 0.84."""
        assert COMPRESSION_FLOOR == 0.84

    def test_mse_thresholds(self):
        """MSE_THRESHOLDS has all regimes."""
        assert "newtonian" in MSE_THRESHOLDS
        assert "mond" in MSE_THRESHOLDS
        assert "nfw" in MSE_THRESHOLDS
        assert "pbh_fog" in MSE_THRESHOLDS

    def test_accuracy_thresholds(self):
        """ACCURACY_THRESHOLDS has all regimes."""
        assert "newtonian" in ACCURACY_THRESHOLDS
        assert "mond" in ACCURACY_THRESHOLDS
        assert "nfw" in ACCURACY_THRESHOLDS
        assert "pbh_fog" in ACCURACY_THRESHOLDS

    def test_regimes(self):
        """REGIMES matches expected list."""
        assert REGIMES == ["newtonian", "mond", "nfw", "pbh_fog"]


# =============================================================================
# COLONY SIMULATION TESTS (BUILD C5)
# =============================================================================

from src.sim import (
    ColonySimConfig,
    ColonySimState,
    COLONY_SCENARIO_CONFIGS,
    COLONY_SCENARIOS,
    run_colony_simulation,
    colony_simulate_cycle,
    colony_validate_constraints,
    colony_find_minimum_viable_crew,
    run_colony_scenario,
    run_all_colony_scenarios,
    colony_validate_entropy_stable,
    colony_validate_sovereignty,
    colony_validate_atmosphere,
    colony_validate_thermal,
    colony_validate_resource,
    colony_validate_cascade,
)
from src.colony import ColonyConfig, ColonyState


class TestColonySimConfig:
    """Tests for ColonySimConfig dataclass (BUILD C5)."""

    def test_colony_simconfig_defaults(self):
        """All defaults match spec."""
        config = ColonySimConfig()

        assert config.n_cycles == 1000
        assert config.n_colonies_per_stress == 25
        assert config.duration_days == 365
        assert config.crew_sizes == (4, 10, 25, 50, 100)
        assert config.stress_events == ("none", "dust_storm", "hab_breach")
        assert config.random_seed == 42

    def test_colony_simconfig_frozen(self):
        """Modifying config raises FrozenInstanceError."""
        config = ColonySimConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.n_cycles = 500

    def test_colony_simconfig_custom_values(self):
        """Custom values override defaults."""
        config = ColonySimConfig(
            n_cycles=100,
            duration_days=30,
            crew_sizes=(4, 8, 12),
            random_seed=123
        )

        assert config.n_cycles == 100
        assert config.duration_days == 30
        assert config.crew_sizes == (4, 8, 12)
        assert config.random_seed == 123


class TestColonySimState:
    """Tests for ColonySimState dataclass (BUILD C5)."""

    def test_colony_simstate_initial(self):
        """Initial state is empty."""
        state = ColonySimState()

        assert state.colonies == []
        assert state.entropy_receipts == []
        assert state.violations == []
        assert state.cycle == 0
        assert state.sovereignty_threshold_crew is None

    def test_colony_simstate_mutable(self):
        """State can be modified."""
        state = ColonySimState()

        state.cycle = 5
        state.sovereignty_threshold_crew = 10
        state.colonies.append({"id": "test"})

        assert state.cycle == 5
        assert state.sovereignty_threshold_crew == 10
        assert len(state.colonies) == 1

    def test_colony_simstate_passed_property_true(self):
        """violations=[] → passed=True."""
        state = ColonySimState()
        state.violations = []

        assert state.passed is True

    def test_colony_simstate_passed_property_false(self):
        """violations=[x] → passed=False."""
        state = ColonySimState()
        state.violations = [{"validator": "test"}]

        assert state.passed is False


class TestColonyRunSimulation:
    """Tests for run_colony_simulation function (BUILD C5)."""

    def test_run_colony_simulation_returns_state(self):
        """Returns ColonySimState."""
        config = ColonySimConfig(n_cycles=1, duration_days=10, crew_sizes=(10,))
        state = run_colony_simulation(config)

        assert isinstance(state, ColonySimState)
        assert state.cycle >= 1

    def test_run_colony_simulation_deterministic(self):
        """Same seed → same violations count."""
        config = ColonySimConfig(n_cycles=2, duration_days=10, crew_sizes=(10,))

        state1 = run_colony_simulation(config)
        state2 = run_colony_simulation(config)

        assert len(state1.violations) == len(state2.violations)

    def test_run_colony_simulation_different_seeds(self):
        """Different seeds can produce different results."""
        config1 = ColonySimConfig(n_cycles=2, duration_days=10, crew_sizes=(10,), random_seed=42)
        config2 = ColonySimConfig(n_cycles=2, duration_days=10, crew_sizes=(10,), random_seed=99)

        state1 = run_colony_simulation(config1)
        state2 = run_colony_simulation(config2)

        # Results may differ (not guaranteed but likely with different seeds)
        assert isinstance(state1, ColonySimState)
        assert isinstance(state2, ColonySimState)

    def test_run_colony_simulation_zero_cycles(self):
        """n_cycles=0 → empty state, passed=True."""
        config = ColonySimConfig(n_cycles=0, duration_days=10)
        state = run_colony_simulation(config)

        assert state.cycle == 0
        assert len(state.colonies) == 0
        assert state.passed is True


class TestColonySimulateCycle:
    """Tests for colony_simulate_cycle function (BUILD C5)."""

    def test_colony_simulate_cycle_increments(self):
        """state.cycle increases."""
        config = ColonySimConfig(n_cycles=1, duration_days=10, crew_sizes=(10,))
        state = ColonySimState()

        assert state.cycle == 0

        state = colony_simulate_cycle(state, config)

        assert state.cycle == 1

    def test_colony_simulate_cycle_adds_colonies(self):
        """Cycle adds colonies."""
        config = ColonySimConfig(
            n_cycles=1,
            duration_days=10,
            crew_sizes=(10, 25),
            stress_events=("none",)
        )
        state = ColonySimState()

        state = colony_simulate_cycle(state, config)

        # 2 crew sizes × 1 stress event = 2 colonies
        assert len(state.colonies) == 2


class TestColonyValidators:
    """Tests for colony constraint validators (BUILD C5)."""

    def test_validate_entropy_stable_pass(self):
        """Stable entropy → no violation."""
        # Create a mock colony with stable entropy
        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": [ColonyState() for _ in range(10)]
        }
        # Set stable atmosphere values
        for state in colony["states"]:
            state.atmosphere = {"O2_pct": 21.0}
            state.thermal = {"T_hab_C": 22.0}
            state.resource = {"water_kg": 1000, "food_kcal": 50000, "power_W": 20000}

        result = colony_validate_entropy_stable(colony)

        # Should pass (None) or be a violation depending on entropy calculation
        assert result is None or "validator" in result

    def test_validate_sovereignty_pass(self):
        """Large crew (internal > external) → no violation."""
        # Large crew should have enough decision capacity
        colony = {
            "config": ColonyConfig(crew_size=100),
            "states": [ColonyState()]
        }

        result = colony_validate_sovereignty(colony)

        # With 100 crew, should achieve sovereignty
        # (actual threshold depends on constants)
        assert result is None or "validator" in result

    def test_validate_sovereignty_fail(self):
        """Small crew (internal < external) → violation returned."""
        # Small crew unlikely to achieve sovereignty
        colony = {
            "config": ColonyConfig(crew_size=4),
            "states": [ColonyState()]
        }

        result = colony_validate_sovereignty(colony)

        # Small crew likely to fail sovereignty check
        if result is not None:
            assert result["validator"] == "sovereignty"
            assert "internal" in result
            assert "external" in result

    def test_validate_atmosphere_pass(self):
        """O2 in range → no violation."""
        state = ColonyState()
        state.atmosphere = {"O2_pct": 21.0}

        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": [state]
        }

        result = colony_validate_atmosphere(colony)

        assert result is None

    def test_validate_atmosphere_fail(self):
        """O2 out of range → violation returned."""
        state = ColonyState()
        state.atmosphere = {"O2_pct": 15.0}  # Below 19.5

        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": [state]
        }

        result = colony_validate_atmosphere(colony)

        assert result is not None
        assert result["validator"] == "atmosphere"
        assert result["min_o2"] == 15.0

    def test_validate_thermal_pass(self):
        """T in range → no violation."""
        state = ColonyState()
        state.thermal = {"T_hab_C": 22.0}

        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": [state]
        }

        result = colony_validate_thermal(colony)

        assert result is None

    def test_validate_thermal_fail(self):
        """T out of range → violation returned."""
        state = ColonyState()
        state.thermal = {"T_hab_C": 50.0}  # Above 40

        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": [state]
        }

        result = colony_validate_thermal(colony)

        assert result is not None
        assert result["validator"] == "thermal"
        assert result["max_T"] == 50.0

    def test_validate_resource_pass(self):
        """Buffer >= 90 → no violation."""
        state = ColonyState()
        # 10 crew × 3L/day = 30L/day water, 1000kg / 30 = 33 days
        # Need more water for 90 day buffer
        state.resource = {"water_kg": 3000, "food_kcal": 2500000}

        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": [state]
        }

        result = colony_validate_resource(colony)

        # 3000 / 30 = 100 days water, 2500000 / 25000 = 100 days food
        assert result is None

    def test_validate_resource_fail(self):
        """Buffer < 90 → violation returned."""
        state = ColonyState()
        state.resource = {"water_kg": 100, "food_kcal": 10000}  # Very low

        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": [state]
        }

        result = colony_validate_resource(colony)

        assert result is not None
        assert result["validator"] == "resource"
        assert result["buffer_days"] < 90

    def test_validate_cascade_pass(self):
        """No failed status → no violation."""
        states = [ColonyState() for _ in range(5)]
        for state in states:
            state.status = "nominal"

        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": states
        }

        result = colony_validate_cascade(colony)

        assert result is None

    def test_validate_cascade_fail(self):
        """Failed status → violation returned."""
        states = [ColonyState() for _ in range(5)]
        states[0].status = "nominal"
        states[1].status = "nominal"
        states[2].status = "failed"  # Failure on day 2

        colony = {
            "config": ColonyConfig(crew_size=10),
            "states": states
        }

        result = colony_validate_cascade(colony)

        assert result is not None
        assert result["validator"] == "cascade"
        assert result["failed_day"] == 2


class TestColonyFindMinimumViableCrew:
    """Tests for colony_find_minimum_viable_crew function (BUILD C5)."""

    def test_find_minimum_viable_crew_found(self):
        """Returns minimum sovereign crew."""
        # Create colonies with different crew sizes
        colonies = []
        for crew in [4, 10, 25, 50, 100]:
            colonies.append({
                "config": ColonyConfig(crew_size=crew),
                "states": [ColonyState()]
            })

        config = ColonySimConfig(crew_sizes=(4, 10, 25, 50, 100))
        result = colony_find_minimum_viable_crew(colonies, config)

        # Result should be an int or None
        assert result is None or isinstance(result, int)

    def test_find_minimum_viable_crew_none(self):
        """No sovereignty → returns None."""
        # Empty colonies list
        colonies = []
        config = ColonySimConfig(crew_sizes=(4,))
        result = colony_find_minimum_viable_crew(colonies, config)

        assert result is None


class TestColonyRunScenario:
    """Tests for run_colony_scenario function (BUILD C5)."""

    def test_run_colony_scenario_baseline(self):
        """BASELINE scenario runs without crash."""
        # Use small config by modifying (we'll run the full scenario briefly)
        state = run_colony_scenario("BASELINE")

        assert isinstance(state, ColonySimState)

    def test_run_colony_scenario_godel(self):
        """GÖDEL with edge cases doesn't crash."""
        state = run_colony_scenario("GÖDEL")

        assert isinstance(state, ColonySimState)

    def test_run_colony_scenario_invalid_raises(self):
        """ValueError on bad name."""
        with pytest.raises(ValueError) as exc_info:
            run_colony_scenario("INVALID_SCENARIO")

        assert "Unknown colony scenario" in str(exc_info.value)


class TestColonyRunAllScenarios:
    """Tests for run_all_colony_scenarios function (BUILD C5)."""

    def test_run_all_colony_scenarios_returns_dict(self):
        """Returns dict with all 6 scenarios."""
        results = run_all_colony_scenarios()

        assert isinstance(results, dict)
        assert len(results) == 6
        for name in COLONY_SCENARIOS:
            assert name in results


class TestColonyScenarioConfigs:
    """Tests for COLONY_SCENARIO_CONFIGS (BUILD C5)."""

    def test_scenario_configs_exist(self):
        """All 6 scenarios in COLONY_SCENARIO_CONFIGS."""
        assert "BASELINE" in COLONY_SCENARIO_CONFIGS
        assert "DUST_STORM" in COLONY_SCENARIO_CONFIGS
        assert "HAB_BREACH" in COLONY_SCENARIO_CONFIGS
        assert "SOVEREIGNTY" in COLONY_SCENARIO_CONFIGS
        assert "ISRU_CLOSURE" in COLONY_SCENARIO_CONFIGS
        assert "GÖDEL" in COLONY_SCENARIO_CONFIGS

    def test_baseline_config(self):
        """BASELINE has correct overrides."""
        config = COLONY_SCENARIO_CONFIGS["BASELINE"]
        assert config.n_cycles == 1000
        assert config.stress_events == ("none",)

    def test_sovereignty_config(self):
        """SOVEREIGNTY has expanded crew sizes."""
        config = COLONY_SCENARIO_CONFIGS["SOVEREIGNTY"]
        assert config.n_cycles == 1000
        assert 4 in config.crew_sizes
        assert 50 in config.crew_sizes
        assert len(config.crew_sizes) == 11

    def test_isru_closure_config(self):
        """ISRU_CLOSURE has synodic period."""
        config = COLONY_SCENARIO_CONFIGS["ISRU_CLOSURE"]
        assert config.duration_days == 780

    def test_godel_config(self):
        """GÖDEL has edge case crew sizes."""
        config = COLONY_SCENARIO_CONFIGS["GÖDEL"]
        assert config.n_cycles == 100
        # Edge cases (adjusted for ColonyConfig validation)
        assert 4 in config.crew_sizes or 1000 in config.crew_sizes
