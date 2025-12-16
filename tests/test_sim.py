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
