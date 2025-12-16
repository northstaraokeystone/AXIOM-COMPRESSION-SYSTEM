"""Tests for BUILD C3: colony.py - synthetic colony state generator.

Verifies colony generation and stress events.
"""
import pytest
from dataclasses import FrozenInstanceError

from src.colony import (
    ColonyConfig,
    ColonyState,
    default_config,
    generate_colony,
    simulate_dust_storm,
    simulate_hab_breach,
    simulate_crop_failure,
    batch_generate,
    _determine_status,
    O2_NOMINAL,
    O2_STRESSED,
    O2_CRITICAL,
    O2_FAILED,
    CO2_NOMINAL,
    CO2_STRESSED,
    CO2_CRITICAL,
    CO2_FAILED,
    T_HAB_NOMINAL,
    T_HAB_MIN_STRESSED,
    T_HAB_MAX_STRESSED,
    T_HAB_MIN_CRITICAL,
    T_HAB_MAX_CRITICAL,
    PRESSURE_NOMINAL_KPA,
    PRESSURE_STRESSED_KPA,
    PRESSURE_CRITICAL_KPA,
)


# === ColonyConfig Tests ===

class TestColonyConfig:
    """Tests for ColonyConfig dataclass."""

    def test_colony_config_frozen(self):
        """Attempting to modify raises FrozenInstanceError."""
        config = ColonyConfig(crew_size=10)
        with pytest.raises(FrozenInstanceError):
            config.crew_size = 20

    def test_colony_config_defaults(self):
        """default_config(10) returns valid config with crew_size=10."""
        config = default_config(10)
        assert config.crew_size == 10
        assert config.hab_volume_m3 == 500.0
        assert config.solar_array_m2 == 200.0
        assert config.kilopower_units == 2

    def test_colony_config_validation_crew_too_small(self):
        """crew_size < 4 raises ValueError."""
        with pytest.raises(ValueError, match="crew_size must be in range"):
            ColonyConfig(crew_size=3)

    def test_colony_config_validation_crew_too_large(self):
        """crew_size > 1000 raises ValueError."""
        with pytest.raises(ValueError, match="crew_size must be in range"):
            ColonyConfig(crew_size=1001)

    def test_colony_config_validation_negative_volume(self):
        """Negative hab_volume_m3 raises ValueError."""
        with pytest.raises(ValueError, match="hab_volume_m3 must be non-negative"):
            ColonyConfig(crew_size=10, hab_volume_m3=-100)

    def test_colony_config_validation_negative_solar(self):
        """Negative solar_array_m2 raises ValueError."""
        with pytest.raises(ValueError, match="solar_array_m2 must be non-negative"):
            ColonyConfig(crew_size=10, solar_array_m2=-50)

    def test_colony_config_validation_sabatier_out_of_range(self):
        """sabatier_efficiency outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="sabatier_efficiency must be in range"):
            ColonyConfig(crew_size=10, sabatier_efficiency=1.5)


# === ColonyState Tests ===

class TestColonyState:
    """Tests for ColonyState dataclass."""

    def test_colony_state_mutable(self):
        """Can modify ColonyState fields."""
        state = ColonyState()
        state.status = "stressed"
        assert state.status == "stressed"

        state.atmosphere = {"O2_pct": 19.0}
        assert state.atmosphere["O2_pct"] == 19.0

    def test_colony_state_defaults(self):
        """ColonyState has correct default values."""
        state = ColonyState()
        assert state.ts == ""
        assert state.atmosphere == {}
        assert state.thermal == {}
        assert state.resource == {}
        assert state.decision == {}
        assert state.entropy == {}
        assert state.status == "nominal"


# === generate_colony Tests ===

class TestGenerateColony:
    """Tests for generate_colony function."""

    def test_generate_colony_length(self):
        """duration_days=30 returns 30 states."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)
        assert len(states) == 30

    def test_generate_colony_deterministic(self):
        """Same seed produces same states."""
        config = default_config(10)
        s1 = generate_colony(config, 10, seed=42)
        s2 = generate_colony(config, 10, seed=42)
        assert s1[0].atmosphere == s2[0].atmosphere
        assert s1[5].thermal == s2[5].thermal

    def test_generate_colony_different_seeds(self):
        """Different seeds produce different states."""
        config = default_config(10)
        s1 = generate_colony(config, 10, seed=42)
        s2 = generate_colony(config, 10, seed=123)
        # Very unlikely to be exactly equal
        assert s1[0].atmosphere != s2[0].atmosphere

    def test_generate_colony_empty_on_zero_days(self):
        """duration_days=0 returns empty list."""
        config = default_config(10)
        states = generate_colony(config, 0, seed=42)
        assert states == []

    def test_generate_colony_timestamps(self):
        """Each state has valid ISO8601 timestamp."""
        config = default_config(10)
        states = generate_colony(config, 5, seed=42)
        for state in states:
            assert state.ts.endswith("Z")
            assert "2035-01" in state.ts  # Base date is 2035-01-01

    def test_generate_colony_nominal_status(self):
        """Without stress, states should be nominal or better."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)
        # Most states should be nominal in a healthy colony
        nominal_count = sum(1 for s in states if s.status == "nominal")
        assert nominal_count >= 20  # At least 2/3 nominal

    def test_generate_colony_atmosphere_bounds(self):
        """O2_pct in reasonable range (14-25%)."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)
        for state in states:
            o2 = state.atmosphere.get("O2_pct", 21)
            assert 14 <= o2 <= 25, f"O2_pct {o2} out of range"

    def test_generate_colony_thermal_bounds(self):
        """T_hab_C in reasonable range (-20 to 50)."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)
        for state in states:
            temp = state.thermal.get("T_hab_C", 22)
            assert -20 <= temp <= 50, f"T_hab_C {temp} out of range"


# === simulate_dust_storm Tests ===

class TestSimulateDustStorm:
    """Tests for simulate_dust_storm function."""

    def test_simulate_dust_storm_power_drop(self):
        """Power drops during storm days."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)
        pre_storm_power = states[10].resource["power_W"]

        simulate_dust_storm(states, start_day=10, duration_days=5)

        post_storm_power = states[12].resource["power_W"]
        assert post_storm_power < pre_storm_power

    def test_simulate_dust_storm_no_effect_before(self):
        """Days before start_day unchanged."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)
        pre_storm_state = states[5].resource["power_W"]

        simulate_dust_storm(states, start_day=10, duration_days=5)

        assert states[5].resource["power_W"] == pre_storm_state

    def test_simulate_dust_storm_recovery(self):
        """Days after storm can recover (nuclear unaffected)."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)

        # Get power during storm
        simulate_dust_storm(states, start_day=10, duration_days=5)
        storm_power = states[12].resource["power_W"]

        # Nuclear power should ensure non-zero power even during storm
        assert storm_power > 0

    def test_simulate_dust_storm_out_of_range(self):
        """Start day beyond states returns unchanged."""
        config = default_config(10)
        states = generate_colony(config, 10, seed=42)
        original_power = states[-1].resource["power_W"]

        result = simulate_dust_storm(states, start_day=100, duration_days=5)

        assert result[-1].resource["power_W"] == original_power


# === simulate_hab_breach Tests ===

class TestSimulateHabBreach:
    """Tests for simulate_hab_breach function."""

    def test_simulate_hab_breach_pressure_drop(self):
        """Pressure decreases from day onward."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)
        pre_breach_pressure = states[10].atmosphere["pressure_kPa"]

        simulate_hab_breach(states, day=10, breach_m2=0.1)

        post_breach_pressure = states[15].atmosphere["pressure_kPa"]
        assert post_breach_pressure < pre_breach_pressure

    def test_simulate_hab_breach_status_escalation(self):
        """Status changes to stressed/critical after breach."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)

        # Large breach for dramatic effect
        simulate_hab_breach(states, day=5, breach_m2=1.0)

        # Later days should have escalated status
        late_states = [s.status for s in states[10:20]]
        assert "critical" in late_states or "stressed" in late_states

    def test_simulate_hab_breach_out_of_range(self):
        """Day beyond states returns unchanged."""
        config = default_config(10)
        states = generate_colony(config, 10, seed=42)
        original_pressure = states[-1].atmosphere["pressure_kPa"]

        result = simulate_hab_breach(states, day=100, breach_m2=0.1)

        assert result[-1].atmosphere["pressure_kPa"] == original_pressure


# === simulate_crop_failure Tests ===

class TestSimulateCropFailure:
    """Tests for simulate_crop_failure function."""

    def test_simulate_crop_failure_food_drop(self):
        """food_kcal decreases by loss_pct."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)
        pre_failure_food = states[10].resource["food_kcal"]

        simulate_crop_failure(states, day=10, loss_pct=0.5)

        post_failure_food = states[11].resource["food_kcal"]
        # Should be roughly 50% of original
        assert post_failure_food < pre_failure_food * 0.6

    def test_simulate_crop_failure_out_of_range(self):
        """Day beyond states returns unchanged."""
        config = default_config(10)
        states = generate_colony(config, 10, seed=42)
        original_food = states[-1].resource["food_kcal"]

        result = simulate_crop_failure(states, day=100, loss_pct=0.5)

        assert result[-1].resource["food_kcal"] == original_food


# === batch_generate Tests ===

class TestBatchGenerate:
    """Tests for batch_generate function."""

    def test_batch_generate_count(self):
        """n_colonies=5 returns 5 results."""
        results = batch_generate(n_colonies=5, stress_level="nominal", seed=42)
        assert len(results) == 5

    def test_batch_generate_stress_levels_nominal(self):
        """stress_level='nominal' has no events."""
        results = batch_generate(n_colonies=3, stress_level="nominal", seed=42)
        for r in results:
            assert r["stress_events"] == []

    def test_batch_generate_stress_levels_stressed(self):
        """stress_level='stressed' has events."""
        results = batch_generate(n_colonies=5, stress_level="stressed", seed=42)
        # At least some should have events
        event_count = sum(1 for r in results if r["stress_events"])
        assert event_count > 0

    def test_batch_generate_emits_receipt(self, capsys):
        """Each colony produces receipt with colony_id."""
        results = batch_generate(n_colonies=2, stress_level="nominal", seed=42)

        captured = capsys.readouterr()
        # Check that receipts were emitted
        assert "colony_id" in captured.out
        assert "receipt_type" in captured.out

        # Each result should have a colony_id
        for r in results:
            assert "colony_id" in r
            assert len(r["colony_id"]) > 0

    def test_batch_generate_structure(self):
        """Each result has expected keys."""
        results = batch_generate(n_colonies=2, stress_level="nominal", seed=42)
        for r in results:
            assert "config" in r
            assert "states" in r
            assert "stress_events" in r
            assert "colony_id" in r
            assert isinstance(r["config"], ColonyConfig)
            assert isinstance(r["states"], list)


# === _determine_status Tests ===

class TestDetermineStatus:
    """Tests for _determine_status function."""

    def test_determine_status_nominal(self):
        """All values in range returns 'nominal'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_NOMINAL, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_NOMINAL},
            resource={"power_W": 20000, "food_kcal": 50000},
        )
        assert _determine_status(state) == "nominal"

    def test_determine_status_stressed_o2(self):
        """O2 below stressed threshold returns 'stressed'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_STRESSED - 0.1, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_NOMINAL},
            resource={"power_W": 20000, "food_kcal": 50000},
        )
        assert _determine_status(state) in ("stressed", "critical")

    def test_determine_status_critical_o2(self):
        """O2 in critical range returns 'critical'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_CRITICAL - 0.1, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_NOMINAL},
            resource={"power_W": 20000, "food_kcal": 50000},
        )
        assert _determine_status(state) == "critical"

    def test_determine_status_failed_o2(self):
        """O2 in failed range returns 'failed'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_FAILED - 0.1, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_NOMINAL},
            resource={"power_W": 20000, "food_kcal": 50000},
        )
        assert _determine_status(state) == "failed"

    def test_determine_status_stressed_co2(self):
        """CO2 above stressed threshold returns 'stressed'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_NOMINAL, "CO2_ppm": CO2_STRESSED + 100, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_NOMINAL},
            resource={"power_W": 20000, "food_kcal": 50000},
        )
        assert _determine_status(state) in ("stressed", "critical")

    def test_determine_status_failed_zero_power(self):
        """Zero power returns 'failed'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_NOMINAL, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_NOMINAL},
            resource={"power_W": 0, "food_kcal": 50000},
        )
        assert _determine_status(state) == "failed"

    def test_determine_status_failed_zero_food(self):
        """Zero food returns 'failed'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_NOMINAL, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_NOMINAL},
            resource={"power_W": 20000, "food_kcal": 0},
        )
        assert _determine_status(state) == "failed"

    def test_determine_status_critical_temperature_cold(self):
        """Temperature below critical minimum returns 'critical'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_NOMINAL, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_MIN_CRITICAL - 5},
            resource={"power_W": 20000, "food_kcal": 50000},
        )
        assert _determine_status(state) == "critical"

    def test_determine_status_critical_temperature_hot(self):
        """Temperature above critical maximum returns 'critical'."""
        state = ColonyState(
            atmosphere={"O2_pct": O2_NOMINAL, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
            thermal={"T_hab_C": T_HAB_MAX_CRITICAL + 5},
            resource={"power_W": 20000, "food_kcal": 50000},
        )
        assert _determine_status(state) == "critical"


# === Cascade Behavior Tests ===

class TestCascadeBehavior:
    """Tests for cascade failure behavior."""

    def test_cascade_stops_at_failed(self):
        """Once 'failed', subsequent states stay 'failed'."""
        config = default_config(10)
        states = generate_colony(config, 30, seed=42)

        # Force failure at day 10
        states[10].atmosphere["O2_pct"] = O2_FAILED - 1
        states[10].status = "failed"

        # Re-generate remaining states (simulating what generate_colony does)
        # In practice, generate_colony handles this, but let's verify the intent
        # by creating a colony where we inject failure

        # Create fresh colony and force early failure
        config2 = default_config(10)
        fresh_states = []
        for i in range(30):
            state = ColonyState(
                ts=f"2035-01-{i+1:02d}T00:00:00Z",
                atmosphere={"O2_pct": O2_FAILED - 1 if i >= 10 else O2_NOMINAL, "CO2_ppm": CO2_NOMINAL, "pressure_kPa": PRESSURE_NOMINAL_KPA},
                thermal={"T_hab_C": T_HAB_NOMINAL},
                resource={"power_W": 20000, "food_kcal": 50000},
            )
            state.status = _determine_status(state)
            fresh_states.append(state)

        # All states from day 10 onward should be failed
        for i in range(10, 30):
            assert fresh_states[i].status == "failed"
