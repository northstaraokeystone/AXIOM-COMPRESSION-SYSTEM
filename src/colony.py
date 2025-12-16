"""BUILD C3: Synthetic colony state generator with stress event injection.

Ground truth for sovereignty threshold validation. Generate colony states
across 4 subsystems: atmosphere, thermal, resource, decision.

Source: CLAUDEME §5, AXIOM_Colony_Build_Strategy_v2.md §2.3
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np

from src.core import emit_receipt, TENANT_ID
from src.entropy import (
    subsystem_entropy,
    total_colony_entropy,
    decision_capacity,
    earth_input_rate,
    sovereignty_threshold,
    entropy_status,
    SOLAR_FLUX_MAX,
    SOLAR_FLUX_DUST,
    KILOPOWER_KW,
    HUMAN_METABOLIC_W,
    MARS_RELAY_MBPS,
    LIGHT_DELAY_MAX,
)


# === COLONY PHYSICS DEFAULTS ===

DEFAULT_HAB_VOLUME_M3 = 500.0
"""~50m³/person for 10 crew. Engineering estimate."""

DEFAULT_SOLAR_ARRAY_M2 = 200.0
"""~20m²/person. Engineering estimate."""

DEFAULT_RADIATOR_AREA_M2 = 100.0
"""Heat rejection area. Engineering estimate."""

DEFAULT_KILOPOWER_UNITS = 2
"""20kW nuclear baseline. NASA reference."""


# === ATMOSPHERIC THRESHOLDS ===

O2_NOMINAL = 21.0
"""Earth atmosphere, percent."""

O2_STRESSED = 19.5
"""NASA safety. Below this = stressed."""

O2_CRITICAL = 16.0
"""Physiology. Below this = critical."""

O2_FAILED = 14.0
"""Biosphere 2 minimum. Below this = failed."""

CO2_NOMINAL = 400.0
"""Earth atmosphere, ppm."""

CO2_STRESSED = 5000.0
"""NASA limit. Above this = stressed."""

CO2_CRITICAL = 20000.0
"""Danger threshold. Above this = critical."""

CO2_FAILED = 40000.0
"""Lethal. Above this = failed."""


# === THERMAL THRESHOLDS ===

T_HAB_NOMINAL = 22.0
"""Comfortable temperature, Celsius."""

T_HAB_MIN_STRESSED = 10.0
"""Cold stress. Below this = stressed."""

T_HAB_MAX_STRESSED = 32.0
"""Heat stress. Above this = stressed."""

T_HAB_MIN_CRITICAL = 0.0
"""Freezing. Below this = critical."""

T_HAB_MAX_CRITICAL = 40.0
"""Hyperthermia. Above this = critical."""


# === PRESSURE THRESHOLDS ===

PRESSURE_NOMINAL_KPA = 101.3
"""Earth sea level, kPa."""

PRESSURE_STRESSED_KPA = 70.0
"""High altitude. Below = stressed."""

PRESSURE_CRITICAL_KPA = 50.0
"""Dangerous. Below = critical."""


# === DATACLASSES ===

@dataclass(frozen=True)
class ColonyConfig:
    """Immutable configuration for colony generation.

    Post-init validation:
    - crew_size must be in range [4, 1000]
    - All float fields must be non-negative
    """
    crew_size: int = 10
    hab_volume_m3: float = DEFAULT_HAB_VOLUME_M3
    solar_array_m2: float = DEFAULT_SOLAR_ARRAY_M2
    radiator_area_m2: float = DEFAULT_RADIATOR_AREA_M2
    kilopower_units: int = DEFAULT_KILOPOWER_UNITS
    sabatier_efficiency: float = 0.85
    earth_bandwidth_mbps: float = 2.0

    def __post_init__(self):
        if not (4 <= self.crew_size <= 1000):
            raise ValueError(f"crew_size must be in range [4, 1000], got {self.crew_size}")
        if self.hab_volume_m3 < 0:
            raise ValueError("hab_volume_m3 must be non-negative")
        if self.solar_array_m2 < 0:
            raise ValueError("solar_array_m2 must be non-negative")
        if self.radiator_area_m2 < 0:
            raise ValueError("radiator_area_m2 must be non-negative")
        if self.kilopower_units < 0:
            raise ValueError("kilopower_units must be non-negative")
        if not (0.0 <= self.sabatier_efficiency <= 1.0):
            raise ValueError("sabatier_efficiency must be in range [0.0, 1.0]")
        if self.earth_bandwidth_mbps <= 0:
            raise ValueError("earth_bandwidth_mbps must be positive")


@dataclass
class ColonyState:
    """Mutable state snapshot for a single day."""
    ts: str = ""
    atmosphere: dict = field(default_factory=dict)
    thermal: dict = field(default_factory=dict)
    resource: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    entropy: dict = field(default_factory=dict)
    status: str = "nominal"


# === FACTORY FUNCTION ===

def default_config(crew_size: int) -> ColonyConfig:
    """Factory function to create ColonyConfig with crew_size set, others default."""
    return ColonyConfig(crew_size=crew_size)


# === INTERNAL HELPERS ===

def _compute_atmosphere(config: ColonyConfig, day: int, rng: np.random.Generator) -> dict:
    """Compute atmosphere state for a nominal day.

    Returns dict with O2_pct, CO2_ppm, pressure_kPa.
    """
    # Nominal values with small random variation
    o2_variation = rng.normal(0, 0.3)  # ±0.3% variation
    co2_variation = rng.normal(0, 50)  # ±50 ppm variation
    pressure_variation = rng.normal(0, 0.5)  # ±0.5 kPa variation

    return {
        "O2_pct": O2_NOMINAL + o2_variation,
        "CO2_ppm": CO2_NOMINAL + co2_variation,
        "pressure_kPa": PRESSURE_NOMINAL_KPA + pressure_variation,
    }


def _compute_thermal(config: ColonyConfig, day: int, rng: np.random.Generator) -> dict:
    """Compute thermal state for a nominal day.

    Returns dict with Q_in_W, Q_out_W, T_hab_C.
    """
    # Solar input varies with day (simplified seasonal/orbital variation)
    solar_factor = 0.8 + 0.2 * np.sin(2 * np.pi * day / 687)  # Mars year
    solar_input = config.solar_array_m2 * SOLAR_FLUX_MAX * solar_factor

    # Nuclear power (constant)
    nuclear_input = config.kilopower_units * KILOPOWER_KW * 1000  # Convert to W

    # Total heat input
    q_in = solar_input + nuclear_input

    # Crew metabolic heat
    metabolic_heat = config.crew_size * HUMAN_METABOLIC_W
    q_in += metabolic_heat

    # Radiative heat rejection
    q_out = config.radiator_area_m2 * 200  # ~200 W/m² typical rejection

    # Temperature: slight variation around nominal
    temp_variation = rng.normal(0, 1.0)  # ±1°C variation

    return {
        "Q_in_W": q_in,
        "Q_out_W": q_out,
        "T_hab_C": T_HAB_NOMINAL + temp_variation,
    }


def _compute_resource(config: ColonyConfig, day: int, rng: np.random.Generator) -> dict:
    """Compute resource state for a nominal day.

    Returns dict with water_kg, food_kcal, power_W.
    """
    # Water: ~3L/person/day baseline, recycled
    water_per_person = 3.0  # kg/day
    water_total = config.crew_size * water_per_person * 30  # 30 day buffer
    water_variation = rng.normal(0, 10)

    # Food: ~2500 kcal/person/day
    food_per_person = 2500  # kcal/day
    food_total = config.crew_size * food_per_person * 30  # 30 day buffer
    food_variation = rng.normal(0, 500)

    # Power calculation
    solar_power = config.solar_array_m2 * SOLAR_FLUX_MAX * 0.2  # 20% efficiency
    nuclear_power = config.kilopower_units * KILOPOWER_KW * 1000  # W
    total_power = solar_power + nuclear_power
    power_variation = rng.normal(0, 100)

    return {
        "water_kg": water_total + water_variation,
        "food_kcal": food_total + food_variation,
        "power_W": total_power + power_variation,
    }


def _compute_decision(config: ColonyConfig, day: int) -> dict:
    """Compute decision state.

    Returns dict with bits_per_sec, latency_sec, expertise_coverage.
    Uses decision_capacity from entropy.py.
    """
    # Expertise coverage: baseline at 0.6 (60% of needed skills covered)
    expertise = {"engineering": 0.7, "medical": 0.5, "science": 0.6, "operations": 0.6}

    # Decision capacity from entropy.py
    bits_per_sec = decision_capacity(
        crew=config.crew_size,
        expertise=expertise,
        bandwidth=config.earth_bandwidth_mbps,
        latency=LIGHT_DELAY_MAX * 60,  # Convert min to sec
    )

    # Latency varies with Mars-Earth distance (simplified orbital model)
    # Range: LIGHT_DELAY_MIN (3 min) to LIGHT_DELAY_MAX (22 min)
    orbital_phase = (day % 780) / 780  # ~780 day synodic period
    latency_min = 3 + 19 * abs(np.sin(np.pi * orbital_phase))  # 3-22 min
    latency_sec = latency_min * 60

    # Expertise coverage mean
    expertise_coverage = sum(expertise.values()) / len(expertise)

    return {
        "bits_per_sec": bits_per_sec,
        "latency_sec": latency_sec,
        "expertise_coverage": expertise_coverage,
    }


def _compute_entropy(state: ColonyState) -> dict:
    """Compute entropy for all subsystems using entropy.py functions.

    Returns dict with H_atmo, H_thermal, H_resource, H_decision, H_total.
    """
    # Convert ColonyState to dict format expected by entropy.py
    entropy_state = {
        "O2_pct": state.atmosphere.get("O2_pct", 0.21) / 100,  # Convert to fraction
        "CO2_pct": state.atmosphere.get("CO2_ppm", 400) / 1e6,  # Convert ppm to fraction
        "N2_pct": 0.78,
        "temperature_C": state.thermal.get("T_hab_C", 22),
        "temp_min_C": T_HAB_MIN_CRITICAL,
        "temp_max_C": T_HAB_MAX_CRITICAL,
        "water_ratio": min(1.0, state.resource.get("water_kg", 1000) / 1000),
        "food_ratio": min(1.0, state.resource.get("food_kcal", 50000) / 50000),
        "power_ratio": min(1.0, state.resource.get("power_W", 20000) / 20000),
        "expertise": {"general": state.decision.get("expertise_coverage", 0.6)},
        "latency_min": state.decision.get("latency_sec", 600) / 60,
    }

    h_atmo = subsystem_entropy(entropy_state, "atmosphere")
    h_thermal = subsystem_entropy(entropy_state, "thermal")
    h_resource = subsystem_entropy(entropy_state, "resource")
    h_decision = subsystem_entropy(entropy_state, "decision")
    h_total = total_colony_entropy(entropy_state)

    return {
        "H_atmo": h_atmo,
        "H_thermal": h_thermal,
        "H_resource": h_resource,
        "H_decision": h_decision,
        "H_total": h_total,
    }


def _determine_status(state: ColonyState) -> str:
    """Check all thresholds and return worst status.

    Returns worst status: failed > critical > stressed > nominal.
    """
    status = "nominal"

    # Check atmosphere
    o2 = state.atmosphere.get("O2_pct", O2_NOMINAL)
    co2 = state.atmosphere.get("CO2_ppm", CO2_NOMINAL)
    pressure = state.atmosphere.get("pressure_kPa", PRESSURE_NOMINAL_KPA)

    # O2 checks (lower is worse)
    if o2 < O2_FAILED:
        return "failed"
    if o2 < O2_CRITICAL:
        status = "critical"
    elif o2 < O2_STRESSED and status not in ("critical",):
        status = "stressed"

    # CO2 checks (higher is worse)
    if co2 > CO2_FAILED:
        return "failed"
    if co2 > CO2_CRITICAL:
        status = "critical"
    elif co2 > CO2_STRESSED and status not in ("critical",):
        status = "stressed"

    # Pressure checks
    if pressure < PRESSURE_CRITICAL_KPA:
        status = "critical"
    elif pressure < PRESSURE_STRESSED_KPA and status not in ("critical",):
        status = "stressed"

    # Check thermal
    temp = state.thermal.get("T_hab_C", T_HAB_NOMINAL)

    if temp < T_HAB_MIN_CRITICAL or temp > T_HAB_MAX_CRITICAL:
        status = "critical"
    elif (temp < T_HAB_MIN_STRESSED or temp > T_HAB_MAX_STRESSED) and status not in ("critical",):
        status = "stressed"

    # Check resources
    power = state.resource.get("power_W", 1)
    food = state.resource.get("food_kcal", 1)

    if power <= 0:
        return "failed"
    if food <= 0:
        return "failed"

    return status


# === MAIN GENERATION FUNCTIONS ===

def generate_colony(config: ColonyConfig, duration_days: int, seed: int) -> List[ColonyState]:
    """Generate colony states for duration_days.

    One state per day. Deterministic given seed.
    If crew_size <= 0 or duration_days <= 0, return empty list.
    """
    if config.crew_size <= 0 or duration_days <= 0:
        return []

    rng = np.random.default_rng(seed)
    states: List[ColonyState] = []
    base_date = datetime(2035, 1, 1)
    failed = False

    for day in range(duration_days):
        # Create timestamp
        ts = (base_date + timedelta(days=day)).isoformat() + "Z"

        # Create state
        state = ColonyState(ts=ts)

        if failed:
            # Once failed, stay failed with degraded values
            if states:
                state.atmosphere = states[-1].atmosphere.copy()
                state.thermal = states[-1].thermal.copy()
                state.resource = states[-1].resource.copy()
                state.decision = states[-1].decision.copy()
            state.status = "failed"
            state.entropy = _compute_entropy(state)
            states.append(state)
            continue

        # Compute subsystem states
        state.atmosphere = _compute_atmosphere(config, day, rng)
        state.thermal = _compute_thermal(config, day, rng)
        state.resource = _compute_resource(config, day, rng)
        state.decision = _compute_decision(config, day)

        # Compute entropy
        state.entropy = _compute_entropy(state)

        # Determine status
        state.status = _determine_status(state)

        if state.status == "failed":
            failed = True

        states.append(state)

    return states


def simulate_dust_storm(
    states: List[ColonyState], start_day: int, duration_days: int
) -> List[ColonyState]:
    """Simulate dust storm effects on colony states.

    Modifies states in place. Solar flux drops to SOLAR_FLUX_DUST (1%).
    Cascade: power drops → thermal stress → potential O2 drop if MOXIE power-limited.
    """
    if start_day >= len(states):
        return states

    end_day = min(start_day + duration_days, len(states))
    solar_reduction = SOLAR_FLUX_DUST / SOLAR_FLUX_MAX  # ~1%

    for day in range(start_day, end_day):
        state = states[day]

        # Reduce solar-derived power
        original_power = state.resource.get("power_W", 20000)
        # Assume 70% of power is solar in nominal conditions
        solar_fraction = 0.7
        nuclear_power = original_power * (1 - solar_fraction)
        reduced_solar = original_power * solar_fraction * solar_reduction
        new_power = nuclear_power + reduced_solar

        state.resource["power_W"] = new_power

        # Thermal cascade: less power = less heating capacity
        if new_power < original_power * 0.5:
            temp_drop = (original_power - new_power) / original_power * 5  # Up to 5°C drop
            state.thermal["T_hab_C"] = state.thermal.get("T_hab_C", 22) - temp_drop

        # O2 cascade: MOXIE may be power-limited
        if new_power < original_power * 0.3:
            o2_drop = 0.5  # 0.5% O2 drop per day if severely power-limited
            state.atmosphere["O2_pct"] = state.atmosphere.get("O2_pct", 21) - o2_drop

        # Recalculate entropy and status
        state.entropy = _compute_entropy(state)
        state.status = _determine_status(state)

    return states


def simulate_hab_breach(
    states: List[ColonyState], day: int, breach_m2: float
) -> List[ColonyState]:
    """Simulate habitat breach effects.

    Modifies states from day onward. Pressure drops rapidly.
    O2 percentage stays same but absolute O2 drops with pressure.
    """
    if day >= len(states):
        return states

    # Larger breach = faster pressure loss
    # Exponential decay rate based on breach size
    decay_rate = 0.1 * breach_m2  # 10% loss per day per m² of breach

    for d in range(day, len(states)):
        state = states[d]
        days_since = d - day

        # Exponential pressure decay
        original_pressure = PRESSURE_NOMINAL_KPA
        current_pressure = original_pressure * np.exp(-decay_rate * days_since)
        state.atmosphere["pressure_kPa"] = max(10, current_pressure)  # Min 10 kPa (not total vacuum)

        # Recalculate entropy and status
        state.entropy = _compute_entropy(state)
        state.status = _determine_status(state)

    return states


def simulate_crop_failure(
    states: List[ColonyState], day: int, loss_pct: float
) -> List[ColonyState]:
    """Simulate crop failure effects.

    Modifies states from day onward. Food production drops by loss_pct.
    Long-term caloric deficit tracked.
    """
    if day >= len(states):
        return states

    cumulative_deficit = 0

    for d in range(day, len(states)):
        state = states[d]

        # Reduce food by loss percentage
        original_food = state.resource.get("food_kcal", 50000)
        state.resource["food_kcal"] = original_food * (1 - loss_pct)

        # Track cumulative deficit (crew needs ~2500 kcal/person/day)
        daily_need = 2500 * 10  # Assume 10 crew
        daily_production = state.resource["food_kcal"] / 30  # Daily from 30-day buffer
        daily_deficit = max(0, daily_need - daily_production)
        cumulative_deficit += daily_deficit

        # Escalate status if severe deficit
        if cumulative_deficit > daily_need * 7:  # 7 days of deficit
            if state.status == "nominal":
                state.status = "stressed"
            elif state.status == "stressed":
                state.status = "critical"

        # Recalculate entropy
        state.entropy = _compute_entropy(state)

    return states


def batch_generate(
    n_colonies: int, stress_level: str, seed: int
) -> List[Dict[str, Any]]:
    """Generate multiple colonies with stress events.

    stress_level: "nominal" (no events), "stressed" (one random event),
    "critical" (multiple events).

    Returns list of {config, states, stress_events, colony_id}.
    Emits colony_receipt per colony.
    """
    base_rng = np.random.default_rng(seed)
    results = []

    stress_events_map = {
        "nominal": 0,
        "stressed": 1,
        "critical": 3,
    }
    num_events = stress_events_map.get(stress_level, 0)

    for i in range(n_colonies):
        # Random crew size variation
        crew_size = int(base_rng.integers(8, 20))
        config = default_config(crew_size)

        # Generate colony
        colony_seed = int(base_rng.integers(0, 2**31))
        states = generate_colony(config, 30, colony_seed)

        # Apply stress events
        stress_events = []

        if num_events >= 1:
            event_type = base_rng.choice(["dust_storm", "hab_breach", "crop_failure"])
            event_day = int(base_rng.integers(5, 15))

            if event_type == "dust_storm":
                simulate_dust_storm(states, event_day, 10)
                stress_events.append("dust_storm")
            elif event_type == "hab_breach":
                simulate_hab_breach(states, event_day, 0.01)
                stress_events.append("hab_breach")
            elif event_type == "crop_failure":
                simulate_crop_failure(states, event_day, 0.5)
                stress_events.append("crop_failure")

        if num_events >= 2:
            # Second event: dust storm if not already
            if "dust_storm" not in stress_events:
                simulate_dust_storm(states, int(base_rng.integers(10, 20)), 7)
                stress_events.append("dust_storm")

        if num_events >= 3:
            # Third event: crop failure
            if "crop_failure" not in stress_events:
                simulate_crop_failure(states, int(base_rng.integers(15, 25)), 0.3)
                stress_events.append("crop_failure")

        # Generate colony ID
        colony_id = str(uuid.uuid4())

        # Determine final status
        final_status = states[-1].status if states else "nominal"

        # Emit colony receipt
        emit_receipt("colony", {
            "colony_id": colony_id,
            "crew_size": crew_size,
            "duration_days": len(states),
            "stress_events": stress_events,
            "final_status": final_status,
        })

        results.append({
            "config": config,
            "states": states,
            "stress_events": stress_events,
            "colony_id": colony_id,
        })

    return results
