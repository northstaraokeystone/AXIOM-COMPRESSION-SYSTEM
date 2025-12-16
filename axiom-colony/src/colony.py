"""AXIOM-COLONY v3.1 Colony Module - State generators with AI/Neuralink config."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
import random

import numpy as np

from src.core import emit_receipt, dual_hash
from src.entropy import (
    internal_compression_rate,
    external_compression_rate,
    compression_advantage,
    sovereignty_threshold,
    total_colony_entropy,
    LIGHT_DELAY_MIN,
    LIGHT_DELAY_MAX,
)


@dataclass(frozen=True)
class ColonyConfig:
    """Colony configuration with v3.1 additions."""
    crew_size: int = 10
    hab_volume_m3: float = 500.0
    solar_array_m2: float = 100.0
    radiator_area_m2: float = 50.0
    kilopower_units: int = 2  # Minimum 2 enforced
    sabatier_efficiency: float = 0.70  # Changed from 0.85
    earth_bandwidth_mbps: float = 2.0
    # v3.1 NEW fields
    compute_mass_kg: float = 100.0
    compute_flops: float = 1e15
    neuralink_fraction: float = 0.0
    expertise_coverage: float = 0.8


@dataclass
class ColonyState:
    """Colony state with v3.1 compression metrics."""
    ts: str = ""
    day: int = 0
    atmosphere: dict = field(default_factory=dict)
    thermal: dict = field(default_factory=dict)
    resource: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)  # v3.1: internal_rate, external_rate, advantage, sovereign
    entropy: dict = field(default_factory=dict)
    status: str = "nominal"


def default_config(crew_size: int) -> ColonyConfig:
    """Create default config with v3.1 fields."""
    return ColonyConfig(
        crew_size=crew_size,
        kilopower_units=max(2, crew_size // 5),  # Scale with crew
        compute_flops=1e15,
        neuralink_fraction=0.0,
        expertise_coverage=0.8,
    )


def _compute_latency(day: int) -> float:
    """Compute light delay based on orbital position (simplified sinusoid)."""
    # Mars-Earth distance varies from ~3 to ~22 light-minutes
    phase = (day % 780) / 780 * 2 * np.pi
    latency_minutes = LIGHT_DELAY_MIN + (LIGHT_DELAY_MAX - LIGHT_DELAY_MIN) * (1 + np.sin(phase)) / 2
    return latency_minutes * 60  # Convert to seconds


def generate_colony(config: ColonyConfig, duration_days: int, seed: int = 42) -> List[ColonyState]:
    """Generate colony states with compression metrics per day."""
    rng = np.random.default_rng(seed)
    states = []

    for day in range(duration_days):
        # Compute latency for this day
        latency = _compute_latency(day)

        # v3.1: Compute compression metrics
        internal = internal_compression_rate(
            config.crew_size,
            config.expertise_coverage,
            config.compute_flops,
            config.neuralink_fraction
        )
        external = external_compression_rate(
            config.earth_bandwidth_mbps,
            latency,
            day
        )
        advantage = compression_advantage(internal, external)
        sovereign = sovereignty_threshold(internal, external)

        # Generate subsystem states
        atmosphere = {
            "O2_pct": 21.0 + rng.normal(0, 0.5),
            "CO2_ppm": 400 + rng.normal(0, 50),
            "pressure_kPa": 101.3 + rng.normal(0, 1),
        }
        thermal = {
            "T_hab_C": 22.0 + rng.normal(0, 2),
            "Q_in_W": 5000 + rng.normal(0, 200),
            "Q_out_W": 4800 + rng.normal(0, 200),
        }
        resource = {
            "water_L": 1000 + rng.normal(0, 20),
            "food_kg": 500 + rng.normal(0, 10),
            "power_W": 10000 + rng.normal(0, 500),
            "buffer_days": 90 + rng.integers(-10, 10),
        }

        # Determine status based on conditions
        status = "nominal"
        if atmosphere["O2_pct"] < 19.5 or atmosphere["O2_pct"] > 23.5:
            status = "warning"
        if thermal["T_hab_C"] < 0 or thermal["T_hab_C"] > 40:
            status = "critical"
        if not sovereign and day > 30:
            status = "dependent"

        state = ColonyState(
            ts=datetime.now(timezone.utc).isoformat(),
            day=day,
            atmosphere=atmosphere,
            thermal=thermal,
            resource=resource,
            decision={
                "internal_rate": internal,
                "external_rate": external,
                "advantage": advantage,
                "sovereign": sovereign,
            },
            entropy={"H_total": 0.0},  # Will be computed by entropy module
            status=status,
        )
        states.append(state)

    return states


def simulate_dust_storm(states: List[ColonyState], start_day: int, duration: int) -> List[ColonyState]:
    """Simulate dust storm reducing solar input."""
    for i, state in enumerate(states):
        if start_day <= state.day < start_day + duration:
            # Reduce solar input, increase thermal stress
            state.thermal["Q_in_W"] *= 0.1  # 90% reduction
            state.thermal["T_hab_C"] -= 5
            state.atmosphere["CO2_ppm"] += 100
            if state.status == "nominal":
                state.status = "stressed"
    return states


def simulate_hab_breach(states: List[ColonyState], day: int, breach_m2: float = 0.01) -> List[ColonyState]:
    """Simulate habitat breach."""
    for state in states:
        if state.day >= day:
            # Pressure loss, O2 drop
            days_since = state.day - day
            state.atmosphere["pressure_kPa"] *= max(0.8, 1 - 0.05 * days_since)
            state.atmosphere["O2_pct"] *= max(0.85, 1 - 0.02 * days_since)
            if state.status == "nominal":
                state.status = "emergency"
    return states


def simulate_crop_failure(states: List[ColonyState], day: int, loss_pct: float = 0.5) -> List[ColonyState]:
    """Simulate crop failure."""
    for state in states:
        if state.day >= day:
            state.resource["food_kg"] *= (1 - loss_pct)
            state.resource["buffer_days"] = int(state.resource["buffer_days"] * (1 - loss_pct))
            if state.status == "nominal":
                state.status = "rationing"
    return states


def simulate_equipment_failure(states: List[ColonyState], day: int, subsystem: str) -> List[ColonyState]:
    """Simulate equipment failure in a subsystem."""
    for state in states:
        if state.day >= day:
            if subsystem == "thermal":
                state.thermal["Q_out_W"] *= 0.5
                state.thermal["T_hab_C"] += 10
            elif subsystem == "atmosphere":
                state.atmosphere["CO2_ppm"] += 500
            elif subsystem == "power":
                state.resource["power_W"] *= 0.5
            if state.status == "nominal":
                state.status = "degraded"
    return states


def batch_generate(n: int, stress: str = "none", seed: int = 42) -> List[dict]:
    """Generate batch of colonies with optional stress events."""
    results = []
    rng = random.Random(seed)

    for i in range(n):
        config = default_config(crew_size=rng.randint(4, 50))
        states = generate_colony(config, duration_days=365, seed=seed + i)

        if stress == "dust_storm":
            start = rng.randint(30, 200)
            states = simulate_dust_storm(states, start, 90)
        elif stress == "hab_breach":
            day = rng.randint(50, 300)
            states = simulate_hab_breach(states, day)

        results.append({
            "config": config,
            "states": states,
            "final_status": states[-1].status,
        })

    return results
