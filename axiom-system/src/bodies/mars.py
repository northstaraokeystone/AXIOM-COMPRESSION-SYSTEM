"""AXIOM-SYSTEM v2 Mars Module - Main colony body.

Status: MIGRATED from colony.py, UPDATED with network fields
Purpose: Mars colony state with full subsystem modeling

Mars is the primary colony:
- delay_s = 180-1320s (3-22 minutes, varies with orbital position)
- Primary target for sovereignty analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
import math

import numpy as np

from ..core import emit_receipt
from ..entropy import (
    internal_compression_rate,
    external_compression_rate,
    compression_advantage,
    is_sovereign,
    total_colony_entropy,
    LIGHT_DELAY_MIN,
    LIGHT_DELAY_MAX,
    MARS_RELAY_MBPS,
    CONJUNCTION_BLACKOUT_DAYS,
    MOON_RELAY_BOOST,
)
from .base import BodyState, update_body_state


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MARS_SOL_SECONDS = 88775               # Mars sol in Earth seconds
MARS_YEAR_SOLS = 668                   # Mars year in sols
CONJUNCTION_CYCLE_SOLS = 760           # ~780 Earth days between conjunctions


# ═══════════════════════════════════════════════════════════════════════════════
# MARS CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarsConfig:
    """Mars colony configuration with v2 network fields.

    Attributes:
        crew_size: Number of crew
        hab_volume_m3: Habitat volume
        solar_array_m2: Solar panel area
        radiator_area_m2: Radiator area
        kilopower_units: Number of Kilopower reactors
        sabatier_efficiency: Sabatier reactor efficiency
        earth_bandwidth_mbps: Bandwidth to Earth
        compute_mass_kg: Computing hardware mass
        compute_flops: Computing capacity
        neuralink_fraction: Fraction with Neuralink
        neuralink_mbps: Neuralink bandwidth per person
        expertise_coverage: Expertise coverage factor
        relay_via_moon: Whether using Moon relay
        conjunction_sol_start: Sol when conjunction blackout starts
        conjunction_duration_sols: Duration of blackout
    """
    crew_size: int = 10
    hab_volume_m3: float = 500.0
    solar_array_m2: float = 100.0
    radiator_area_m2: float = 50.0
    kilopower_units: int = 2
    sabatier_efficiency: float = 0.70
    earth_bandwidth_mbps: float = MARS_RELAY_MBPS
    compute_mass_kg: float = 100.0
    compute_flops: float = 1e15
    neuralink_fraction: float = 0.0
    neuralink_mbps: float = 1.0
    expertise_coverage: float = 0.8
    relay_via_moon: bool = False
    conjunction_sol_start: int = 373
    conjunction_duration_sols: int = CONJUNCTION_BLACKOUT_DAYS


@dataclass
class MarsSubsystems:
    """Mars colony subsystem states."""
    atmosphere: dict = field(default_factory=dict)
    thermal: dict = field(default_factory=dict)
    resource: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mars_latency(sol: int) -> float:
    """Compute light delay based on orbital position (simplified sinusoid).

    Mars-Earth distance varies from ~3 to ~22 light-minutes.

    Args:
        sol: Current sol

    Returns:
        Latency in seconds
    """
    phase = (sol % CONJUNCTION_CYCLE_SOLS) / CONJUNCTION_CYCLE_SOLS * 2 * np.pi
    latency_minutes = LIGHT_DELAY_MIN + (LIGHT_DELAY_MAX - LIGHT_DELAY_MIN) * (1 + np.sin(phase)) / 2
    return latency_minutes * 60  # Convert to seconds


def conjunction_mask(sol: int, config: MarsConfig = None) -> float:
    """Check if Mars is in solar conjunction blackout.

    Returns 0.0 during 14-day blackout, else 1.0.

    Args:
        sol: Current sol
        config: Mars configuration (for custom conjunction timing)

    Returns:
        1.0 (nominal) or 0.0 (blackout)
    """
    if config is None:
        start = 373
        duration = CONJUNCTION_BLACKOUT_DAYS
    else:
        start = config.conjunction_sol_start
        duration = config.conjunction_duration_sols

    sol_in_cycle = sol % CONJUNCTION_CYCLE_SOLS
    if start <= sol_in_cycle <= start + duration:
        return 0.0
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# MARS STATE
# ═══════════════════════════════════════════════════════════════════════════════

def create_mars_state(config: MarsConfig = None, sol: int = 0) -> BodyState:
    """Create Mars colony state.

    Args:
        config: Mars configuration (uses defaults if None)
        sol: Current sol

    Returns:
        BodyState for Mars
    """
    if config is None:
        config = MarsConfig()

    # Calculate light delay
    delay_s = compute_mars_latency(sol)

    # Calculate internal compression rate
    neuralink_mbps = config.neuralink_mbps if config.neuralink_fraction > 0 else 0.0
    internal = internal_compression_rate(
        config.crew_size,
        config.expertise_coverage,
        config.compute_flops,
        neuralink_mbps * config.neuralink_fraction
    )

    # Calculate external compression rate
    external = external_compression_rate(
        config.earth_bandwidth_mbps,
        delay_s,
        sol
    )

    # Apply Moon relay boost if active
    if config.relay_via_moon:
        external *= (1 + MOON_RELAY_BOOST)

    advantage = compression_advantage(internal, external)
    sovereign = is_sovereign(advantage)

    return BodyState(
        id="mars",
        delay_s=delay_s,
        bandwidth_share=0.0,
        relay_path=["mars", "moon", "earth"] if config.relay_via_moon else ["mars", "earth"],
        internal_rate=internal,
        external_rate=external,
        advantage=advantage,
        sovereign=sovereign,
        entropy=100.0,  # Initial entropy
        entropy_rate=0.0,
        status="nominal" if sovereign else "dependent",
    )


def generate_mars_subsystems(config: MarsConfig, sol: int, rng: np.random.Generator = None) -> MarsSubsystems:
    """Generate Mars subsystem states.

    Args:
        config: Mars configuration
        sol: Current sol
        rng: Random generator

    Returns:
        MarsSubsystems with all subsystem states
    """
    if rng is None:
        rng = np.random.default_rng(sol)

    # Atmosphere subsystem
    atmosphere = {
        "O2_pct": 21.0 + rng.normal(0, 0.5),
        "CO2_ppm": 400 + rng.normal(0, 50),
        "pressure_kPa": 101.3 + rng.normal(0, 1),
    }

    # Thermal subsystem
    thermal = {
        "T_hab_C": 22.0 + rng.normal(0, 2),
        "Q_in_W": 5000 + rng.normal(0, 200),
        "Q_out_W": 4800 + rng.normal(0, 200),
        "radiator_efficiency": config.radiator_area_m2 / 100.0,
    }

    # Resource subsystem
    resource = {
        "water_L": 1000 + rng.normal(0, 20),
        "food_kg": 500 + rng.normal(0, 10),
        "power_W": config.kilopower_units * 10000 + rng.normal(0, 500),
        "buffer_days": 90 + rng.integers(-10, 10),
    }

    # Decision subsystem (compression rates)
    delay_s = compute_mars_latency(sol)
    neuralink_mbps = config.neuralink_mbps if config.neuralink_fraction > 0 else 0.0
    internal = internal_compression_rate(
        config.crew_size,
        config.expertise_coverage,
        config.compute_flops,
        neuralink_mbps * config.neuralink_fraction
    )
    external = external_compression_rate(config.earth_bandwidth_mbps, delay_s, sol)
    if config.relay_via_moon:
        external *= (1 + MOON_RELAY_BOOST)

    decision = {
        "internal_rate": internal,
        "external_rate": external,
        "advantage": compression_advantage(internal, external),
        "sovereign": is_sovereign(compression_advantage(internal, external)),
    }

    return MarsSubsystems(
        atmosphere=atmosphere,
        thermal=thermal,
        resource=resource,
        decision=decision,
    )


def evolve_mars_state(state: BodyState, config: MarsConfig, sol: int,
                      bandwidth_share: float = 0.0,
                      congestion: float = 0.0,
                      solar_factor: float = 1.0,
                      relay_efficiency: float = 1.0,
                      moon_relay_active: bool = False) -> BodyState:
    """Evolve Mars state for one sol.

    Args:
        state: Current Mars state
        config: Mars configuration
        sol: Current sol
        bandwidth_share: Allocated bandwidth share
        congestion: Network congestion factor
        solar_factor: Solar activity factor
        relay_efficiency: Relay path efficiency
        moon_relay_active: Whether Moon relay is boosting external rate

    Returns:
        Updated BodyState
    """
    # Update light delay
    state.delay_s = compute_mars_latency(sol)

    # Calculate internal rate
    neuralink_mbps = config.neuralink_mbps if config.neuralink_fraction > 0 else 0.0
    internal = internal_compression_rate(
        config.crew_size,
        config.expertise_coverage,
        config.compute_flops,
        neuralink_mbps * config.neuralink_fraction
    )

    # Calculate external rate with network adjustments
    mask = conjunction_mask(sol, config)
    external = external_compression_rate(
        config.earth_bandwidth_mbps,
        state.delay_s,
        sol,
        relay_efficiency=relay_efficiency,
        congestion_factor=congestion,
        solar_factor=solar_factor
    )

    # Apply Moon relay boost
    if moon_relay_active or config.relay_via_moon:
        external *= (1 + MOON_RELAY_BOOST)

    # Apply conjunction mask
    external *= mask

    # Calculate entropy flows
    entropy_generated = internal * 86400 / 1000  # bits/sol (simplified)
    entropy_exported = external * 86400 / 1000 if state.sovereign else 0.0
    new_entropy = state.entropy + entropy_generated - entropy_exported

    # Update state
    state.bandwidth_share = bandwidth_share
    state.relay_path = ["mars", "moon", "earth"] if moon_relay_active else ["mars", "earth"]

    state = update_body_state(
        state, internal, external,
        entropy=new_entropy,
        entropy_generated=entropy_generated,
        entropy_exported=entropy_exported
    )

    return state


def assess_mars_status(state: BodyState, subsystems: MarsSubsystems) -> str:
    """Assess overall Mars status.

    Args:
        state: Current Mars body state
        subsystems: Mars subsystem states

    Returns:
        Status string
    """
    atmosphere = subsystems.atmosphere
    thermal = subsystems.thermal
    resource = subsystems.resource

    # Critical conditions
    if atmosphere.get("O2_pct", 21.0) < 18.0 or atmosphere.get("O2_pct", 21.0) > 25.0:
        return "critical"
    if thermal.get("T_hab_C", 22.0) < -10 or thermal.get("T_hab_C", 22.0) > 45:
        return "critical"
    if resource.get("buffer_days", 90) < 30:
        return "critical"

    # Warning conditions
    if atmosphere.get("O2_pct", 21.0) < 19.5 or atmosphere.get("O2_pct", 21.0) > 23.5:
        return "stressed"
    if thermal.get("T_hab_C", 22.0) < 0 or thermal.get("T_hab_C", 22.0) > 40:
        return "stressed"
    if resource.get("buffer_days", 90) < 60:
        return "stressed"

    # Sovereignty check
    if not state.sovereign:
        return "dependent"

    return "nominal"
