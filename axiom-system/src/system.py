"""AXIOM-SYSTEM v2 System Module - Unified SystemState and simulation.

Status: NEW (replaces sim.py)
Purpose: One simulation. Entropy flows. Everything connects.

THE PARADIGM:
  - All bodies in ONE SystemState
  - Entropy conservation: sum_generated = sum_exported + sum_stored
  - System sovereignty: sum(advantages) > 0
  - Events cascade through relay graph
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import random

import numpy as np

from .core import emit_receipt, merkle, stoprule_conservation_violation
from .entropy import (
    entropy_conservation_check,
    system_entropy_budget,
    QUEUE_DELAY_SOLS,
)
from .network import (
    NetworkState,
    create_network_state,
    enable_moon_relay,
    update_network_state,
    relay_efficiency,
    emit_network_receipt,
)
from .orbital import (
    OrbitalState,
    create_orbital_state,
    evolve_orbital_state,
    emit_orbital_receipt,
)
from .bodies.base import BodyState, emit_body_receipt
from .bodies.earth import create_earth_state, Mission, queue_starship, launch_starship
from .bodies.moon import create_moon_state, evolve_moon_state, MoonConfig
from .bodies.mars import create_mars_state, evolve_mars_state, MarsConfig


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """System simulation configuration.

    Attributes:
        duration_sols: Simulation duration
        bodies_enabled: Which bodies to simulate
        moon_relay_enabled: Whether Moon relay to Mars is active
        neuralink_enabled: Whether Neuralink is available
        neuralink_mbps: Neuralink bandwidth per person
        mars_crew_size: Initial Mars crew
        moon_crew_size: Initial Moon crew
        random_seed: Reproducibility seed
        emit_receipts: Whether to emit receipts (for quiet runs)
    """
    duration_sols: int = 365
    bodies_enabled: List[str] = field(default_factory=lambda: ["earth", "moon", "mars", "orbital"])
    moon_relay_enabled: bool = False
    neuralink_enabled: bool = False
    neuralink_mbps: float = 1.0
    mars_crew_size: int = 10
    moon_crew_size: int = 4
    random_seed: int = 42
    emit_receipts: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# SOLAR STATE (CME tracking)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolarState:
    """Solar activity state.

    Attributes:
        cme_active: Whether CME is currently affecting system
        cme_intensity: Current CME intensity
        cme_remaining_sols: Sols until CME effects end
        flare_active: Whether solar flare is active
    """
    cme_active: bool = False
    cme_intensity: float = 0.0
    cme_remaining_sols: int = 0
    flare_active: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemState:
    """Unified system state.

    THE HEART OF AXIOM-SYSTEM v2.

    Attributes:
        sol: Current sol
        bodies: dict[str, BodyState] for all bodies
        orbital: OrbitalState
        solar: SolarState
        network: NetworkState
        starship_queue: List of pending missions
        total_entropy: System-wide entropy
        entropy_rate: System entropy change rate
        system_sovereign: Sum of advantages > 0
        cascade_risk: Risk factors by event type
        receipts: Accumulated receipts
        sovereignty_history: Track sovereignty by sol
    """
    sol: int = 0
    bodies: Dict[str, BodyState] = field(default_factory=dict)
    orbital: OrbitalState = field(default_factory=OrbitalState)
    solar: SolarState = field(default_factory=SolarState)
    network: NetworkState = field(default_factory=NetworkState)
    starship_queue: List[Mission] = field(default_factory=list)
    total_entropy: float = 0.0
    entropy_rate: float = 0.0
    system_sovereign: bool = False
    cascade_risk: Dict[str, float] = field(default_factory=dict)
    receipts: List[dict] = field(default_factory=list)
    sovereignty_history: List[Dict[str, bool]] = field(default_factory=list)
    entropy_generated: float = 0.0
    entropy_exported: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_system(config: SystemConfig) -> SystemState:
    """Create initial system state.

    Args:
        config: System configuration

    Returns:
        Initialized SystemState
    """
    state = SystemState()

    # Initialize bodies
    if "earth" in config.bodies_enabled:
        state.bodies["earth"] = create_earth_state()

    if "moon" in config.bodies_enabled:
        moon_config = MoonConfig(
            crew_size=config.moon_crew_size,
            neuralink_fraction=1.0 if config.neuralink_enabled else 0.0,
        )
        state.bodies["moon"] = create_moon_state(moon_config, sol=0)

    if "mars" in config.bodies_enabled:
        mars_config = MarsConfig(
            crew_size=config.mars_crew_size,
            neuralink_fraction=1.0 if config.neuralink_enabled else 0.0,
            neuralink_mbps=config.neuralink_mbps,
            relay_via_moon=config.moon_relay_enabled,
        )
        state.bodies["mars"] = create_mars_state(mars_config, sol=0)

    if "orbital" in config.bodies_enabled:
        # Orbital stations (simplified)
        state.bodies["orbital"] = BodyState(
            id="orbital",
            delay_s=0.0,
            bandwidth_share=0.1,
            relay_path=["orbital", "earth"],
            internal_rate=1.0,
            external_rate=10.0,
            advantage=-9.0,  # Dependent on Earth
            sovereign=False,
            entropy=10.0,
            status="nominal",
        )

    # Initialize network
    bodies_list = [b for b in config.bodies_enabled if b != "earth"]
    state.network = create_network_state(bodies_list)

    # Enable Moon relay if configured
    if config.moon_relay_enabled and "moon" in config.bodies_enabled:
        state.network = enable_moon_relay(state.network)

    # Initialize orbital environment
    state.orbital = create_orbital_state()

    # Initialize solar state
    state.solar = SolarState()

    # Calculate initial system values
    state = compute_system_entropy(state)
    state = compute_system_sovereignty(state)
    state = assess_cascade_risk_state(state)

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# TICK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def update_network(state: SystemState, config: SystemConfig) -> SystemState:
    """Recalculate bandwidth, relay efficiency, congestion.

    Args:
        state: Current system state
        config: System config

    Returns:
        Updated SystemState
    """
    # Calculate demands based on body needs
    demands = {}
    for body_id, body_state in state.bodies.items():
        if body_id == "earth":
            continue
        # Demand proportional to internal rate (more processing = more comms needed)
        demands[body_id] = body_state.internal_rate / 10.0

    # Update network allocation
    state.network = update_network_state(state.network, demands)

    # Update body bandwidth shares
    for body_id in state.network.bandwidth_allocation:
        if body_id in state.bodies:
            state.bodies[body_id].bandwidth_share = state.network.bandwidth_allocation[body_id]

    return state


def evolve_bodies(state: SystemState, config: SystemConfig) -> SystemState:
    """Evolve each body for one sol.

    Args:
        state: Current system state
        config: System config

    Returns:
        Updated SystemState
    """
    # Get solar factor
    solar_factor = 1.0 if not state.solar.cme_active else 0.7

    # Moon relay active?
    moon_relay_active = config.moon_relay_enabled and "moon" in state.bodies

    for body_id, body_state in state.bodies.items():
        if body_id == "earth":
            continue  # Earth doesn't evolve

        if body_id == "moon":
            moon_config = MoonConfig(
                crew_size=config.moon_crew_size,
                neuralink_fraction=1.0 if config.neuralink_enabled else 0.0,
            )
            congestion = state.network.congestion.get("moon", 0.0)
            state.bodies["moon"] = evolve_moon_state(
                body_state, moon_config, state.sol,
                bandwidth_share=state.network.bandwidth_allocation.get("moon", 0.0),
                congestion=congestion,
                solar_factor=solar_factor,
            )

        elif body_id == "mars":
            mars_config = MarsConfig(
                crew_size=config.mars_crew_size,
                neuralink_fraction=1.0 if config.neuralink_enabled else 0.0,
                neuralink_mbps=config.neuralink_mbps,
                relay_via_moon=moon_relay_active,
            )
            congestion = state.network.congestion.get("mars", 0.0)
            eff = relay_efficiency(state.network.relay_graph, "mars")
            state.bodies["mars"] = evolve_mars_state(
                body_state, mars_config, state.sol,
                bandwidth_share=state.network.bandwidth_allocation.get("mars", 0.0),
                congestion=congestion,
                solar_factor=solar_factor,
                relay_efficiency=eff,
                moon_relay_active=moon_relay_active,
            )

        elif body_id == "orbital":
            # Simple evolution for orbital
            body_state.entropy += 0.1  # Small entropy accumulation
            body_state.entropy_generated = 0.1
            body_state.entropy_exported = 0.0

    return state


def evolve_orbital(state: SystemState, config: SystemConfig) -> SystemState:
    """Update debris, check Kessler.

    Args:
        state: Current system state
        config: System config

    Returns:
        Updated SystemState
    """
    # Random launches (0-2 per sol)
    rng = random.Random(config.random_seed + state.sol)
    n_launches = rng.randint(0, 2) if not state.orbital.launch_blackout else 0

    state.orbital = evolve_orbital_state(
        state.orbital,
        state.sol,
        n_launches=n_launches,
        seed=config.random_seed + state.sol
    )

    return state


def check_events(state: SystemState, config: SystemConfig) -> SystemState:
    """Check for and handle events.

    Args:
        state: Current system state
        config: System config

    Returns:
        Updated SystemState with events applied
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from events.solar import roll_cme, create_cme_event
    from events.debris import roll_collision, create_collision_event
    from .cascade import propagate_event

    seed = config.random_seed + state.sol

    # Check for CME
    if roll_cme(state.sol, seed):
        cme = create_cme_event(state.sol, seed)
        state.solar.cme_active = True
        state.solar.cme_intensity = cme.intensity
        state.solar.cme_remaining_sols = cme.duration_sols
        state = propagate_event(state, cme)

    # Decay CME effects
    if state.solar.cme_active:
        state.solar.cme_remaining_sols -= 1
        if state.solar.cme_remaining_sols <= 0:
            state.solar.cme_active = False
            state.solar.cme_intensity = 0.0

    # Check for collision
    if roll_collision(state.orbital, seed + 1000):
        collision = create_collision_event(state.sol, state.orbital, seed + 1000)
        state = propagate_event(state, collision)

    return state


def compute_system_entropy(state: SystemState) -> SystemState:
    """Sum body entropies, compute rate.

    Args:
        state: Current system state

    Returns:
        Updated SystemState
    """
    prev_total = state.total_entropy

    # Sum all body entropies (excluding earth)
    body_entropies = {
        body_id: body.entropy
        for body_id, body in state.bodies.items()
        if body_id != "earth"
    }
    state.total_entropy = system_entropy_budget(body_entropies)

    # Compute rate
    state.entropy_rate = state.total_entropy - prev_total

    # Sum generated and exported
    state.entropy_generated = sum(
        body.entropy_generated for body in state.bodies.values() if body.id != "earth"
    )
    state.entropy_exported = sum(
        body.entropy_exported for body in state.bodies.values() if body.id != "earth"
    )

    return state


def compute_system_sovereignty(state: SystemState) -> SystemState:
    """Sum advantages, set system_sovereign.

    Grok: "sum(advantage for body in active) > 0"

    Args:
        state: Current system state

    Returns:
        Updated SystemState
    """
    total_advantage = sum(
        body.advantage for body in state.bodies.values()
        if body.id != "earth"
    )
    state.system_sovereign = total_advantage > 0

    # Record sovereignty history
    sovereignty_snapshot = {
        body_id: body.sovereign
        for body_id, body in state.bodies.items()
    }
    state.sovereignty_history.append(sovereignty_snapshot)

    return state


def assess_cascade_risk_state(state: SystemState) -> SystemState:
    """Update cascade_risk dict.

    Args:
        state: Current system state

    Returns:
        Updated SystemState
    """
    from .cascade import assess_cascade_risk
    state.cascade_risk = assess_cascade_risk(state)
    return state


def emit_system_receipt(state: SystemState, config: SystemConfig) -> dict:
    """Emit CLAUDEME-compliant system_tick_receipt.

    Args:
        state: Current system state
        config: System config

    Returns:
        Receipt dict
    """
    data = {
        "sol": state.sol,
        "total_entropy": state.total_entropy,
        "entropy_rate": state.entropy_rate,
        "entropy_generated": state.entropy_generated,
        "entropy_exported": state.entropy_exported,
        "system_sovereign": state.system_sovereign,
        "bodies_sovereign": {
            body_id: body.sovereign
            for body_id, body in state.bodies.items()
        },
        "cascade_risk": state.cascade_risk,
        "orbital_kessler_risk": state.orbital.kessler_risk,
        "solar_cme_active": state.solar.cme_active,
    }
    receipt = emit_receipt("system_tick", data)
    state.receipts.append(receipt)
    return receipt


def tick(state: SystemState, config: SystemConfig) -> SystemState:
    """One sol: solar->network->bodies->orbital->aggregate->receipts.

    Args:
        state: Current system state
        config: System config

    Returns:
        Updated SystemState
    """
    # Check and apply events
    state = check_events(state, config)

    # Update network
    state = update_network(state, config)

    # Evolve bodies
    state = evolve_bodies(state, config)

    # Evolve orbital
    state = evolve_orbital(state, config)

    # Aggregate system values
    state = compute_system_entropy(state)
    state = compute_system_sovereignty(state)
    state = assess_cascade_risk_state(state)

    # Emit receipts
    if config.emit_receipts:
        emit_system_receipt(state, config)

    # Advance sol
    state.sol += 1

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(config: SystemConfig) -> SystemState:
    """Run full simulation.

    Args:
        config: System configuration

    Returns:
        Final SystemState
    """
    # Initialize
    state = initialize_system(config)

    # Main loop
    for _ in range(config.duration_sols):
        state = tick(state, config)

    return state


def inject_event(state: SystemState, event) -> SystemState:
    """Inject event into running simulation.

    Args:
        state: Current system state
        event: Event to inject

    Returns:
        Updated SystemState
    """
    from .cascade import propagate_event
    return propagate_event(state, event)


# ═══════════════════════════════════════════════════════════════════════════════
# SOVEREIGNTY FLIP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def emit_sovereignty_flip_receipt(body_id: str, sol: int, new_sovereign: bool) -> dict:
    """Emit receipt when body sovereignty changes.

    Args:
        body_id: Body identifier
        sol: Sol when flip occurred
        new_sovereign: New sovereignty status

    Returns:
        Receipt dict
    """
    data = {
        "body_id": body_id,
        "sol": sol,
        "new_sovereign": new_sovereign,
        "flip_direction": "SOVEREIGN" if new_sovereign else "DEPENDENT",
    }
    return emit_receipt("sovereignty_flip", data)


def find_sovereignty_sol(state: SystemState, body_id: str) -> Optional[int]:
    """Find first sol when body became sovereign.

    Args:
        state: System state with sovereignty history
        body_id: Body to check

    Returns:
        Sol number or None if never sovereign
    """
    for sol, snapshot in enumerate(state.sovereignty_history):
        if snapshot.get(body_id, False):
            return sol
    return None
