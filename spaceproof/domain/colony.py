"""colony.py - Mars Colony Domain Generator

SpaceProof domain generator for Mars colony state simulation.

THE COLONY INSIGHT:
    A colony is an information processing system.
    Its survival depends on decision quality under stress.
    Psychology entropy is the hidden variable.

Source: SpaceProof D20 Production Evolution
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..core import emit_receipt, dual_hash

# === CONSTANTS ===

TENANT_ID = "spaceproof"

# Colony parameters
DEFAULT_CREW_SIZE = 6
NOMINAL_STRESS = 0.3
CRITICAL_STRESS = 0.8

# Resource constants (kg per person per day)
O2_PER_PERSON_DAY = 0.84
WATER_PER_PERSON_DAY = 2.5
FOOD_PER_PERSON_DAY = 1.77

# Decision rate constants
HUMAN_DECISION_RATE_BPS = 10  # bits/sec per person
AUTONOMOUS_DECISION_RATE_BPS = 1000  # bits/sec for autonomous systems

# === NETWORK CONSTANTS (v3.0) ===

# Starship fleet (Grok: "500t payload", "1000 flights/year target")
STARSHIP_PAYLOAD_KG = 500000
STARSHIP_FLIGHTS_PER_YEAR = 1000

# Network scale (Grok: "1M colonists by 2050")
MARS_COLONIST_TARGET_2050 = 1_000_000
COLONY_NETWORK_SIZE_TARGET = 1000  # 1M @ 1000/colony

# Augmentation factors (Grok: "xAI autonomy")
AI_AUGMENTATION_FACTOR = 5.0
NEURALINK_AUGMENTATION_FACTOR = 20.0

# Inter-colony bandwidth
INTER_COLONY_BANDWIDTH_MBPS = 10.0


class ColonyPhase(Enum):
    """Colony development phase."""

    INITIAL = "initial"
    GROWTH = "growth"
    SELF_SUFFICIENT = "self_sufficient"
    SOVEREIGN = "sovereign"


@dataclass
class ColonyConfig:
    """Configuration for colony simulation.

    v3.0: Added network parameters for multi-colony support.
    """

    name: str = "Alpha Base"
    crew_size: int = DEFAULT_CREW_SIZE
    initial_stress: float = NOMINAL_STRESS
    autonomy_level: float = 0.3
    o2_kg: float = 1000.0
    water_kg: float = 5000.0
    food_kg: float = 10000.0
    # Network parameters (v3.0)
    colony_id: str = "C0001"
    network_id: Optional[str] = None
    position: Tuple[float, float] = (0.0, 0.0)  # km from reference
    inter_colony_bandwidth_mbps: float = INTER_COLONY_BANDWIDTH_MBPS
    augmentation_type: str = "human_only"  # human_only | ai_assisted | neuralink_assisted
    augmentation_factor: float = 1.0


@dataclass
class ColonyState:
    """State of a Mars colony at a point in time.

    v3.0: Added network-aware fields for multi-colony support.
    """

    sol: int = 0
    crew_count: int = DEFAULT_CREW_SIZE
    stress_level: float = NOMINAL_STRESS
    isolation_days: int = 0
    resources_sufficient: bool = True
    phase: str = "initial"
    sovereignty_ratio: float = 0.0
    h_total: float = 0.0
    # Network-aware fields (v3.0)
    colony_id: str = "C0001"
    network_connected: bool = True
    effective_crew: float = 0.0  # crew * augmentation_factor
    decision_capacity_bps: float = 0.0


def generate(config: ColonyConfig, days: int = 30) -> List[Dict]:
    """Generate colony state time series.

    Args:
        config: ColonyConfig with initial parameters
        days: Number of sols to simulate

    Returns:
        List of state dicts, one per sol
    """
    states = []

    # Calculate effective crew with augmentation (v3.0)
    effective_crew = config.crew_size * config.augmentation_factor
    decision_capacity = effective_crew * HUMAN_DECISION_RATE_BPS

    # Initialize state
    state = ColonyState(
        sol=0,
        crew_count=config.crew_size,
        stress_level=config.initial_stress,
        isolation_days=0,
        resources_sufficient=True,
        phase="initial",
        sovereignty_ratio=0.0,
        colony_id=config.colony_id,
        network_connected=config.network_id is not None,
        effective_crew=effective_crew,
        decision_capacity_bps=decision_capacity,
    )

    # Current resources
    o2 = config.o2_kg
    water = config.water_kg
    food = config.food_kg

    for sol in range(days):
        # Update state
        state.sol = sol

        # Consume resources
        o2 -= O2_PER_PERSON_DAY * config.crew_size
        water -= WATER_PER_PERSON_DAY * config.crew_size
        food -= FOOD_PER_PERSON_DAY * config.crew_size

        state.resources_sufficient = o2 > 0 and water > 0 and food > 0

        # Update stress
        if not state.resources_sufficient:
            state.stress_level = min(1.0, state.stress_level + 0.05)
        else:
            # Gradual stress recovery
            state.stress_level = max(0.1, state.stress_level - 0.01)

        # Update isolation
        state.isolation_days = sol

        # Compute entropy components
        h_thermal = 1.0
        h_atmospheric = 1.0
        h_resource = 1.0 if state.resources_sufficient else 1.5
        h_information = 1.0 + np.log2(1 + sol / 30)
        h_psychology = 1.0 + state.stress_level * 0.5

        state.h_total = h_thermal + h_atmospheric + h_resource + h_information + h_psychology

        # Compute sovereignty ratio with augmentation (v3.0)
        augmented_crew = config.crew_size * config.augmentation_factor
        internal_bps = (
            augmented_crew * HUMAN_DECISION_RATE_BPS * (1 - 0.3 * state.stress_level)
            + config.autonomy_level * AUTONOMOUS_DECISION_RATE_BPS
        )
        external_bps = 1e6 / (1 + sol / 100)  # Degrades over time
        state.sovereignty_ratio = internal_bps / external_bps if external_bps > 0 else float("inf")

        # Update effective crew and decision capacity
        state.effective_crew = augmented_crew * (1 - 0.3 * state.stress_level)
        state.decision_capacity_bps = internal_bps

        # Update phase
        if state.sovereignty_ratio > 1.0:
            state.phase = "sovereign"
        elif state.sovereignty_ratio > 0.5:
            state.phase = "self_sufficient"
        elif state.sovereignty_ratio > 0.2:
            state.phase = "growth"
        else:
            state.phase = "initial"

        states.append(
            {
                "sol": state.sol,
                "crew_count": state.crew_count,
                "stress_level": state.stress_level,
                "isolation_days": state.isolation_days,
                "resources_sufficient": state.resources_sufficient,
                "phase": state.phase,
                "sovereignty_ratio": state.sovereignty_ratio,
                "h_total": state.h_total,
                # Network-aware fields (v3.0)
                "colony_id": state.colony_id,
                "network_connected": state.network_connected,
                "effective_crew": state.effective_crew,
                "decision_capacity_bps": state.decision_capacity_bps,
            }
        )

    # Emit domain receipt
    data_hash = dual_hash(str(states))
    emit_receipt(
        "domain_receipt",
        {
            "tenant_id": TENANT_ID,
            "domain": "colony",
            "crew_size": config.crew_size,
            "duration_days": days,
            "events": [],
            "data_hash": data_hash,
        },
    )

    return states


def simulate_dust_storm(states: List[Dict], params: Dict) -> List[Dict]:
    """Inject dust storm event into colony state series.

    Args:
        states: List of colony state dicts
        params: Dust storm parameters (start_sol, duration, intensity)

    Returns:
        Modified states with dust storm effects
    """
    start_sol = params.get("start_sol", 10)
    duration = params.get("duration", 5)
    intensity = params.get("intensity", 0.5)

    for i, state in enumerate(states):
        if start_sol <= state["sol"] < start_sol + duration:
            # Dust storm effects: increased stress, reduced resources
            state["stress_level"] = min(1.0, state["stress_level"] + intensity * 0.2)
            state["h_total"] += intensity * 2.0

    emit_receipt(
        "colony_event",
        {
            "tenant_id": TENANT_ID,
            "event_type": "dust_storm",
            "start_sol": start_sol,
            "duration": duration,
            "intensity": intensity,
        },
    )

    return states


def simulate_hab_breach(states: List[Dict], params: Dict) -> List[Dict]:
    """Inject hab breach event into colony state series.

    Args:
        states: List of colony state dicts
        params: Breach parameters (sol, severity)

    Returns:
        Modified states with breach effects
    """
    breach_sol = params.get("sol", 15)
    severity = params.get("severity", 0.3)

    for i, state in enumerate(states):
        if state["sol"] >= breach_sol:
            # Breach effects: stress spike, reduced resources
            if state["sol"] == breach_sol:
                state["stress_level"] = min(1.0, state["stress_level"] + severity)
            state["resources_sufficient"] = state["resources_sufficient"] and (np.random.random() > severity * 0.1)

    emit_receipt(
        "colony_event",
        {
            "tenant_id": TENANT_ID,
            "event_type": "hab_breach",
            "sol": breach_sol,
            "severity": severity,
        },
    )

    return states
