"""colony.py - Mars Colony State and Psychology

THE COLONY INSIGHT:
    A colony is an information processing system.
    Its survival depends on decision quality under stress.
    Psychology entropy is the hidden variable.

Source: AXIOM Validation Lock v1
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import numpy as np
from pathlib import Path

# Import from src
try:
    from src.core import emit_receipt
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.core import emit_receipt

from .entropy import (
    crew_psychology_entropy,
    total_colony_entropy,
    landauer_mass_equivalent,
    CrewPsychologyState,
)


# === CONSTANTS ===

TENANT_ID = "axiom-colony"

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


class ColonyPhase(Enum):
    """Colony development phase."""
    INITIAL = "initial"
    GROWTH = "growth"
    SELF_SUFFICIENT = "self_sufficient"
    SOVEREIGN = "sovereign"


class CrewRole(Enum):
    """Crew member roles."""
    COMMANDER = "commander"
    PILOT = "pilot"
    ENGINEER = "engineer"
    SCIENTIST = "scientist"
    MEDICAL = "medical"
    SPECIALIST = "specialist"


# === DATA STRUCTURES ===

@dataclass
class CrewMember:
    """Individual crew member state."""
    id: str
    role: CrewRole
    stress_level: float = 0.3
    fatigue: float = 0.0
    morale: float = 1.0
    decision_rate_bps: float = HUMAN_DECISION_RATE_BPS


@dataclass
class CrewState:
    """Aggregate crew state."""
    members: List[CrewMember] = field(default_factory=list)
    psychology: CrewPsychologyState = field(default_factory=CrewPsychologyState)
    cohesion: float = 1.0
    communication_quality: float = 1.0


@dataclass
class ResourceState:
    """Colony resource levels."""
    o2_kg: float = 1000.0
    water_kg: float = 5000.0
    food_kg: float = 10000.0
    power_kw: float = 100.0
    spare_parts: int = 100


@dataclass
class EnvironmentState:
    """Mars environment state."""
    sol: int = 0
    temperature_c: float = -60.0
    dust_opacity: float = 0.3
    radiation_msv_per_day: float = 0.7
    comms_delay_min: float = 12.0
    is_conjunction: bool = False


@dataclass
class ColonyState:
    """Complete colony state."""
    name: str = "Alpha Base"
    phase: ColonyPhase = ColonyPhase.INITIAL
    crew: CrewState = field(default_factory=CrewState)
    resources: ResourceState = field(default_factory=ResourceState)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    autonomy_level: float = 0.3
    sovereignty_ratio: float = 0.0
    entropy_history: List[float] = field(default_factory=list)


# === INITIALIZATION ===

def initialize_crew(
    crew_size: int = DEFAULT_CREW_SIZE,
    initial_stress: float = NOMINAL_STRESS
) -> CrewState:
    """Initialize crew state.

    Args:
        crew_size: Number of crew members
        initial_stress: Initial stress level

    Returns:
        Initialized CrewState
    """
    roles = list(CrewRole)
    members = []

    for i in range(crew_size):
        role = roles[i % len(roles)]
        member = CrewMember(
            id=f"crew_{i:02d}",
            role=role,
            stress_level=initial_stress,
            fatigue=0.1 * np.random.rand(),
            morale=0.8 + 0.2 * np.random.rand(),
        )
        members.append(member)

    return CrewState(
        members=members,
        psychology=CrewPsychologyState(stress_level=initial_stress),
        cohesion=0.9,
        communication_quality=1.0,
    )


def initialize_colony(
    name: str = "Alpha Base",
    crew_size: int = DEFAULT_CREW_SIZE,
    initial_resources: ResourceState = None
) -> ColonyState:
    """Initialize complete colony state.

    Args:
        name: Colony name
        crew_size: Number of crew members
        initial_resources: Optional resource state

    Returns:
        Initialized ColonyState
    """
    crew = initialize_crew(crew_size)
    resources = initial_resources or ResourceState()

    colony = ColonyState(
        name=name,
        phase=ColonyPhase.INITIAL,
        crew=crew,
        resources=resources,
        environment=EnvironmentState(),
        autonomy_level=0.3,
        sovereignty_ratio=0.0,
    )

    # Emit initialization receipt
    emit_receipt("colony_init", {
        "tenant_id": TENANT_ID,
        "colony_name": name,
        "crew_size": crew_size,
        "initial_phase": colony.phase.value,
    })

    return colony


# === STATE UPDATES ===

def update_crew_stress(
    crew: CrewState,
    stress_delta: float,
    crisis_event: bool = False
) -> CrewState:
    """Update crew stress levels.

    Args:
        crew: Current crew state
        stress_delta: Change in stress
        crisis_event: Whether a crisis occurred

    Returns:
        Updated CrewState
    """
    # Update individual members
    for member in crew.members:
        member.stress_level = max(0, min(1, member.stress_level + stress_delta))
        if crisis_event:
            member.stress_level = min(1, member.stress_level + 0.2)
            member.morale = max(0, member.morale - 0.1)

    # Update aggregate psychology
    avg_stress = np.mean([m.stress_level for m in crew.members])
    crew.psychology.stress_level = avg_stress
    if crisis_event:
        crew.psychology.crisis_count += 1

    # Update cohesion (degrades under high stress)
    if avg_stress > 0.6:
        crew.cohesion = max(0.3, crew.cohesion - 0.05)
    elif avg_stress < 0.3:
        crew.cohesion = min(1.0, crew.cohesion + 0.02)

    return crew


def update_environment(
    env: EnvironmentState,
    days_elapsed: int = 1
) -> EnvironmentState:
    """Update Mars environment state.

    Args:
        env: Current environment state
        days_elapsed: Sols elapsed

    Returns:
        Updated EnvironmentState
    """
    env.sol += days_elapsed

    # Simulate dust storms (random)
    if np.random.rand() < 0.01:
        env.dust_opacity = min(1.0, env.dust_opacity + 0.3)
    else:
        env.dust_opacity = max(0.1, env.dust_opacity - 0.02)

    # Temperature variation
    env.temperature_c = -60 + 20 * np.sin(2 * np.pi * env.sol / 668)

    # Communication delay (varies with orbital position)
    # Simplified: 3-22 minutes
    env.comms_delay_min = 3 + 19 * (0.5 + 0.5 * np.sin(2 * np.pi * env.sol / 780))

    # Solar conjunction (roughly every 26 months, lasting ~2 weeks)
    conjunction_phase = (env.sol % 780) / 780
    env.is_conjunction = 0.48 < conjunction_phase < 0.52

    return env


def consume_resources(
    resources: ResourceState,
    crew_size: int,
    days: int = 1
) -> Tuple[ResourceState, bool]:
    """Consume daily resources.

    Args:
        resources: Current resource state
        crew_size: Number of crew members
        days: Number of days

    Returns:
        Tuple of (updated_resources, sufficient)
    """
    o2_needed = O2_PER_PERSON_DAY * crew_size * days
    water_needed = WATER_PER_PERSON_DAY * crew_size * days
    food_needed = FOOD_PER_PERSON_DAY * crew_size * days

    resources.o2_kg -= o2_needed
    resources.water_kg -= water_needed
    resources.food_kg -= food_needed

    sufficient = (
        resources.o2_kg > 0 and
        resources.water_kg > 0 and
        resources.food_kg > 0
    )

    return resources, sufficient


def compute_decision_capacity(crew: CrewState, autonomy_level: float) -> float:
    """Compute total colony decision capacity.

    Args:
        crew: Crew state
        autonomy_level: Level of autonomous systems (0-1)

    Returns:
        Total decision capacity in bits/sec
    """
    # Human contribution (degraded by stress and fatigue)
    human_bps = sum(
        m.decision_rate_bps * (1 - 0.3 * m.stress_level) * (1 - 0.2 * m.fatigue)
        for m in crew.members
    )

    # Autonomous systems contribution
    auto_bps = autonomy_level * AUTONOMOUS_DECISION_RATE_BPS

    # Communication quality affects both
    total_bps = (human_bps + auto_bps) * crew.communication_quality

    return total_bps


def update_colony_state(
    colony: ColonyState,
    days_elapsed: int = 1,
    crisis_event: bool = False,
    stress_delta: float = 0.0
) -> ColonyState:
    """Update complete colony state.

    Args:
        colony: Current colony state
        days_elapsed: Days elapsed
        crisis_event: Whether a crisis occurred
        stress_delta: External stress change

    Returns:
        Updated ColonyState
    """
    # Update environment
    colony.environment = update_environment(colony.environment, days_elapsed)

    # Communication blackout increases stress
    if colony.environment.is_conjunction:
        stress_delta += 0.02 * days_elapsed

    # Update crew
    colony.crew = update_crew_stress(colony.crew, stress_delta, crisis_event)
    colony.crew.psychology.isolation_days += days_elapsed

    # Consume resources
    colony.resources, sufficient = consume_resources(
        colony.resources,
        len(colony.crew.members),
        days_elapsed
    )

    if not sufficient:
        # Resource crisis
        colony.crew = update_crew_stress(colony.crew, 0.1, crisis_event=True)

    # Compute entropy components
    h_thermal = 1.0 + 0.01 * abs(colony.environment.temperature_c + 20)
    h_atmospheric = 1.0 + colony.environment.dust_opacity
    h_resource = 1.0 + (0.1 if not sufficient else 0)
    h_information = 1.0 + colony.environment.comms_delay_min / 22
    h_psychology = crew_psychology_entropy(
        colony.crew.psychology.stress_level,
        colony.crew.psychology.isolation_days,
        colony.crew.psychology.crisis_count,
        colony.crew.cohesion
    )

    h_total = total_colony_entropy(
        h_thermal, h_atmospheric, h_resource, h_information, h_psychology
    )
    colony.entropy_history.append(h_total)

    # Compute sovereignty
    decision_capacity = compute_decision_capacity(colony.crew, colony.autonomy_level)
    colony_mass_eq = landauer_mass_equivalent(decision_capacity)

    # External dependency (decreases as autonomy increases)
    external_support_kg = 60000 * (1 - colony.autonomy_level)
    colony.sovereignty_ratio = colony_mass_eq / external_support_kg if external_support_kg > 0 else float("inf")

    # Update phase based on sovereignty
    if colony.sovereignty_ratio > 1.0:
        colony.phase = ColonyPhase.SOVEREIGN
    elif colony.sovereignty_ratio > 0.5:
        colony.phase = ColonyPhase.SELF_SUFFICIENT
    elif colony.sovereignty_ratio > 0.2:
        colony.phase = ColonyPhase.GROWTH
    else:
        colony.phase = ColonyPhase.INITIAL

    # Emit update receipt
    emit_receipt("colony_update", {
        "tenant_id": TENANT_ID,
        "sol": colony.environment.sol,
        "crew_stress": colony.crew.psychology.stress_level,
        "h_psychology": h_psychology,
        "h_total": h_total,
        "sovereignty_ratio": colony.sovereignty_ratio,
        "phase": colony.phase.value,
        "is_conjunction": colony.environment.is_conjunction,
    })

    return colony


def simulate_blackout(
    colony: ColonyState,
    blackout_days: int = 43
) -> ColonyState:
    """Simulate communication blackout (Mars conjunction).

    Args:
        colony: Colony state
        blackout_days: Duration of blackout

    Returns:
        Colony state after blackout
    """
    # Force conjunction state
    colony.environment.is_conjunction = True
    colony.crew.communication_quality = 0.1  # Minimal relay only

    for day in range(blackout_days):
        # Higher stress accumulation during blackout
        stress_delta = 0.01 if day < 20 else 0.02
        colony = update_colony_state(
            colony,
            days_elapsed=1,
            stress_delta=stress_delta
        )

    # End conjunction
    colony.environment.is_conjunction = False
    colony.crew.communication_quality = 1.0

    emit_receipt("blackout_complete", {
        "tenant_id": TENANT_ID,
        "blackout_days": blackout_days,
        "final_stress": colony.crew.psychology.stress_level,
        "final_sovereignty": colony.sovereignty_ratio,
        "survived": colony.sovereignty_ratio > 0,
    })

    return colony
