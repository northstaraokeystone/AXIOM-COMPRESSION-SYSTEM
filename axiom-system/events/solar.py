"""AXIOM-SYSTEM v2 Solar Events Module - CME and flare events.

Status: NEW
Purpose: Solar activity event generation and impact

Grok: "P(CME)=0.02/day [NOAA 2025]" - Cycle 25 peak
"""

from dataclasses import dataclass
from typing import Optional
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import emit_receipt
from src.entropy import CME_PROBABILITY_PER_DAY
from src.bodies.base import BodyState


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CME_RADIATION_FACTOR = 1.5             # 50% increase in radiation
CME_BANDWIDTH_FACTOR = 0.7             # 30% reduction in bandwidth
SHELTER_ENTROPY_COST = 10              # bits/sol per body during shelter
CME_DURATION_SOLS = 3                  # How long CME effects last
FLARE_PROBABILITY_PER_DAY = 0.05       # Minor flares more common


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CMEEvent:
    """Coronal Mass Ejection event.

    Attributes:
        sol: Sol when CME occurred
        intensity: 1.0 = normal, higher = more severe
        propagation_times: dict[str, float] body -> arrival time in sols
        duration_sols: How long effects last
    """
    sol: int = 0
    intensity: float = 1.0
    propagation_times: dict = None
    duration_sols: int = CME_DURATION_SOLS

    def __post_init__(self):
        if self.propagation_times is None:
            self.propagation_times = {}


@dataclass
class FlareEvent:
    """Solar flare event (less severe than CME)."""
    sol: int = 0
    intensity: float = 0.5
    bandwidth_impact: float = 0.1  # 10% reduction


# ═══════════════════════════════════════════════════════════════════════════════
# SOLAR EVENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def roll_cme(sol: int, seed: int = None) -> bool:
    """Roll for CME event.

    Grok: "P(CME)=0.02/day" from NOAA Cycle 25 data.

    Args:
        sol: Current sol
        seed: Random seed for reproducibility

    Returns:
        True if CME occurs
    """
    rng = random.Random(seed if seed is not None else sol)
    return rng.random() < CME_PROBABILITY_PER_DAY


def roll_flare(sol: int, seed: int = None) -> bool:
    """Roll for solar flare event.

    Args:
        sol: Current sol
        seed: Random seed

    Returns:
        True if flare occurs
    """
    rng = random.Random(seed if seed is not None else sol + 1000)
    return rng.random() < FLARE_PROBABILITY_PER_DAY


def create_cme_event(sol: int, seed: int = None) -> CMEEvent:
    """Create CME event with propagation times.

    Args:
        sol: Sol when CME occurred
        seed: Random seed

    Returns:
        CMEEvent with propagation times to each body
    """
    rng = random.Random(seed if seed is not None else sol)

    # CME intensity varies
    intensity = 0.8 + rng.random() * 0.4  # 0.8 to 1.2

    # Propagation times (CME travels at 400-1000 km/s)
    # Earth: 1-4 days, Moon: same as Earth, Mars: 3-10 days
    propagation_times = {
        "earth": 0.0,  # Origin
        "moon": rng.uniform(0.001, 0.005),  # ~1-5 hours
        "orbital": 0.0,  # LEO, same as Earth
        "mars": rng.uniform(0.5, 2.0),  # 0.5-2 sols for Mars
    }

    return CMEEvent(
        sol=sol,
        intensity=intensity,
        propagation_times=propagation_times,
        duration_sols=CME_DURATION_SOLS,
    )


def cme_arrival_time(body_delay_s: float, cme_speed_km_s: float = 700) -> float:
    """Calculate CME arrival time based on distance.

    Args:
        body_delay_s: Light-time delay to body in seconds
        cme_speed_km_s: CME speed in km/s (default 700)

    Returns:
        Arrival time in sols
    """
    # Light travels at 299,792 km/s
    distance_km = body_delay_s * 299792
    travel_time_s = distance_km / cme_speed_km_s
    return travel_time_s / 86400  # Convert to sols


def cme_impact(body_state: BodyState, cme: CMEEvent, current_sol: int) -> BodyState:
    """Apply CME impact to body state.

    Grok: "shelter decisions increase entropy export to Earth"

    Args:
        body_state: Current body state
        cme: CME event
        current_sol: Current sol

    Returns:
        Updated BodyState
    """
    arrival_sol = cme.sol + cme.propagation_times.get(body_state.id, 1.0)

    # Check if CME has arrived and is still active
    if current_sol < arrival_sol:
        return body_state  # Not arrived yet
    if current_sol > arrival_sol + cme.duration_sols:
        return body_state  # Effects have passed

    # Apply CME effects
    # Reduce external rate due to bandwidth impact
    body_state.external_rate *= CME_BANDWIDTH_FACTOR

    # Add shelter entropy cost
    body_state.entropy += SHELTER_ENTROPY_COST * cme.intensity
    body_state.entropy_generated += SHELTER_ENTROPY_COST * cme.intensity

    # Update status
    if body_state.status == "nominal":
        body_state.status = "stressed"

    return body_state


def get_solar_factor(cme_active: bool, cme_intensity: float = 1.0) -> float:
    """Get solar factor for external rate calculation.

    Args:
        cme_active: Whether CME is currently affecting system
        cme_intensity: CME intensity

    Returns:
        Solar factor (0-1), lower = more impact
    """
    if not cme_active:
        return 1.0
    return CME_BANDWIDTH_FACTOR / cme_intensity


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION
# ═══════════════════════════════════════════════════════════════════════════════

def emit_cme_receipt(cme: CMEEvent, affected_bodies: list) -> dict:
    """Emit CME event receipt.

    Args:
        cme: CME event
        affected_bodies: List of affected body IDs

    Returns:
        Receipt dict
    """
    data = {
        "event_type": "cme",
        "sol": cme.sol,
        "intensity": cme.intensity,
        "duration_sols": cme.duration_sols,
        "propagation_times": cme.propagation_times,
        "affected_bodies": affected_bodies,
        "probability_source": "NOAA Cycle 25",
    }
    return emit_receipt("solar_event", data)
