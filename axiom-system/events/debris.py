"""AXIOM-SYSTEM v2 Debris Events Module - Collision events, Kessler cascade.

Status: NEW
Purpose: Orbital collision event generation and cascade effects

Grok: "73% threshold...10^5 objects >10cm [ESA 2025]"
"""

from dataclasses import dataclass
from typing import Optional
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import emit_receipt
from src.orbital import OrbitalState
from src.entropy import KESSLER_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BASE_COLLISION_PROBABILITY = 0.001     # Base daily collision probability
DEBRIS_PER_COLLISION = 50              # Average debris created per collision
CASCADE_MULTIPLIER = 1.5               # Debris multiplication in cascade


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CollisionEvent:
    """Orbital collision event.

    Attributes:
        sol: Sol when collision occurred
        debris_added: Number of debris pieces created
        satellites_destroyed: Number of satellites destroyed
        cascade: Whether this triggered a cascade
    """
    sol: int = 0
    debris_added: int = 0
    satellites_destroyed: int = 0
    cascade: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# DEBRIS EVENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def roll_collision(orbital_state: OrbitalState, seed: int = None) -> bool:
    """Roll for collision event.

    Probability scales with debris_ratio^2 (collision probability increases
    quadratically with debris density).

    Args:
        orbital_state: Current orbital state
        seed: Random seed

    Returns:
        True if collision occurs
    """
    rng = random.Random(seed)

    # Quadratic scaling: more debris = much higher collision probability
    probability = BASE_COLLISION_PROBABILITY * (orbital_state.debris_ratio ** 2)

    return rng.random() < probability


def create_collision_event(sol: int, orbital_state: OrbitalState,
                           seed: int = None) -> CollisionEvent:
    """Create collision event.

    Args:
        sol: Sol when collision occurred
        orbital_state: Current orbital state
        seed: Random seed

    Returns:
        CollisionEvent with debris and satellite impact
    """
    rng = random.Random(seed)

    # Base debris from collision
    debris_added = int(DEBRIS_PER_COLLISION * (0.5 + rng.random()))

    # Cascade effect if near Kessler threshold
    cascade = False
    if orbital_state.debris_ratio > KESSLER_THRESHOLD * 0.9:
        cascade = True
        debris_added = int(debris_added * CASCADE_MULTIPLIER)

    # Satellite destruction (1-5 satellites per collision)
    satellites_destroyed = rng.randint(1, 5)

    return CollisionEvent(
        sol=sol,
        debris_added=debris_added,
        satellites_destroyed=satellites_destroyed,
        cascade=cascade,
    )


def collision_impact(orbital_state: OrbitalState, event: CollisionEvent) -> OrbitalState:
    """Apply collision impact to orbital state.

    Args:
        orbital_state: Current orbital state
        event: Collision event

    Returns:
        Updated OrbitalState
    """
    from src.orbital import add_collision_debris, check_kessler, activate_blackout

    # Add debris
    orbital_state = add_collision_debris(orbital_state, event.debris_added)

    # Check for Kessler syndrome
    if check_kessler(orbital_state) and not orbital_state.kessler_active:
        orbital_state = activate_blackout(orbital_state)

    return orbital_state


def assess_kessler_risk(orbital_state: OrbitalState) -> dict:
    """Assess Kessler syndrome risk.

    Args:
        orbital_state: Current orbital state

    Returns:
        Risk assessment dict
    """
    return {
        "debris_ratio": orbital_state.debris_ratio,
        "kessler_threshold": KESSLER_THRESHOLD,
        "risk_level": orbital_state.kessler_risk,
        "status": "CRITICAL" if orbital_state.kessler_active else
                  "HIGH" if orbital_state.debris_ratio > 0.6 else
                  "MODERATE" if orbital_state.debris_ratio > 0.4 else "LOW",
        "sols_to_kessler": estimate_sols_to_kessler(orbital_state),
    }


def estimate_sols_to_kessler(orbital_state: OrbitalState,
                              launches_per_sol: float = 0.4) -> Optional[int]:
    """Estimate sols until Kessler threshold reached.

    Args:
        orbital_state: Current orbital state
        launches_per_sol: Average launches per sol

    Returns:
        Estimated sols, or None if already at/past threshold
    """
    if orbital_state.debris_ratio >= KESSLER_THRESHOLD:
        return 0

    from src.orbital import DEBRIS_PER_LAUNCH, DEBRIS_COUNT_2025

    # Estimate debris growth rate
    debris_per_sol = launches_per_sol * DEBRIS_PER_LAUNCH * 5  # Average 5 debris per event

    # Current gap to threshold
    current_debris = orbital_state.debris_count
    threshold_debris = int(DEBRIS_COUNT_2025 * KESSLER_THRESHOLD)
    gap = threshold_debris - current_debris

    if debris_per_sol <= 0:
        return None

    return int(gap / debris_per_sol)


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION
# ═══════════════════════════════════════════════════════════════════════════════

def emit_collision_receipt(event: CollisionEvent, orbital_state: OrbitalState) -> dict:
    """Emit collision event receipt.

    Args:
        event: Collision event
        orbital_state: Current orbital state

    Returns:
        Receipt dict
    """
    data = {
        "event_type": "collision",
        "sol": event.sol,
        "debris_added": event.debris_added,
        "satellites_destroyed": event.satellites_destroyed,
        "cascade": event.cascade,
        "debris_ratio_after": orbital_state.debris_ratio,
        "kessler_active": orbital_state.kessler_active,
    }
    return emit_receipt("debris_event", data)
