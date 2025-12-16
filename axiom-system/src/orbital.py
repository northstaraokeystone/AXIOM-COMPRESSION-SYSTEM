"""AXIOM-SYSTEM v2 Orbital Module - Debris field and Kessler threshold.

Status: NEW
Purpose: Track debris, satellite constellation, Kessler cascade risk

Grok: "73% threshold...10^5 objects >10cm [ESA 2025]"
"""

from dataclasses import dataclass
import random

from .core import emit_receipt
from .entropy import KESSLER_THRESHOLD, DEBRIS_COUNT_2025


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEBRIS_PER_LAUNCH = 0.01               # Probability of adding debris per launch
KESSLER_BLACKOUT_SOLS = 100            # Duration of launch blackout after Kessler
DEBRIS_NATURAL_DECAY = 0.0001          # Natural debris decay per sol


# ═══════════════════════════════════════════════════════════════════════════════
# ORBITAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrbitalState:
    """Orbital environment state.

    Attributes:
        debris_count: Tracked objects >10cm
        debris_ratio: debris_count / DEBRIS_COUNT_2025
        kessler_risk: debris_ratio / KESSLER_THRESHOLD
        kessler_active: debris_ratio >= KESSLER_THRESHOLD
        satellites_active: Active satellites (Starlink etc.)
        launch_blackout: Whether launches are blocked
        blackout_remaining_sols: Sols until blackout lifts
    """
    debris_count: int = 0
    debris_ratio: float = 0.0
    kessler_risk: float = 0.0
    kessler_active: bool = False
    satellites_active: int = 0
    launch_blackout: bool = False
    blackout_remaining_sols: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ORBITAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_orbital_state(debris_count: int = None, satellites: int = 5000) -> OrbitalState:
    """Create initial orbital state.

    Args:
        debris_count: Initial debris count (default DEBRIS_COUNT_2025 * 0.65)
        satellites: Active satellite count

    Returns:
        Initialized OrbitalState
    """
    if debris_count is None:
        debris_count = int(DEBRIS_COUNT_2025 * 0.65)  # 65% of threshold

    debris_ratio = debris_count / DEBRIS_COUNT_2025
    kessler_risk = debris_ratio / KESSLER_THRESHOLD

    return OrbitalState(
        debris_count=debris_count,
        debris_ratio=debris_ratio,
        kessler_risk=kessler_risk,
        kessler_active=debris_ratio >= KESSLER_THRESHOLD,
        satellites_active=satellites,
        launch_blackout=False,
        blackout_remaining_sols=0,
    )


def add_launch_debris(state: OrbitalState, n_launches: int, seed: int = None) -> OrbitalState:
    """Increment debris from launches.

    Each launch has DEBRIS_PER_LAUNCH probability of adding debris.

    Args:
        state: Current orbital state
        n_launches: Number of launches
        seed: Random seed for reproducibility

    Returns:
        Updated OrbitalState
    """
    rng = random.Random(seed)

    debris_added = 0
    for _ in range(n_launches):
        if rng.random() < DEBRIS_PER_LAUNCH:
            debris_added += rng.randint(1, 10)  # 1-10 debris pieces per event

    state.debris_count += debris_added
    state.debris_ratio = state.debris_count / DEBRIS_COUNT_2025
    state.kessler_risk = state.debris_ratio / KESSLER_THRESHOLD

    # Check Kessler threshold
    if state.debris_ratio >= KESSLER_THRESHOLD and not state.kessler_active:
        state = activate_blackout(state)

    return state


def add_collision_debris(state: OrbitalState, debris_added: int) -> OrbitalState:
    """Add debris from collision event.

    Args:
        state: Current orbital state
        debris_added: Number of debris pieces added

    Returns:
        Updated OrbitalState
    """
    state.debris_count += debris_added
    state.debris_ratio = state.debris_count / DEBRIS_COUNT_2025
    state.kessler_risk = state.debris_ratio / KESSLER_THRESHOLD

    # Destroy some satellites in collision
    satellites_destroyed = min(state.satellites_active, debris_added // 10)
    state.satellites_active -= satellites_destroyed

    # Check Kessler threshold
    if state.debris_ratio >= KESSLER_THRESHOLD and not state.kessler_active:
        state = activate_blackout(state)

    return state


def check_kessler(state: OrbitalState) -> bool:
    """Check if Kessler threshold is reached.

    Grok: "73% threshold"

    Args:
        state: Current orbital state

    Returns:
        True if debris_ratio >= KESSLER_THRESHOLD
    """
    return state.debris_ratio >= KESSLER_THRESHOLD


def activate_blackout(state: OrbitalState) -> OrbitalState:
    """Activate launch blackout due to Kessler syndrome.

    Args:
        state: Current orbital state

    Returns:
        Updated OrbitalState with blackout active
    """
    state.kessler_active = True
    state.launch_blackout = True
    state.blackout_remaining_sols = KESSLER_BLACKOUT_SOLS
    return state


def evolve_orbital_state(state: OrbitalState, sol: int, n_launches: int = 0,
                         seed: int = None) -> OrbitalState:
    """Evolve orbital state for one sol.

    Args:
        state: Current orbital state
        sol: Current sol
        n_launches: Number of launches this sol
        seed: Random seed

    Returns:
        Updated OrbitalState
    """
    # Natural debris decay
    decay = int(state.debris_count * DEBRIS_NATURAL_DECAY)
    state.debris_count = max(0, state.debris_count - decay)

    # Add launch debris if not in blackout
    if n_launches > 0 and not state.launch_blackout:
        state = add_launch_debris(state, n_launches, seed)

    # Update blackout countdown
    if state.launch_blackout:
        state.blackout_remaining_sols -= 1
        if state.blackout_remaining_sols <= 0:
            state.launch_blackout = False
            state.blackout_remaining_sols = 0

    # Update derived values
    state.debris_ratio = state.debris_count / DEBRIS_COUNT_2025
    state.kessler_risk = state.debris_ratio / KESSLER_THRESHOLD
    state.kessler_active = state.debris_ratio >= KESSLER_THRESHOLD

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION
# ═══════════════════════════════════════════════════════════════════════════════

def emit_orbital_receipt(state: OrbitalState) -> dict:
    """Emit CLAUDEME-compliant orbital_state_receipt.

    Args:
        state: Current orbital state

    Returns:
        Receipt dict
    """
    data = {
        "debris_count": state.debris_count,
        "debris_ratio": state.debris_ratio,
        "kessler_risk": state.kessler_risk,
        "kessler_active": state.kessler_active,
        "kessler_threshold": KESSLER_THRESHOLD,
        "satellites_active": state.satellites_active,
        "launch_blackout": state.launch_blackout,
        "blackout_remaining_sols": state.blackout_remaining_sols,
    }
    return emit_receipt("orbital_state", data)
