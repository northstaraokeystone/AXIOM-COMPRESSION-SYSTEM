"""AXIOM-SYSTEM v2 Earth Module - Anchor node.

Status: NEW
Purpose: Infinite bandwidth source, launch capacity

Earth is the anchor:
- delay_s = 0
- internal = infinity (always sovereign)
- external = infinity (source of all comms)
- sovereign = True always
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..core import emit_receipt
from .base import BodyState


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

EARTH_BANDWIDTH_TOTAL = 1.0            # Normalized anchor capacity
STARSHIP_CAPACITY_PER_YEAR = 150       # Grok: "150 flights/yr projected"
STARSHIP_PAYLOAD_KG = 150000           # 150 tons to orbit


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION QUEUE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Mission:
    """Starship mission in queue.

    Attributes:
        id: Mission identifier
        destination: Target body (moon, mars, orbital)
        type: "crew" | "cargo"
        payload_kg: Payload mass
        scheduled_sol: Scheduled launch sol
        priority: Priority (higher = more important)
    """
    id: str = ""
    destination: str = ""
    type: str = "cargo"
    payload_kg: float = 0.0
    scheduled_sol: int = 0
    priority: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# EARTH STATE
# ═══════════════════════════════════════════════════════════════════════════════

def create_earth_state() -> BodyState:
    """Create Earth anchor state.

    Earth is always sovereign with infinite capacity.

    Returns:
        BodyState for Earth
    """
    return BodyState(
        id="earth",
        delay_s=0.0,
        bandwidth_share=EARTH_BANDWIDTH_TOTAL,
        relay_path=["earth"],
        internal_rate=float('inf'),
        external_rate=float('inf'),
        advantage=0.0,  # N/A for anchor
        sovereign=True,  # Always
        entropy=0.0,
        entropy_rate=0.0,
        status="nominal",
    )


def launch_starship(queue: List[Mission], destination: str) -> tuple:
    """Launch next Starship to destination.

    Removes mission from queue and returns launch info.

    Grok: "Mars cargo priority shifts Moon crew by 7 sols avg"

    Args:
        queue: Current mission queue
        destination: Target body

    Returns:
        (updated_queue, launched_mission or None)
    """
    # Find first mission to destination
    for i, mission in enumerate(queue):
        if mission.destination == destination:
            launched = queue.pop(i)
            return queue, launched

    return queue, None


def queue_starship(queue: List[Mission], mission: Mission) -> List[Mission]:
    """Add mission to queue.

    Sorted by priority then scheduled_sol.

    Args:
        queue: Current mission queue
        mission: Mission to add

    Returns:
        Updated queue
    """
    queue.append(mission)
    queue.sort(key=lambda m: (-m.priority, m.scheduled_sol))
    return queue


def emit_queue_entropy_receipt(mission: Mission, queue_length: int, delay_sols: int) -> dict:
    """Emit queue_entropy_receipt when launch affects queue.

    Grok: Queue entropy has measured delay impact.

    Args:
        mission: Launched mission
        queue_length: Current queue length
        delay_sols: Delay caused to other missions

    Returns:
        Receipt dict
    """
    data = {
        "mission_id": mission.id,
        "destination": mission.destination,
        "mission_type": mission.type,
        "payload_kg": mission.payload_kg,
        "queue_length": queue_length,
        "delay_sols": delay_sols,
    }
    return emit_receipt("queue_entropy", data)
