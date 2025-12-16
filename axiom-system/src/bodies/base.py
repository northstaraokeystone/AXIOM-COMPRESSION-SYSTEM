"""AXIOM-SYSTEM v2 Bodies Base Module - Shared BodyState template.

Status: NEW
Purpose: Common body state dataclass and utility functions
"""

from dataclasses import dataclass, field
from typing import List

from ..core import emit_receipt


@dataclass
class BodyState:
    """Shared body state template.

    Attributes:
        id: Body identifier (earth, moon, mars, orbital)
        delay_s: Light-time to Earth in seconds
        bandwidth_share: Allocated bandwidth from system (0-1)
        relay_path: Path to Earth via relay graph
        internal_rate: Internal compression rate (bits/sec)
        external_rate: Network-adjusted external rate (bits/sec)
        advantage: internal - external (THE KEY)
        sovereign: advantage > 0
        entropy: Local entropy (bits)
        entropy_rate: Change in entropy (bits/sol)
        entropy_generated: Entropy generated this sol
        entropy_exported: Entropy exported this sol
        status: "nominal" | "stressed" | "critical" | "failed"
    """
    id: str = ""
    delay_s: float = 0.0
    bandwidth_share: float = 0.0
    relay_path: List[str] = field(default_factory=list)
    internal_rate: float = 0.0
    external_rate: float = 0.0
    advantage: float = 0.0
    sovereign: bool = False
    entropy: float = 0.0
    entropy_rate: float = 0.0
    entropy_generated: float = 0.0
    entropy_exported: float = 0.0
    status: str = "nominal"


def compute_advantage(internal: float, external: float) -> float:
    """Compute compression advantage.

    THE KEY: internal - external

    Args:
        internal: Internal compression rate
        external: External compression rate

    Returns:
        Compression advantage
    """
    return internal - external


def is_sovereign(advantage: float) -> bool:
    """Check if body is sovereign.

    Sovereign when advantage > 0.

    Args:
        advantage: Compression advantage

    Returns:
        True if sovereign
    """
    return advantage > 0


def update_body_state(state: BodyState, internal: float, external: float,
                      entropy: float = None, entropy_generated: float = 0.0,
                      entropy_exported: float = 0.0) -> BodyState:
    """Update body state with new rates.

    Args:
        state: Current body state
        internal: New internal compression rate
        external: New external compression rate
        entropy: New entropy value (optional)
        entropy_generated: Entropy generated this sol
        entropy_exported: Entropy exported this sol

    Returns:
        Updated BodyState
    """
    state.internal_rate = internal
    state.external_rate = external
    state.advantage = compute_advantage(internal, external)
    state.sovereign = is_sovereign(state.advantage)
    state.entropy_generated = entropy_generated
    state.entropy_exported = entropy_exported

    if entropy is not None:
        prev_entropy = state.entropy
        state.entropy = entropy
        state.entropy_rate = entropy - prev_entropy

    # Update status based on sovereignty
    if state.sovereign:
        if state.status == "dependent":
            state.status = "nominal"
    else:
        if state.status == "nominal":
            state.status = "dependent"

    return state


def emit_body_receipt(state: BodyState) -> dict:
    """Emit CLAUDEME-compliant body_state_receipt.

    Args:
        state: Current body state

    Returns:
        Receipt dict
    """
    data = {
        "body_id": state.id,
        "delay_s": state.delay_s,
        "bandwidth_share": state.bandwidth_share,
        "relay_path": state.relay_path,
        "internal_rate": state.internal_rate,
        "external_rate": state.external_rate,
        "advantage": state.advantage,
        "sovereign": state.sovereign,
        "entropy": state.entropy,
        "entropy_rate": state.entropy_rate,
        "entropy_generated": state.entropy_generated,
        "entropy_exported": state.entropy_exported,
        "status": state.status,
    }
    return emit_receipt("body_state", data)
