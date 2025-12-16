"""AXIOM-SYSTEM v2 Moon Module - Low-latency relay node.

Status: NEW
Purpose: 1.3s delay relay capability

Moon is a relay node:
- delay_s = 1.3 (1.3 light-seconds)
- Can relay to Mars, improving Mars external rate by 40%
"""

from dataclasses import dataclass
from typing import Optional

from ..core import emit_receipt
from ..entropy import (
    internal_compression_rate,
    external_compression_rate,
    compression_advantage,
    is_sovereign,
    MOON_RELAY_BOOST,
)
from .base import BodyState, update_body_state


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MOON_LIGHT_DELAY_S = 1.3               # 1.3 light-seconds to Earth
MOON_RELAY_CAPABILITY = True           # Can act as relay
MOON_RELAY_EFFICIENCY = 0.85           # Relay efficiency Moon->Mars


# ═══════════════════════════════════════════════════════════════════════════════
# MOON CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoonConfig:
    """Moon colony configuration.

    Attributes:
        crew_size: Number of crew
        compute_flops: Computing capacity
        neuralink_fraction: Fraction with Neuralink
        expertise_coverage: Expertise coverage factor
        earth_bandwidth_mbps: Bandwidth to Earth
        relay_enabled: Whether relay to Mars is active
    """
    crew_size: int = 4
    compute_flops: float = 1e14
    neuralink_fraction: float = 0.0
    expertise_coverage: float = 0.8
    earth_bandwidth_mbps: float = 10.0  # Higher than Mars due to proximity
    relay_enabled: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# MOON STATE
# ═══════════════════════════════════════════════════════════════════════════════

def create_moon_state(config: MoonConfig = None, sol: int = 0) -> BodyState:
    """Create Moon colony state.

    Args:
        config: Moon configuration (uses defaults if None)
        sol: Current sol (for rate calculations)

    Returns:
        BodyState for Moon
    """
    if config is None:
        config = MoonConfig()

    # Calculate compression rates
    internal = internal_compression_rate(
        config.crew_size,
        config.expertise_coverage,
        config.compute_flops,
        config.neuralink_fraction
    )

    external = external_compression_rate(
        config.earth_bandwidth_mbps,
        MOON_LIGHT_DELAY_S,
        sol
    )

    advantage = compression_advantage(internal, external)
    sovereign = is_sovereign(advantage)

    return BodyState(
        id="moon",
        delay_s=MOON_LIGHT_DELAY_S,
        bandwidth_share=0.0,
        relay_path=["moon", "earth"],
        internal_rate=internal,
        external_rate=external,
        advantage=advantage,
        sovereign=sovereign,
        entropy=50.0,  # Initial entropy
        entropy_rate=0.0,
        status="nominal" if sovereign else "dependent",
    )


def enable_relay(network_state) -> None:
    """Enable Moon relay to Mars.

    Adds Moon->Mars edge with efficiency 0.85.
    Grok: "Moon relay upgrade -> Mars external_rate +40%"

    Args:
        network_state: NetworkState to modify (modified in place)
    """
    from ..network import add_relay, emit_relay_update_receipt

    add_relay(network_state.relay_graph, "moon", "mars", MOON_RELAY_EFFICIENCY)
    emit_relay_update_receipt("moon", "mars", MOON_RELAY_EFFICIENCY, "add")


def evolve_moon_state(state: BodyState, config: MoonConfig, sol: int,
                      bandwidth_share: float = 0.0,
                      congestion: float = 0.0,
                      solar_factor: float = 1.0) -> BodyState:
    """Evolve Moon state for one sol.

    Args:
        state: Current Moon state
        config: Moon configuration
        sol: Current sol
        bandwidth_share: Allocated bandwidth share
        congestion: Network congestion factor
        solar_factor: Solar activity factor

    Returns:
        Updated BodyState
    """
    # Calculate internal rate
    internal = internal_compression_rate(
        config.crew_size,
        config.expertise_coverage,
        config.compute_flops,
        config.neuralink_fraction
    )

    # Calculate external rate with network adjustments
    external = external_compression_rate(
        config.earth_bandwidth_mbps,
        MOON_LIGHT_DELAY_S,
        sol,
        relay_efficiency=1.0,  # Direct to Earth
        congestion_factor=congestion,
        solar_factor=solar_factor
    )

    # Calculate entropy flows
    entropy_generated = internal * 86400 / 1000  # bits/sol
    entropy_exported = external * 86400 / 1000 if state.sovereign else 0.0
    new_entropy = state.entropy + entropy_generated - entropy_exported

    # Update state
    state.bandwidth_share = bandwidth_share
    state = update_body_state(
        state, internal, external,
        entropy=new_entropy,
        entropy_generated=entropy_generated,
        entropy_exported=entropy_exported
    )

    return state
