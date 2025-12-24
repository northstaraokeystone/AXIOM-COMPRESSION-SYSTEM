"""Adaptive gravity functions for multi-planet operations.

Provides functions to adjust operational parameters based on
planetary gravity conditions.
"""

from typing import Any, Dict

from .constants import GRAVITY_EARTH, PLANET_GRAVITY_MAP


def get_gravity_for_planet(planet: str) -> float:
    """Get gravity for a planet or moon.

    Args:
        planet: Planet or moon name.

    Returns:
        float: Gravity in Earth g units.
    """
    return PLANET_GRAVITY_MAP.get(planet.lower(), GRAVITY_EARTH)


def calculate_adjustment_factor(gravity_g: float) -> float:
    """Calculate timing adjustment factor for gravity.

    Lower gravity = longer physical process times = higher factor.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        float: Adjustment factor (1.0 = Earth).
    """
    if gravity_g <= 0:
        return 1.0

    gravity_ratio = gravity_g / GRAVITY_EARTH
    return 1.0 / (gravity_ratio**0.5)


def get_consensus_timing(gravity_g: float) -> Dict[str, int]:
    """Get consensus timing parameters for gravity.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        dict: Consensus timing in milliseconds.
    """
    factor = calculate_adjustment_factor(gravity_g)

    # Base values (Earth)
    base_heartbeat_ms = 60000  # 1 minute
    base_election_timeout_ms = 300000  # 5 minutes
    base_batch_interval_ms = 10000  # 10 seconds

    return {
        "heartbeat_ms": int(base_heartbeat_ms * factor),
        "election_timeout_ms": int(base_election_timeout_ms * factor),
        "batch_interval_ms": int(base_batch_interval_ms * factor),
        "adjustment_factor": factor,
    }


def get_packet_timing(gravity_g: float) -> Dict[str, int]:
    """Get packet timing parameters for gravity.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        dict: Packet timing in milliseconds.
    """
    factor = calculate_adjustment_factor(gravity_g)

    # Cap packet timing at 2x to prevent excessive delays
    packet_factor = min(2.0, factor)

    # Base values (Earth)
    base_timeout_ms = 5000  # 5 seconds
    base_retry_delay_ms = 1000  # 1 second
    base_ack_timeout_ms = 2000  # 2 seconds

    return {
        "timeout_ms": int(base_timeout_ms * packet_factor),
        "retry_delay_ms": int(base_retry_delay_ms * packet_factor),
        "ack_timeout_ms": int(base_ack_timeout_ms * packet_factor),
        "adjustment_factor": packet_factor,
    }


def get_autonomy_threshold(gravity_g: float) -> float:
    """Get autonomy threshold for gravity.

    Lower gravity environments require higher autonomy.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        float: Autonomy threshold (0-1).
    """
    base_autonomy = 0.999

    # Lower gravity = higher autonomy needed
    gravity_ratio = gravity_g / GRAVITY_EARTH if gravity_g > 0 else 1.0
    adjustment = max(0.0, (1.0 - gravity_ratio) * 0.001)

    return min(1.0, base_autonomy + adjustment)


def get_full_adjustment(planet: str) -> Dict[str, Any]:
    """Get full operational adjustment for a planet.

    Args:
        planet: Planet or moon name.

    Returns:
        dict: Complete adjustment parameters.
    """
    gravity_g = get_gravity_for_planet(planet)

    return {
        "planet": planet,
        "gravity_g": gravity_g,
        "timing_factor": calculate_adjustment_factor(gravity_g),
        "consensus": get_consensus_timing(gravity_g),
        "packet": get_packet_timing(gravity_g),
        "autonomy_threshold": get_autonomy_threshold(gravity_g),
    }
