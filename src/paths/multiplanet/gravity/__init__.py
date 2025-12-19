"""Gravity package for multi-planet adaptive operations.

Provides gravity-aware timing and parameter adjustments for different
planetary environments.
"""

from .constants import (
    GRAVITY_EARTH,
    GRAVITY_MARS,
    GRAVITY_VENUS,
    GRAVITY_MERCURY,
    GRAVITY_MOON,
    GRAVITY_TITAN,
    GRAVITY_EUROPA,
    GRAVITY_GANYMEDE,
    GRAVITY_CALLISTO,
    GRAVITY_IO,
    PLANET_GRAVITY_MAP,
)

from .adaptive import (
    get_gravity_for_planet,
    calculate_adjustment_factor,
    get_consensus_timing,
    get_packet_timing,
    get_autonomy_threshold,
)

__all__ = [
    # Constants
    "GRAVITY_EARTH",
    "GRAVITY_MARS",
    "GRAVITY_VENUS",
    "GRAVITY_MERCURY",
    "GRAVITY_MOON",
    "GRAVITY_TITAN",
    "GRAVITY_EUROPA",
    "GRAVITY_GANYMEDE",
    "GRAVITY_CALLISTO",
    "GRAVITY_IO",
    "PLANET_GRAVITY_MAP",
    # Functions
    "get_gravity_for_planet",
    "calculate_adjustment_factor",
    "get_consensus_timing",
    "get_packet_timing",
    "get_autonomy_threshold",
]
