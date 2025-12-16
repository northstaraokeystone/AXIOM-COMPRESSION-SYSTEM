"""AXIOM-SYSTEM v2 Bodies Module - Celestial body state management."""

from .base import BodyState, compute_advantage, is_sovereign, emit_body_receipt
from .earth import create_earth_state, EARTH_BANDWIDTH_TOTAL, STARSHIP_CAPACITY_PER_YEAR
from .moon import create_moon_state, enable_relay, MOON_LIGHT_DELAY_S, MOON_RELAY_EFFICIENCY
from .mars import create_mars_state, MarsConfig

__all__ = [
    'BodyState', 'compute_advantage', 'is_sovereign', 'emit_body_receipt',
    'create_earth_state', 'EARTH_BANDWIDTH_TOTAL', 'STARSHIP_CAPACITY_PER_YEAR',
    'create_moon_state', 'enable_relay', 'MOON_LIGHT_DELAY_S', 'MOON_RELAY_EFFICIENCY',
    'create_mars_state', 'MarsConfig',
]
