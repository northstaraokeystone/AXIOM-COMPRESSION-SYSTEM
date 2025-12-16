"""AXIOM-SYSTEM v2 Events Module - Solar, debris, and network events."""

from .solar import roll_cme, create_cme_event, cme_arrival_time, cme_impact, CME_PROBABILITY_PER_DAY
from .debris import roll_collision, create_collision_event, collision_impact
from .network import roll_relay_failure, create_failure_event, failure_impact

__all__ = [
    'roll_cme', 'create_cme_event', 'cme_arrival_time', 'cme_impact', 'CME_PROBABILITY_PER_DAY',
    'roll_collision', 'create_collision_event', 'collision_impact',
    'roll_relay_failure', 'create_failure_event', 'failure_impact',
]
