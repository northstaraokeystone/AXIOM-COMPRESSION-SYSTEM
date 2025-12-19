"""Moon-specific hybrid autonomy modules.

Each moon has specialized autonomy requirements:
- titan: Methane lakes, 99% autonomy, 70-90 min latency
- europa: Ice drilling, 95% autonomy, 33-53 min latency
- ganymede: Magnetic navigation, 97% autonomy
- callisto: Hub coordination, radiation shielding
"""

from .titan import (
    integrate_titan,
    compute_titan_autonomy,
    simulate_titan_methane,
)
from .europa import (
    integrate_europa,
    compute_europa_autonomy,
    simulate_europa_drilling,
)
from .ganymede import (
    integrate_ganymede,
    compute_ganymede_autonomy,
    simulate_ganymede_navigation,
)
from .callisto import (
    integrate_callisto,
    compute_callisto_autonomy,
)

__all__ = [
    # Titan
    "integrate_titan", "compute_titan_autonomy", "simulate_titan_methane",
    # Europa
    "integrate_europa", "compute_europa_autonomy", "simulate_europa_drilling",
    # Ganymede
    "integrate_ganymede", "compute_ganymede_autonomy", "simulate_ganymede_navigation",
    # Callisto
    "integrate_callisto", "compute_callisto_autonomy",
]

RECEIPT_SCHEMA = {
    "module": "src.paths.multiplanet.moons",
    "receipt_types": [
        "mp_titan_integrate", "mp_titan_autonomy",
        "mp_europa_integrate", "mp_europa_autonomy",
        "mp_ganymede_integrate", "mp_ganymede_autonomy",
        "mp_callisto_integrate", "mp_callisto_autonomy",
    ],
    "version": "1.0.0",
}
