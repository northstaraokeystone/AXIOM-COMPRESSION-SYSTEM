"""Planet-specific hybrid autonomy modules.

Each planet has specialized requirements:
- venus: Acid cloud navigation, thermal management
- mercury: Extreme thermal cycling, solar proximity
- mars: Dust/ISRU integration (handled in paths/mars/)
"""

from .venus import (
    integrate_venus,
    compute_venus_autonomy,
    coordinate_inner_planets,
    compute_solar_system_coverage,
)

__all__ = [
    # Venus
    "integrate_venus",
    "compute_venus_autonomy",
    "coordinate_inner_planets",
    "compute_solar_system_coverage",
]

RECEIPT_SCHEMA = {
    "module": "src.paths.multiplanet.planets",
    "receipt_types": [
        "mp_venus_integrate",
        "mp_venus_autonomy",
        "mp_inner_planets",
        "mp_solar_coverage",
    ],
    "version": "1.0.0",
}
