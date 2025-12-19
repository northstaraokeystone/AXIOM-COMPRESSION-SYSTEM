"""Coordination hub modules for multi-body systems.

Hub types:
- jovian: 4-moon coordination (Io, Europa, Ganymede, Callisto)
- solar: 3-planet coordination (inner system)
- interstellar: 7-body backbone for deep space
"""

from .jovian import (
    coordinate_jovian_moons,
    integrate_unified_rl,
    coordinate_titan_europa,
    compute_jovian_autonomy,
    coordinate_jovian_system,
    compute_system_autonomy,
    integrate_jovian_hub,
    coordinate_four_moons,
    select_hub_location,
)
from .solar import (
    integrate_solar_hub,
    compute_solar_hub_autonomy,
    coordinate_inner_system,
    compute_full_system_coverage,
)
from .interstellar import (
    integrate_interstellar_backbone,
    compute_interstellar_autonomy,
    coordinate_full_system,
    get_backbone_status,
)

__all__ = [
    # Jovian
    "coordinate_jovian_moons", "integrate_unified_rl", "coordinate_titan_europa",
    "compute_jovian_autonomy", "coordinate_jovian_system", "compute_system_autonomy",
    "integrate_jovian_hub", "coordinate_four_moons", "select_hub_location",
    # Solar
    "integrate_solar_hub", "compute_solar_hub_autonomy",
    "coordinate_inner_system", "compute_full_system_coverage",
    # Interstellar
    "integrate_interstellar_backbone", "compute_interstellar_autonomy",
    "coordinate_full_system", "get_backbone_status",
]

RECEIPT_SCHEMA = {
    "module": "src.paths.multiplanet.hubs",
    "receipt_types": [
        "mp_jovian_coordinate", "mp_jovian_hub",
        "mp_solar_hub", "mp_solar_coordinate",
        "mp_interstellar_backbone", "mp_interstellar_coordinate",
    ],
    "version": "1.0.0",
}
