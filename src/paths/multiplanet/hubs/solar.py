"""Solar system coordination hub.

Placeholder - functions imported from parent modules.
"""

from ....solar_orbital_hub import (
    compute_solar_hub_autonomy as _compute_solar_hub_autonomy,
    coordinate_inner_system as _coordinate_inner_system,
    compute_full_system_coverage as _compute_full_system_coverage,
)

MP_TENANT_ID = "axiom-multiplanet"


def integrate_solar_hub(config=None):
    """Wire solar hub to multiplanet path."""
    return {"integrated": True, "hub": "solar"}


def compute_solar_hub_autonomy():
    """Compute solar hub autonomy."""
    return _compute_solar_hub_autonomy()


def coordinate_inner_system(**kwargs):
    """Coordinate inner solar system."""
    return _coordinate_inner_system(**kwargs)


def compute_full_system_coverage(**kwargs):
    """Compute full system coverage."""
    return _compute_full_system_coverage(**kwargs)


__all__ = [
    "integrate_solar_hub",
    "compute_solar_hub_autonomy",
    "coordinate_inner_system",
    "compute_full_system_coverage",
]

RECEIPT_SCHEMA = {
    "module": "src.paths.multiplanet.hubs.solar",
    "receipt_types": ["mp_solar_hub", "mp_solar_coordinate"],
    "version": "1.0.0",
}
