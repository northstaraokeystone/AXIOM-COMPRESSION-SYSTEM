"""Interstellar backbone coordination hub.

Re-exports functions from src.interstellar_backbone for modular organization.
"""

from ....interstellar_backbone import (
    integrate_interstellar_backbone,
    compute_interstellar_autonomy,
    coordinate_full_system,
    get_backbone_status,
)

MP_TENANT_ID = "axiom-multiplanet"


__all__ = [
    "integrate_interstellar_backbone",
    "compute_interstellar_autonomy",
    "coordinate_full_system",
    "get_backbone_status",
]

RECEIPT_SCHEMA = {
    "module": "src.paths.multiplanet.hubs.interstellar",
    "receipt_types": ["mp_interstellar_backbone", "mp_interstellar_coordinate"],
    "version": "1.0.0",
}
