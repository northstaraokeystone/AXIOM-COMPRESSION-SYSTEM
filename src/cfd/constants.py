"""CFD constants for Mars and Atacama conditions.

PARADIGM:
    Physical constants and configuration parameters for CFD simulations.

Exports:
    - CFD constants for Mars atmosphere
    - LES (Large Eddy Simulation) constants
    - Atacama field validation constants
"""

from typing import Any, Dict

# === RECEIPT SCHEMA ===

RECEIPT_SCHEMA: Dict[str, Any] = {
    "cfd_constants": {
        "tenant_id": "axiom-cfd",
        "receipt_type": "cfd_constants",
        "description": "CFD and LES constants",
    }
}

# === CFD CONSTANTS ===

CFD_TENANT_ID = "axiom-cfd"
"""Tenant ID for CFD receipts."""

CFD_REYNOLDS_NUMBER_MARS = 50
"""Reynolds number for Mars atmosphere (laminar)."""

CFD_GRAVITY_MARS_M_S2 = 3.71
"""Mars gravity in m/s^2."""

CFD_VISCOSITY_MARS_PA_S = 1.1e-5
"""Mars atmosphere dynamic viscosity in Pa*s."""

CFD_DENSITY_MARS_KG_M3 = 0.02
"""Mars atmosphere density in kg/m^3."""

CFD_PARTICLE_DENSITY_KG_M3 = 2500
"""Dust particle density (basalt) in kg/m^3."""

CFD_PARTICLE_SIZE_UM = (1, 100)
"""Particle size range in micrometers."""

CFD_SETTLING_MODEL = "stokes"
"""Settling velocity model (Stokes for low-Re)."""

CFD_TURBULENCE_MODEL = "laminar"
"""Turbulence model (laminar for low-Re)."""


# === LES (LARGE EDDY SIMULATION) CONSTANTS ===

LES_SUBGRID_MODEL = "smagorinsky"
"""LES subgrid-scale model (Smagorinsky-Lilly)."""

LES_SMAGORINSKY_CONSTANT = 0.17
"""Smagorinsky constant (standard value)."""

LES_FILTER_WIDTH_M = 10
"""LES grid filter width in meters."""

LES_REYNOLDS_THRESHOLD = 10000
"""Reynolds number threshold for RANS→LES transition."""

LES_DUST_DEVIL_REYNOLDS = 50000
"""Typical Reynolds number for Mars dust devils (high-Re)."""

LES_DUST_DEVIL_DIAMETER_M = (1, 100)
"""Dust devil diameter range in meters."""

LES_DUST_DEVIL_HEIGHT_M = (10, 1000)
"""Dust devil height range in meters."""


# === ATACAMA REAL-TIME LES CONSTANTS ===

ATACAMA_LES_REALTIME = True
"""Atacama real-time LES mode enabled."""

ATACAMA_DRONE_SAMPLING_HZ = 100
"""Drone sampling frequency in Hz (upgraded from 10)."""

ATACAMA_LES_CORRELATION_TARGET = 0.95
"""Target correlation between LES and field data."""

ATACAMA_DUST_DEVIL_TRACKING = True
"""Dust devil tracking enabled."""

ATACAMA_REYNOLDS_NUMBER = 1090000
"""Atacama dust devil Reynolds number (Re=1.09M)."""

ATACAMA_TERRAIN_MODEL = "atacama"
"""Terrain model for Atacama simulations."""


# === EXPORTS ===

__all__ = [
    "RECEIPT_SCHEMA",
    "CFD_TENANT_ID",
    "CFD_REYNOLDS_NUMBER_MARS",
    "CFD_GRAVITY_MARS_M_S2",
    "CFD_VISCOSITY_MARS_PA_S",
    "CFD_DENSITY_MARS_KG_M3",
    "CFD_PARTICLE_DENSITY_KG_M3",
    "CFD_PARTICLE_SIZE_UM",
    "CFD_SETTLING_MODEL",
    "CFD_TURBULENCE_MODEL",
    "LES_SUBGRID_MODEL",
    "LES_SMAGORINSKY_CONSTANT",
    "LES_FILTER_WIDTH_M",
    "LES_REYNOLDS_THRESHOLD",
    "LES_DUST_DEVIL_REYNOLDS",
    "LES_DUST_DEVIL_DIAMETER_M",
    "LES_DUST_DEVIL_HEIGHT_M",
    "ATACAMA_LES_REALTIME",
    "ATACAMA_DRONE_SAMPLING_HZ",
    "ATACAMA_LES_CORRELATION_TARGET",
    "ATACAMA_DUST_DEVIL_TRACKING",
    "ATACAMA_REYNOLDS_NUMBER",
    "ATACAMA_TERRAIN_MODEL",
]
