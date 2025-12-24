"""SpaceProof Moons Package - Consolidated Jovian moon hybrid simulations.

This package consolidates:
- titan_methane_hybrid.py -> moons.titan
- europa_ice_hybrid.py -> moons.europa
- ganymede_mag_hybrid.py -> moons.ganymede
- callisto_ice.py -> moons.callisto

All modules share ~60% identical structure through MoonHybridBase.
"""

from .base import MoonHybridBase

__all__ = [
    "MoonHybridBase",
]
