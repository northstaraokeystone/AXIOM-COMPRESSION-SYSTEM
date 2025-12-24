"""SpaceProof Mars Analog Package - Consolidated Mars validation modules.

This package consolidates:
- atacama_validation.py -> mars_analog.atacama
- atacama_drone.py -> mars_analog.drone
- atacama_dust_dynamics.py -> mars_analog.dust
- cfd_dust_dynamics.py -> mars_analog.cfd
- nrel_validation.py -> mars_analog.nrel

All Mars analog validation shares common patterns for:
- Dust calibration
- Efficiency projection
- Solar flux correction
"""

from .base import MarsAnalogBase

__all__ = [
    "MarsAnalogBase",
]
