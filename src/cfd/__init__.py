"""CFD dust dynamics modules for Mars simulation.

Provides computational fluid dynamics functions organized by Reynolds regime:
- Laminar: Re < 2300 (Stokes settling)
- Turbulent: Re 2300-10000 (k-epsilon model)
- LES: Large Eddy Simulation (Smagorinsky subgrid)
- Atacama: Field validation with real-time mode
"""

from .core import (
    load_cfd_config,
    get_cfd_info,
    compute_reynolds_number,
    run_cfd_validation,
)
from .laminar import (
    stokes_settling,
    simulate_particle_trajectory,
)
from .turbulent import (
    simulate_dust_storm,
    compute_deposition_rate,
)
from .les import (
    load_les_config,
    get_les_info,
    smagorinsky_viscosity,
    compute_subgrid_stress,
    simulate_les,
    simulate_les_dust_devil,
    les_vs_rans_comparison,
    run_les_validation,
)
from .atacama import (
    load_atacama_realtime_config,
    get_atacama_realtime_info,
    atacama_les_realtime,
    track_dust_devil,
    realtime_feedback_loop,
    compute_realtime_correlation,
    run_atacama_validation,
    validate_against_atacama,
    project_mars_dynamics,
)

__all__ = [
    # Core
    "load_cfd_config",
    "get_cfd_info",
    "compute_reynolds_number",
    "run_cfd_validation",
    # Laminar
    "stokes_settling",
    "simulate_particle_trajectory",
    # Turbulent
    "simulate_dust_storm",
    "compute_deposition_rate",
    # LES
    "load_les_config",
    "get_les_info",
    "smagorinsky_viscosity",
    "compute_subgrid_stress",
    "simulate_les",
    "simulate_les_dust_devil",
    "les_vs_rans_comparison",
    "run_les_validation",
    # Atacama
    "load_atacama_realtime_config",
    "get_atacama_realtime_info",
    "atacama_les_realtime",
    "track_dust_devil",
    "realtime_feedback_loop",
    "compute_realtime_correlation",
    "run_atacama_validation",
    "validate_against_atacama",
    "project_mars_dynamics",
]

RECEIPT_SCHEMA = {
    "module": "src.cfd",
    "receipt_types": [
        "cfd_config",
        "cfd_info",
        "cfd_reynolds",
        "cfd_settling",
        "cfd_trajectory",
        "cfd_dust_storm",
        "les_config",
        "les_simulation",
        "atacama_realtime",
        "atacama_validation",
    ],
    "version": "1.0.0",
}
