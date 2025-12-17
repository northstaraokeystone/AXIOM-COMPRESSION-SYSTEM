"""real_data - Real Dataset Loaders with Provenance Receipts

PARADIGM: Every claim traceable to measured data.

Modules:
    sparc: SPARC galaxy rotation curves (175 galaxies)
    nasa_pds: Perseverance MOXIE telemetry
    iss_eclss: ISS life support system data

Source: AXIOM Validation Lock v1 (Dec 17, 2025)
"""

from .sparc import load_sparc, get_galaxy, list_available, verify_checksum
from .nasa_pds import load_moxie, get_run, list_runs
from .iss_eclss import load_eclss, get_water_recovery, get_o2_closure

__all__ = [
    # SPARC
    "load_sparc",
    "get_galaxy",
    "list_available",
    "verify_checksum",
    # MOXIE
    "load_moxie",
    "get_run",
    "list_runs",
    # ECLSS
    "load_eclss",
    "get_water_recovery",
    "get_o2_closure",
]
