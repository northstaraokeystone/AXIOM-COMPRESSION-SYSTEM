"""real_data - Real data loaders with provenance receipts.

Modules:
    sparc: SPARC galaxy rotation curves (with reproducible seed)
    nasa_pds: NASA Planetary Data System MOXIE telemetry
    iss_eclss: ISS Environmental Control and Life Support System data
"""

from .sparc import (
    SPARC_RANDOM_SEED,
    SPARC_TOTAL_GALAXIES,
    load_sparc,
    get_galaxy,
    list_available,
    verify_checksum,
)
from .nasa_pds import (
    PDS_RELEASE_VERSION,
    MOXIE_RUN_COUNT,
    MOXIE_EFFICIENCY_MIN,
    MOXIE_EFFICIENCY_MAX,
    MOXIE_EFFICIENCY_MEAN,
    MOXIE_EFFICIENCY_VARIANCE_PCT,
    load_moxie,
    get_run,
    list_runs,
    get_efficiency_stats,
)
from .iss_eclss import (
    ISS_WATER_RECOVERY,
    ISS_O2_CLOSURE,
    load_eclss,
    get_water_recovery,
    get_o2_closure,
)

__all__ = [
    # SPARC
    "SPARC_RANDOM_SEED",
    "SPARC_TOTAL_GALAXIES",
    "load_sparc",
    "get_galaxy",
    "list_available",
    "verify_checksum",
    # NASA PDS
    "PDS_RELEASE_VERSION",
    "MOXIE_RUN_COUNT",
    "MOXIE_EFFICIENCY_MIN",
    "MOXIE_EFFICIENCY_MAX",
    "MOXIE_EFFICIENCY_MEAN",
    "MOXIE_EFFICIENCY_VARIANCE_PCT",
    "load_moxie",
    "get_run",
    "list_runs",
    "get_efficiency_stats",
    # ISS ECLSS
    "ISS_WATER_RECOVERY",
    "ISS_O2_CLOSURE",
    "load_eclss",
    "get_water_recovery",
    "get_o2_closure",
]
