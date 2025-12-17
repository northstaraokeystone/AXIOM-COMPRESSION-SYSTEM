"""Mars habitat autonomous optimization path.

Evolution path: stub -> simulate -> optimize -> autonomous

CURRENT STAGE: stub

Capabilities:
- stub_status(): Returns current stub status
- simulate_dome(): Dome simulation (stub)
- compute_isru_closure(): ISRU closure calculation
- compute_sovereignty(): Sovereignty check
- optimize_resources(): RL optimization (stub)

Source: AXIOM scalable paths architecture - Mars autonomous habitat
"""

from .core import (
    stub_status,
    simulate_dome,
    compute_isru_closure,
    compute_sovereignty,
    optimize_resources,
    get_mars_info,
    MARS_TENANT_ID,
    DEFAULT_CREW,
    ISRU_CLOSURE_TARGET,
    ISRU_UPLIFT_TARGET,
    DECISION_RATE_TARGET_BPS,
    DOME_RESOURCES,
)

from .receipts import (
    emit_mars_status,
    emit_mars_dome,
    emit_mars_isru,
    emit_mars_sovereignty,
    emit_mars_optimize,
)

from . import cli

__all__ = [
    # Core functions
    "stub_status",
    "simulate_dome",
    "compute_isru_closure",
    "compute_sovereignty",
    "optimize_resources",
    "get_mars_info",
    # Constants
    "MARS_TENANT_ID",
    "DEFAULT_CREW",
    "ISRU_CLOSURE_TARGET",
    "ISRU_UPLIFT_TARGET",
    "DECISION_RATE_TARGET_BPS",
    "DOME_RESOURCES",
    # Receipt helpers
    "emit_mars_status",
    "emit_mars_dome",
    "emit_mars_isru",
    "emit_mars_sovereignty",
    "emit_mars_optimize",
    # CLI module
    "cli",
]
