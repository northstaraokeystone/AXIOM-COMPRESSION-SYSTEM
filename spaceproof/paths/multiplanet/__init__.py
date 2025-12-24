"""Multi-planet expansion sequence path.

Evolution path: stub -> sequence -> body_sim -> integrated

EXPANSION SEQUENCE:
1. Asteroid (3-20 min latency, 70% autonomy)
2. Mars (3-22 min latency, 85% autonomy)
3. Europa (33-53 min latency, 95% autonomy)
4. Titan (70-90 min latency, 99% autonomy)

Source: SpaceProof scalable paths architecture - Multi-planet expansion
"""

from .core import (
    stub_status,
    get_sequence,
    get_body_config,
    compute_latency_budget,
    compute_autonomy_requirement,
    simulate_body,
    compute_telemetry_compression,
    get_multiplanet_info,
    MULTIPLANET_TENANT_ID,
    EXPANSION_SEQUENCE,
    LATENCY_BOUNDS_MIN,
    LATENCY_BOUNDS_MAX,
    AUTONOMY_REQUIREMENT,
    BANDWIDTH_BUDGET_MBPS,
    TELEMETRY_COMPRESSION_TARGET,
)

from .receipts import (
    emit_mp_status,
    emit_mp_sequence,
    emit_mp_body,
    emit_mp_telemetry,
    emit_mp_latency,
    emit_mp_autonomy,
)

from . import cli

__all__ = [
    # Core functions
    "stub_status",
    "get_sequence",
    "get_body_config",
    "compute_latency_budget",
    "compute_autonomy_requirement",
    "simulate_body",
    "compute_telemetry_compression",
    "get_multiplanet_info",
    # Constants
    "MULTIPLANET_TENANT_ID",
    "EXPANSION_SEQUENCE",
    "LATENCY_BOUNDS_MIN",
    "LATENCY_BOUNDS_MAX",
    "AUTONOMY_REQUIREMENT",
    "BANDWIDTH_BUDGET_MBPS",
    "TELEMETRY_COMPRESSION_TARGET",
    # Receipt helpers
    "emit_mp_status",
    "emit_mp_sequence",
    "emit_mp_body",
    "emit_mp_telemetry",
    "emit_mp_latency",
    "emit_mp_autonomy",
    # CLI module
    "cli",
]
