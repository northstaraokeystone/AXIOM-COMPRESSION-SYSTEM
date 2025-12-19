"""src/paths/multiplanet/federation/__init__.py - Multi-star federation package."""

from src.paths.multiplanet.federation.stub import (
    load_federation_config,
    initialize_federation,
    consensus_with_lag,
    autonomous_arbitration,
    federation_status,
)

__all__ = [
    "load_federation_config",
    "initialize_federation",
    "consensus_with_lag",
    "autonomous_arbitration",
    "federation_status",
]
