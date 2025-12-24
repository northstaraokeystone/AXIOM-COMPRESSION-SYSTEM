"""SpaceProof Utilities Package - Shared utilities for config, specs, and validation.

This package provides:
- config: Generic config loading with dual-hash verification
- spec_loader: Spec file loading with base + delta merging
- constants: Shared constants across modules
- autonomy: Common autonomy computation patterns
"""

from .config import load_spec, get_spec_path
from .spec_loader import load_depth_spec, get_depth_config
from .constants import (
    DEFAULT_TENANT_ID,
    DEFAULT_GATE,
    JOVIAN_MOONS,
    DEPTH_SPECS,
)
from .autonomy import compute_autonomy_from_latency

__all__ = [
    "load_spec",
    "get_spec_path",
    "load_depth_spec",
    "get_depth_config",
    "compute_autonomy_from_latency",
    "DEFAULT_TENANT_ID",
    "DEFAULT_GATE",
    "JOVIAN_MOONS",
    "DEPTH_SPECS",
]
