"""spec_loader.py - Unified spec loading for depth configurations.

Consolidates the repetitive get_d*_spec() and get_d*_uplift() functions.
"""

import json
from typing import Any, Dict

from .config import load_spec
from .constants import DEPTH_SPECS, DEFAULT_TENANT_ID
from ..core import emit_receipt, dual_hash


def load_depth_spec(depth: int) -> Dict[str, Any]:
    """Load spec for a given depth level.

    Replaces the repetitive get_d4_spec(), get_d5_spec(), ..., get_d10_spec() functions.

    Args:
        depth: Recursion depth (4-10)

    Returns:
        Dict with depth configuration

    Receipt: d{depth}_spec_load
    """
    if depth not in DEPTH_SPECS:
        raise ValueError(f"Invalid depth: {depth}. Must be 4-10.")

    depth_info = DEPTH_SPECS[depth]
    spec_file = depth_info["spec_file"]

    spec = load_spec(spec_file, emit=False)

    emit_receipt(
        f"d{depth}_spec_load",
        {
            "tenant_id": DEFAULT_TENANT_ID,
            "depth": depth,
            "version": spec.get("version", "1.0.0"),
            "alpha_floor": spec.get(f"d{depth}_config", {}).get(
                "alpha_floor", depth_info["alpha_floor"]
            ),
            "alpha_target": spec.get(f"d{depth}_config", {}).get(
                "alpha_target", depth_info["alpha_target"]
            ),
            "payload_hash": dual_hash(json.dumps(spec, sort_keys=True)),
        },
    )

    return spec


def get_depth_config(depth: int) -> Dict[str, Any]:
    """Get the configuration for a specific depth.

    Args:
        depth: Recursion depth (4-10)

    Returns:
        Dict with depth configuration including floor, target, ceiling
    """
    if depth not in DEPTH_SPECS:
        raise ValueError(f"Invalid depth: {depth}. Must be 4-10.")

    spec = load_depth_spec(depth)
    depth_info = DEPTH_SPECS[depth]

    return spec.get(f"d{depth}_config", {
        "alpha_floor": depth_info["alpha_floor"],
        "alpha_target": depth_info["alpha_target"],
        "alpha_ceiling": depth_info["alpha_ceiling"],
        "instability_max": 0.00,
    })


def get_depth_uplift(depth: int, level: int = None) -> float:
    """Get uplift value for a depth level.

    Args:
        depth: Recursion depth (4-10)
        level: Optional specific level (defaults to depth)

    Returns:
        Uplift value

    Note: This replaces get_d4_uplift(), get_d5_uplift(), etc.
    """
    if level is None:
        level = depth

    spec = load_depth_spec(depth)
    uplift_map = spec.get("uplift_by_depth", {})

    return float(uplift_map.get(str(level), DEPTH_SPECS.get(depth, {}).get("uplift", 0.0)))


def get_all_depth_constants(depth: int) -> Dict[str, Any]:
    """Get all constants for a depth level.

    Args:
        depth: Recursion depth (4-10)

    Returns:
        Dict with all depth constants
    """
    if depth not in DEPTH_SPECS:
        raise ValueError(f"Invalid depth: {depth}. Must be 4-10.")

    info = DEPTH_SPECS[depth]

    return {
        "alpha_floor": info["alpha_floor"],
        "alpha_target": info["alpha_target"],
        "alpha_ceiling": info["alpha_ceiling"],
        "instability_max": 0.00,
        "tree_min": 10**12,
        "uplift": info["uplift"],
        "spec_file": info["spec_file"],
    }
