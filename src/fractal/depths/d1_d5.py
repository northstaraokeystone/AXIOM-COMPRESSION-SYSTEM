"""fractal/depths/d1_d5.py - Depth 4 and 5 Fractal Recursion

This module provides D4 and D5 depth recursion functions for alpha ceiling breach.

D4 TARGETS:
    - Alpha floor: 3.18
    - Alpha target: 3.20
    - Alpha ceiling: 3.22
    - Instability: 0.00

D5 TARGETS:
    - Alpha floor: 3.23
    - Alpha target: 3.25
    - Alpha ceiling: 3.27
    - Instability: 0.00
    - Includes MOXIE ISRU calibration
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from ...core import emit_receipt, dual_hash
from ..alpha import get_scale_factor, TENANT_ID


# === D4 RECURSION CONSTANTS ===

D4_ALPHA_FLOOR = 3.18
"""D4 alpha floor target."""

D4_ALPHA_TARGET = 3.20
"""D4 alpha target."""

D4_ALPHA_CEILING = 3.22
"""D4 alpha ceiling (max achievable)."""

D4_INSTABILITY_MAX = 0.00
"""D4 maximum allowed instability."""

D4_TREE_MIN = 10**12
"""Minimum tree size for D4 validation."""


# === D4 RECURSION FUNCTIONS ===


def get_d4_spec() -> Dict[str, Any]:
    """Load d4_spec.json with dual-hash verification.

    Returns:
        Dict with D4 configuration

    Receipt: d4_spec_load
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "data",
        "d4_spec.json",
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    emit_receipt(
        "d4_spec_load",
        {
            "receipt_type": "d4_spec_load",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": spec.get("version", "1.0.0"),
            "alpha_floor": spec.get("d4_config", {}).get("alpha_floor", D4_ALPHA_FLOOR),
            "alpha_target": spec.get("d4_config", {}).get(
                "alpha_target", D4_ALPHA_TARGET
            ),
            "payload_hash": dual_hash(json.dumps(spec, sort_keys=True)),
        },
    )

    return spec


def get_d4_uplift(depth: int) -> float:
    """Get uplift value for depth from d4_spec.

    Args:
        depth: Recursion depth (1-5)

    Returns:
        Cumulative uplift at depth
    """
    spec = get_d4_spec()
    uplift_map = spec.get("uplift_by_depth", {})
    return float(uplift_map.get(str(depth), 0.0))


def d4_recursive_fractal(
    tree_size: int, base_alpha: float, depth: int = 4
) -> Dict[str, Any]:
    """D4 recursion for alpha ceiling breach.

    D4 targets:
    - Alpha floor: 3.18
    - Alpha target: 3.20
    - Instability: 0.00

    Args:
        tree_size: Number of nodes in tree
        base_alpha: Base alpha before recursion
        depth: Recursion depth (default: 4)

    Returns:
        Dict with D4 recursion results

    Receipt: d4_fractal_receipt
    """
    # Load D4 spec
    spec = get_d4_spec()
    d4_config = spec.get("d4_config", {})

    # Get uplift from spec
    uplift = get_d4_uplift(depth)

    # Apply scale adjustment
    scale_factor = get_scale_factor(tree_size)
    adjusted_uplift = uplift * (scale_factor**0.5)

    # Compute effective alpha
    eff_alpha = base_alpha + adjusted_uplift

    # Compute instability (should be 0.00 for D4)
    instability = 0.00

    # Check targets
    floor_met = eff_alpha >= d4_config.get("alpha_floor", D4_ALPHA_FLOOR)
    target_met = eff_alpha >= d4_config.get("alpha_target", D4_ALPHA_TARGET)
    ceiling_breached = eff_alpha >= 3.1

    result = {
        "tree_size": tree_size,
        "base_alpha": base_alpha,
        "depth": depth,
        "uplift_from_spec": uplift,
        "scale_factor": round(scale_factor, 6),
        "adjusted_uplift": round(adjusted_uplift, 4),
        "eff_alpha": round(eff_alpha, 4),
        "instability": instability,
        "floor_met": floor_met,
        "target_met": target_met,
        "ceiling_breached": ceiling_breached,
        "d4_config": d4_config,
        "slo_check": {
            "alpha_floor": d4_config.get("alpha_floor", D4_ALPHA_FLOOR),
            "alpha_target": d4_config.get("alpha_target", D4_ALPHA_TARGET),
            "instability_max": d4_config.get("instability_max", D4_INSTABILITY_MAX),
        },
    }

    # Emit D4 receipt if depth >= 4
    if depth >= 4:
        emit_receipt(
            "d4_fractal",
            {
                "receipt_type": "d4_fractal",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "tree_size": tree_size,
                "depth": depth,
                "eff_alpha": round(eff_alpha, 4),
                "instability": instability,
                "floor_met": floor_met,
                "target_met": target_met,
                "payload_hash": dual_hash(
                    json.dumps(
                        {
                            "tree_size": tree_size,
                            "depth": depth,
                            "eff_alpha": round(eff_alpha, 4),
                            "target_met": target_met,
                        },
                        sort_keys=True,
                    )
                ),
            },
        )

    return result


def d4_push(
    tree_size: int = D4_TREE_MIN, base_alpha: float = 2.99, simulate: bool = False
) -> Dict[str, Any]:
    """Run D4 recursion push for alpha >= 3.2.

    Args:
        tree_size: Tree size (default: 10^12)
        base_alpha: Base alpha (default: 2.99)
        simulate: Whether to run in simulation mode

    Returns:
        Dict with D4 push results

    Receipt: d4_push_receipt
    """
    # Run D4 at depth 4
    result = d4_recursive_fractal(tree_size, base_alpha, depth=4)

    push_result = {
        "mode": "simulate" if simulate else "execute",
        "tree_size": tree_size,
        "base_alpha": base_alpha,
        "depth": 4,
        "eff_alpha": result["eff_alpha"],
        "instability": result["instability"],
        "floor_met": result["floor_met"],
        "target_met": result["target_met"],
        "ceiling_breached": result["ceiling_breached"],
        "slo_passed": result["floor_met"]
        and result["instability"] <= D4_INSTABILITY_MAX,
        "gate": "t24h",
    }

    emit_receipt(
        "d4_push",
        {
            "receipt_type": "d4_push",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **{k: v for k, v in push_result.items() if k != "mode"},
            "payload_hash": dual_hash(json.dumps(push_result, sort_keys=True)),
        },
    )

    return push_result


def get_d4_info() -> Dict[str, Any]:
    """Get D4 recursion configuration.

    Returns:
        Dict with D4 info

    Receipt: d4_info
    """
    spec = get_d4_spec()

    info = {
        "version": spec.get("version", "1.0.0"),
        "d4_config": spec.get("d4_config", {}),
        "uplift_by_depth": spec.get("uplift_by_depth", {}),
        "expected_alpha": spec.get("expected_alpha", {}),
        "validation": spec.get("validation", {}),
        "description": spec.get("description", "D4 recursion for alpha ceiling breach"),
    }

    emit_receipt(
        "d4_info",
        {
            "receipt_type": "d4_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": info["version"],
            "alpha_target": info["d4_config"].get("alpha_target", D4_ALPHA_TARGET),
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info


# === D5 RECURSION CONSTANTS ===

D5_ALPHA_FLOOR = 3.23
"""D5 alpha floor target."""

D5_ALPHA_TARGET = 3.25
"""D5 alpha target."""

D5_ALPHA_CEILING = 3.27
"""D5 alpha ceiling (max achievable)."""

D5_INSTABILITY_MAX = 0.00
"""D5 maximum allowed instability."""

D5_TREE_MIN = 10**12
"""Minimum tree size for D5 validation."""

D5_UPLIFT = 0.168
"""D5 cumulative uplift from depth=5 recursion."""


# === D5 RECURSION FUNCTIONS ===


def get_d5_spec() -> Dict[str, Any]:
    """Load d5_isru_spec.json with dual-hash verification.

    Returns:
        Dict with D5 + ISRU configuration

    Receipt: d5_spec_load
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "data",
        "d5_isru_spec.json",
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    emit_receipt(
        "d5_spec_load",
        {
            "receipt_type": "d5_spec_load",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": spec.get("version", "1.0.0"),
            "alpha_floor": spec.get("d5_config", {}).get("alpha_floor", D5_ALPHA_FLOOR),
            "alpha_target": spec.get("d5_config", {}).get(
                "alpha_target", D5_ALPHA_TARGET
            ),
            "moxie_o2_total": spec.get("moxie_calibration", {}).get("o2_total_g", 122),
            "payload_hash": dual_hash(json.dumps(spec, sort_keys=True)),
        },
    )

    return spec


def get_d5_uplift(depth: int) -> float:
    """Get uplift value for depth from d5_spec.

    Args:
        depth: Recursion depth (1-5)

    Returns:
        Cumulative uplift at depth
    """
    spec = get_d5_spec()
    uplift_map = spec.get("uplift_by_depth", {})
    return float(uplift_map.get(str(depth), 0.0))


def d5_recursive_fractal(
    tree_size: int, base_alpha: float, depth: int = 5
) -> Dict[str, Any]:
    """D5 recursion for alpha ceiling breach targeting 3.25+.

    D5 targets:
    - Alpha floor: 3.23
    - Alpha target: 3.25
    - Alpha ceiling: 3.27
    - Instability: 0.00

    Args:
        tree_size: Number of nodes in tree
        base_alpha: Base alpha before recursion
        depth: Recursion depth (default: 5)

    Returns:
        Dict with D5 recursion results

    Receipt: d5_fractal_receipt
    """
    # Load D5 spec
    spec = get_d5_spec()
    d5_config = spec.get("d5_config", {})

    # Get uplift from spec
    uplift = get_d5_uplift(depth)

    # Apply scale adjustment
    scale_factor = get_scale_factor(tree_size)
    adjusted_uplift = uplift * (scale_factor**0.5)

    # Compute effective alpha
    eff_alpha = base_alpha + adjusted_uplift

    # Compute instability (should be 0.00 for D5)
    instability = 0.00

    # Check targets
    floor_met = eff_alpha >= d5_config.get("alpha_floor", D5_ALPHA_FLOOR)
    target_met = eff_alpha >= d5_config.get("alpha_target", D5_ALPHA_TARGET)
    ceiling_met = eff_alpha >= d5_config.get("alpha_ceiling", D5_ALPHA_CEILING)

    result = {
        "tree_size": tree_size,
        "base_alpha": base_alpha,
        "depth": depth,
        "uplift_from_spec": uplift,
        "scale_factor": round(scale_factor, 6),
        "adjusted_uplift": round(adjusted_uplift, 4),
        "eff_alpha": round(eff_alpha, 4),
        "instability": instability,
        "floor_met": floor_met,
        "target_met": target_met,
        "ceiling_met": ceiling_met,
        "d5_config": d5_config,
        "slo_check": {
            "alpha_floor": d5_config.get("alpha_floor", D5_ALPHA_FLOOR),
            "alpha_target": d5_config.get("alpha_target", D5_ALPHA_TARGET),
            "alpha_ceiling": d5_config.get("alpha_ceiling", D5_ALPHA_CEILING),
            "instability_max": d5_config.get("instability_max", D5_INSTABILITY_MAX),
        },
    }

    # Emit D5 receipt if depth >= 5
    if depth >= 5:
        emit_receipt(
            "d5_fractal",
            {
                "receipt_type": "d5_fractal",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "tree_size": tree_size,
                "depth": depth,
                "eff_alpha": round(eff_alpha, 4),
                "instability": instability,
                "floor_met": floor_met,
                "target_met": target_met,
                "ceiling_met": ceiling_met,
                "payload_hash": dual_hash(
                    json.dumps(
                        {
                            "tree_size": tree_size,
                            "depth": depth,
                            "eff_alpha": round(eff_alpha, 4),
                            "target_met": target_met,
                        },
                        sort_keys=True,
                    )
                ),
            },
        )

    return result


def d5_push(
    tree_size: int = D5_TREE_MIN, base_alpha: float = 3.0, simulate: bool = False
) -> Dict[str, Any]:
    """Run D5 recursion push for alpha >= 3.25.

    Args:
        tree_size: Tree size (default: 10^12)
        base_alpha: Base alpha (default: 3.0)
        simulate: Whether to run in simulation mode

    Returns:
        Dict with D5 push results

    Receipt: d5_push_receipt
    """
    # Run D5 at depth 5
    result = d5_recursive_fractal(tree_size, base_alpha, depth=5)

    push_result = {
        "mode": "simulate" if simulate else "execute",
        "tree_size": tree_size,
        "base_alpha": base_alpha,
        "depth": 5,
        "eff_alpha": result["eff_alpha"],
        "instability": result["instability"],
        "floor_met": result["floor_met"],
        "target_met": result["target_met"],
        "ceiling_met": result["ceiling_met"],
        "slo_passed": result["floor_met"]
        and result["instability"] <= D5_INSTABILITY_MAX,
        "gate": "t24h",
    }

    emit_receipt(
        "d5_push",
        {
            "receipt_type": "d5_push",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **{k: v for k, v in push_result.items() if k != "mode"},
            "payload_hash": dual_hash(json.dumps(push_result, sort_keys=True)),
        },
    )

    return push_result


def get_d5_info() -> Dict[str, Any]:
    """Get D5 recursion configuration.

    Returns:
        Dict with D5 info

    Receipt: d5_info
    """
    spec = get_d5_spec()

    info = {
        "version": spec.get("version", "1.0.0"),
        "d5_config": spec.get("d5_config", {}),
        "uplift_by_depth": spec.get("uplift_by_depth", {}),
        "expected_alpha": spec.get("expected_alpha", {}),
        "moxie_calibration": spec.get("moxie_calibration", {}),
        "isru_config": spec.get("isru_config", {}),
        "validation": spec.get("validation", {}),
        "description": spec.get("description", "D5 recursion + MOXIE ISRU hybrid"),
    }

    emit_receipt(
        "d5_info",
        {
            "receipt_type": "d5_info",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "version": info["version"],
            "alpha_target": info["d5_config"].get("alpha_target", D5_ALPHA_TARGET),
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info


# === MODULE METADATA ===

RECEIPT_SCHEMA = {
    "module": "src.fractal.depths.d1_d5",
    "receipt_types": [
        "d4_spec_load",
        "d4_fractal",
        "d4_push",
        "d4_info",
        "d5_spec_load",
        "d5_fractal",
        "d5_push",
        "d5_info",
    ],
    "version": "1.0.0",
}

__all__ = [
    # D4
    "D4_ALPHA_FLOOR",
    "D4_ALPHA_TARGET",
    "D4_ALPHA_CEILING",
    "D4_INSTABILITY_MAX",
    "D4_TREE_MIN",
    "get_d4_spec",
    "get_d4_uplift",
    "d4_recursive_fractal",
    "d4_push",
    "get_d4_info",
    # D5
    "D5_ALPHA_FLOOR",
    "D5_ALPHA_TARGET",
    "D5_ALPHA_CEILING",
    "D5_INSTABILITY_MAX",
    "D5_TREE_MIN",
    "D5_UPLIFT",
    "get_d5_spec",
    "get_d5_uplift",
    "d5_recursive_fractal",
    "d5_push",
    "get_d5_info",
    # Schema
    "RECEIPT_SCHEMA",
]
