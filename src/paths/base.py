"""Shared path primitives for parallel exploration paths.

All exploration paths (Mars, Multi-planet, AGI) share this infrastructure:
- load_path_spec(): Load path-specific spec.json with dual-hash verification
- emit_path_receipt(): Emit receipt with path prefix
- get_path_status(): Return path readiness status
- PathStopRule: Path-specific exception class

Source: AXIOM scalable paths architecture
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

from ..core import dual_hash, emit_receipt, StopRule


# === CONSTANTS ===

REGISTERED_PATHS = ["mars", "multiplanet", "agi"]
"""List of all registered exploration paths."""

PATH_RECEIPT_PREFIX = {"mars": "mars_", "multiplanet": "mp_", "agi": "agi_"}
"""Receipt type prefix for each path."""

PATH_TENANT_ID = "axiom-paths"
"""Tenant ID for path receipts."""


# === EXCEPTION CLASS ===


class PathStopRule(StopRule):
    """Raised when a path-specific stoprule triggers.

    Includes path context for debugging.
    """

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"[{path}] {message}")


# === PATH PRIMITIVES ===


def load_path_spec(path_name: str) -> Dict[str, Any]:
    """Load path-specific spec.json with dual-hash verification.

    Args:
        path_name: Name of the path (mars, multiplanet, agi)

    Returns:
        Dict with spec configuration

    Raises:
        PathStopRule: If spec file not found or invalid

    Receipt: {path}_spec_load
    """
    if path_name not in REGISTERED_PATHS:
        raise PathStopRule(path_name, f"Unknown path: {path_name}")

    spec_path = os.path.join(os.path.dirname(__file__), path_name, "spec.json")

    if not os.path.exists(spec_path):
        raise PathStopRule(path_name, f"Spec file not found: {spec_path}")

    with open(spec_path, "r") as f:
        spec = json.load(f)

    # Compute and verify dual-hash
    spec_hash = dual_hash(json.dumps(spec, sort_keys=True))

    receipt_type = f"{PATH_RECEIPT_PREFIX[path_name]}spec_load"
    emit_receipt(
        receipt_type,
        {
            "path": path_name,
            "version": spec.get("version", "0.0.0"),
            "status": spec.get("status", "unknown"),
            "spec_hash": spec_hash,
            "tenant_id": PATH_TENANT_ID,
        },
    )

    return spec


def emit_path_receipt(
    path: str, receipt_type: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    """Emit receipt with path prefix.

    Args:
        path: Path name (mars, multiplanet, agi)
        receipt_type: Receipt type (without prefix)
        data: Receipt data

    Returns:
        Complete receipt dict

    Receipt: {prefix}{receipt_type}
    """
    if path not in PATH_RECEIPT_PREFIX:
        path_prefix = f"{path}_"
    else:
        path_prefix = PATH_RECEIPT_PREFIX[path]

    full_type = f"{path_prefix}{receipt_type}"

    # Add path context to data
    receipt_data = {
        "path": path,
        "tenant_id": data.get("tenant_id", PATH_TENANT_ID),
        **data,
    }

    return emit_receipt(full_type, receipt_data)


def get_path_status(path_name: str) -> Dict[str, Any]:
    """Return path readiness status.

    Args:
        path_name: Name of the path

    Returns:
        Dict with:
            - ready: True if path is operational
            - version: Path version
            - last_receipt: Timestamp of last receipt
            - status: Current path status (stub/active/deprecated)

    Receipt: {path}_status
    """
    if path_name not in REGISTERED_PATHS:
        return {
            "ready": False,
            "version": "0.0.0",
            "last_receipt": None,
            "status": "unknown",
            "error": f"Unknown path: {path_name}",
        }

    try:
        spec = load_path_spec(path_name)
        status = {
            "ready": True,
            "version": spec.get("version", "0.0.0"),
            "last_receipt": datetime.utcnow().isoformat() + "Z",
            "status": spec.get("status", "stub"),
            "dependencies": spec.get("dependencies", []),
            "receipts": spec.get("receipts", []),
        }
    except (PathStopRule, FileNotFoundError, json.JSONDecodeError) as e:
        status = {
            "ready": False,
            "version": "0.0.0",
            "last_receipt": None,
            "status": "error",
            "error": str(e),
        }

    emit_path_receipt(path_name, "status", status)
    return status


def get_all_path_status() -> Dict[str, Any]:
    """Return status of all registered paths.

    Returns:
        Dict with status for each path

    Receipt: paths_status_all
    """
    statuses = {}
    for path in REGISTERED_PATHS:
        statuses[path] = get_path_status(path)

    summary = {
        "total_paths": len(REGISTERED_PATHS),
        "ready_count": sum(1 for s in statuses.values() if s.get("ready")),
        "paths": statuses,
        "tenant_id": PATH_TENANT_ID,
    }

    emit_receipt("paths_status_all", summary)
    return summary


def validate_path_dependencies(path_name: str) -> Dict[str, Any]:
    """Validate that path dependencies are available.

    Args:
        path_name: Name of the path

    Returns:
        Dict with validation results

    Raises:
        PathStopRule: If required dependencies not met
    """
    spec = load_path_spec(path_name)
    dependencies = spec.get("dependencies", [])

    results = {
        "path": path_name,
        "dependencies": dependencies,
        "validated": [],
        "missing": [],
        "all_met": True,
    }

    for dep in dependencies:
        # Check if dependency is a path
        if dep in REGISTERED_PATHS:
            dep_status = get_path_status(dep)
            if dep_status.get("ready"):
                results["validated"].append(dep)
            else:
                results["missing"].append(dep)
                results["all_met"] = False
        else:
            # Check if dependency is a module
            try:
                if dep == "fractal_layers":
                    from ..fractal_layers import recursive_fractal  # noqa: F401
                elif dep == "quantum_rl_hybrid":
                    from ..quantum_rl_hybrid import quantum_rl_hybrid  # noqa: F401
                results["validated"].append(dep)
            except ImportError:
                results["missing"].append(dep)
                results["all_met"] = False

    emit_path_receipt(path_name, "dependencies_validated", results)

    if not results["all_met"]:
        raise PathStopRule(path_name, f"Missing dependencies: {results['missing']}")

    return results
