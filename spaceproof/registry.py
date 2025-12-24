"""Central path discovery and routing for SpaceProof exploration paths.

This module provides:
- discover_paths(): Auto-discover paths in src/paths/
- route_to_path(): Route CLI commands to path modules
- aggregate_receipts(): Collect receipts across paths

Source: SpaceProof scalable paths architecture
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from .core import emit_receipt
from .paths import (
    REGISTERED_PATHS,
    get_path,
    get_path_status,
    get_all_path_status,
    PathStopRule,
)


# === CONSTANTS ===

REGISTRY_TENANT_ID = "spaceproof-registry"
"""Tenant ID for registry receipts."""


# === DISCOVERY FUNCTIONS ===


def discover_paths() -> List[str]:
    """Auto-discover paths in src/paths/.

    Scans the paths directory for valid path modules.
    A valid path has:
    - __init__.py
    - spec.json
    - core.py

    Returns:
        List of discovered path names

    Receipt: paths_discovered
    """
    paths_dir = os.path.join(os.path.dirname(__file__), "paths")
    discovered = []

    for name in os.listdir(paths_dir):
        path_dir = os.path.join(paths_dir, name)

        # Skip non-directories and private modules
        if not os.path.isdir(path_dir) or name.startswith("_"):
            continue

        # Check for required files
        has_init = os.path.exists(os.path.join(path_dir, "__init__.py"))
        has_spec = os.path.exists(os.path.join(path_dir, "spec.json"))
        has_core = os.path.exists(os.path.join(path_dir, "core.py"))

        if has_init and has_spec and has_core:
            discovered.append(name)

    emit_receipt(
        "paths_discovered",
        {
            "tenant_id": REGISTRY_TENANT_ID,
            "discovered_paths": discovered,
            "count": len(discovered),
            "registered_paths": REGISTERED_PATHS,
        },
    )

    return discovered


def validate_registry() -> Dict[str, Any]:
    """Validate that all registered paths are discoverable.

    Returns:
        Dict with validation results

    Receipt: registry_validation
    """
    discovered = discover_paths()

    missing = [p for p in REGISTERED_PATHS if p not in discovered]
    extra = [p for p in discovered if p not in REGISTERED_PATHS]

    result = {
        "registered": REGISTERED_PATHS,
        "discovered": discovered,
        "missing": missing,
        "extra": extra,
        "valid": len(missing) == 0,
        "tenant_id": REGISTRY_TENANT_ID,
    }

    emit_receipt("registry_validation", result)
    return result


# === ROUTING FUNCTIONS ===


def route_to_path(
    path: str, command: str, args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Route CLI command to path module.

    Args:
        path: Path name (mars, multiplanet, agi)
        command: Command name to execute
        args: Optional arguments dict

    Returns:
        Command result dict

    Raises:
        PathStopRule: If path or command not found

    Receipt: path_command_routed
    """
    if path not in REGISTERED_PATHS:
        raise PathStopRule(path, f"Unknown path: {path}")

    path_module = get_path(path)
    if path_module is None:
        raise PathStopRule(path, f"Failed to load path module: {path}")

    # Look for CLI module
    try:
        cli_module = path_module.cli
    except AttributeError:
        raise PathStopRule(path, f"Path has no CLI module: {path}")

    # Look for command handler
    cmd_name = f"cmd_{path}_{command}"
    if not hasattr(cli_module, cmd_name):
        # Try generic command name
        cmd_name = f"cmd_{command}"
        if not hasattr(cli_module, cmd_name):
            raise PathStopRule(path, f"Unknown command: {command}")

    handler = getattr(cli_module, cmd_name)

    # Execute command
    emit_receipt(
        "path_command_routed",
        {
            "tenant_id": REGISTRY_TENANT_ID,
            "path": path,
            "command": command,
            "args": args or {},
            "handler": cmd_name,
        },
    )

    if args:
        return handler(args)
    else:
        return handler()


def get_path_commands(path: str) -> List[str]:
    """Get available commands for a path.

    Args:
        path: Path name

    Returns:
        List of command names

    Receipt: path_commands_listed
    """
    if path not in REGISTERED_PATHS:
        raise PathStopRule(path, f"Unknown path: {path}")

    path_module = get_path(path)
    if path_module is None:
        return []

    try:
        cli_module = path_module.cli
    except AttributeError:
        return []

    # Find all cmd_ functions
    commands = []
    for name in dir(cli_module):
        if name.startswith("cmd_"):
            # Extract command name
            cmd_name = name[4:]  # Remove "cmd_" prefix
            if cmd_name.startswith(f"{path}_"):
                cmd_name = cmd_name[len(path) + 1 :]  # Remove path prefix
            commands.append(cmd_name)

    emit_receipt(
        "path_commands_listed",
        {
            "tenant_id": REGISTRY_TENANT_ID,
            "path": path,
            "commands": commands,
            "count": len(commands),
        },
    )

    return commands


# === RECEIPT AGGREGATION ===


def aggregate_receipts(paths: Optional[List[str]] = None) -> Dict[str, Any]:
    """Collect receipts across paths.

    Args:
        paths: List of paths to aggregate (default: all)

    Returns:
        Dict with aggregated receipt info

    Receipt: receipts_aggregated
    """
    if paths is None:
        paths = list(REGISTERED_PATHS)

    receipts_by_path = {}
    total_receipts = 0

    for path in paths:
        if path not in REGISTERED_PATHS:
            continue

        status = get_path_status(path)
        path_receipts = status.get("receipts", [])
        receipts_by_path[path] = path_receipts
        total_receipts += len(path_receipts)

    result = {
        "tenant_id": REGISTRY_TENANT_ID,
        "paths_queried": paths,
        "receipts_by_path": receipts_by_path,
        "total_receipt_types": total_receipts,
        "ts": datetime.utcnow().isoformat() + "Z",
    }

    emit_receipt("receipts_aggregated", result)
    return result


def get_registry_info() -> Dict[str, Any]:
    """Get registry configuration and status.

    Returns:
        Dict with registry info

    Receipt: registry_info
    """
    validation = validate_registry()
    all_status = get_all_path_status()

    info = {
        "tenant_id": REGISTRY_TENANT_ID,
        "registered_paths": REGISTERED_PATHS,
        "discovered_paths": validation["discovered"],
        "registry_valid": validation["valid"],
        "paths_ready": all_status["ready_count"],
        "paths_total": all_status["total_paths"],
        "path_statuses": {
            path: status.get("status", "unknown")
            for path, status in all_status.get("paths", {}).items()
        },
    }

    emit_receipt("registry_info", info)
    return info
