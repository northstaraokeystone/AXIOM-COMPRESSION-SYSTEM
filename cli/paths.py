"""Unified path CLI routing for AXIOM exploration paths.

Single entry point for all path commands:
- cmd_path_status: Show status of one or all paths
- cmd_path_list: List registered paths
- cmd_path_run: Route command to specific path

Routing pattern:
  --path mars --path_cmd status    -> src/paths/mars/cli.py::cmd_mars_status
  --path multiplanet --path_cmd sequence -> src/paths/multiplanet/cli.py::cmd_multiplanet_sequence
  --path agi --path_cmd policy     -> src/paths/agi/cli.py::cmd_agi_policy

Source: AXIOM scalable paths architecture
"""

import json
from typing import Dict, Any, Optional

from src.paths import (
    REGISTERED_PATHS,
    list_paths,
    get_path,
    get_path_status,
    get_all_path_status,
    PathStopRule,
)
from src.registry import (
    discover_paths,
    validate_registry,
    route_to_path,
    get_path_commands,
    get_registry_info,
)
from src.fractal_layers import (
    d4_push,
    get_d4_info,
    D4_TREE_MIN,
)


# === PATH STATUS COMMANDS ===

def cmd_path_status(path: Optional[str] = None) -> Dict[str, Any]:
    """Show status of one or all paths.

    Args:
        path: Optional path name (if None, show all)

    Returns:
        Status dict
    """
    if path is None:
        # Show all paths
        status = get_all_path_status()
        print("=" * 60)
        print("ALL PATHS STATUS")
        print("=" * 60)
        print(f"\nTotal paths: {status['total_paths']}")
        print(f"Ready: {status['ready_count']}")
        print("\nPath details:")
        for path_name, path_status in status.get("paths", {}).items():
            ready = "READY" if path_status.get("ready") else "NOT READY"
            stage = path_status.get("status", "unknown")
            version = path_status.get("version", "0.0.0")
            print(f"  {path_name}: {ready} (v{version}, {stage})")
        return status
    else:
        # Show specific path
        if path not in REGISTERED_PATHS:
            print(f"Unknown path: {path}")
            print(f"Available paths: {', '.join(REGISTERED_PATHS)}")
            return {"error": f"Unknown path: {path}"}

        status = get_path_status(path)
        print("=" * 60)
        print(f"{path.upper()} PATH STATUS")
        print("=" * 60)
        print(f"Ready: {status.get('ready')}")
        print(f"Version: {status.get('version')}")
        print(f"Status: {status.get('status')}")
        if status.get("dependencies"):
            print(f"Dependencies: {status.get('dependencies')}")
        if status.get("receipts"):
            print(f"Receipts: {status.get('receipts')}")
        return status


def cmd_path_list() -> Dict[str, Any]:
    """List registered paths.

    Returns:
        List of path names
    """
    paths = list_paths()
    discovered = discover_paths()

    print("=" * 60)
    print("REGISTERED PATHS")
    print("=" * 60)
    print(f"\nRegistered: {', '.join(paths)}")
    print(f"Discovered: {', '.join(discovered)}")

    # Validate registry
    validation = validate_registry()
    if validation["valid"]:
        print("\nRegistry: VALID")
    else:
        print("\nRegistry: INVALID")
        if validation["missing"]:
            print(f"  Missing: {validation['missing']}")
        if validation["extra"]:
            print(f"  Extra: {validation['extra']}")

    return {"paths": paths, "discovered": discovered, "valid": validation["valid"]}


def cmd_path_run(path: str, command: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Route command to specific path.

    Args:
        path: Path name
        command: Command to run
        args: Optional arguments

    Returns:
        Command result
    """
    if path not in REGISTERED_PATHS:
        print(f"Unknown path: {path}")
        print(f"Available paths: {', '.join(REGISTERED_PATHS)}")
        return {"error": f"Unknown path: {path}"}

    try:
        result = route_to_path(path, command, args)
        return result
    except PathStopRule as e:
        print(f"Error: {e}")
        return {"error": str(e)}


def cmd_path_commands(path: str) -> Dict[str, Any]:
    """List available commands for a path.

    Args:
        path: Path name

    Returns:
        List of commands
    """
    if path not in REGISTERED_PATHS:
        print(f"Unknown path: {path}")
        print(f"Available paths: {', '.join(REGISTERED_PATHS)}")
        return {"error": f"Unknown path: {path}"}

    commands = get_path_commands(path)

    print("=" * 60)
    print(f"{path.upper()} COMMANDS")
    print("=" * 60)
    for cmd in commands:
        print(f"  - {cmd}")

    return {"path": path, "commands": commands}


# === PATH SHORTCUT COMMANDS ===

def cmd_mars_status() -> Dict[str, Any]:
    """Shortcut: Show Mars path status."""
    path_module = get_path("mars")
    return path_module.cli.cmd_mars_status()


def cmd_multiplanet_status() -> Dict[str, Any]:
    """Shortcut: Show multi-planet path status."""
    path_module = get_path("multiplanet")
    return path_module.cli.cmd_multiplanet_status()


def cmd_agi_status() -> Dict[str, Any]:
    """Shortcut: Show AGI path status."""
    path_module = get_path("agi")
    return path_module.cli.cmd_agi_status()


# === D4 COMMANDS ===

def cmd_d4_push(tree_size: int = D4_TREE_MIN, base_alpha: float = 2.99, simulate: bool = False) -> Dict[str, Any]:
    """Run D4 recursion push.

    Args:
        tree_size: Tree size (default: 10^12)
        base_alpha: Base alpha (default: 2.99)
        simulate: Simulation mode

    Returns:
        D4 push results
    """
    result = d4_push(tree_size, base_alpha, simulate)

    print("=" * 60)
    print(f"D4 PUSH {'(SIMULATE)' if simulate else ''}")
    print("=" * 60)
    print(f"Tree size: {tree_size:,}")
    print(f"Base alpha: {base_alpha}")
    print(f"Depth: {result['depth']}")
    print(f"\nEffective alpha: {result['eff_alpha']}")
    print(f"Instability: {result['instability']}")
    print(f"\nFloor met (3.18): {'YES' if result['floor_met'] else 'NO'}")
    print(f"Target met (3.20): {'YES' if result['target_met'] else 'NO'}")
    print(f"Ceiling breached (3.1): {'YES' if result['ceiling_breached'] else 'NO'}")
    print(f"\nSLO passed: {'YES' if result['slo_passed'] else 'NO'}")
    print(f"Gate: {result['gate']}")

    return result


def cmd_d4_info() -> Dict[str, Any]:
    """Show D4 configuration.

    Returns:
        D4 info dict
    """
    info = get_d4_info()

    print("=" * 60)
    print("D4 RECURSION INFO")
    print("=" * 60)
    print(f"Version: {info['version']}")
    print("\nD4 Config:")
    print(json.dumps(info["d4_config"], indent=2))
    print("\nUplift by depth:")
    for depth, uplift in info["uplift_by_depth"].items():
        print(f"  Depth {depth}: +{uplift}")
    print("\nExpected alpha:")
    for depth, alpha in info["expected_alpha"].items():
        print(f"  {depth}: {alpha}")

    return info


# === REGISTRY INFO ===

def cmd_registry_info() -> Dict[str, Any]:
    """Show registry configuration.

    Returns:
        Registry info dict
    """
    info = get_registry_info()

    print("=" * 60)
    print("PATH REGISTRY INFO")
    print("=" * 60)
    print(f"Registered paths: {info['registered_paths']}")
    print(f"Discovered paths: {info['discovered_paths']}")
    print(f"Registry valid: {info['registry_valid']}")
    print(f"Paths ready: {info['paths_ready']}/{info['paths_total']}")
    print("\nPath statuses:")
    for path, status in info.get("path_statuses", {}).items():
        print(f"  {path}: {status}")

    return info
