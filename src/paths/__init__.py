"""AXIOM Exploration Paths - Parallel path registry and discovery.

This package provides modular exploration paths:
- mars: Mars habitat autonomous optimization
- multiplanet: Asteroid -> Titan expansion sequence
- agi: Post-AGI ethics modeling via fractal policies

Each path follows the ProofPack module pattern:
    path/
    ├── spec.json          # Immutable config (dual-hash)
    ├── core.py            # Domain logic
    ├── receipts.py        # Path-specific receipts
    └── cli.py             # Path CLI commands

Source: AXIOM scalable paths architecture
"""

from .base import (
    REGISTERED_PATHS,
    PATH_RECEIPT_PREFIX,
    PATH_TENANT_ID,
    PathStopRule,
    load_path_spec,
    emit_path_receipt,
    get_path_status,
    get_all_path_status,
    validate_path_dependencies,
)

# Lazy imports for path modules to avoid circular dependencies
_path_modules = {}


def get_path(name: str):
    """Return path module by name.

    Args:
        name: Path name (mars, multiplanet, agi)

    Returns:
        Path module

    Raises:
        PathStopRule: If path not found
    """
    if name not in REGISTERED_PATHS:
        raise PathStopRule(name, f"Unknown path: {name}")

    if name not in _path_modules:
        if name == "mars":
            from . import mars

            _path_modules[name] = mars
        elif name == "multiplanet":
            from . import multiplanet

            _path_modules[name] = multiplanet
        elif name == "agi":
            from . import agi

            _path_modules[name] = agi

    return _path_modules.get(name)


def list_paths() -> list:
    """Return all registered paths.

    Returns:
        List of path names
    """
    return list(REGISTERED_PATHS)


def path_status_all() -> dict:
    """Return status of all paths.

    Returns:
        Dict with status for each path
    """
    return get_all_path_status()


__all__ = [
    # Constants
    "REGISTERED_PATHS",
    "PATH_RECEIPT_PREFIX",
    "PATH_TENANT_ID",
    # Exception
    "PathStopRule",
    # Functions
    "load_path_spec",
    "emit_path_receipt",
    "get_path_status",
    "get_all_path_status",
    "validate_path_dependencies",
    "get_path",
    "list_paths",
    "path_status_all",
]
