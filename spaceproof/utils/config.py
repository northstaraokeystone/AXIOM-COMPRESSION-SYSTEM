"""config.py - Generic configuration loading utilities.

Provides unified config loading with dual-hash verification.
Reduces boilerplate across all SpaceProof modules.
"""

import json
import os
from typing import Any, Dict, Optional

from ..core import emit_receipt, dual_hash


def get_spec_path(spec_name: str) -> str:
    """Get absolute path to a spec file in the data directory.

    Args:
        spec_name: Name of the spec file (e.g., 'd8_multi_spec.json')

    Returns:
        Absolute path to the spec file
    """
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(repo_root, "data", spec_name)


def load_spec(
    spec_name: str,
    config_key: Optional[str] = None,
    tenant_id: str = "spaceproof-core",
    emit: bool = True,
) -> Dict[str, Any]:
    """Load a spec file with optional config key extraction.

    This is the unified spec loading function that replaces the pattern:
        spec_path = os.path.join(...)
        with open(spec_path, "r") as f:
            spec = json.load(f)
        config = spec.get("some_config", {})
        emit_receipt(...)

    Args:
        spec_name: Name of the spec file (e.g., 'd8_multi_spec.json')
        config_key: Optional key to extract from spec (e.g., 'atacama_config')
        tenant_id: Tenant ID for the receipt
        emit: Whether to emit a receipt

    Returns:
        The spec dict, or the config dict if config_key is provided

    Receipt: spec_load
    """
    spec_path = get_spec_path(spec_name)

    with open(spec_path, "r") as f:
        spec = json.load(f)

    if emit:
        emit_receipt(
            "spec_load",
            {
                "tenant_id": tenant_id,
                "spec_name": spec_name,
                "config_key": config_key,
                "payload_hash": dual_hash(json.dumps(spec, sort_keys=True)),
            },
        )

    if config_key:
        return spec.get(config_key, {})

    return spec


def load_spec_with_defaults(
    spec_name: str,
    config_key: str,
    defaults: Dict[str, Any],
    tenant_id: str = "spaceproof-core",
) -> Dict[str, Any]:
    """Load a spec config with defaults applied.

    Args:
        spec_name: Name of the spec file
        config_key: Key to extract from spec
        defaults: Default values to apply
        tenant_id: Tenant ID for the receipt

    Returns:
        Config dict with defaults merged in
    """
    config = load_spec(spec_name, config_key, tenant_id, emit=False)

    # Apply defaults
    result = {**defaults}
    result.update(config)

    emit_receipt(
        "spec_config_load",
        {
            "tenant_id": tenant_id,
            "spec_name": spec_name,
            "config_key": config_key,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result
