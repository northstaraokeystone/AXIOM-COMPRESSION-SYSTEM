"""adaptive.py - Runtime Scaling Logic Module

THE PHYSICS (from Grok analysis):
    - "Adaptive depth layers... RL feedback" - depth should scale with problem size
    - Depth scales with log(tree_size), not fixed
    - Deeper networks need lower learning rates (LR / sqrt(depth))
    - Higher entropy → more aggressive pruning

KEY INSIGHT:
    Static configs leave retention on the table. Dynamic scaling finds optimal
    combinations automatically based on runtime conditions.

FUNCTIONS:
    compute_adaptive_depth(tree_size_n, base_depth) → adaptive_depth
    scale_lr_to_depth(depth, base_lr) → scaled_lr
    adaptive_prune_factor(entropy_level, base_factor) → adaptive_factor
    get_dynamic_config(tree_size, entropy, rl_feedback) → unified config

Source: Grok - "Adaptive depth layers... RL feedback"
"""

import json
import math
from typing import Dict, Any, Optional

from .core import emit_receipt, dual_hash


# === CONSTANTS (Adaptive Scaling) ===

ADAPTIVE_DEPTH_BASE = 5
"""Base GNN depth before scaling."""

ADAPTIVE_DEPTH_MIN = 3
"""Minimum adaptive depth."""

ADAPTIVE_DEPTH_MAX = 12
"""Maximum adaptive depth."""

ADAPTIVE_DEPTH_SCALING = "log_n"
"""Scaling formula identifier: depth scales with log(tree_size)."""

LR_BASE = 0.002
"""Base learning rate."""

LR_MIN = 0.0005
"""Minimum learning rate."""

LR_MAX = 0.005
"""Maximum learning rate."""

ENTROPY_BASE = 0.3
"""Base entropy-based prune factor."""

ENTROPY_SCALING_FACTOR = 0.5
"""How much entropy affects pruning aggressiveness."""

PRUNE_FACTOR_MIN = 0.2
"""Minimum prune factor (conservative)."""

PRUNE_FACTOR_MAX = 0.5
"""Maximum prune factor (aggressive)."""


def compute_adaptive_depth(
    tree_size_n: int, base_depth: int = ADAPTIVE_DEPTH_BASE
) -> int:
    """Compute adaptive depth based on tree size.

    Formula: adaptive_depth = base + floor(log2(n) / 10)
    Larger trees get deeper networks for better representation.

    Args:
        tree_size_n: Number of entries in Merkle tree
        base_depth: Base depth before scaling (default: 5)

    Returns:
        Adaptive depth (clamped to min/max bounds)

    Receipt: adaptive_depth_receipt
    """
    if tree_size_n <= 0:
        adaptive_depth = base_depth
    else:
        # log2(n) / 10 gives reasonable scaling
        # 1e6 → ~2 extra layers, 1e8 → ~2.7 extra, 1e10 → ~3.3 extra
        log_factor = math.log2(max(1, tree_size_n)) / 10
        adaptive_depth = base_depth + int(log_factor)

    # Clamp to bounds
    adaptive_depth = max(ADAPTIVE_DEPTH_MIN, min(ADAPTIVE_DEPTH_MAX, adaptive_depth))

    result = {
        "tree_size_n": tree_size_n,
        "base_depth": base_depth,
        "computed_depth": adaptive_depth,
        "scaling_formula": ADAPTIVE_DEPTH_SCALING,
        "log_factor": round(math.log2(max(1, tree_size_n)) / 10, 4)
        if tree_size_n > 0
        else 0,
    }

    emit_receipt(
        "adaptive_depth",
        {
            "receipt_type": "adaptive_depth",
            "tenant_id": "axiom-adaptive",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return adaptive_depth


def scale_lr_to_depth(depth: int, base_lr: float = LR_BASE) -> float:
    """Scale learning rate based on network depth.

    Deeper networks need lower learning rates to prevent instability.
    Formula: scaled_lr = base_lr / sqrt(depth)

    Args:
        depth: Network depth (number of layers)
        base_lr: Base learning rate (default: 0.002)

    Returns:
        Scaled learning rate (clamped to min/max bounds)
    """
    if depth <= 0:
        return base_lr

    # Scale inversely with sqrt(depth)
    scaled_lr = base_lr / math.sqrt(depth)

    # Clamp to bounds
    scaled_lr = max(LR_MIN, min(LR_MAX, scaled_lr))

    return round(scaled_lr, 6)


def adaptive_prune_factor(
    entropy_level: float, base_factor: float = ENTROPY_BASE
) -> float:
    """Compute adaptive pruning factor based on entropy level.

    Higher entropy → more aggressive pruning (more low-information content).
    Formula: adaptive_factor = base + (entropy_level * scaling_factor)

    Args:
        entropy_level: Normalized entropy (0-1 range)
        base_factor: Base prune factor (default: 0.3)

    Returns:
        Adaptive prune factor (clamped to min/max bounds)
    """
    # Scale factor based on entropy
    # Higher entropy means more low-value content to prune
    entropy_adjustment = entropy_level * ENTROPY_SCALING_FACTOR

    adaptive_factor = base_factor + entropy_adjustment

    # Clamp to bounds
    adaptive_factor = max(PRUNE_FACTOR_MIN, min(PRUNE_FACTOR_MAX, adaptive_factor))

    return round(adaptive_factor, 4)


def get_dynamic_config(
    tree_size: int,
    entropy: float,
    rl_feedback: Optional[Dict[str, Any]] = None,
    blackout_days: int = 0,
) -> Dict[str, Any]:
    """Combine all adaptive signals into unified dynamic config.

    This is the main entry point for getting runtime-optimized configuration.
    Combines:
    1. Adaptive depth based on tree size
    2. Scaled LR based on depth
    3. Adaptive prune factor based on entropy
    4. RL feedback overrides (if provided)

    Args:
        tree_size: Number of entries in Merkle tree
        entropy: Average entropy level (0-1)
        rl_feedback: Optional RL tuner feedback dict
        blackout_days: Current blackout duration

    Returns:
        Dict with:
            - gnn_layers: Computed GNN layer count
            - lr_decay: Scaled learning rate
            - prune_aggressiveness: Adaptive prune factor
            - adaptive_depth_enabled: Whether adaptive depth is active
            - source: Where each value came from

    Receipt: dynamic_config_receipt
    """
    # Compute adaptive values
    adaptive_depth = compute_adaptive_depth(tree_size)
    scaled_lr = scale_lr_to_depth(adaptive_depth)
    adaptive_prune = adaptive_prune_factor(entropy)

    # Initialize config with adaptive values
    config = {
        "gnn_layers": adaptive_depth,
        "lr_decay": scaled_lr,
        "prune_aggressiveness": adaptive_prune,
        "adaptive_depth_enabled": True,
        "tree_size": tree_size,
        "entropy_level": entropy,
        "blackout_days": blackout_days,
    }

    # Track sources
    sources = {
        "gnn_layers": "adaptive",
        "lr_decay": "adaptive",
        "prune_aggressiveness": "adaptive",
    }

    # Apply RL feedback overrides if provided
    if rl_feedback is not None:
        # RL can override layer count
        if "gnn_layers_delta" in rl_feedback:
            delta = rl_feedback["gnn_layers_delta"]
            config["gnn_layers"] = max(
                ADAPTIVE_DEPTH_MIN, min(ADAPTIVE_DEPTH_MAX, adaptive_depth + delta)
            )
            sources["gnn_layers"] = "rl"

        # RL can override LR
        if "lr_decay" in rl_feedback:
            config["lr_decay"] = max(LR_MIN, min(LR_MAX, rl_feedback["lr_decay"]))
            sources["lr_decay"] = "rl"

        # RL can override prune aggressiveness
        if "prune_aggressiveness" in rl_feedback:
            config["prune_aggressiveness"] = max(
                PRUNE_FACTOR_MIN,
                min(PRUNE_FACTOR_MAX, rl_feedback["prune_aggressiveness"]),
            )
            sources["prune_aggressiveness"] = "rl"

    config["sources"] = sources

    emit_receipt(
        "dynamic_config",
        {
            "receipt_type": "dynamic_config",
            "tenant_id": "axiom-adaptive",
            "gnn_layers": config["gnn_layers"],
            "lr_decay": config["lr_decay"],
            "prune_aggressiveness": config["prune_aggressiveness"],
            "adaptive_depth_enabled": config["adaptive_depth_enabled"],
            "sources": sources,
            "tree_size": tree_size,
            "entropy_level": entropy,
            "rl_feedback_applied": rl_feedback is not None,
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "gnn_layers": config["gnn_layers"],
                        "lr_decay": config["lr_decay"],
                        "prune_aggressiveness": config["prune_aggressiveness"],
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return config


def apply_config_delta(
    current_config: Dict[str, Any], delta_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply delta changes to current configuration.

    Used by RL tuner to incrementally adjust config.

    Args:
        current_config: Current dynamic configuration
        delta_config: Changes to apply

    Returns:
        Updated configuration dict

    Receipt: config_delta_receipt
    """
    updated = current_config.copy()

    for key, delta in delta_config.items():
        if key in updated and isinstance(delta, (int, float)):
            old_value = updated[key]

            if key == "gnn_layers":
                new_value = max(
                    ADAPTIVE_DEPTH_MIN, min(ADAPTIVE_DEPTH_MAX, int(old_value + delta))
                )
            elif key == "lr_decay":
                new_value = max(LR_MIN, min(LR_MAX, old_value + delta))
            elif key == "prune_aggressiveness":
                new_value = max(
                    PRUNE_FACTOR_MIN, min(PRUNE_FACTOR_MAX, old_value + delta)
                )
            else:
                new_value = old_value + delta

            updated[key] = new_value

            emit_receipt(
                "config_delta",
                {
                    "receipt_type": "config_delta",
                    "tenant_id": "axiom-adaptive",
                    "param_name": key,
                    "old_value": old_value,
                    "delta": delta,
                    "new_value": new_value,
                    "source": "rl",
                    "payload_hash": dual_hash(
                        json.dumps(
                            {"param": key, "old": old_value, "new": new_value},
                            sort_keys=True,
                        )
                    ),
                },
            )

    return updated


def validate_config_bounds(config: Dict[str, Any]) -> Dict[str, bool]:
    """Validate that config values are within physics bounds.

    Args:
        config: Configuration dict to validate

    Returns:
        Dict with validation status for each parameter
    """
    validations = {}

    if "gnn_layers" in config:
        val = config["gnn_layers"]
        validations["gnn_layers"] = ADAPTIVE_DEPTH_MIN <= val <= ADAPTIVE_DEPTH_MAX

    if "lr_decay" in config:
        val = config["lr_decay"]
        validations["lr_decay"] = LR_MIN <= val <= LR_MAX

    if "prune_aggressiveness" in config:
        val = config["prune_aggressiveness"]
        validations["prune_aggressiveness"] = (
            PRUNE_FACTOR_MIN <= val <= PRUNE_FACTOR_MAX
        )

    return validations


def get_adaptive_info() -> Dict[str, Any]:
    """Get adaptive module configuration info.

    Returns:
        Dict with all adaptive constants and configuration

    Receipt: adaptive_info
    """
    info = {
        "adaptive_depth_base": ADAPTIVE_DEPTH_BASE,
        "adaptive_depth_min": ADAPTIVE_DEPTH_MIN,
        "adaptive_depth_max": ADAPTIVE_DEPTH_MAX,
        "adaptive_depth_scaling": ADAPTIVE_DEPTH_SCALING,
        "lr_base": LR_BASE,
        "lr_min": LR_MIN,
        "lr_max": LR_MAX,
        "prune_factor_min": PRUNE_FACTOR_MIN,
        "prune_factor_max": PRUNE_FACTOR_MAX,
        "entropy_scaling_factor": ENTROPY_SCALING_FACTOR,
        "formulas": {
            "depth": "base + floor(log2(n) / 10)",
            "lr": "base_lr / sqrt(depth)",
            "prune": "base + (entropy * scaling_factor)",
        },
        "description": "Runtime scaling logic for adaptive depth, LR, and pruning. "
        "Kill static configs - go dynamic.",
    }

    emit_receipt(
        "adaptive_info",
        {
            "tenant_id": "axiom-adaptive",
            **info,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info
