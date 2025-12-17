"""adaptive_depth.py - Dynamic n-based GNN Layer Scaling

PARADIGM SHIFT:
    OLD: Fixed GNN layers (static depth regardless of tree size)
    NEW: Dynamic depth scales with tree size n and entropy h

    layers = base_layers + round(scale_factor * log(n / baseline_n))

THE PHYSICS:
    - Tree size correlates with buffered decisions
    - Deeper GNN captures longer-range predictions without fixed over-parameterization
    - Informs RL policy -> converges to 1.05+ in 300-500 runs vs 1000+ blind

KEY INSIGHT:
    Static depth leaves retention on the table. A 10^6 tree doesn't need the same
    depth as a 10^12 tree. Over-parameterizing wastes compute. Under-parameterizing
    misses patterns. Adaptive depth finds the Goldilocks zone automatically.

FUNCTIONS:
    load_depth_spec() -> dict: Load adaptive_depth_spec.json, emit receipt
    compute_depth(tree_size_n, entropy_h) -> int: Calculate layers using scaling formula
    validate_depth(layers) -> bool: Ensure layers within bounds
    get_depth_scaling_info() -> dict: Get module configuration

Source: Grok - "Let the tree tell you how deep to look"
"""

import json
import math
import os
from typing import Dict, Any

from .core import emit_receipt, dual_hash, StopRule


# === CONSTANTS (Adaptive Depth Scaling) ===

ADAPTIVE_DEPTH_SPEC_PATH = "data/adaptive_depth_spec.json"
"""Path to adaptive depth specification file."""

# Default values (overridden by spec file)
BASE_LAYERS = 4
"""Base GNN depth before scaling."""

SCALE_FACTOR = 0.5
"""Entropy density scaling factor."""

BASELINE_N = 1000000
"""Typical early Merkle tree size (10^6)."""

MAX_LAYERS = 12
"""Compute safety cap."""

MIN_LAYERS = 4
"""Minimum viable depth."""

SWEEP_LIMIT = 500
"""Informed RL sweep limit (vs 1000 blind)."""

QUICK_TARGET = 1.05
"""First retention milestone."""


# === SPEC LOADING ===

_cached_spec = None


def load_depth_spec(path: str = None) -> Dict[str, Any]:
    """Load and verify adaptive depth specification file.

    Loads data/adaptive_depth_spec.json and emits depth_spec_receipt
    with dual_hash per CLAUDEME S4.1.

    Args:
        path: Optional path override (default: ADAPTIVE_DEPTH_SPEC_PATH)

    Returns:
        Dict containing adaptive depth specification

    Receipt: depth_spec_receipt
    """
    global _cached_spec

    if _cached_spec is not None and path is None:
        return _cached_spec

    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, ADAPTIVE_DEPTH_SPEC_PATH)

    with open(path, 'r') as f:
        data = json.load(f)

    content_hash = dual_hash(json.dumps(data, sort_keys=True))

    emit_receipt("depth_spec", {
        "receipt_type": "depth_spec",
        "tenant_id": "axiom-colony",
        "base_layers": data["base_layers"],
        "scale_factor": data["scale_factor"],
        "baseline_n": data["baseline_n"],
        "max_layers": data["max_layers"],
        "sweep_limit": data["sweep_limit"],
        "spec_hash": content_hash,
        "payload_hash": content_hash
    })

    _cached_spec = data
    return data


def _get_spec_value(key: str, default: Any = None) -> Any:
    """Get value from spec with fallback to default."""
    try:
        spec = load_depth_spec()
        return spec.get(key, default)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# === STOPRULES ===

def stoprule_invalid_depth(layers: int) -> None:
    """StopRule if computed depth is invalid.

    Args:
        layers: Computed layer count

    Raises:
        StopRule: If layers < 1 or layers > max_layers
    """
    max_l = _get_spec_value("max_layers", MAX_LAYERS)
    min_l = _get_spec_value("min_layers", MIN_LAYERS)

    if layers < 1 or layers > max_l:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-colony",
            "metric": "invalid_depth",
            "baseline": f"[{min_l}, {max_l}]",
            "delta": layers,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(f"Invalid depth: {layers} not in [{min_l}, {max_l}]")


def stoprule_negative_entropy(entropy_h: float) -> None:
    """StopRule if entropy is negative.

    Args:
        entropy_h: Entropy value

    Raises:
        StopRule: If entropy_h < 0
    """
    if entropy_h < 0:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-colony",
            "metric": "negative_entropy",
            "baseline": 0.0,
            "delta": entropy_h,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(f"Negative entropy: {entropy_h}")


# === CORE FUNCTIONS ===

def compute_depth(tree_size_n: int, entropy_h: float = 0.5) -> int:
    """Compute adaptive GNN layer count from tree size and entropy.

    Formula: layers = base_layers + round(scale_factor * log(n / baseline_n))

    Larger trees need deeper networks to capture longer-range predictions.
    Entropy can modulate the scaling (higher entropy -> deeper for better dedup).

    Args:
        tree_size_n: Number of entries in Merkle tree
        entropy_h: Average entropy level (0-1 range, default: 0.5)

    Returns:
        Adaptive layer count (clamped to min/max bounds)

    Raises:
        StopRule: If entropy_h < 0 or computed depth invalid

    Receipt: adaptive_depth_receipt
    """
    # Validate entropy
    stoprule_negative_entropy(entropy_h)

    # Load spec values
    base_layers = _get_spec_value("base_layers", BASE_LAYERS)
    scale_factor = _get_spec_value("scale_factor", SCALE_FACTOR)
    baseline_n = _get_spec_value("baseline_n", BASELINE_N)
    max_layers = _get_spec_value("max_layers", MAX_LAYERS)
    min_layers = _get_spec_value("min_layers", MIN_LAYERS)

    # Handle edge cases
    if tree_size_n <= 0:
        layers = base_layers
    elif tree_size_n <= baseline_n:
        # Small trees use base depth
        layers = base_layers
    else:
        # Apply scaling formula: base + scale * ln(n / baseline)
        ratio = tree_size_n / baseline_n
        raw_depth = base_layers + scale_factor * math.log(ratio)

        # Optional: entropy modulation (higher entropy -> slightly deeper)
        # This helps with dedup prediction in high-entropy trees
        entropy_mod = 1.0 + (entropy_h - 0.5) * 0.1  # +-5% at extremes
        raw_depth *= entropy_mod

        layers = round(raw_depth)

    # Clamp to bounds
    layers = max(min_layers, min(max_layers, layers))

    # Validate
    stoprule_invalid_depth(layers)

    # Emit receipt
    result = {
        "tree_size_n": tree_size_n,
        "entropy_h": entropy_h,
        "computed_layers": layers,
        "scaling_formula": "base + scale * log(n/baseline)",
        "base_layers": base_layers,
        "scale_factor": scale_factor,
        "baseline_n": baseline_n
    }

    emit_receipt("adaptive_depth", {
        "receipt_type": "adaptive_depth",
        "tenant_id": "axiom-colony",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return layers


def validate_depth(layers: int) -> bool:
    """Ensure layers within [min_layers, max_layers].

    Args:
        layers: Layer count to validate

    Returns:
        True if valid

    Raises:
        StopRule: If layers invalid
    """
    max_layers = _get_spec_value("max_layers", MAX_LAYERS)
    min_layers = _get_spec_value("min_layers", MIN_LAYERS)

    if layers < min_layers or layers > max_layers:
        stoprule_invalid_depth(layers)
        return False

    return True


def get_depth_for_sweep_run(
    tree_size_n: int,
    entropy_h: float,
    run_number: int,
    total_runs: int = 500
) -> int:
    """Get depth with sweep progress context.

    Useful for RL sweeps to track depth decisions across runs.

    Args:
        tree_size_n: Tree size for depth calculation
        entropy_h: Entropy level
        run_number: Current run in sweep (1-indexed)
        total_runs: Total runs in sweep (default: 500)

    Returns:
        Computed layer count
    """
    depth = compute_depth(tree_size_n, entropy_h)

    # Log progress at milestones
    if run_number in [1, 50, 100, 250, 500] or run_number == total_runs:
        emit_receipt("depth_sweep_milestone", {
            "receipt_type": "depth_sweep_milestone",
            "tenant_id": "axiom-colony",
            "run_number": run_number,
            "total_runs": total_runs,
            "tree_size_n": tree_size_n,
            "computed_depth": depth,
            "progress_pct": round(run_number / total_runs * 100, 1),
            "payload_hash": dual_hash(json.dumps({
                "run": run_number, "depth": depth
            }, sort_keys=True))
        })

    return depth


def get_depth_scaling_info() -> Dict[str, Any]:
    """Get adaptive depth module configuration info.

    Returns:
        Dict with all adaptive depth constants and formulas

    Receipt: depth_scaling_info
    """
    spec = load_depth_spec()

    # Compute example depths
    examples = {}
    for n_exp in [4, 6, 8, 9, 12, 15]:
        n = 10 ** n_exp
        depth = compute_depth(n, 0.5)
        examples[f"n_1e{n_exp}"] = depth

    info = {
        "base_layers": spec["base_layers"],
        "scale_factor": spec["scale_factor"],
        "baseline_n": spec["baseline_n"],
        "max_layers": spec["max_layers"],
        "min_layers": spec.get("min_layers", MIN_LAYERS),
        "sweep_limit": spec["sweep_limit"],
        "quick_target": spec["quick_target"],
        "formula": spec["scaling_formula"]["formula"],
        "example_depths": examples,
        "description": "Dynamic n-based GNN layer scaling. "
                       "Tree size correlates with buffered decisions. "
                       "Kill static layers - go dynamic."
    }

    emit_receipt("depth_scaling_info", {
        "tenant_id": "axiom-colony",
        **info,
        "payload_hash": dual_hash(json.dumps(info, sort_keys=True, default=str))
    })

    return info


def clear_spec_cache() -> None:
    """Clear cached spec for testing."""
    global _cached_spec
    _cached_spec = None
