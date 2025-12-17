"""alpha_compute.py - Explicit α Formula Computation Module

THE PHYSICS (from Grok clarification):
    - α and H measure different things
    - H ≤ e·ln(n) is Shannon entropy bound (invariant)
    - α = (min_eff / baseline) * retention_factor is engineered resilience
    - e is FLOOR (baseline), not ceiling
    - Ceiling is ~3.0 with ML optimization

KEY FORMULA:
    α = (min_eff / baseline) * retention_factor

    Where:
    - min_eff: minimum effective redundancy (from GNN asymptote, typically ~2.718)
    - baseline: normalized to 1.0
    - retention_factor: product of all layer contributions (GNN, pruning, etc.)
    - retention_factor > 1.0 means engineering adds resilience above Shannon floor

EXAMPLE:
    min_eff = 2.7185, baseline = 1.0, retention = 1.01
    α = (2.7185 / 1.0) * 1.01 = 2.745 ✓

CONSTANTS:
    SHANNON_FLOOR_ALPHA = 2.71828 (Baseline α without engineering = e)
    ALPHA_CEILING_TARGET = 3.0 (e * max_factor where max_factor ≈ 1.1)
    RETENTION_FACTOR_MAX = 1.10 (Ceiling / floor = 3.0 / 2.718 ≈ 1.10)

Source: Grok - "α and H measure different things... retention_factor >1 from GNN boosts"
"""

import json
import os
from functools import reduce
from typing import Dict, Any, List

from .core import emit_receipt, dual_hash, StopRule


# === PHYSICS CONSTANTS ===

SHANNON_FLOOR_ALPHA = 2.71828
"""physics: Baseline α without engineering = e (Shannon bound on resilience). NOT tunable."""

ALPHA_CEILING_TARGET = 3.0
"""physics: e * max_factor where max_factor ≈ 1.1 with full ML optimization."""

RETENTION_FACTOR_MAX = 1.10
"""physics: Ceiling / floor = 3.0 / 2.718 ≈ 1.10. Theoretical max compounding."""

RETENTION_FACTOR_MIN = 0.95
"""physics: Minimum valid retention factor. Below this indicates bug."""

RETENTION_FACTOR_STOPRULE_MAX = 1.15
"""physics: StopRule if retention exceeds this. Unphysical value indicates bug."""

RETENTION_FACTOR_GNN_RANGE = (1.008, 1.015)
"""physics: Isolated GNN contribution per Grok ablation."""

RETENTION_FACTOR_PRUNE_RANGE = (1.008, 1.015)
"""physics: Isolated pruning contribution per Grok ablation."""

ABLATION_MODES = ["full", "no_cache", "no_prune", "baseline"]
"""physics: Four-mode isolation testing."""

ALPHA_FORMULA_VERSION = "v1.0"
"""Track formula evolution."""

ALPHA_FORMULA_SPEC_PATH = "data/alpha_formula_spec.json"
"""Path to alpha formula specification file."""


def load_alpha_formula_spec(path: str = None) -> Dict[str, Any]:
    """Load and verify alpha formula specification file.

    Loads data/alpha_formula_spec.json and emits ingest receipt
    with dual_hash per CLAUDEME S4.1.

    Args:
        path: Optional path override (default: ALPHA_FORMULA_SPEC_PATH)

    Returns:
        Dict containing alpha formula specification

    Receipt: alpha_formula_spec_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, ALPHA_FORMULA_SPEC_PATH)

    with open(path, 'r') as f:
        data = json.load(f)

    content_hash = dual_hash(json.dumps(data, sort_keys=True))

    emit_receipt("alpha_formula_spec_ingest", {
        "tenant_id": "axiom-alpha-compute",
        "file_path": path,
        "formula_version": data["formula_version"],
        "formula": data["formula"],
        "shannon_floor_alpha": data["constants"]["shannon_floor_alpha"],
        "alpha_ceiling_target": data["constants"]["alpha_ceiling_target"],
        "ablation_modes": data["ablation_modes"],
        "payload_hash": content_hash
    })

    return data


def stoprule_invalid_retention(factor: float) -> None:
    """StopRule if retention factor is unphysical.

    Args:
        factor: Retention factor to validate

    Raises:
        StopRule: If factor < 0.95 or > 1.15
    """
    if factor < RETENTION_FACTOR_MIN or factor > RETENTION_FACTOR_STOPRULE_MAX:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-alpha-compute",
            "metric": "retention_factor",
            "baseline": 1.0,
            "delta": factor - 1.0,
            "classification": "violation",
            "action": "halt"
        })
        raise StopRule(
            f"Invalid retention factor {factor:.4f}: "
            f"must be in range [{RETENTION_FACTOR_MIN}, {RETENTION_FACTOR_STOPRULE_MAX}]"
        )


def stoprule_alpha_below_floor(alpha: float) -> None:
    """StopRule if alpha drops below Shannon floor.

    Args:
        alpha: Computed alpha to validate

    Raises:
        StopRule: If alpha < 2.70
    """
    if alpha < 2.70:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-alpha-compute",
            "metric": "computed_alpha",
            "baseline": SHANNON_FLOOR_ALPHA,
            "delta": alpha - SHANNON_FLOOR_ALPHA,
            "classification": "deviation",
            "action": "investigate"
        })
        raise StopRule(
            f"Alpha {alpha:.4f} below Shannon floor {SHANNON_FLOOR_ALPHA:.4f}"
        )


def stoprule_alpha_above_ceiling(alpha: float) -> None:
    """StopRule if alpha exceeds theoretical ceiling.

    Args:
        alpha: Computed alpha to validate

    Raises:
        StopRule: If alpha > 3.1
    """
    ceiling_plus_margin = ALPHA_CEILING_TARGET + 0.1
    if alpha > ceiling_plus_margin:
        emit_receipt("anomaly", {
            "tenant_id": "axiom-alpha-compute",
            "metric": "computed_alpha",
            "baseline": ALPHA_CEILING_TARGET,
            "delta": alpha - ALPHA_CEILING_TARGET,
            "classification": "deviation",
            "action": "investigate"
        })
        raise StopRule(
            f"Alpha {alpha:.4f} exceeds ceiling {ALPHA_CEILING_TARGET:.1f} + margin"
        )


def alpha_calc(
    min_eff: float,
    baseline: float,
    retention_factor: float,
    validate: bool = True
) -> Dict[str, Any]:
    """Compute alpha using explicit formula.

    THE FORMULA: α = (min_eff / baseline) * retention_factor

    This is the single source of truth for α calculation.

    Args:
        min_eff: Minimum effective redundancy (from GNN asymptote, typically ~2.718)
        baseline: Normalized baseline (typically 1.0)
        retention_factor: Product of all layer contributions (GNN, pruning, etc.)
        validate: Whether to apply stoprules (default: True)

    Returns:
        Dict with:
            - computed_alpha: The calculated α value
            - formula_used: String representation of formula
            - components: Dict with min_eff, baseline, retention_factor
            - floor_reference: Shannon floor (e)
            - ceiling_reference: Ceiling target (3.0)
            - gap_to_ceiling_pct: Percentage gap to ceiling

    Raises:
        StopRule: If retention_factor is invalid or alpha is out of bounds

    Receipt: alpha_formula
    """
    # Validate retention factor
    if validate:
        stoprule_invalid_retention(retention_factor)

    # THE FORMULA
    computed_alpha = (min_eff / baseline) * retention_factor
    computed_alpha = round(computed_alpha, 5)

    # Validate computed alpha
    if validate:
        stoprule_alpha_below_floor(computed_alpha)
        stoprule_alpha_above_ceiling(computed_alpha)

    # Compute gap to ceiling
    gap_absolute = ALPHA_CEILING_TARGET - computed_alpha
    gap_pct = (gap_absolute / ALPHA_CEILING_TARGET) * 100

    result = {
        "computed_alpha": computed_alpha,
        "formula_used": f"({min_eff} / {baseline}) * {retention_factor}",
        "components": {
            "min_eff": min_eff,
            "baseline": baseline,
            "retention_factor": retention_factor
        },
        "floor_reference": SHANNON_FLOOR_ALPHA,
        "ceiling_reference": ALPHA_CEILING_TARGET,
        "gap_to_ceiling_absolute": round(gap_absolute, 4),
        "gap_to_ceiling_pct": round(gap_pct, 2),
        "formula_version": ALPHA_FORMULA_VERSION
    }

    emit_receipt("alpha_formula", {
        "receipt_type": "alpha_formula",
        "tenant_id": "axiom-alpha-compute",
        "min_eff": min_eff,
        "baseline": baseline,
        "retention_factor": retention_factor,
        "computed_alpha": computed_alpha,
        "formula_version": ALPHA_FORMULA_VERSION,
        "floor_reference": SHANNON_FLOOR_ALPHA,
        "ceiling_reference": ALPHA_CEILING_TARGET,
        "gap_to_ceiling_pct": round(gap_pct, 2),
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def compound_retention(factors: List[float]) -> float:
    """Compute compound retention factor from multiple layers.

    Multiplicative: Π(factors). E.g., [1.01, 1.02] → 1.0302

    Args:
        factors: List of retention factors from each layer

    Returns:
        Compound retention factor
    """
    if not factors:
        return 1.0

    compound = reduce(lambda x, y: x * y, factors, 1.0)
    return round(compound, 5)


def isolate_layer_contribution(
    full_alpha: float,
    ablated_alpha: float,
    floor: float = SHANNON_FLOOR_ALPHA
) -> float:
    """Compute isolated contribution of a layer.

    Formula: contribution_factor = (full - ablated) / (full - floor)

    Args:
        full_alpha: Alpha with all layers active
        ablated_alpha: Alpha with one layer disabled
        floor: Shannon floor (default: e)

    Returns:
        Contribution factor (0-1 range, where 1 = 100% contribution)
    """
    if full_alpha <= floor:
        return 0.0

    contribution = (full_alpha - ablated_alpha) / (full_alpha - floor)
    return round(max(0.0, min(1.0, contribution)), 4)


def ceiling_gap(
    current_alpha: float,
    ceiling_target: float = ALPHA_CEILING_TARGET
) -> Dict[str, Any]:
    """Track progress toward ceiling target.

    Args:
        current_alpha: Current alpha value
        ceiling_target: Target ceiling (default: 3.0)

    Returns:
        Dict with:
            - gap_absolute: Absolute gap to ceiling
            - gap_pct: Percentage gap
            - retention_factor_current: Current effective retention
            - retention_factor_needed: Retention needed to reach ceiling
            - path_to_ceiling: Description of path

    Receipt: ceiling_track
    """
    gap_absolute = ceiling_target - current_alpha
    gap_pct = (gap_absolute / ceiling_target) * 100

    # Compute current retention factor (relative to floor)
    retention_current = current_alpha / SHANNON_FLOOR_ALPHA

    # Compute retention needed to reach ceiling
    retention_needed = ceiling_target / SHANNON_FLOOR_ALPHA

    # Retention delta
    retention_delta = retention_needed - retention_current

    # Path description
    if gap_pct <= 0:
        path = "Ceiling reached"
    elif gap_pct <= 5:
        path = "Near ceiling - minor optimization needed"
    elif gap_pct <= 10:
        path = "Close to ceiling - moderate optimization needed"
    else:
        path = f"Far from ceiling - {gap_pct:.1f}% optimization needed"

    result = {
        "current_alpha": current_alpha,
        "ceiling_target": ceiling_target,
        "gap_absolute": round(gap_absolute, 4),
        "gap_pct": round(gap_pct, 2),
        "retention_factor_current": round(retention_current, 4),
        "retention_factor_needed": round(retention_needed, 4),
        "retention_factor_delta": round(retention_delta, 4),
        "path_to_ceiling": path
    }

    emit_receipt("ceiling_track", {
        "receipt_type": "ceiling_track",
        "tenant_id": "axiom-alpha-compute",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def validate_formula(
    min_eff: float,
    retention: float,
    expected: float,
    tolerance: float = 0.01
) -> bool:
    """Validate formula correctness within tolerance.

    Args:
        min_eff: Minimum effective redundancy
        retention: Retention factor
        expected: Expected alpha value
        tolerance: Acceptable deviation (default: 0.01)

    Returns:
        True if computed alpha matches expected within tolerance
    """
    result = alpha_calc(min_eff, 1.0, retention, validate=False)
    computed = result["computed_alpha"]
    return abs(computed - expected) <= tolerance


def get_ablation_expected(mode: str) -> Dict[str, Any]:
    """Get expected results for an ablation mode.

    Args:
        mode: One of "full", "no_cache", "no_prune", "baseline"

    Returns:
        Dict with alpha_range, retention, description
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode}. Valid: {ABLATION_MODES}")

    expected = {
        "baseline": {
            "alpha_range": (2.71, 2.72),
            "retention": 1.0,
            "description": "No engineering - Shannon floor"
        },
        "no_cache": {
            "alpha_range": (2.76, 2.80),
            "retention": (1.015, 1.03),
            "description": "Pruning only - no GNN caching"
        },
        "no_prune": {
            "alpha_range": (2.72, 2.74),
            "retention": (1.003, 1.01),
            "description": "GNN only - no pruning"
        },
        "full": {
            "alpha_range": (2.80, 2.85),
            "retention": (1.03, 1.05),
            "description": "Full stack - GNN + pruning"
        }
    }

    return expected[mode]


def compute_alpha_from_layers(
    gnn_retention: float = 1.0,
    prune_retention: float = 1.0,
    base_min_eff: float = SHANNON_FLOOR_ALPHA,
    ablation_mode: str = "full"
) -> Dict[str, Any]:
    """Compute alpha from individual layer retention factors.

    Args:
        gnn_retention: GNN layer retention factor (default: 1.0)
        prune_retention: Pruning layer retention factor (default: 1.0)
        base_min_eff: Base minimum efficiency (default: e)
        ablation_mode: Ablation mode (default: "full")

    Returns:
        Dict with computed_alpha, layer_contributions, ablation_mode

    Receipt: alpha_from_layers
    """
    # Apply ablation mode
    if ablation_mode == "baseline":
        gnn_active = 1.0
        prune_active = 1.0
    elif ablation_mode == "no_cache":
        gnn_active = 1.0
        prune_active = prune_retention
    elif ablation_mode == "no_prune":
        gnn_active = gnn_retention
        prune_active = 1.0
    else:  # full
        gnn_active = gnn_retention
        prune_active = prune_retention

    # Compound retention
    compound = compound_retention([gnn_active, prune_active])

    # Compute alpha
    alpha_result = alpha_calc(base_min_eff, 1.0, compound)

    result = {
        "computed_alpha": alpha_result["computed_alpha"],
        "compound_retention": compound,
        "layer_contributions": {
            "gnn_retention": gnn_retention,
            "gnn_active": gnn_active,
            "prune_retention": prune_retention,
            "prune_active": prune_active
        },
        "ablation_mode": ablation_mode,
        "base_min_eff": base_min_eff,
        "gap_to_ceiling_pct": alpha_result["gap_to_ceiling_pct"]
    }

    emit_receipt("alpha_from_layers", {
        "receipt_type": "alpha_from_layers",
        "tenant_id": "axiom-alpha-compute",
        **result,
        "payload_hash": dual_hash(json.dumps(result, sort_keys=True))
    })

    return result


def get_alpha_compute_info() -> Dict[str, Any]:
    """Get alpha compute module configuration info.

    Returns:
        Dict with all alpha compute constants and configuration

    Receipt: alpha_compute_info
    """
    info = {
        "shannon_floor_alpha": SHANNON_FLOOR_ALPHA,
        "alpha_ceiling_target": ALPHA_CEILING_TARGET,
        "retention_factor_max": RETENTION_FACTOR_MAX,
        "retention_factor_min": RETENTION_FACTOR_MIN,
        "retention_factor_gnn_range": RETENTION_FACTOR_GNN_RANGE,
        "retention_factor_prune_range": RETENTION_FACTOR_PRUNE_RANGE,
        "ablation_modes": ABLATION_MODES,
        "alpha_formula_version": ALPHA_FORMULA_VERSION,
        "formula": "alpha = (min_eff / baseline) * retention_factor",
        "physics_clarification": {
            "alpha_vs_entropy": "α and H measure different things",
            "shannon_bound": "H ≤ e·ln(n) is Shannon entropy bound",
            "alpha_definition": "α is engineered resilience metric",
            "floor_is_e": "e is the FLOOR (baseline), not ceiling",
            "ceiling_is_3": "Ceiling is ~3.0 with ML optimization"
        },
        "description": "Explicit α formula computation module. Single source of truth for α calculation."
    }

    emit_receipt("alpha_compute_info", {
        "tenant_id": "axiom-alpha-compute",
        **info,
        "payload_hash": dual_hash(json.dumps(info, sort_keys=True))
    })

    return info


# === DYNAMIC RETENTION SUPPORT (Dec 2025) ===
# Source: Grok - "Stop: Static baselines - go dynamic"


def alpha_calc_dynamic(
    min_eff: float,
    baseline: float,
    rl_tuner_retention: float = None,
    static_retention: float = 1.01
) -> Dict[str, Any]:
    """Compute alpha with dynamic retention from RL feedback.

    Uses RL-optimized retention if provided, otherwise falls back to static.

    Args:
        min_eff: Minimum effective redundancy
        baseline: Baseline efficiency
        rl_tuner_retention: Optional RL-tuned retention factor
        static_retention: Fallback static retention (default: 1.01)

    Returns:
        Dict with computed_alpha, retention_source, gap analysis

    Receipt: alpha_calc_dynamic
    """
    # Determine retention source
    if rl_tuner_retention is not None:
        retention = rl_tuner_retention
        retention_source = "rl"
    else:
        retention = static_retention
        retention_source = "static"

    # Compute alpha using standard formula
    result = alpha_calc(min_eff, baseline, retention)

    # Add source tracking
    result["retention_source"] = retention_source
    result["dynamic_mode"] = rl_tuner_retention is not None
    result["rl_retention"] = rl_tuner_retention
    result["static_retention"] = static_retention

    # Add milestone tracking
    milestone_1 = 1.05
    milestone_2 = 1.08
    result["milestone_1_achieved"] = retention >= milestone_1
    result["milestone_2_achieved"] = retention >= milestone_2
    result["retention_target_next"] = milestone_1 if not result["milestone_1_achieved"] else (
        milestone_2 if not result["milestone_2_achieved"] else RETENTION_FACTOR_MAX
    )

    emit_receipt("alpha_calc_dynamic", {
        "receipt_type": "alpha_calc_dynamic",
        "tenant_id": "axiom-alpha-compute",
        "computed_alpha": result["computed_alpha"],
        "retention_factor": result["retention_factor"],
        "retention_source": retention_source,
        "gap_to_ceiling_pct": result["gap_to_ceiling_pct"],
        "milestone_1_achieved": result["milestone_1_achieved"],
        "payload_hash": dual_hash(json.dumps({
            "alpha": result["computed_alpha"],
            "retention": retention,
            "source": retention_source
        }, sort_keys=True))
    })

    return result


def ceiling_gap_with_rl_path(
    current_alpha: float,
    rl_projected_retention: float = None
) -> Dict[str, Any]:
    """Analyze ceiling gap with RL-projected path.

    Args:
        current_alpha: Current alpha value
        rl_projected_retention: Optional RL-projected retention improvement

    Returns:
        Dict with gap analysis and RL path projection

    Receipt: ceiling_gap_rl_path
    """
    # Get base gap analysis
    gap_result = ceiling_gap(current_alpha)

    # Add RL path projection
    if rl_projected_retention is not None:
        # Compute projected alpha with RL retention
        projected_alpha = SHANNON_FLOOR_ALPHA * rl_projected_retention
        projected_gap_pct = ((ALPHA_CEILING_TARGET - projected_alpha) / ALPHA_CEILING_TARGET) * 100

        gap_result["rl_projected_retention"] = rl_projected_retention
        gap_result["rl_projected_alpha"] = round(projected_alpha, 4)
        gap_result["rl_projected_gap_pct"] = round(projected_gap_pct, 2)
        gap_result["rl_improvement_pct"] = round(gap_result["gap_pct"] - projected_gap_pct, 2)

        # Update path with RL info
        if rl_projected_retention >= RETENTION_FACTOR_MAX:
            gap_result["path_to_ceiling"] = "RL reaches ceiling - quantum validates"
        elif rl_projected_retention >= 1.08:
            gap_result["path_to_ceiling"] = f"RL → {rl_projected_retention:.3f} (+quantum hybrid)"
        elif rl_projected_retention >= 1.05:
            gap_result["path_to_ceiling"] = f"RL → {rl_projected_retention:.3f} (+RL2 → 1.08 +quantum)"
        else:
            gap_result["path_to_ceiling"] = f"RL → {rl_projected_retention:.3f} (continue tuning)"

    emit_receipt("ceiling_gap_rl_path", {
        "receipt_type": "ceiling_gap_rl_path",
        "tenant_id": "axiom-alpha-compute",
        "current_alpha": current_alpha,
        "gap_pct": gap_result["gap_pct"],
        "rl_projected_retention": rl_projected_retention,
        "rl_projected_alpha": gap_result.get("rl_projected_alpha"),
        "path_to_ceiling": gap_result["path_to_ceiling"],
        "payload_hash": dual_hash(json.dumps({
            "alpha": current_alpha,
            "rl_retention": rl_projected_retention
        }, sort_keys=True, default=str))
    })

    return gap_result


def get_retention_milestones() -> Dict[str, Any]:
    """Get retention milestone definitions.

    Returns:
        Dict with milestone targets and their alpha equivalents
    """
    milestones = {
        "current": {
            "retention": 1.01,
            "alpha": round(SHANNON_FLOOR_ALPHA * 1.01, 4),
            "description": "Current baseline"
        },
        "milestone_1": {
            "retention": 1.05,
            "alpha": round(SHANNON_FLOOR_ALPHA * 1.05, 4),
            "description": "First RL target (this build)"
        },
        "milestone_2": {
            "retention": 1.08,
            "alpha": round(SHANNON_FLOOR_ALPHA * 1.08, 4),
            "description": "Second RL target (next build)"
        },
        "ceiling": {
            "retention": RETENTION_FACTOR_MAX,
            "alpha": ALPHA_CEILING_TARGET,
            "description": "Physics ceiling (quantum hybrid)"
        }
    }

    return milestones


def get_alpha_compute_dynamic_info() -> Dict[str, Any]:
    """Get alpha compute module info with dynamic retention support.

    Returns:
        Dict with alpha compute info and RL integration details

    Receipt: alpha_compute_dynamic_info
    """
    base_info = get_alpha_compute_info()
    milestones = get_retention_milestones()

    info = {
        **base_info,
        "retention_milestones": milestones,
        "rl_integration": {
            "rl_retention_source": "rl_tune.RLTuner.best_retention",
            "static_fallback": 1.01,
            "dynamic_mode_default": False
        },
        "kill_list": [
            "Fixed retention_factor defaults"
        ],
        "description_dynamic": "Alpha compute with dynamic RL retention input. "
                               "Kill static baselines - go dynamic."
    }

    emit_receipt("alpha_compute_dynamic_info", {
        "tenant_id": "axiom-alpha-compute",
        **{k: v for k, v in info.items() if k not in ["description", "kill_list", "physics_clarification"]},
        "payload_hash": dual_hash(json.dumps(milestones, sort_keys=True))
    })

    return info
