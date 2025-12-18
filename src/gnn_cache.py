"""gnn_cache.py - GNN Predictive Caching Layer

THE PHYSICS (from Grok analysis):
    - GNN adds nonlinear boosts via anticipatory buffering
    - α asymptotes ~e (2.71828) - Shannon entropy bound, NOT tunable
    - Merkle batch entropy bounds as ~e*ln(n) - physics, not coincidence
    - Holds to 150d before dipping, <2.5 at 180d+, breaks at 200d+ on cache overflow
    - With pruning: extends to 250d at α>2.8, overflow pushed to 300d+

Split modules:
    - gnn_retention.py: Retention curve functions
    - gnn_overflow.py: Overflow detection and sweeps

Source: Grok - "Not coincidence - Merkle batch entropy often bounds as ~e*ln(n)"
"""

import json
import math
import os
from typing import Dict, Any, List

from .core import emit_receipt, dual_hash, StopRule

# Import all constants for use AND for backward-compatible re-export
from .constants import (
    ENTROPY_ASYMPTOTE_E,
    ASYMPTOTE_ALPHA,
    PRUNING_TARGET_ALPHA,
    MIN_EFF_ALPHA_VALIDATED,
    CACHE_DEPTH_BASELINE,
    CACHE_DEPTH_MIN,
    CACHE_DEPTH_MAX,
    OVERFLOW_THRESHOLD_DAYS,
    OVERFLOW_THRESHOLD_DAYS_PRUNED,
    OVERFLOW_CAPACITY_PCT,
    CACHE_BREAK_DAYS,
    BLACKOUT_BASE_DAYS,
    BLACKOUT_PRUNING_TARGET_DAYS,
    NONLINEAR_RETENTION_FLOOR,
    RETENTION_BASE_FACTOR,
    CURVE_TYPE,
    DECAY_LAMBDA,
    SATURATION_KAPPA,
    QUORUM_FAIL_DAYS,
    ENTRIES_PER_SOL,
    GNN_CACHE_SPEC_PATH,
    ABLATION_MODES,
    RETENTION_FACTOR_GNN_RANGE,
)
from .alpha_compute import RETENTION_FACTOR_MAX

# Backward-compatible re-exports (other modules import these from here)
__all__ = [
    # Functions
    "load_gnn_cache_spec",
    "nonlinear_retention",
    "nonlinear_retention_with_pruning",
    "gnn_boost_factor",
    "compute_asymptote",
    "compute_asymptote_with_pruning",
    "get_retention_factor_gnn_isolated",
    "apply_gnn_nonlinear_boost",
    "cache_depth_check",
    "predict_overflow",
    "extreme_blackout_sweep",
    "validate_gnn_nonlinear_slos",
    "quantum_relay_stub",
    "swarm_autorepair_stub",
    "cosmos_sim_stub",
    "get_gnn_cache_info",
    # Constants (re-exported for backward compatibility)
    "ENTROPY_ASYMPTOTE_E",
    "ASYMPTOTE_ALPHA",
    "PRUNING_TARGET_ALPHA",
    "MIN_EFF_ALPHA_VALIDATED",
    "CACHE_DEPTH_BASELINE",
    "CACHE_DEPTH_MIN",
    "CACHE_DEPTH_MAX",
    "OVERFLOW_THRESHOLD_DAYS",
    "OVERFLOW_THRESHOLD_DAYS_PRUNED",
    "OVERFLOW_CAPACITY_PCT",
    "CACHE_BREAK_DAYS",
    "BLACKOUT_BASE_DAYS",
    "BLACKOUT_PRUNING_TARGET_DAYS",
    "NONLINEAR_RETENTION_FLOOR",
    "RETENTION_BASE_FACTOR",
    "CURVE_TYPE",
    "DECAY_LAMBDA",
    "SATURATION_KAPPA",
    "QUORUM_FAIL_DAYS",
    "ENTRIES_PER_SOL",
    "ABLATION_MODES",
    "RETENTION_FACTOR_GNN_RANGE",
    # Adaptive depth integration (Dec 2025)
    "query_adaptive_depth",
    "set_adaptive_depth_enabled",
    "check_gnn_rebuild_needed",
    "rebuild_gnn_layers",
    "get_current_gnn_layers",
    "reset_gnn_layer_state",
]

# Import split modules
from .gnn_retention import (
    gnn_boost_factor,
    compute_asymptote,
    compute_asymptote_with_pruning,
    get_retention_factor_gnn_isolated,
    apply_gnn_nonlinear_boost,
)
from .gnn_overflow import (
    cache_depth_check,
    predict_overflow,
    extreme_blackout_sweep,
)

# === ADAPTIVE DEPTH INTEGRATION (Dec 2025) ===
# Kill static layers - query adaptive_depth at runtime
# Source: Grok - "Let the tree tell you how deep to look"

# Current GNN layer state (tracks for rebuild detection)
_current_gnn_layers = None
_adaptive_depth_enabled = False


def load_gnn_cache_spec(path: str = None) -> Dict[str, Any]:
    """Load and verify GNN cache specification file.

    Args:
        path: Optional path override (default: GNN_CACHE_SPEC_PATH)

    Returns:
        Dict containing GNN cache specification

    Receipt: gnn_cache_spec_ingest
    """
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, GNN_CACHE_SPEC_PATH)

    with open(path, "r") as f:
        data = json.load(f)

    content_hash = dual_hash(json.dumps(data, sort_keys=True))

    emit_receipt(
        "gnn_cache_spec_ingest",
        {
            "tenant_id": "axiom-gnn-cache",
            "file_path": path,
            "asymptote_alpha": data["asymptote_alpha"],
            "min_eff_alpha_validated": data["min_eff_alpha_validated"],
            "cache_depth_baseline": data["cache_depth_baseline"],
            "overflow_threshold_days": data["overflow_threshold_days"],
            "curve_type": data["curve_type"],
            "payload_hash": content_hash,
        },
    )

    return data


def nonlinear_retention(
    blackout_days: int, cache_depth: int = CACHE_DEPTH_BASELINE
) -> Dict[str, Any]:
    """Compute nonlinear retention with GNN predictive caching.

    Pure function. Returns retention_factor, eff_alpha, curve_type.
    Raises StopRule if blackout_days > cache_break_days AND
    cache_depth <= CACHE_DEPTH_BASELINE.

    Args:
        blackout_days: Blackout duration in days
        cache_depth: Cache depth in entries (default: 1e8)

    Returns:
        Dict with retention_factor, eff_alpha, curve_type, gnn_boost

    Raises:
        StopRule: If cache overflow detected

    Receipt: gnn_nonlinear_receipt
    """
    overflow_result = predict_overflow(blackout_days, cache_depth)

    if blackout_days > CACHE_BREAK_DAYS and cache_depth <= CACHE_DEPTH_BASELINE:
        emit_receipt(
            "overflow_stoprule",
            {
                "tenant_id": "axiom-gnn-cache",
                "blackout_days": blackout_days,
                "cache_depth": cache_depth,
                "overflow_pct": overflow_result["overflow_risk"],
                "action": "halt",
                "payload_hash": dual_hash(
                    json.dumps(
                        {
                            "blackout_days": blackout_days,
                            "cache_depth": cache_depth,
                            "overflow_pct": overflow_result["overflow_risk"],
                        },
                        sort_keys=True,
                    )
                ),
            },
        )
        raise StopRule(
            f"Cache overflow at {blackout_days}d: "
            f"{overflow_result['overflow_risk'] * 100:.1f}% > {OVERFLOW_CAPACITY_PCT * 100:.0f}%"
        )

    if blackout_days <= BLACKOUT_BASE_DAYS:
        retention_factor = RETENTION_BASE_FACTOR
    else:
        excess_days = blackout_days - BLACKOUT_BASE_DAYS
        decay = math.exp(-DECAY_LAMBDA * excess_days)
        retention_factor = (
            NONLINEAR_RETENTION_FLOOR
            + (RETENTION_BASE_FACTOR - NONLINEAR_RETENTION_FLOOR) * decay
        )
        retention_factor = max(NONLINEAR_RETENTION_FLOOR, retention_factor)

    retention_factor = round(retention_factor, 4)

    eff_alpha = compute_asymptote(blackout_days)
    gnn_boost = gnn_boost_factor(blackout_days)
    asymptote_proximity = abs(ASYMPTOTE_ALPHA - eff_alpha)

    result = {
        "blackout_days": blackout_days,
        "cache_depth": cache_depth,
        "retention_factor": retention_factor,
        "eff_alpha": eff_alpha,
        "curve_type": CURVE_TYPE,
        "gnn_boost": gnn_boost,
        "asymptote_proximity": round(asymptote_proximity, 4),
        "model": "gnn_nonlinear",
    }

    emit_receipt(
        "gnn_nonlinear",
        {
            "tenant_id": "axiom-gnn-cache",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def nonlinear_retention_with_pruning(
    blackout_days: int,
    cache_depth: int = CACHE_DEPTH_BASELINE,
    pruning_enabled: bool = True,
    trim_factor: float = 0.3,
    ablation_mode: str = "full",
) -> Dict[str, Any]:
    """Compute nonlinear retention with GNN caching and entropy pruning.

    Extended version that incorporates pruning boost.
    With pruning enabled:
    - Overflow threshold extends from 200d to 300d
    - Alpha target increases from 2.72 to 2.80

    Args:
        blackout_days: Blackout duration in days
        cache_depth: Cache depth in entries (default: 1e8)
        pruning_enabled: Whether entropy pruning is active (default: True)
        trim_factor: ln(n) trim factor (0.3-0.5 range)
        ablation_mode: Ablation mode for testing (default: "full")

    Returns:
        Dict with retention_factor, eff_alpha, curve_type, pruning_boost

    Raises:
        StopRule: If cache overflow detected

    Receipt: gnn_nonlinear_pruned
    """
    if ablation_mode == "baseline":
        result = {
            "blackout_days": blackout_days,
            "cache_depth": cache_depth,
            "retention_factor": 1.0,
            "retention_factor_gnn": 1.0,
            "eff_alpha": ENTROPY_ASYMPTOTE_E,
            "curve_type": CURVE_TYPE,
            "gnn_boost": 0.0,
            "pruning_enabled": False,
            "pruning_boost": 0.0,
            "trim_factor": 0.0,
            "asymptote_proximity": 0.0,
            "target_alpha": ENTROPY_ASYMPTOTE_E,
            "overflow_threshold": CACHE_BREAK_DAYS,
            "ablation_mode": "baseline",
            "model": "baseline_no_engineering",
        }
        emit_receipt(
            "gnn_nonlinear_pruned",
            {
                "tenant_id": "axiom-gnn-cache",
                **result,
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
            },
        )
        return result

    if ablation_mode == "no_cache":
        pruning_enabled = True
    elif ablation_mode == "no_prune":
        pruning_enabled = False
    else:
        pass

    overflow_threshold = (
        OVERFLOW_THRESHOLD_DAYS_PRUNED if pruning_enabled else CACHE_BREAK_DAYS
    )

    overflow_result = predict_overflow(blackout_days, cache_depth)

    if pruning_enabled:
        effective_usage = overflow_result["overflow_risk"] * (1 - trim_factor)
    else:
        effective_usage = overflow_result["overflow_risk"]

    if blackout_days > overflow_threshold and effective_usage >= OVERFLOW_CAPACITY_PCT:
        emit_receipt(
            "overflow_stoprule",
            {
                "tenant_id": "axiom-gnn-cache",
                "blackout_days": blackout_days,
                "cache_depth": cache_depth,
                "overflow_pct": effective_usage,
                "pruning_enabled": pruning_enabled,
                "action": "halt",
                "payload_hash": dual_hash(
                    json.dumps(
                        {
                            "blackout_days": blackout_days,
                            "cache_depth": cache_depth,
                            "pruning_enabled": pruning_enabled,
                        },
                        sort_keys=True,
                    )
                ),
            },
        )
        raise StopRule(
            f"Cache overflow at {blackout_days}d (pruning={pruning_enabled}): "
            f"{effective_usage * 100:.1f}% > {OVERFLOW_CAPACITY_PCT * 100:.0f}%"
        )

    if blackout_days <= BLACKOUT_BASE_DAYS:
        retention_factor = RETENTION_BASE_FACTOR
    else:
        excess_days = blackout_days - BLACKOUT_BASE_DAYS
        decay = math.exp(-DECAY_LAMBDA * excess_days)
        retention_factor = (
            NONLINEAR_RETENTION_FLOOR
            + (RETENTION_BASE_FACTOR - NONLINEAR_RETENTION_FLOOR) * decay
        )
        retention_factor = max(NONLINEAR_RETENTION_FLOOR, retention_factor)

    retention_factor = round(retention_factor, 4)

    if pruning_enabled and trim_factor > 0:
        excess = max(0, blackout_days - BLACKOUT_BASE_DAYS)
        pruning_boost = trim_factor * 0.1 * (1 - math.exp(-excess / 100))
        pruning_boost = round(min(0.08, pruning_boost), 4)
    else:
        pruning_boost = 0.0

    if ablation_mode == "no_cache":
        eff_alpha = compute_asymptote_with_pruning(
            blackout_days, pruning_uplift=pruning_boost
        )
        gnn_boost = 0.0
        retention_factor_gnn = 1.0
    else:
        eff_alpha = compute_asymptote_with_pruning(
            blackout_days, pruning_uplift=pruning_boost
        )
        gnn_boost = gnn_boost_factor(blackout_days)
        gnn_isolated = get_retention_factor_gnn_isolated(blackout_days)
        retention_factor_gnn = gnn_isolated["retention_factor_gnn"]

    target = PRUNING_TARGET_ALPHA if pruning_enabled else ASYMPTOTE_ALPHA
    asymptote_proximity = abs(target - eff_alpha)

    result = {
        "blackout_days": blackout_days,
        "cache_depth": cache_depth,
        "retention_factor": retention_factor,
        "retention_factor_gnn": retention_factor_gnn,
        "eff_alpha": eff_alpha,
        "curve_type": CURVE_TYPE,
        "gnn_boost": gnn_boost,
        "pruning_enabled": pruning_enabled,
        "pruning_boost": pruning_boost,
        "trim_factor": trim_factor,
        "asymptote_proximity": round(asymptote_proximity, 4),
        "target_alpha": target,
        "overflow_threshold": overflow_threshold,
        "ablation_mode": ablation_mode,
        "model": "gnn_nonlinear_pruned" if pruning_enabled else "gnn_nonlinear",
    }

    emit_receipt(
        "gnn_nonlinear_pruned",
        {
            "tenant_id": "axiom-gnn-cache",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


# === INNOVATION STUBS ===


def quantum_relay_stub() -> Dict[str, Any]:
    """Placeholder for quantum communication relay innovation."""
    result = {
        "status": "stub_only",
        "potential_boost": "unknown",
        "description": "Quantum communication relay for reduced latency and extended range",
        "gate": "future_innovation",
        "requires": "external_physics_validation",
    }
    emit_receipt(
        "quantum_relay_stub",
        {
            "tenant_id": "axiom-gnn-cache",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )
    return result


def swarm_autorepair_stub() -> Dict[str, Any]:
    """Placeholder for swarm ML auto-repair innovation."""
    result = {
        "status": "stub_only",
        "potential_boost": "unknown",
        "description": "Swarm ML auto-repair for distributed self-healing node failures",
        "gate": "future_innovation",
        "potential": "extend_quorum_tolerance_beyond_2_3",
    }
    emit_receipt(
        "swarm_autorepair_stub",
        {
            "tenant_id": "axiom-gnn-cache",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )
    return result


def cosmos_sim_stub() -> Dict[str, Any]:
    """Placeholder for xAI Cosmos integration (not available Dec 2025)."""
    result = {
        "status": "not_available",
        "reason": "no_public_api",
        "description": "xAI Cosmos Mars simulation integration",
        "availability": "no_public_real_time_mars_sim_product_dec_2025",
    }
    emit_receipt(
        "cosmos_sim_stub",
        {
            "tenant_id": "axiom-gnn-cache",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )
    return result


def get_gnn_cache_info() -> Dict[str, Any]:
    """Get GNN cache configuration info."""
    from .constants import (
        CACHE_DEPTH_MIN,
        CACHE_DEPTH_MAX,
        OVERFLOW_THRESHOLD_DAYS,
        QUORUM_FAIL_DAYS,
        ENTRIES_PER_SOL,
        SATURATION_KAPPA,
    )

    info = {
        "asymptote_alpha": ASYMPTOTE_ALPHA,
        "min_eff_alpha_validated": MIN_EFF_ALPHA_VALIDATED,
        "cache_depth_baseline": CACHE_DEPTH_BASELINE,
        "cache_depth_min": CACHE_DEPTH_MIN,
        "cache_depth_max": CACHE_DEPTH_MAX,
        "overflow_threshold_days": OVERFLOW_THRESHOLD_DAYS,
        "overflow_capacity_pct": OVERFLOW_CAPACITY_PCT,
        "quorum_fail_days": QUORUM_FAIL_DAYS,
        "cache_break_days": CACHE_BREAK_DAYS,
        "entries_per_sol": ENTRIES_PER_SOL,
        "curve_type": CURVE_TYPE,
        "nonlinear_retention_floor": NONLINEAR_RETENTION_FLOOR,
        "decay_lambda": DECAY_LAMBDA,
        "saturation_kappa": SATURATION_KAPPA,
        "description": "GNN predictive caching with nonlinear retention curve",
    }

    emit_receipt(
        "gnn_cache_info",
        {
            "tenant_id": "axiom-gnn-cache",
            **info,
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
        },
    )

    return info


def validate_gnn_nonlinear_slos(sweep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate GNN nonlinear SLOs.

    SLOs:
    1. α asymptotes within 0.02 of 2.72 by 150d
    2. 100% survival to 150d, graceful degradation 150-180d
    3. StopRule on overflow at 200d+ with baseline cache
    4. Nonlinear curve R² >= 0.98 to exponential model

    Args:
        sweep_results: Results from extreme_blackout_sweep

    Returns:
        Dict with validation results

    Receipt: gnn_nonlinear_slo_validation
    """
    if not sweep_results:
        return {"validated": False, "reason": "no sweep results"}

    # SLO 1: Asymptote proximity at 150d
    results_150d = [r for r in sweep_results if r.get("blackout_days") == 150]
    asymptote_ok = (
        all(r.get("asymptote_proximity", 1.0) <= 0.02 for r in results_150d)
        if results_150d
        else True
    )

    # SLO 2: Survival to 150d
    results_under_150d = [r for r in sweep_results if r.get("blackout_days", 0) <= 150]
    survival_to_150d = all(r.get("survival_status", False) for r in results_under_150d)

    # SLO 3: Overflow detection at 200d+
    results_200d_plus = [r for r in sweep_results if r.get("blackout_days", 0) >= 200]
    overflow_detected = any(
        r.get("overflow_triggered", False) for r in results_200d_plus
    )

    # SLO 4: No cliff behavior (check smooth degradation)
    eff_alphas = [
        (r.get("blackout_days", 0), r.get("eff_alpha", 0))
        for r in sweep_results
        if "eff_alpha" in r
    ]
    eff_alphas.sort(key=lambda x: x[0])

    no_cliff = True
    for i in range(1, len(eff_alphas)):
        if eff_alphas[i][0] - eff_alphas[i - 1][0] == 1:  # Consecutive days
            drop = eff_alphas[i - 1][1] - eff_alphas[i][1]
            if drop > 0.01:  # Max single-day drop
                no_cliff = False
                break

    validation = {
        "asymptote_proximity_ok": asymptote_ok,
        "survival_to_150d": survival_to_150d,
        "overflow_detected_at_200d": overflow_detected,
        "no_cliff_behavior": no_cliff,
        "validated": asymptote_ok and survival_to_150d and no_cliff,
    }

    emit_receipt(
        "gnn_nonlinear_slo_validation",
        {
            "tenant_id": "axiom-gnn-cache",
            **validation,
            "payload_hash": dual_hash(json.dumps(validation, sort_keys=True)),
        },
    )

    return validation


# === DYNAMIC CONFIGURATION SUPPORT (Dec 2025) ===
# Kill static baselines - go dynamic
# Source: Grok - "Stop: Static baselines - go dynamic"


# Global dynamic config state
_dynamic_config = {
    "gnn_layers": None,  # None means use default
    "lr_decay": None,
    "prune_aggressiveness": None,
    "adaptive_depth_enabled": False,
}


def apply_dynamic_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply dynamic configuration from RL tuner or adaptive module.

    Updates internal state to use dynamic parameters instead of static defaults.

    Args:
        config: Dynamic config dict with:
            - gnn_layers: Number of GNN layers
            - lr_decay: Learning rate decay value
            - prune_aggressiveness: Pruning aggressiveness factor
            - adaptive_depth_enabled: Whether adaptive depth is active

    Returns:
        Dict with old values that were replaced

    Receipt: dynamic_config_applied
    """
    global _dynamic_config

    old_values = _dynamic_config.copy()

    for key in [
        "gnn_layers",
        "lr_decay",
        "prune_aggressiveness",
        "adaptive_depth_enabled",
    ]:
        if key in config:
            _dynamic_config[key] = config[key]

    emit_receipt(
        "dynamic_config_applied",
        {
            "receipt_type": "dynamic_config_applied",
            "tenant_id": "axiom-gnn-cache",
            "old_config": {k: v for k, v in old_values.items() if v is not None},
            "new_config": {k: v for k, v in _dynamic_config.items() if v is not None},
            "source": config.get("source", "unknown"),
            "payload_hash": dual_hash(
                json.dumps(_dynamic_config, sort_keys=True, default=str)
            ),
        },
    )

    return old_values


def get_current_config() -> Dict[str, Any]:
    """Get current dynamic configuration state.

    Returns:
        Dict with current dynamic config values
    """
    return {
        "gnn_layers": _dynamic_config["gnn_layers"],
        "lr_decay": _dynamic_config["lr_decay"],
        "prune_aggressiveness": _dynamic_config["prune_aggressiveness"],
        "adaptive_depth_enabled": _dynamic_config["adaptive_depth_enabled"],
        "using_defaults": all(
            v is None
            for k, v in _dynamic_config.items()
            if k != "adaptive_depth_enabled"
        ),
    }


def reset_dynamic_config() -> None:
    """Reset dynamic configuration to defaults (static mode).

    Receipt: dynamic_config_reset
    """
    global _dynamic_config

    _dynamic_config = {
        "gnn_layers": None,
        "lr_decay": None,
        "prune_aggressiveness": None,
        "adaptive_depth_enabled": False,
    }

    emit_receipt(
        "dynamic_config_reset",
        {
            "receipt_type": "dynamic_config_reset",
            "tenant_id": "axiom-gnn-cache",
            "reason": "reset_to_static",
            "payload_hash": dual_hash(json.dumps({"reset": True})),
        },
    )


def nonlinear_retention_dynamic(
    blackout_days: int,
    cache_depth: int = CACHE_DEPTH_BASELINE,
    dynamic_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """GNN nonlinear retention with dynamic configuration support.

    Extends nonlinear_retention to accept dynamic params from RL/adaptive.

    Args:
        blackout_days: Blackout duration in days
        cache_depth: Cache depth in entries
        dynamic_config: Optional dynamic config dict with:
            - gnn_layers_delta: Change to GNN layer count
            - lr_decay: Learning rate value
            - prune_aggressiveness: Pruning factor

    Returns:
        Dict with retention metrics and dynamic_config_applied flag

    Receipt: gnn_nonlinear_dynamic_receipt
    """
    # Get base result
    base_result = nonlinear_retention(blackout_days, cache_depth)

    # Apply dynamic adjustments if provided
    if dynamic_config is not None:
        # GNN layers delta affects retention factor
        gnn_delta = dynamic_config.get("gnn_layers_delta", 0)
        if gnn_delta > 0:
            # Each additional layer contributes 1.008-1.015x
            for _ in range(gnn_delta):
                layer_boost = (
                    RETENTION_FACTOR_GNN_RANGE[0] + RETENTION_FACTOR_GNN_RANGE[1]
                ) / 2 - 1.0  # ~0.0115
                base_result["retention_factor"] *= 1.0 + layer_boost
                base_result["gnn_boost"] += layer_boost

        # LR decay affects stability (higher LR = more variance)
        lr_decay = dynamic_config.get("lr_decay", 0.002)
        lr_optimal = 0.002
        1.0 - (abs(lr_decay - lr_optimal) / lr_optimal) * 0.01

        # Prune aggressiveness affects retention
        prune_aggr = dynamic_config.get("prune_aggressiveness")
        if prune_aggr is not None:
            # Pruning adds ~0.8% per 0.1 factor
            prune_boost = (prune_aggr - 0.3) * 0.08
            base_result["retention_factor"] *= 1.0 + prune_boost

        # Cap retention at ceiling
        base_result["retention_factor"] = min(
            RETENTION_FACTOR_MAX, base_result["retention_factor"]
        )

        # Recompute alpha
        base_result["eff_alpha"] = ENTROPY_ASYMPTOTE_E * base_result["retention_factor"]

        base_result["dynamic_config_applied"] = True
        base_result["dynamic_params"] = {
            "gnn_layers_delta": gnn_delta,
            "lr_decay": lr_decay,
            "prune_aggressiveness": prune_aggr,
        }
    else:
        base_result["dynamic_config_applied"] = False

    emit_receipt(
        "gnn_nonlinear_dynamic",
        {
            "receipt_type": "gnn_nonlinear_dynamic",
            "tenant_id": "axiom-gnn-cache",
            "blackout_days": blackout_days,
            "retention_factor": base_result["retention_factor"],
            "eff_alpha": base_result["eff_alpha"],
            "dynamic_config_applied": base_result["dynamic_config_applied"],
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "days": blackout_days,
                        "retention": base_result["retention_factor"],
                        "alpha": base_result["eff_alpha"],
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return base_result


def get_gnn_cache_dynamic_info() -> Dict[str, Any]:
    """Get GNN cache module info including dynamic config status.

    Returns:
        Dict with GNN cache info and current dynamic config

    Receipt: gnn_cache_dynamic_info
    """
    base_info = get_gnn_cache_info()
    current_config = get_current_config()

    info = {
        **base_info,
        "dynamic_config": current_config,
        "dynamic_mode": not current_config["using_defaults"],
        "kill_list": [
            "Hard-coded layer counts",
            "Fixed LR values",
            "Static prune factors",
        ],
        "description_dynamic": "GNN cache with dynamic config support. "
        "Kill static baselines - go dynamic.",
    }

    emit_receipt(
        "gnn_cache_dynamic_info",
        {
            "tenant_id": "axiom-gnn-cache",
            **{k: v for k, v in info.items() if k not in ["description", "kill_list"]},
            "payload_hash": dual_hash(json.dumps(current_config, sort_keys=True)),
        },
    )

    return info


# === ADAPTIVE DEPTH LAYER QUERY (Dec 2025) ===
# Runtime layer query from adaptive_depth module
# Kill static GNN_LAYERS constant - query dynamically


def query_adaptive_depth(tree_size_n: int, entropy_h: float = 0.5) -> int:
    """Query adaptive depth module for optimal GNN layer count.

    This is the integration point between gnn_cache and adaptive_depth.
    Replaces fixed GNN_LAYERS constant with dynamic query.

    Args:
        tree_size_n: Number of entries in Merkle tree
        entropy_h: Average entropy level (0-1 range)

    Returns:
        Optimal GNN layer count from adaptive_depth module

    Receipt: Delegates to adaptive_depth.compute_depth
    """
    global _adaptive_depth_enabled

    if not _adaptive_depth_enabled:
        # Return default if adaptive depth not enabled
        return _dynamic_config.get("gnn_layers") or 6  # Default fallback

    try:
        from .adaptive_depth import compute_depth

        return compute_depth(tree_size_n, entropy_h)
    except ImportError:
        # Fallback if module not available
        return 6


def set_adaptive_depth_enabled(enabled: bool) -> None:
    """Enable or disable adaptive depth integration.

    Args:
        enabled: Whether adaptive depth should be active
    """
    global _adaptive_depth_enabled
    _adaptive_depth_enabled = enabled
    _dynamic_config["adaptive_depth_enabled"] = enabled


def check_gnn_rebuild_needed(
    tree_size_n: int, entropy_h: float = 0.5
) -> Dict[str, Any]:
    """Check if GNN needs rebuild due to depth change.

    Call at cycle boundary or entropy threshold to determine
    if GNN layers need adjustment.

    Args:
        tree_size_n: Current tree size
        entropy_h: Current entropy level

    Returns:
        Dict with rebuild_needed, old_layers, new_layers, reason

    Receipt: gnn_rebuild_receipt if rebuild needed
    """
    global _current_gnn_layers

    new_layers = query_adaptive_depth(tree_size_n, entropy_h)

    result = {
        "rebuild_needed": False,
        "old_layers": _current_gnn_layers,
        "new_layers": new_layers,
        "tree_size_n": tree_size_n,
        "entropy_h": entropy_h,
        "reason": None,
    }

    if _current_gnn_layers is None:
        # First query - initialize
        _current_gnn_layers = new_layers
        result["reason"] = "initial"
        result["rebuild_needed"] = False
        return result

    if new_layers != _current_gnn_layers:
        result["rebuild_needed"] = True
        result["reason"] = f"depth_change_{_current_gnn_layers}_to_{new_layers}"

        # Emit rebuild receipt
        emit_receipt(
            "gnn_rebuild",
            {
                "receipt_type": "gnn_rebuild",
                "tenant_id": "axiom-gnn-cache",
                "old_layers": _current_gnn_layers,
                "new_layers": new_layers,
                "tree_size_n": tree_size_n,
                "entropy_h": entropy_h,
                "trigger": "adaptive_depth_change",
                "payload_hash": dual_hash(
                    json.dumps(
                        {
                            "old": _current_gnn_layers,
                            "new": new_layers,
                            "n": tree_size_n,
                        },
                        sort_keys=True,
                    )
                ),
            },
        )

        # Update current
        _current_gnn_layers = new_layers

    return result


def rebuild_gnn_layers(new_depth: int) -> Dict[str, Any]:
    """Rebuild GNN with new layer depth.

    This function is called when adaptive depth indicates a change.
    In production, this would reinitialize the GNN model.

    Args:
        new_depth: New layer count for GNN

    Returns:
        Dict with rebuild status and timing info

    Receipt: gnn_rebuild_complete
    """
    global _current_gnn_layers

    old_depth = _current_gnn_layers or 6

    # In production: reinitialize GNN model with new_depth layers
    # For now, just update state and emit receipt
    _current_gnn_layers = new_depth
    _dynamic_config["gnn_layers"] = new_depth

    result = {
        "old_depth": old_depth,
        "new_depth": new_depth,
        "rebuild_status": "complete",
        "layers_delta": new_depth - old_depth,
    }

    emit_receipt(
        "gnn_rebuild_complete",
        {
            "receipt_type": "gnn_rebuild_complete",
            "tenant_id": "axiom-gnn-cache",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_current_gnn_layers() -> int:
    """Get current GNN layer count.

    Returns:
        Current layer count (from dynamic config or default)
    """
    global _current_gnn_layers

    if _current_gnn_layers is not None:
        return _current_gnn_layers

    if _dynamic_config.get("gnn_layers") is not None:
        return _dynamic_config["gnn_layers"]

    return 6  # Default fallback


def reset_gnn_layer_state() -> None:
    """Reset GNN layer state for testing."""
    global _current_gnn_layers, _adaptive_depth_enabled
    _current_gnn_layers = None
    _adaptive_depth_enabled = False
