"""reasoning/ablation.py - Ablation Testing Functions.

Functions for ablation testing, layer isolation, and contribution analysis.
"""

from typing import Any, Dict, List, Optional
import json

from ..core import emit_receipt, StopRule, dual_hash
from ..gnn_cache import (
    nonlinear_retention_with_pruning,
    get_retention_factor_gnn_isolated,
    CACHE_DEPTH_BASELINE,
    RETENTION_FACTOR_GNN_RANGE,
)
from ..alpha_compute import (
    alpha_calc,
    compound_retention,
    isolate_layer_contribution,
    ceiling_gap,
    SHANNON_FLOOR_ALPHA,
    ALPHA_CEILING_TARGET,
    ABLATION_MODES,
)
from ..pruning import (
    generate_sample_merkle_tree,
    get_retention_factor_prune_isolated,
    RETENTION_FACTOR_PRUNE_RANGE,
)


def ablation_sweep(
    modes: List[str] = None,
    blackout_days: int = 150,
    iterations: int = 100,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Run ablation sweep across all 4 modes.

    Ablation modes isolate layer contributions:
    - baseline: No engineering, alpha = e (Shannon floor)
    - no_cache: Pruning only, no GNN caching
    - no_prune: GNN only, no pruning
    - full: All layers active

    Args:
        modes: List of ablation modes (default: all 4)
        blackout_days: Blackout duration for testing (default: 150)
        iterations: Number of iterations per mode (default: 100)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dict with results per mode, ordering validation, layer contributions

    Receipt: ablation_sweep
    """
    import random

    if modes is None:
        modes = ABLATION_MODES

    if seed is not None:
        random.seed(seed)

    results_by_mode = {}
    merkle_tree = generate_sample_merkle_tree(n_leaves=100, duplicate_ratio=0.2)

    for mode in modes:
        mode_results = []

        for i in range(iterations):
            try:
                # Run with ablation mode
                retention_result = nonlinear_retention_with_pruning(
                    blackout_days,
                    CACHE_DEPTH_BASELINE,
                    pruning_enabled=(mode != "no_prune" and mode != "baseline"),
                    trim_factor=0.3,
                    ablation_mode=mode,
                )

                # Get isolated factors
                gnn_isolated = get_retention_factor_gnn_isolated(blackout_days)
                prune_isolated = get_retention_factor_prune_isolated(merkle_tree, 0.3)

                mode_results.append(
                    {
                        "iteration": i,
                        "ablation_mode": mode,
                        "eff_alpha": retention_result["eff_alpha"],
                        "retention_factor_gnn": gnn_isolated["retention_factor_gnn"],
                        "retention_factor_prune": prune_isolated[
                            "retention_factor_prune"
                        ],
                        "success": True,
                    }
                )
            except StopRule as e:
                mode_results.append(
                    {
                        "iteration": i,
                        "ablation_mode": mode,
                        "eff_alpha": 0.0,
                        "success": False,
                        "stoprule_reason": str(e),
                    }
                )

        # Aggregate stats for mode
        successful = [r for r in mode_results if r["success"]]
        alpha_values = [r["eff_alpha"] for r in successful]

        results_by_mode[mode] = {
            "mode": mode,
            "iterations": iterations,
            "successful": len(successful),
            "failed": len(mode_results) - len(successful),
            "avg_alpha": round(sum(alpha_values) / max(1, len(alpha_values)), 4)
            if alpha_values
            else 0.0,
            "min_alpha": round(min(alpha_values), 4) if alpha_values else 0.0,
            "max_alpha": round(max(alpha_values), 4) if alpha_values else 0.0,
            "results": mode_results,
        }

    # Validate ordering: baseline < no_prune < no_cache < full
    ordering_valid = True
    expected_order = ["baseline", "no_prune", "no_cache", "full"]
    prev_alpha = 0.0

    for mode in expected_order:
        if mode in results_by_mode:
            current_alpha = results_by_mode[mode]["avg_alpha"]
            if current_alpha < prev_alpha:
                ordering_valid = False
            prev_alpha = current_alpha

    # Compute layer contributions
    baseline_alpha = results_by_mode.get("baseline", {}).get(
        "avg_alpha", SHANNON_FLOOR_ALPHA
    )
    full_alpha = results_by_mode.get("full", {}).get("avg_alpha", SHANNON_FLOOR_ALPHA)
    no_cache_alpha = results_by_mode.get("no_cache", {}).get(
        "avg_alpha", SHANNON_FLOOR_ALPHA
    )
    no_prune_alpha = results_by_mode.get("no_prune", {}).get(
        "avg_alpha", SHANNON_FLOOR_ALPHA
    )

    gnn_contribution = isolate_layer_contribution(
        full_alpha, no_cache_alpha, baseline_alpha
    )
    prune_contribution = isolate_layer_contribution(
        full_alpha, no_prune_alpha, baseline_alpha
    )

    result = {
        "blackout_days": blackout_days,
        "iterations": iterations,
        "modes_tested": modes,
        "results_by_mode": {
            m: {k: v for k, v in r.items() if k != "results"}
            for m, r in results_by_mode.items()
        },
        "ordering_valid": ordering_valid,
        "expected_ordering": expected_order,
        "layer_contributions": {
            "gnn_contribution": gnn_contribution,
            "prune_contribution": prune_contribution,
            "total_uplift": round(full_alpha - baseline_alpha, 4),
        },
        "shannon_floor": SHANNON_FLOOR_ALPHA,
        "ceiling_target": ALPHA_CEILING_TARGET,
        "gap_to_ceiling": ceiling_gap(full_alpha),
    }

    emit_receipt(
        "ablation_sweep",
        {
            "receipt_type": "ablation_sweep",
            "tenant_id": "axiom-reasoning",
            **{k: v for k, v in result.items() if k != "results_by_mode"},
            "mode_summary": {
                m: {"avg_alpha": r["avg_alpha"], "successful": r["successful"]}
                for m, r in results_by_mode.items()
            },
            "payload_hash": dual_hash(
                json.dumps(
                    {k: v for k, v in result.items() if k != "results_by_mode"},
                    sort_keys=True,
                )
            ),
        },
    )

    return result


def compute_alpha_with_isolation(
    gnn_result: Dict[str, Any],
    prune_result: Dict[str, Any],
    base_min_eff: float = SHANNON_FLOOR_ALPHA,
) -> Dict[str, Any]:
    """Compute alpha combining isolated layer factors via explicit formula.

    Uses the explicit formula: alpha = (min_eff / baseline) * retention_factor
    where retention_factor = gnn_factor * prune_factor (compound)

    Args:
        gnn_result: Result from get_retention_factor_gnn_isolated
        prune_result: Result from get_retention_factor_prune_isolated
        base_min_eff: Base minimum efficiency (default: e)

    Returns:
        Dict with computed_alpha, compound_retention, layer breakdown

    Receipt: alpha_with_isolation
    """
    gnn_factor = gnn_result.get("retention_factor_gnn", 1.0)
    prune_factor = prune_result.get("retention_factor_prune", 1.0)

    # Compound retention
    compound = compound_retention([gnn_factor, prune_factor])

    # Compute alpha using explicit formula
    alpha_result = alpha_calc(base_min_eff, 1.0, compound)

    result = {
        "computed_alpha": alpha_result["computed_alpha"],
        "compound_retention": compound,
        "gnn_retention_factor": gnn_factor,
        "gnn_contribution_pct": gnn_result.get("contribution_pct", 0.0),
        "prune_retention_factor": prune_factor,
        "prune_contribution_pct": prune_result.get("contribution_pct", 0.0),
        "base_min_eff": base_min_eff,
        "gap_to_ceiling_pct": alpha_result["gap_to_ceiling_pct"],
        "formula_used": alpha_result["formula_used"],
    }

    emit_receipt(
        "alpha_with_isolation",
        {
            "receipt_type": "alpha_with_isolation",
            "tenant_id": "axiom-reasoning",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_layer_contributions(
    blackout_days: int = 150, trim_factor: float = 0.3
) -> Dict[str, Any]:
    """Get isolated contribution from each layer.

    Returns breakdown of GNN and pruning contributions with percentages.

    Args:
        blackout_days: Blackout duration for testing
        trim_factor: Pruning trim factor

    Returns:
        Dict with gnn_contribution, prune_contribution, compound breakdown

    Receipt: layer_contributions
    """
    merkle_tree = generate_sample_merkle_tree(n_leaves=100, duplicate_ratio=0.2)

    # Get isolated factors
    gnn_isolated = get_retention_factor_gnn_isolated(blackout_days)
    prune_isolated = get_retention_factor_prune_isolated(merkle_tree, trim_factor)

    # Compute compound
    gnn_factor = gnn_isolated["retention_factor_gnn"]
    prune_factor = prune_isolated["retention_factor_prune"]
    compound = compound_retention([gnn_factor, prune_factor])

    # Compute alphas at each level
    baseline_alpha = SHANNON_FLOOR_ALPHA
    gnn_only_alpha = alpha_calc(baseline_alpha, 1.0, gnn_factor, validate=False)[
        "computed_alpha"
    ]
    prune_only_alpha = alpha_calc(baseline_alpha, 1.0, prune_factor, validate=False)[
        "computed_alpha"
    ]
    full_alpha = alpha_calc(baseline_alpha, 1.0, compound, validate=False)[
        "computed_alpha"
    ]

    result = {
        "blackout_days": blackout_days,
        "trim_factor": trim_factor,
        "gnn_layer": {
            "retention_factor": gnn_factor,
            "contribution_pct": gnn_isolated["contribution_pct"],
            "alpha_with_gnn_only": gnn_only_alpha,
            "range_expected": RETENTION_FACTOR_GNN_RANGE,
        },
        "prune_layer": {
            "retention_factor": prune_factor,
            "contribution_pct": prune_isolated["contribution_pct"],
            "alpha_with_prune_only": prune_only_alpha,
            "range_expected": RETENTION_FACTOR_PRUNE_RANGE,
        },
        "compound": {
            "compound_retention": compound,
            "full_alpha": full_alpha,
            "total_uplift_from_floor": round(full_alpha - baseline_alpha, 4),
        },
        "ceiling_analysis": ceiling_gap(full_alpha),
        "shannon_floor": baseline_alpha,
    }

    emit_receipt(
        "layer_contributions",
        {
            "receipt_type": "layer_contributions",
            "tenant_id": "axiom-reasoning",
            **{k: v for k, v in result.items() if k != "ceiling_analysis"},
            "gap_to_ceiling_pct": result["ceiling_analysis"]["gap_pct"],
            "payload_hash": dual_hash(
                json.dumps(
                    {k: v for k, v in result.items() if k != "ceiling_analysis"},
                    sort_keys=True,
                )
            ),
        },
    )

    return result


__all__ = [
    "ablation_sweep",
    "compute_alpha_with_isolation",
    "get_layer_contributions",
]
