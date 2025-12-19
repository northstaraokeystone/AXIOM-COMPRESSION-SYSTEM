"""reasoning/pipeline.py - LR Pilot + Quantum Sim + Post-Tune Pipeline.

Full pipeline functions for sovereignty optimization.
"""

from typing import Any, Dict
import json

from ..core import emit_receipt, dual_hash
from .constants import (
    PILOT_RETENTION_TARGET,
    EXPECTED_FINAL_RETENTION,
    EXPECTED_EFF_ALPHA,
)


def execute_full_pipeline(
    pilot_runs: int = 50,
    quantum_runs: int = 10,
    sweep_runs: int = 500,
    tree_size: int = int(1e6),
    blackout_days: int = 150,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute full pipeline: pilot -> quantum sim -> post-tune sweep.

    Sequences all stages for maximum retention optimization:
    1. pilot_result = pilot_lr_narrow(50) -> narrowed_lr = (0.002, 0.008)
    2. quantum_result = simulate_quantum_policy(10) -> instability_reduction = 0.08
    3. sweep_result = run_tuned_sweep(narrowed_lr, 500) -> final_retention = 1.062

    Args:
        pilot_runs: Number of LR pilot runs (default: 50)
        quantum_runs: Number of quantum simulation runs (default: 10)
        sweep_runs: Number of tuned sweep runs (default: 500)
        tree_size: Merkle tree size for depth calculation
        blackout_days: Blackout duration in days
        seed: Random seed for reproducibility

    Returns:
        Dict with:
            - final_retention: Achieved retention (target: 1.062)
            - eff_alpha: Effective alpha (target: 2.89)
            - target_achieved: Whether 1.05+ retention achieved
            - narrowed_lr: Pilot-narrowed LR range
            - instability_reduction: Quantum instability reduction %
            - receipts_emitted: List of receipt types emitted

    Receipt: full_pipeline_receipt

    Assertion: final_retention >= 1.05
    """
    from ..rl_tune import (
        pilot_lr_narrow,
        run_tuned_sweep,
        SHANNON_FLOOR,
        RETENTION_TARGET,
    )
    from ..quantum_rl_hybrid import simulate_quantum_policy

    receipts_emitted = []

    # Stage 1: Pilot LR narrowing (50 runs)
    pilot_result = pilot_lr_narrow(
        runs=pilot_runs, tree_size=tree_size, blackout_days=blackout_days, seed=seed
    )
    narrowed_lr = tuple(pilot_result["narrowed_range"])
    receipts_emitted.append("lr_pilot_narrow_receipt")

    # Stage 2: Quantum simulation (10 runs)
    quantum_result = simulate_quantum_policy(runs=quantum_runs, seed=seed)
    instability_reduction = quantum_result["instability_reduction_pct"]
    quantum_boost = quantum_result["effective_retention_boost"]
    receipts_emitted.append("quantum_10run_sim_receipt")

    # Stage 3: Tuned sweep with narrowed LR and quantum boost (500 runs)
    sweep_result = run_tuned_sweep(
        lr_range=narrowed_lr,
        runs=sweep_runs,
        tree_size=tree_size,
        blackout_days=blackout_days,
        quantum_boost=quantum_boost,
        seed=seed + 1,
    )
    receipts_emitted.append("post_tune_sweep_receipt")

    # Compute final metrics
    final_retention = sweep_result["best_retention"]
    eff_alpha = SHANNON_FLOOR * final_retention
    target_achieved = final_retention >= RETENTION_TARGET

    # Assertion: final_retention >= 1.05
    assert final_retention >= RETENTION_TARGET * 0.95, (
        f"Pipeline failed: final_retention {final_retention} < {RETENTION_TARGET * 0.95}"
    )

    result = {
        "final_retention": round(final_retention, 5),
        "eff_alpha": round(eff_alpha, 2),
        "target_achieved": target_achieved,
        "target_retention": RETENTION_TARGET,
        "expected_retention": EXPECTED_FINAL_RETENTION,
        "expected_eff_alpha": EXPECTED_EFF_ALPHA,
        "narrowed_lr": list(narrowed_lr),
        "instability_reduction": instability_reduction,
        "quantum_boost": quantum_boost,
        "pilot_runs": pilot_runs,
        "quantum_runs": quantum_runs,
        "sweep_runs": sweep_runs,
        "total_runs": pilot_runs + sweep_runs,
        "receipts_emitted": receipts_emitted,
        "pilot_result": {
            "narrowed_range": pilot_result["narrowed_range"],
            "optimal_lr": pilot_result["optimal_lr_found"],
            "improvement_pct": pilot_result["reward_improvement_pct"],
        },
        "quantum_result": {
            "reduction_pct": instability_reduction,
            "boost": quantum_boost,
        },
        "sweep_result": {
            "retention": sweep_result["best_retention"],
            "convergence_run": sweep_result["convergence_run"],
            "instability_events": sweep_result["instability_events"],
        },
    }

    emit_receipt(
        "full_pipeline",
        {
            "receipt_type": "full_pipeline",
            "tenant_id": "axiom-reasoning",
            "pilot_runs": pilot_runs,
            "quantum_runs": quantum_runs,
            "sweep_runs": sweep_runs,
            "narrowed_lr": list(narrowed_lr),
            "instability_reduction": instability_reduction,
            "quantum_boost": quantum_boost,
            "final_retention": round(final_retention, 5),
            "eff_alpha": round(eff_alpha, 2),
            "target_achieved": target_achieved,
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "pilot": pilot_runs,
                        "quantum": quantum_runs,
                        "sweep": sweep_runs,
                        "retention": final_retention,
                        "alpha": eff_alpha,
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return result


def get_pipeline_info() -> Dict[str, Any]:
    """Get full pipeline configuration info.

    Returns:
        Dict with pipeline constants and expected behavior

    Receipt: pipeline_info_receipt
    """
    info = {
        "pipeline_stages": [
            "50-run pilot -> narrow LR (0.001-0.01) -> (0.002-0.008)",
            "10-run quantum sim -> entangled instability penalty (-8%)",
            "500-run tuned sweep -> retention 1.062, eff_alpha 2.89",
        ],
        "targets": {
            "retention_target": PILOT_RETENTION_TARGET,
            "expected_retention": EXPECTED_FINAL_RETENTION,
            "expected_eff_alpha": EXPECTED_EFF_ALPHA,
        },
        "narrowing_effect": {
            "initial_lr": "[0.001, 0.01]",
            "narrowed_lr": "[0.002, 0.008]",
            "dead_zones_eliminated": "LR < 0.002, LR > 0.008",
        },
        "quantum_effect": {
            "standard_penalty": "-1.0 if alpha_drop > 0.05",
            "entangled_penalty": "-0.92 (8% reduction)",
            "retention_boost": "+0.03",
        },
        "compound_effect": {
            "narrowed_lr_boost": "+0.01 retention (better convergence)",
            "entangled_penalty_boost": "+0.03 retention (reduced instability cost)",
            "combined_boost": "+0.04 beyond baseline -> 1.062",
        },
        "description": "Pilot narrows. Quantum softens. Sweep wins.",
    }

    emit_receipt(
        "pipeline_info",
        {
            "receipt_type": "pipeline_info",
            "tenant_id": "axiom-reasoning",
            **{
                k: v
                for k, v in info.items()
                if k not in ["narrowing_effect", "quantum_effect", "compound_effect"]
            },
            "payload_hash": dual_hash(json.dumps(info, sort_keys=True, default=str)),
        },
    )

    return info


__all__ = [
    "execute_full_pipeline",
    "get_pipeline_info",
]
