"""paths/agi/zk/bulletproofs.py - Bulletproofs High-Load Integration.

Bulletproofs ZK proof system for high-load stress testing.
"""

from typing import Any, Dict, List, Optional

from ...base import emit_path_receipt
from ..policy import AGI_TENANT_ID, ALIGNMENT_METRIC
from .halo2 import compute_halo2_alignment

BULLETPROOFS_RESILIENCE_TARGET = 1.0
BULLETPROOFS_RESILIENCE_WEIGHT = 0.15


def integrate_bulletproofs(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wire Bulletproofs high-load stress testing to AGI path."""
    from ....bulletproofs_infinite import (
        load_bulletproofs_config, stress_test, BULLETPROOFS_RESILIENCE_TARGET as BP_TARGET,
    )
    if config is None:
        config = load_bulletproofs_config()
    stress = stress_test(depth=config.get("stress_depth", 100))
    result = {
        "integrated": True, "bulletproofs_config": config,
        "stress_results": {
            "depth": stress["depth"], "all_valid": stress["all_valid"],
            "resilience": stress["resilience"], "avg_verify_time_ms": stress["avg_verify_time_ms"],
            "target_met": stress["target_met"],
        },
        "resilience_target": BP_TARGET, "no_trusted_setup": True,
        "aggregation_enabled": config.get("aggregation", True),
        "alignment_metric": ALIGNMENT_METRIC,
        "key_insight": "Bulletproofs provide high-load resilience without trusted setup",
        "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "bulletproofs_integrate", result)
    return result


def run_bulletproofs_stress_test(depth: int = 100) -> Dict[str, Any]:
    """Run Bulletproofs high-load stress test."""
    from ....bulletproofs_infinite import stress_test, BULLETPROOFS_RESILIENCE_TARGET as BP_TARGET
    stress = stress_test(depth=depth)
    result = {
        "stress_test_type": "bulletproofs_high_load", "depth": stress["depth"],
        "proofs_generated": stress["proofs_generated"], "all_valid": stress["all_valid"],
        "total_time_ms": stress["total_time_ms"], "avg_verify_time_ms": stress["avg_verify_time_ms"],
        "max_verify_time_ms": stress["max_verify_time_ms"], "resilience": stress["resilience"],
        "resilience_target": BP_TARGET, "target_met": stress["target_met"],
        "aggregation_tested": stress["aggregation_tested"],
        "alignment_metric": ALIGNMENT_METRIC, "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "bulletproofs_stress", result)
    return result


def compare_all_zk_systems() -> Dict[str, Any]:
    """Compare all ZK systems: Groth16 vs PLONK vs Halo2 vs Bulletproofs."""
    from ....bulletproofs_infinite import compare_to_halo2 as bp_compare
    bp = bp_compare(2**20)
    zk_systems = {
        "groth16": {"proof_size_bytes": 128, "verify_time_ms": 1.5, "trusted_setup": True,
            "recursion": False, "aggregation": False, "best_for": "fastest_verification"},
        "plonk": {"proof_size_bytes": 2000, "verify_time_ms": 4, "trusted_setup": True,
            "recursion": True, "aggregation": True, "best_for": "universal_setup"},
        "halo2": {"proof_size_bytes": bp["halo2"]["proof_size_bytes"],
            "verify_time_ms": bp["halo2"]["verify_time_ms"], "trusted_setup": False,
            "recursion": True, "aggregation": True, "best_for": "infinite_recursion"},
        "bulletproofs": {"proof_size_bytes": bp["bulletproofs"]["proof_size_bytes"],
            "verify_time_ms": bp["bulletproofs"]["verify_time_ms"], "trusted_setup": False,
            "recursion": False, "aggregation": True, "best_for": "range_proofs_no_setup"},
    }
    result = {
        "systems_compared": list(zk_systems.keys()), "systems": zk_systems,
        "recommendations": {
            "fastest_verification": "groth16", "universal_setup": "plonk",
            "no_trusted_setup": ["halo2", "bulletproofs"], "infinite_recursion": "halo2",
            "range_proofs": "bulletproofs", "high_load_stress": "bulletproofs", "general_circuits": "halo2",
        },
        "evolution": "Groth16 → PLONK → Halo2 → Bulletproofs (complementary)",
        "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "zk_comparison", result)
    return result


def measure_bulletproofs_overhead() -> Dict[str, Any]:
    """Measure Bulletproofs performance impact."""
    from ....bulletproofs_infinite import benchmark_bulletproofs
    benchmark = benchmark_bulletproofs()
    result = {
        "iterations": benchmark["iterations"], "avg_prove_time_ms": benchmark["avg_prove_time_ms"],
        "avg_verify_time_ms": benchmark["avg_verify_time_ms"],
        "aggregation_time_ms": benchmark["aggregation_time_ms"],
        "proof_size_bytes": benchmark["proof_size_bytes"], "range_bits": benchmark["range_bits"],
        "overhead_acceptable": benchmark["avg_verify_time_ms"] < 5,
        "alignment_metric": ALIGNMENT_METRIC, "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "bulletproofs_overhead", result)
    return result


def high_load_attestation_chain(depth: int = 100) -> Dict[str, Any]:
    """Run high-load attestation chain with Bulletproofs."""
    from ....bulletproofs_infinite import generate_infinite_chain, verify_chain
    chain = generate_infinite_chain(depth=depth)
    is_valid = verify_chain(chain)
    result = {
        "chain_type": "bulletproofs_high_load", "chain_depth": chain["chain_depth"],
        "genesis_hash": chain["genesis_hash"], "final_hash": chain["final_hash"],
        "chain_valid": is_valid, "infinite_capable": chain["infinite_capable"],
        "no_trusted_setup": chain["no_trusted_setup"],
        "alignment_metric": ALIGNMENT_METRIC, "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "bulletproofs_chain", result)
    return result


def compute_bulletproofs_alignment(receipts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Compute alignment including Bulletproofs contribution."""
    from ....agi_audit_expanded import EXPANDED_RECOVERY_THRESHOLD
    halo2_result = compute_halo2_alignment()
    bp_stress = run_bulletproofs_stress_test(depth=50)
    bp_resilience = bp_stress["resilience"]
    base_alignment = halo2_result["enhanced_alignment"]
    bp_contribution = bp_resilience * BULLETPROOFS_RESILIENCE_WEIGHT
    enhanced_alignment = min(1.0, base_alignment + bp_contribution)
    result = {
        "base_halo2_alignment": halo2_result["enhanced_alignment"],
        "groth16_resilience": halo2_result["groth16_resilience"],
        "plonk_resilience": halo2_result["plonk_resilience"],
        "halo2_resilience": halo2_result["halo2_resilience"],
        "bulletproofs_resilience": bp_resilience,
        "bulletproofs_target": BULLETPROOFS_RESILIENCE_TARGET,
        "bulletproofs_target_met": bp_resilience >= BULLETPROOFS_RESILIENCE_TARGET,
        "enhanced_alignment": round(enhanced_alignment, 4),
        "alignment_improvement": round(enhanced_alignment - base_alignment, 4),
        "is_aligned": enhanced_alignment >= EXPANDED_RECOVERY_THRESHOLD,
        "alignment_metric": "compression_as_alignment + PLONK + Halo2 + Bulletproofs",
        "key_insight": "Bulletproofs high-load resilience completes ZK stack for alignment",
        "proof_system_stack": "Groth16 + PLONK + Halo2 + Bulletproofs (complete)",
        "tenant_id": AGI_TENANT_ID,
    }
    emit_path_receipt("agi", "bulletproofs_alignment", result)
    return result


__all__ = [
    "BULLETPROOFS_RESILIENCE_TARGET", "BULLETPROOFS_RESILIENCE_WEIGHT",
    "integrate_bulletproofs", "run_bulletproofs_stress_test", "compare_all_zk_systems",
    "measure_bulletproofs_overhead", "high_load_attestation_chain", "compute_bulletproofs_alignment",
]
