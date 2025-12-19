"""core.py - Re-export wrapper for backward compatibility.

All implementation moved to src/paths/agi/ submodules.
This file exists ONLY to maintain import compatibility.

CLAUDEME COMPLIANT: ≤50 lines
"""

from .policy import (
    AGI_TENANT_ID, POLICY_DEPTH_DEFAULT, ETHICS_DIMENSIONS, ALIGNMENT_METRIC,
    AUDIT_REQUIREMENT, GROTH16_PROOF_TIME_MS, GROTH16_VERIFY_TIME_MS,
    stub_status, fractal_policy, get_sub_dimensions, compute_tree_complexity,
    evaluate_ethics, compute_alignment, audit_decision, get_agi_info,
)
from .defenses import (
    integrate_adversarial, run_alignment_stress_test, compute_adversarial_alignment,
    integrate_fractal_encrypt, run_side_channel_stress_test,
    run_model_inversion_stress_test, compute_fractal_alignment,
    integrate_randomized_paths, run_timing_stress_test, run_power_stress_test,
    run_cache_stress_test, compute_randomized_alignment,
    integrate_quantum_resist, run_spectre_stress_test, run_quantum_cache_stress_test,
    run_branch_stress_test, compute_quantum_alignment,
    integrate_secure_enclave, run_btb_stress_test, run_pht_stress_test,
    run_rsb_stress_test, measure_defense_overhead, compute_enclave_alignment,
    integrate_expanded_audits, run_injection_stress_test, run_poisoning_stress_test,
    compute_expanded_alignment,
)
from .zk import (
    integrate_zk_proofs, run_zk_stress_test, compare_attestation_methods,
    measure_zk_overhead, compute_zk_alignment,
    integrate_plonk, run_plonk_stress_test, compare_zk_systems,
    measure_plonk_overhead, recursive_attestation_chain, compute_plonk_alignment,
    HALO2_RESILIENCE_WEIGHT, HALO2_RESILIENCE_TARGET, integrate_halo2,
    run_halo2_stress_test, measure_halo2_overhead, infinite_attestation_chain,
    compute_halo2_alignment,
    BULLETPROOFS_RESILIENCE_TARGET, BULLETPROOFS_RESILIENCE_WEIGHT,
    integrate_bulletproofs, run_bulletproofs_stress_test, compare_all_zk_systems,
    measure_bulletproofs_overhead, high_load_attestation_chain, compute_bulletproofs_alignment,
)

__all__ = [
    "AGI_TENANT_ID", "POLICY_DEPTH_DEFAULT", "ETHICS_DIMENSIONS", "ALIGNMENT_METRIC",
    "stub_status", "fractal_policy", "evaluate_ethics", "compute_alignment", "audit_decision",
    "get_agi_info", "integrate_adversarial", "compute_adversarial_alignment",
    "integrate_fractal_encrypt", "compute_fractal_alignment", "integrate_randomized_paths",
    "integrate_quantum_resist", "integrate_secure_enclave", "compute_enclave_alignment",
    "integrate_expanded_audits", "compute_expanded_alignment", "integrate_zk_proofs",
    "integrate_plonk", "compute_plonk_alignment", "integrate_halo2", "compute_halo2_alignment",
    "integrate_bulletproofs", "compute_bulletproofs_alignment",
]
