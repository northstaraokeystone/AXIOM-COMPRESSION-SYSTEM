"""core.py - Re-export wrapper for backward compatibility.

All implementation moved to src/paths/agi/ submodules.
This file exists ONLY to maintain import compatibility.

CLAUDEME COMPLIANT: ≤50 lines
"""

from .policy import (
    AGI_TENANT_ID,
    POLICY_DEPTH_DEFAULT,
    ETHICS_DIMENSIONS,
    ALIGNMENT_METRIC,
    stub_status,
    fractal_policy,
    evaluate_ethics,
    compute_alignment,
    audit_decision,
    get_agi_info,
)
from .defenses import (
    integrate_adversarial,
    compute_adversarial_alignment,
    integrate_fractal_encrypt,
    compute_fractal_alignment,
    integrate_randomized_paths,
    integrate_quantum_resist,
    integrate_secure_enclave,
    compute_enclave_alignment,
    integrate_expanded_audits,
    compute_expanded_alignment,
)
from .zk import (
    integrate_zk_proofs,
    integrate_plonk,
    compute_plonk_alignment,
    integrate_halo2,
    compute_halo2_alignment,
    integrate_bulletproofs,
    compute_bulletproofs_alignment,
)

__all__ = [
    "AGI_TENANT_ID",
    "POLICY_DEPTH_DEFAULT",
    "ETHICS_DIMENSIONS",
    "ALIGNMENT_METRIC",
    "stub_status",
    "fractal_policy",
    "evaluate_ethics",
    "compute_alignment",
    "audit_decision",
    "get_agi_info",
    "integrate_adversarial",
    "compute_adversarial_alignment",
    "integrate_fractal_encrypt",
    "compute_fractal_alignment",
    "integrate_randomized_paths",
    "integrate_quantum_resist",
    "integrate_secure_enclave",
    "compute_enclave_alignment",
    "integrate_expanded_audits",
    "compute_expanded_alignment",
    "integrate_zk_proofs",
    "integrate_plonk",
    "compute_plonk_alignment",
    "integrate_halo2",
    "compute_halo2_alignment",
    "integrate_bulletproofs",
    "compute_bulletproofs_alignment",
]
