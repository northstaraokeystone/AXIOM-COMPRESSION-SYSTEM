"""Zero-knowledge proof modules for AGI attestation.

Provides ZK proof systems for verifiable computation:
- base: Common ZK functions and utilities
- groth16: Groth16 SNARK proofs
- plonk: PLONK universal proofs
"""

from .base import (
    integrate_zk_proofs,
    run_zk_stress_test,
    compare_attestation_methods,
    measure_zk_overhead,
    compute_zk_alignment,
)
from .plonk import (
    integrate_plonk,
    run_plonk_stress_test,
    compare_zk_systems,
    measure_plonk_overhead,
    recursive_attestation_chain,
    compute_plonk_alignment,
)

__all__ = [
    # Base ZK
    "integrate_zk_proofs", "run_zk_stress_test", "compare_attestation_methods",
    "measure_zk_overhead", "compute_zk_alignment",
    # PLONK
    "integrate_plonk", "run_plonk_stress_test", "compare_zk_systems",
    "measure_plonk_overhead", "recursive_attestation_chain", "compute_plonk_alignment",
]

RECEIPT_SCHEMA = {
    "module": "src.paths.agi.zk",
    "receipt_types": [
        "agi_zk_integrate", "agi_zk_stress", "agi_attestation_compare",
        "agi_plonk_integrate", "agi_plonk_stress", "agi_attestation_chain",
    ],
    "version": "1.0.0",
}
