"""Zero-knowledge proof modules for AGI attestation.

Provides ZK proof systems for verifiable computation:
- base: Common ZK functions and utilities
- plonk: PLONK universal proofs
- halo2: Halo2 infinite recursive proofs
- bulletproofs: Bulletproofs high-load stress testing
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
from .halo2 import (
    HALO2_RESILIENCE_WEIGHT,
    HALO2_RESILIENCE_TARGET,
    integrate_halo2,
    run_halo2_stress_test,
    measure_halo2_overhead,
    infinite_attestation_chain,
    compute_halo2_alignment,
)
from .bulletproofs import (
    BULLETPROOFS_RESILIENCE_TARGET,
    BULLETPROOFS_RESILIENCE_WEIGHT,
    integrate_bulletproofs,
    run_bulletproofs_stress_test,
    compare_all_zk_systems,
    measure_bulletproofs_overhead,
    high_load_attestation_chain,
    compute_bulletproofs_alignment,
)

__all__ = [
    # Base ZK
    "integrate_zk_proofs",
    "run_zk_stress_test",
    "compare_attestation_methods",
    "measure_zk_overhead",
    "compute_zk_alignment",
    # PLONK
    "integrate_plonk",
    "run_plonk_stress_test",
    "compare_zk_systems",
    "measure_plonk_overhead",
    "recursive_attestation_chain",
    "compute_plonk_alignment",
    # Halo2
    "HALO2_RESILIENCE_WEIGHT",
    "HALO2_RESILIENCE_TARGET",
    "integrate_halo2",
    "run_halo2_stress_test",
    "measure_halo2_overhead",
    "infinite_attestation_chain",
    "compute_halo2_alignment",
    # Bulletproofs
    "BULLETPROOFS_RESILIENCE_TARGET",
    "BULLETPROOFS_RESILIENCE_WEIGHT",
    "integrate_bulletproofs",
    "run_bulletproofs_stress_test",
    "compare_all_zk_systems",
    "measure_bulletproofs_overhead",
    "high_load_attestation_chain",
    "compute_bulletproofs_alignment",
]

RECEIPT_SCHEMA = {
    "module": "src.paths.agi.zk",
    "receipt_types": [
        "agi_zk_integrate",
        "agi_zk_stress",
        "agi_attestation_compare",
        "agi_plonk_integrate",
        "agi_plonk_stress",
        "agi_attestation_chain",
        "agi_halo2_integration",
        "agi_halo2_stress",
        "agi_infinite_chain",
        "agi_bulletproofs_integrate",
        "agi_bulletproofs_stress",
        "agi_bulletproofs_chain",
    ],
    "version": "1.0.0",
}
