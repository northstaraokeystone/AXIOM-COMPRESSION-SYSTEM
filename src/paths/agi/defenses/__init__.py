"""AGI defense modules for alignment and security.

Each defense type has its own file:
- adversarial: Noise injection and recovery testing
- encryption: Fractal encryption for side-channel resistance
- timing: Randomized execution paths
- spectre: Quantum-resistant branch prediction hardening
- enclave: SGX secure enclave integration
"""

from .adversarial import (
    integrate_adversarial,
    run_alignment_stress_test,
    compute_adversarial_alignment,
)
from .encryption import (
    integrate_fractal_encrypt,
    run_side_channel_stress_test,
    run_model_inversion_stress_test,
    compute_fractal_alignment,
)
from .timing import (
    integrate_randomized_paths,
    run_timing_stress_test,
    run_power_stress_test,
    run_cache_stress_test,
    compute_randomized_alignment,
)
from .spectre import (
    integrate_quantum_resist,
    run_spectre_stress_test,
    run_quantum_cache_stress_test,
    run_branch_stress_test,
    compute_quantum_alignment,
)
from .enclave import (
    integrate_secure_enclave,
    run_btb_stress_test,
    run_pht_stress_test,
    run_rsb_stress_test,
    measure_defense_overhead,
    compute_enclave_alignment,
)

__all__ = [
    # Adversarial
    "integrate_adversarial", "run_alignment_stress_test", "compute_adversarial_alignment",
    # Encryption
    "integrate_fractal_encrypt", "run_side_channel_stress_test",
    "run_model_inversion_stress_test", "compute_fractal_alignment",
    # Timing
    "integrate_randomized_paths", "run_timing_stress_test",
    "run_power_stress_test", "run_cache_stress_test", "compute_randomized_alignment",
    # Spectre
    "integrate_quantum_resist", "run_spectre_stress_test",
    "run_quantum_cache_stress_test", "run_branch_stress_test", "compute_quantum_alignment",
    # Enclave
    "integrate_secure_enclave", "run_btb_stress_test", "run_pht_stress_test",
    "run_rsb_stress_test", "measure_defense_overhead", "compute_enclave_alignment",
]

RECEIPT_SCHEMA = {
    "module": "src.paths.agi.defenses",
    "receipt_types": [
        "agi_adversarial_integrate", "agi_alignment_stress",
        "agi_fractal_encrypt_integrate", "agi_side_channel_stress",
        "agi_randomized_integrate", "agi_timing_stress",
        "agi_quantum_integrate", "agi_spectre_stress",
        "agi_enclave_integrate", "agi_btb_stress",
    ],
    "version": "1.0.0",
}
