"""AXIOM Audits Package - Consolidated AGI audit modules.

This package consolidates:
- agi_audit_expanded.py -> audits.adversarial
- fractal_encrypt_audit.py -> audits.encryption
- randomized_paths_audit.py -> audits.timing
- quantum_resist_random.py -> audits.quantum
- secure_enclave_audit.py -> audits.enclave

All modules share similar patterns through AuditModuleBase.
"""

from .base import AuditModuleBase

__all__ = [
    "AuditModuleBase",
]
