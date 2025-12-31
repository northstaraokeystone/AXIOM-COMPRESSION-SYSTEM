"""medical - Medical Device/Drug Counterfeit Detection Domain.

Detects:
- Counterfeit GLP-1 pens (Ozempic, Wegovy) - $37-162B annual
- Counterfeit Botox vials
- Fake cancer drugs (no API)

Compliance:
- FDA 21 CFR Part 820 (QSR)
- ISO 13485:2016

Detection target: ≥99.9% recall (life-critical)
Critical miss tolerance: ZERO

Source: Grok Research - Medical counterfeit detection via entropy analysis
"""

from .glp1 import verify_glp1_pen, compute_fill_entropy, validate_lot_format
from .botox import verify_botox_vial, compute_surface_entropy, compute_solution_entropy
from .cancer_drugs import verify_cancer_drug, compute_api_distribution_entropy, detect_no_api
from .entropy import (
    structural_entropy,
    surface_topography_entropy,
    dimensional_entropy,
    api_distribution_entropy,
)

__all__ = [
    # GLP-1 verification
    "verify_glp1_pen",
    "compute_fill_entropy",
    "validate_lot_format",
    # Botox verification
    "verify_botox_vial",
    "compute_surface_entropy",
    "compute_solution_entropy",
    # Cancer drug verification
    "verify_cancer_drug",
    "compute_api_distribution_entropy",
    "detect_no_api",
    # Entropy calculators
    "structural_entropy",
    "surface_topography_entropy",
    "dimensional_entropy",
    "api_distribution_entropy",
]


class MedicalVerificationError(Exception):
    """Raised when medical verification fails unexpectedly."""

    pass


class CriticalCounterfeitDetected(Exception):
    """Raised when a counterfeit is detected for a life-critical drug.

    This exception carries the full receipt for audit trail.
    """

    def __init__(self, message: str, receipt: dict):
        super().__init__(message)
        self.receipt = receipt
