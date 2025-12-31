"""food - Food Safety Verification Domain.

Detects:
- Olive oil adulteration ($40B annual fraud)
- Honey syrup mixing ($10-40B annual fraud)
- Seafood species substitution

Compliance:
- FSMA Section 204 traceability
- 21 CFR Part 11 electronic records

Detection target: ≥99.9% recall, <1% false positive rate

Source: Grok Research - Food fraud detection via entropy analysis
"""

from .olive_oil import verify_olive_oil, compute_spectral_entropy
from .honey import verify_honey, compute_texture_entropy, compute_pollen_entropy
from .seafood import verify_seafood, compute_tissue_entropy
from .entropy import (
    spectral_entropy,
    texture_entropy,
    gradient_entropy,
    pollen_diversity_entropy,
)

__all__ = [
    # Verification functions
    "verify_olive_oil",
    "verify_honey",
    "verify_seafood",
    # Entropy calculators
    "compute_spectral_entropy",
    "compute_texture_entropy",
    "compute_pollen_entropy",
    "compute_tissue_entropy",
    "spectral_entropy",
    "texture_entropy",
    "gradient_entropy",
    "pollen_diversity_entropy",
]


class FoodVerificationError(Exception):
    """Raised when food verification fails unexpectedly."""

    pass
