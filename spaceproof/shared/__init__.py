"""shared - Domain-Agnostic Verification Infrastructure.

Provides universal entropy-based verification patterns used across:
- Aerospace (hardware supply chain)
- Food (adulteration detection)
- Medical (counterfeit device/drug detection)

The key insight: entropy detection works across domains because
genuine products exhibit natural randomness while counterfeits
show abnormal uniformity or material deviations.
"""

from .verification_engine import (
    VerificationEngine,
    VerificationResult,
    EntropyAnalysis,
    BaselineConfig,
)

__all__ = [
    "VerificationEngine",
    "VerificationResult",
    "EntropyAnalysis",
    "BaselineConfig",
]
