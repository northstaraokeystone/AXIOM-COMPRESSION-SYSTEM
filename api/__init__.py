"""api - REST API for SpaceProof Multi-Domain Verification.

Exposes verification endpoints for:
- Aerospace (hardware counterfeit detection)
- Food (adulteration detection)
- Medical (counterfeit device/drug detection)

Target: Unblock Jay Lewis test bench integration (2-4 hours work).
"""

from .server import app

__all__ = ["app"]
