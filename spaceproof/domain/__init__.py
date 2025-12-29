"""SpaceProof Domain - Pluggable Domain Generators

D20 Production Evolution: Domain generators for stakeholder-specific use cases.

Available domains:
- galaxy: Galaxy rotation curve generation (cosmos physics)
- colony: Mars colony state simulation
- telemetry: Fleet telemetry generation (Tesla/Starlink/SpaceX)

Defense Expansion Domains:
- orbital_compute: Starcloud orbital compute provenance
- constellation_ops: Starlink maneuver audit chains
- autonomous_decision: Defense DOD 3000.09 compliance
- firmware_integrity: Supply chain verification

Usage:
    from spaceproof.domain import galaxy, colony, telemetry

    # Generate galaxy rotation curves
    curves = galaxy.generate("dark_matter", params)

    # Generate colony states
    states = colony.generate(config, days=30)

    # Generate telemetry streams
    stream = telemetry.generate("tesla", params)

    # Defense expansion
    from spaceproof.domain import orbital_compute, constellation_ops
    from spaceproof.domain import autonomous_decision, firmware_integrity
"""

from . import galaxy
from . import colony
from . import telemetry

# Defense expansion domains
from . import orbital_compute
from . import constellation_ops
from . import autonomous_decision
from . import firmware_integrity

__all__ = [
    # Original domains
    "galaxy",
    "colony",
    "telemetry",
    # Defense expansion domains
    "orbital_compute",
    "constellation_ops",
    "autonomous_decision",
    "firmware_integrity",
]
