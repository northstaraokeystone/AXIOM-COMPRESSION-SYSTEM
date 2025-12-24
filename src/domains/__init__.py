"""AXIOM Domains - Pluggable Domain Generators

D20 Production Evolution: Domain generators for stakeholder-specific use cases.

Available domains:
- galaxy: Galaxy rotation curve generation (cosmos physics)
- colony: Mars colony state simulation
- telemetry: Fleet telemetry generation (Tesla/Starlink/SpaceX)

Usage:
    from src.domains import galaxy, colony, telemetry

    # Generate galaxy rotation curves
    curves = galaxy.generate("dark_matter", params)

    # Generate colony states
    states = colony.generate(config, days=30)

    # Generate telemetry streams
    stream = telemetry.generate("tesla", params)
"""

from . import galaxy
from . import colony
from . import telemetry

__all__ = ["galaxy", "colony", "telemetry"]
