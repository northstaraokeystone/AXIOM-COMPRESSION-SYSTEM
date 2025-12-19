"""CLI command modules organized by domain.

Each file handles one command group:
- fractal: Fractal recursion commands
- cfd: CFD simulation commands
- agi: AGI alignment commands
- multiplanet: Multi-planet expansion commands
- mars: Mars-specific commands
- zk: Zero-knowledge proof commands
"""

__all__ = []

RECEIPT_SCHEMA = {
    "module": "cli.commands",
    "receipt_types": ["cli_command"],
    "version": "1.0.0",
}
