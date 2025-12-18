"""Atacama validation CLI commands.

Commands for Atacama Mars dust analog validation.
"""

import json

from src.atacama_validation import (
    run_atacama_validation,
    get_atacama_info,
)


def cmd_atacama_info():
    """Show Atacama configuration."""
    info = get_atacama_info()
    print(json.dumps(info, indent=2))


def cmd_atacama_validate(simulate: bool = True):
    """Run Atacama validation."""
    result = run_atacama_validation(simulate)
    print(json.dumps(result, indent=2))
