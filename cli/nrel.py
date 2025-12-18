"""NREL perovskite validation CLI commands.

Commands for NREL perovskite efficiency validation.
"""

import json

from src.nrel_validation import (
    load_nrel_config,
    validate_efficiency,
    project_degradation,
    compare_to_moxie,
    get_nrel_info,
    NREL_LAB_EFFICIENCY,
    MOXIE_EFFICIENCY,
)


def cmd_nrel_info():
    """Show NREL validation configuration."""
    info = get_nrel_info()
    print(json.dumps(info, indent=2))


def cmd_nrel_config():
    """Show NREL configuration from spec."""
    config = load_nrel_config()
    print(json.dumps(config, indent=2))


def cmd_nrel_validate(efficiency: float, simulate: bool):
    """Validate efficiency against NREL lab data."""
    result = validate_efficiency(efficiency)
    print(json.dumps(result, indent=2))


def cmd_nrel_project(years: int, initial_eff: float, simulate: bool):
    """Project efficiency degradation over time."""
    result = project_degradation(years, initial_eff if initial_eff > 0 else None)
    print(json.dumps(result, indent=2))


def cmd_nrel_compare(nrel_eff: float, moxie_eff: float, simulate: bool):
    """Compare perovskite to MOXIE efficiency."""
    result = compare_to_moxie(
        nrel_eff if nrel_eff > 0 else NREL_LAB_EFFICIENCY,
        moxie_eff if moxie_eff > 0 else MOXIE_EFFICIENCY,
    )
    print(json.dumps(result, indent=2))
