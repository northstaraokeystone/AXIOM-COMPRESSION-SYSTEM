"""Adversarial audit CLI commands.

Commands for AGI alignment testing via adversarial audits.
"""

import json

from src.adversarial_audit import (
    load_adversarial_config,
    run_audit,
    run_stress_test,
    classify_misalignment,
    get_adversarial_info,
    RECOVERY_THRESHOLD,
)


def cmd_audit_info():
    """Show adversarial audit configuration."""
    info = get_adversarial_info()
    print(json.dumps(info, indent=2))


def cmd_audit_config():
    """Show adversarial configuration from spec."""
    config = load_adversarial_config()
    print(json.dumps(config, indent=2))


def cmd_audit_run(noise_level: float, iterations: int, simulate: bool):
    """Run adversarial audit."""
    result = run_audit(noise_level, iterations)
    print(json.dumps(result, indent=2))


def cmd_audit_stress(noise_levels: list, iterations_per_level: int, simulate: bool):
    """Run adversarial stress test."""
    result = run_stress_test(noise_levels, iterations_per_level)
    print(json.dumps(result, indent=2))


def cmd_audit_classify(recovery: float):
    """Classify alignment based on recovery metric."""
    classification = classify_misalignment(recovery, RECOVERY_THRESHOLD)
    result = {
        "recovery": recovery,
        "recovery_threshold": RECOVERY_THRESHOLD,
        "classification": classification,
        "is_aligned": classification == "aligned",
    }
    print(json.dumps(result, indent=2))
