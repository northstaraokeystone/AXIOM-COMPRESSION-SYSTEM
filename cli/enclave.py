"""Secure enclave CLI commands."""

import json


def cmd_enclave_info():
    """Show secure enclave configuration."""
    from src.secure_enclave_audit import get_enclave_info

    info = get_enclave_info()
    print(json.dumps(info, indent=2))


def cmd_enclave_init(memory_mb: int = 128):
    """Initialize secure enclave."""
    from src.secure_enclave_audit import init_enclave

    result = init_enclave(memory_mb)
    print(json.dumps(result, indent=2))


def cmd_enclave_audit(iterations: int = 100):
    """Run full secure enclave audit."""
    from src.secure_enclave_audit import run_enclave_audit

    result = run_enclave_audit(iterations=iterations)
    print(json.dumps(result, indent=2))


def cmd_enclave_btb(iterations: int = 100):
    """Test BTB injection defense."""
    from src.secure_enclave_audit import test_btb_injection

    result = test_btb_injection(iterations)
    print(json.dumps(result, indent=2))


def cmd_enclave_pht(iterations: int = 100):
    """Test PHT poisoning defense."""
    from src.secure_enclave_audit import test_pht_poisoning

    result = test_pht_poisoning(iterations)
    print(json.dumps(result, indent=2))


def cmd_enclave_rsb(iterations: int = 100):
    """Test RSB stuffing defense."""
    from src.secure_enclave_audit import test_rsb_stuffing

    result = test_rsb_stuffing(iterations)
    print(json.dumps(result, indent=2))


def cmd_enclave_overhead():
    """Measure enclave defense overhead."""
    from src.secure_enclave_audit import measure_enclave_overhead

    result = measure_enclave_overhead()
    print(json.dumps(result, indent=2))
