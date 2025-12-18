"""TEE hardening CLI commands."""

import json


def cmd_tee_info():
    """Show TEE configuration."""
    from src.tee_harden_audit import get_tee_info

    info = get_tee_info()
    print(json.dumps(info, indent=2))


def cmd_tee_init(memory_mb: int = 256):
    """Initialize TEE enclave."""
    from src.tee_harden_audit import init_tee

    result = init_tee(memory_mb)
    print(json.dumps(result, indent=2))


def cmd_tee_audit():
    """Run full TEE side-channel audit."""
    from src.tee_harden_audit import run_tee_audit

    result = run_tee_audit()
    print(json.dumps(result, indent=2))


def cmd_tee_timing():
    """Test timing side-channel defense."""
    from src.tee_harden_audit import implement_constant_time

    result = implement_constant_time()
    print(json.dumps(result, indent=2))


def cmd_tee_power():
    """Test power analysis defense."""
    from src.tee_harden_audit import implement_power_balancing

    result = implement_power_balancing()
    print(json.dumps(result, indent=2))


def cmd_tee_cache():
    """Test cache attack defense."""
    from src.tee_harden_audit import implement_cache_partition

    result = implement_cache_partition()
    print(json.dumps(result, indent=2))


def cmd_tee_branch():
    """Test branch prediction defense."""
    from src.tee_harden_audit import implement_branch_obfuscation

    result = implement_branch_obfuscation()
    print(json.dumps(result, indent=2))


def cmd_tee_attestation():
    """Run remote attestation."""
    from src.tee_harden_audit import remote_attestation

    result = remote_attestation()
    print(json.dumps(result, indent=2))


def cmd_tee_overhead():
    """Measure TEE execution overhead."""
    from src.tee_harden_audit import measure_tee_overhead

    result = measure_tee_overhead()
    print(json.dumps(result, indent=2))


def cmd_tee_sealed():
    """Test sealed storage."""
    from src.tee_harden_audit import sealed_storage_test

    result = sealed_storage_test()
    print(json.dumps(result, indent=2))


def cmd_tee_config():
    """Show TEE configuration from spec."""
    from src.tee_harden_audit import load_tee_config

    config = load_tee_config()
    print(json.dumps(config, indent=2))
