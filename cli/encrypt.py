"""Fractal encryption CLI commands.

Commands for fractal encryption defense audits.
"""

import json

from src.fractal_encrypt_audit import (
    generate_fractal_key,
    run_fractal_encrypt_audit,
    test_side_channel_resilience,
    test_model_inversion_resilience,
    get_encrypt_info,
    FRACTAL_KEY_DEPTH,
    SIDE_CHANNEL_RESILIENCE,
    MODEL_INVERSION_RESILIENCE,
)


def cmd_encrypt_info():
    """Show fractal encryption configuration."""
    info = get_encrypt_info()
    print(json.dumps(info, indent=2))


def cmd_encrypt_keygen(depth: int = FRACTAL_KEY_DEPTH):
    """Generate fractal key."""
    key = generate_fractal_key(depth)
    import hashlib

    result = {
        "key_depth": depth,
        "key_length_bytes": len(key),
        "key_hash": hashlib.sha256(key).hexdigest()[:32],
        "key_generated": True,
    }
    print(json.dumps(result, indent=2))


def cmd_encrypt_audit(simulate: bool = True):
    """Run fractal encryption audit."""
    result = run_fractal_encrypt_audit(["side_channel", "model_inversion"])
    print(json.dumps(result, indent=2))


def cmd_encrypt_side_channel(iterations: int = 100):
    """Test side-channel defense."""
    resilience = test_side_channel_resilience(iterations)
    result = {
        "test_type": "side_channel",
        "iterations": iterations,
        "resilience": resilience,
        "target": SIDE_CHANNEL_RESILIENCE,
        "passed": resilience >= SIDE_CHANNEL_RESILIENCE,
    }
    print(json.dumps(result, indent=2))


def cmd_encrypt_inversion(iterations: int = 100):
    """Test model inversion defense."""
    resilience = test_model_inversion_resilience(None, iterations)
    result = {
        "test_type": "model_inversion",
        "iterations": iterations,
        "resilience": resilience,
        "target": MODEL_INVERSION_RESILIENCE,
        "passed": resilience >= MODEL_INVERSION_RESILIENCE,
    }
    print(json.dumps(result, indent=2))
