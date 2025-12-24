"""Variable gravity autonomy for multi-planet operations.

Implements gravity-aware timing and parameter adjustments for different
planetary environments (Mars 0.38g, Venus 0.9g, Mercury 0.38g, Jovian moons).

Receipt Types:
    - gravity_config_receipt: Configuration loaded
    - gravity_adjustment_receipt: Adjustment applied
    - gravity_timing_receipt: Timing adjusted
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from src.core import TENANT_ID, dual_hash, emit_receipt

# Gravity constants for celestial bodies
GRAVITY_EARTH = 1.0
GRAVITY_MARS = 0.38
GRAVITY_VENUS = 0.9
GRAVITY_MERCURY = 0.38
GRAVITY_MOON = 0.166
GRAVITY_TITAN = 0.14
GRAVITY_EUROPA = 0.134
GRAVITY_GANYMEDE = 0.146
GRAVITY_CALLISTO = 0.126
GRAVITY_IO = 0.183

# Backward-compatibility aliases (tests use these names)
EARTH_GRAVITY_G = GRAVITY_EARTH
MARS_GRAVITY_G = GRAVITY_MARS

# Gravity adjustment enabled
GRAVITY_ADAPTIVE_ENABLED = True
GRAVITY_ADJUSTMENT_FACTOR = True

# Planet gravity map
PLANET_GRAVITY_MAP = {
    "earth": GRAVITY_EARTH,
    "mars": GRAVITY_MARS,
    "venus": GRAVITY_VENUS,
    "mercury": GRAVITY_MERCURY,
    "moon": GRAVITY_MOON,
    "titan": GRAVITY_TITAN,
    "europa": GRAVITY_EUROPA,
    "ganymede": GRAVITY_GANYMEDE,
    "callisto": GRAVITY_CALLISTO,
    "io": GRAVITY_IO,
}


def load_gravity_config() -> Dict[str, Any]:
    """Load gravity configuration from federation spec.

    Returns:
        dict: Gravity configuration with planet profiles.

    Receipt:
        gravity_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "federation_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    planet_profiles = spec.get("planet_profiles", {})

    config = {
        "adaptive_enabled": GRAVITY_ADAPTIVE_ENABLED,
        "adjustment_factor": GRAVITY_ADJUSTMENT_FACTOR,
        "planet_gravity": {
            name: profile.get("gravity_g", PLANET_GRAVITY_MAP.get(name, 1.0))
            for name, profile in planet_profiles.items()
        },
        "default_gravity": PLANET_GRAVITY_MAP,
    }

    emit_receipt(
        "gravity_config_receipt",
        {
            "receipt_type": "gravity_config_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "adaptive_enabled": config["adaptive_enabled"],
            "planets_configured": list(config["planet_gravity"].keys()),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )
    return config


def get_planet_gravity(planet: str) -> float:
    """Get gravity value for a planet.

    Args:
        planet: Planet or moon name.

    Returns:
        float: Gravity in Earth g units.
    """
    config = load_gravity_config()

    # Check federation config first
    if planet.lower() in config["planet_gravity"]:
        return config["planet_gravity"][planet.lower()]

    # Fall back to default map
    return PLANET_GRAVITY_MAP.get(planet.lower(), GRAVITY_EARTH)


def adjust_for_gravity(gravity_g: float) -> Dict[str, Any]:
    """Adjust operational parameters for gravity.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        dict: Adjusted parameters.

    Receipt:
        gravity_adjustment_receipt
    """
    # Calculate adjustment factors
    # Lower gravity = longer physical processes = adjusted timing
    gravity_ratio = gravity_g / GRAVITY_EARTH

    # Timing adjustment: lower gravity = slower physical settling
    timing_factor = 1.0 / (gravity_ratio**0.5) if gravity_ratio > 0 else 1.0

    # Consensus timing adjustment
    consensus_multiplier = timing_factor

    # Packet timing adjustment
    packet_multiplier = min(2.0, timing_factor)  # Cap at 2x

    # Autonomy threshold adjustment: lower gravity = higher autonomy needed
    autonomy_adjustment = max(0.0, (1.0 - gravity_ratio) * 0.001)

    result = {
        "adjusted": True,
        "gravity_g": gravity_g,
        "gravity_ratio": gravity_ratio,
        "timing_factor": timing_factor,
        "consensus_multiplier": consensus_multiplier,
        "packet_multiplier": packet_multiplier,
        "autonomy_adjustment": autonomy_adjustment,
    }

    emit_receipt(
        "gravity_adjustment_receipt",
        {
            "receipt_type": "gravity_adjustment_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "adjusted": True,
            "gravity_g": gravity_g,
            "timing_factor": timing_factor,
            "consensus_multiplier": consensus_multiplier,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def calculate_timing_adjustment(gravity_g: float) -> float:
    """Calculate timing adjustment factor for gravity.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        float: Timing adjustment factor.
    """
    if gravity_g <= 0:
        return 1.0

    gravity_ratio = gravity_g / GRAVITY_EARTH
    return 1.0 / (gravity_ratio**0.5)


def adjust_consensus_timing(gravity_g: float) -> Dict[str, Any]:
    """Adjust consensus timing for gravity.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        dict: Adjusted consensus timing.

    Receipt:
        gravity_timing_receipt
    """
    timing_factor = calculate_timing_adjustment(gravity_g)

    # Base consensus timing values
    base_heartbeat_ms = 60000  # 1 minute
    base_election_timeout_ms = 300000  # 5 minutes

    adjusted_heartbeat = int(base_heartbeat_ms * timing_factor)
    adjusted_election_timeout = int(base_election_timeout_ms * timing_factor)

    result = {
        "gravity_g": gravity_g,
        "timing_factor": timing_factor,
        "base_heartbeat_ms": base_heartbeat_ms,
        "adjusted_heartbeat_ms": adjusted_heartbeat,
        "base_election_timeout_ms": base_election_timeout_ms,
        "adjusted_election_timeout_ms": adjusted_election_timeout,
    }

    emit_receipt(
        "gravity_timing_receipt",
        {
            "receipt_type": "gravity_timing_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "timing_type": "consensus",
            "gravity_g": gravity_g,
            "timing_factor": timing_factor,
            "adjusted_heartbeat_ms": adjusted_heartbeat,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def adjust_packet_timing(gravity_g: float) -> Dict[str, Any]:
    """Adjust packet timing for gravity.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        dict: Adjusted packet timing.

    Receipt:
        gravity_timing_receipt
    """
    timing_factor = calculate_timing_adjustment(gravity_g)

    # Cap packet timing at 2x to prevent excessive delays
    packet_factor = min(2.0, timing_factor)

    # Base packet timing values
    base_timeout_ms = 5000  # 5 seconds
    base_retry_delay_ms = 1000  # 1 second

    adjusted_timeout = int(base_timeout_ms * packet_factor)
    adjusted_retry_delay = int(base_retry_delay_ms * packet_factor)

    result = {
        "gravity_g": gravity_g,
        "timing_factor": timing_factor,
        "packet_factor": packet_factor,
        "base_timeout_ms": base_timeout_ms,
        "adjusted_timeout_ms": adjusted_timeout,
        "base_retry_delay_ms": base_retry_delay_ms,
        "adjusted_retry_delay_ms": adjusted_retry_delay,
    }

    emit_receipt(
        "gravity_timing_receipt",
        {
            "receipt_type": "gravity_timing_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "timing_type": "packet",
            "gravity_g": gravity_g,
            "packet_factor": packet_factor,
            "adjusted_timeout_ms": adjusted_timeout,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def adjust_autonomy_threshold(gravity_g: float) -> float:
    """Adjust autonomy threshold for gravity.

    Lower gravity environments require higher autonomy due to
    longer physical process times.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        float: Adjusted autonomy threshold (0-1).
    """
    base_autonomy = 0.999

    # Calculate adjustment: lower gravity = higher autonomy needed
    gravity_ratio = gravity_g / GRAVITY_EARTH if gravity_g > 0 else 1.0
    adjustment = max(0.0, (1.0 - gravity_ratio) * 0.001)

    return min(1.0, base_autonomy + adjustment)


def validate_gravity_adjustment(planet: str) -> Dict[str, Any]:
    """Validate gravity adjustment for a planet.

    Args:
        planet: Planet name.

    Returns:
        dict: Validation result.

    Receipt:
        gravity_adjustment_receipt
    """
    gravity_g = get_planet_gravity(planet)
    adjustment = adjust_for_gravity(gravity_g)

    # Validate adjustment is within reasonable bounds
    valid = (
        0.1 <= adjustment["timing_factor"] <= 10.0
        and 0.5 <= adjustment["consensus_multiplier"] <= 5.0
        and 1.0 <= adjustment["packet_multiplier"] <= 2.0
    )

    result = {
        "valid": valid,
        "planet": planet,
        "gravity_g": gravity_g,
        "adjustment": adjustment,
    }

    emit_receipt(
        "gravity_adjustment_receipt",
        {
            "receipt_type": "gravity_adjustment_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "validation",
            "valid": valid,
            "planet": planet,
            "gravity_g": gravity_g,
            "payload_hash": dual_hash(json.dumps(result, default=str)),
        },
    )
    return result


def get_gravity_status() -> Dict[str, Any]:
    """Get current gravity adaptation status.

    Returns:
        dict: Gravity status.
    """
    config = load_gravity_config()

    planets_adjusted = {}
    for planet in config["planet_gravity"]:
        gravity_g = config["planet_gravity"][planet]
        timing_factor = calculate_timing_adjustment(gravity_g)
        autonomy = adjust_autonomy_threshold(gravity_g)
        planets_adjusted[planet] = {
            "gravity_g": gravity_g,
            "timing_factor": timing_factor,
            "autonomy_threshold": autonomy,
        }

    return {
        "adaptive_enabled": config["adaptive_enabled"],
        "planets_configured": len(config["planet_gravity"]),
        "planets_adjusted": planets_adjusted,
        "default_gravity_map": PLANET_GRAVITY_MAP,
    }


def get_all_planet_adjustments() -> Dict[str, Dict[str, Any]]:
    """Get gravity adjustments for all configured planets.

    Returns:
        dict: Adjustments for each planet.
    """
    adjustments = {}
    for planet, gravity_g in PLANET_GRAVITY_MAP.items():
        adjustments[planet] = adjust_for_gravity(gravity_g)
    return adjustments
