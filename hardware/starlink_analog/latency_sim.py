"""Latency simulation for Starlink analog testing.

Provides utilities for simulating various latency scenarios including
interstellar distances, Mars communications, and relay chain delays.

Receipt Types:
    - latency_sim_hop_receipt: Single hop latency
    - latency_sim_chain_receipt: Relay chain latency
    - latency_sim_mars_receipt: Mars communication latency
"""

import json
import random
from datetime import datetime
from typing import Dict, Any

from src.core import TENANT_ID, dual_hash, emit_receipt

# Physical constants
SPEED_OF_LIGHT_KM_S = 299792.458
LIGHT_YEAR_KM = 9.461e12
AU_KM = 1.496e8

# Mars orbital parameters
MARS_MIN_DISTANCE_AU = 0.38  # Opposition
MARS_MAX_DISTANCE_AU = 2.67  # Conjunction
MARS_AVG_DISTANCE_AU = 1.52

# Default jitter percentage
DEFAULT_JITTER_PCT = 0.05


def simulate_hop_latency(distance_km: float) -> float:
    """Calculate single hop latency based on distance.

    Args:
        distance_km: Distance in kilometers.

    Returns:
        float: Latency in milliseconds.

    Receipt:
        latency_sim_hop_receipt
    """
    latency_s = distance_km / SPEED_OF_LIGHT_KM_S
    latency_ms = latency_s * 1000

    emit_receipt(
        "latency_sim_hop_receipt",
        {
            "receipt_type": "latency_sim_hop_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "distance_km": distance_km,
            "latency_ms": latency_ms,
            "payload_hash": dual_hash(
                json.dumps({"distance_km": distance_km, "latency_ms": latency_ms})
            ),
        },
    )
    return latency_ms


def simulate_chain_latency(hops: int, per_hop_ms: float) -> float:
    """Calculate total latency for a relay chain.

    Args:
        hops: Number of relay hops.
        per_hop_ms: Latency per hop in milliseconds.

    Returns:
        float: Total latency in milliseconds.

    Receipt:
        latency_sim_chain_receipt
    """
    total_ms = hops * per_hop_ms

    emit_receipt(
        "latency_sim_chain_receipt",
        {
            "receipt_type": "latency_sim_chain_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "hops": hops,
            "per_hop_ms": per_hop_ms,
            "total_ms": total_ms,
            "payload_hash": dual_hash(
                json.dumps(
                    {"hops": hops, "per_hop_ms": per_hop_ms, "total_ms": total_ms}
                )
            ),
        },
    )
    return total_ms


def add_jitter(base_ms: float, jitter_pct: float = DEFAULT_JITTER_PCT) -> float:
    """Add realistic jitter to a latency value.

    Args:
        base_ms: Base latency in milliseconds.
        jitter_pct: Jitter as percentage of base (0-1).

    Returns:
        float: Latency with jitter applied.
    """
    jitter = base_ms * jitter_pct * (2 * random.random() - 1)
    return max(0.0, base_ms + jitter)


def simulate_mars_latency(phase: str = "average") -> float:
    """Simulate Mars communication latency based on orbital phase.

    Args:
        phase: Orbital phase - "opposition", "conjunction", or "average".

    Returns:
        float: One-way latency in minutes.

    Receipt:
        latency_sim_mars_receipt
    """
    if phase == "opposition":
        distance_au = MARS_MIN_DISTANCE_AU
        latency_min = 3.0  # ~3 minutes at opposition
    elif phase == "conjunction":
        distance_au = MARS_MAX_DISTANCE_AU
        latency_min = 22.0  # ~22 minutes at conjunction
    else:  # average
        distance_au = MARS_AVG_DISTANCE_AU
        latency_min = 12.5  # Average latency

    # More precise calculation
    distance_km = distance_au * AU_KM
    latency_s = distance_km / SPEED_OF_LIGHT_KM_S
    latency_min_calc = latency_s / 60.0

    emit_receipt(
        "latency_sim_mars_receipt",
        {
            "receipt_type": "latency_sim_mars_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "phase": phase,
            "distance_au": distance_au,
            "latency_min_simplified": latency_min,
            "latency_min_calculated": latency_min_calc,
            "round_trip_min": latency_min * 2,
            "payload_hash": dual_hash(
                json.dumps({"phase": phase, "latency_min": latency_min})
            ),
        },
    )
    return latency_min


def simulate_proxima_latency() -> float:
    """Simulate Proxima Centauri communication latency.

    Returns:
        float: One-way latency in years.

    Receipt:
        latency_sim_proxima_receipt
    """
    distance_ly = 4.24
    latency_years = distance_ly  # Light travels 1 ly in 1 year

    emit_receipt(
        "latency_sim_proxima_receipt",
        {
            "receipt_type": "latency_sim_proxima_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "distance_ly": distance_ly,
            "latency_years": latency_years,
            "latency_days": latency_years * 365.25,
            "round_trip_years": latency_years * 2,
            "payload_hash": dual_hash(
                json.dumps({"distance_ly": distance_ly, "latency_years": latency_years})
            ),
        },
    )
    return latency_years


def calculate_latency_multiplier(
    earth_latency_ms: float, target_latency_ms: float
) -> float:
    """Calculate latency multiplier for scaling.

    Args:
        earth_latency_ms: Earth-based network latency in ms.
        target_latency_ms: Target interstellar latency in ms.

    Returns:
        float: Multiplier to scale Earth latency to target.
    """
    return target_latency_ms / max(1.0, earth_latency_ms)


def get_latency_config() -> Dict[str, Any]:
    """Get default latency configuration.

    Returns:
        dict: Latency configuration parameters.
    """
    return {
        "speed_of_light_km_s": SPEED_OF_LIGHT_KM_S,
        "light_year_km": LIGHT_YEAR_KM,
        "au_km": AU_KM,
        "mars_min_distance_au": MARS_MIN_DISTANCE_AU,
        "mars_max_distance_au": MARS_MAX_DISTANCE_AU,
        "default_jitter_pct": DEFAULT_JITTER_PCT,
        "proxima_distance_ly": 4.24,
        "proxima_latency_multiplier": 6300,  # vs Earth networks
    }
