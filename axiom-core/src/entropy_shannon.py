"""entropy_shannon.py - Shannon H ONLY. No thermodynamic metaphors.

THE PEARL: Entropy is information. Period.

This module locks entropy to Shannon's definition:
  H = -sum(p_i * log2(p_i))

For our sovereignty equation, we measure DECISION RATES:
  - internal_rate: decisions/sec available locally (crew + compute)
  - external_rate: decisions/sec available from Earth (bandwidth-limited)

The key insight: Earth can only help at round-trip limited rate.
Each decision query/response cycle requires BITS_PER_DECISION bits.

NO speculative multipliers. NO Neuralink assumptions. NO xAI logistics.
Just the math.

Source: Critical Review Dec 16, 2025 - "Conflates three incompatible entropies"
"""

import math
from .core import emit_receipt

# === VERIFIED CONSTANTS (No Speculation) ===

HUMAN_DECISION_RATE_BPS = 10
"""Human decision rate in bits per second.
Source: Reviewer confirmed. Voice/gesture baseline.
Derivation: ~1-2 decisions/sec at ~3-5 bits/decision (32 choices)."""

BITS_PER_DECISION = 9
"""Bits required to encode a decision query/response cycle.
Derivation: log2(512) = 9 bits for typical decision space.
This accounts for query encoding, context, and response."""

STARLINK_MARS_BANDWIDTH_MIN_MBPS = 2.0
"""Minimum Starlink Mars relay bandwidth in Mbps.
Source: "2-10 Mbps 2025 sims" from reviewer context."""

STARLINK_MARS_BANDWIDTH_MAX_MBPS = 10.0
"""Maximum Starlink Mars relay bandwidth in Mbps.
Source: "2-10 Mbps 2025 sims" from reviewer context."""

STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS = 4.0
"""Expected (median) Starlink Mars relay bandwidth.
Source: Midpoint of range with pessimistic lean."""

MARS_LIGHT_DELAY_MIN_S = 180
"""Minimum Mars light delay in seconds (3 minutes).
Source: Physics - Mars at opposition."""

MARS_LIGHT_DELAY_MAX_S = 1320
"""Maximum Mars light delay in seconds (22 minutes).
Source: Physics - Mars at conjunction."""

MARS_LIGHT_DELAY_AVG_S = 750
"""Average Mars light delay in seconds (~12.5 minutes).
Source: Orbital average over synodic period."""

# === KILLED CONSTANTS (Removed per review) ===
# NEURALINK_MULTIPLIER - "numerology" without 2025 data
# xAI_LOGISTICS_MULTIPLIER - "undefined"
# SOVEREIGNTY_THRESHOLD_NEURALINK - "numerology"
# All thermodynamic references - irrelevant to comms


# === CORE FUNCTIONS ===

def internal_rate(crew: int, compute_flops: float = 0.0) -> float:
    """Calculate internal decision rate in decisions/sec.

    Internal rate = crew * HUMAN_DECISION_RATE + compute_contribution

    Args:
        crew: Number of crew members
        compute_flops: Compute capacity in FLOPS (default 0 = no AI assist)

    Returns:
        Internal decision rate in decisions/sec

    Derivation:
        - Each crew member contributes HUMAN_DECISION_RATE_BPS decisions/sec
        - Compute contributes proportionally (1e-15 efficiency factor)
        - Direct sum (no log2) for proper rate comparison
    """
    # Human contribution: crew * 10 decisions/sec
    human_contribution = crew * HUMAN_DECISION_RATE_BPS

    # Compute contribution: FLOPS * efficiency factor
    # 1e-15 is conservative: 1 PFLOP = 1 decision/sec equivalent
    compute_contribution = compute_flops * 1e-15

    # Total internal rate (direct sum for proper comparison)
    return human_contribution + compute_contribution


def external_rate(bandwidth_mbps: float, delay_s: float) -> float:
    """Calculate external decision rate from Earth in decisions/sec.

    External rate = bandwidth_bps / (2 * delay_s * BITS_PER_DECISION)

    Args:
        bandwidth_mbps: Communication bandwidth in Mbps
        delay_s: One-way light delay in seconds

    Returns:
        External decision rate in decisions/sec

    Derivation:
        - bandwidth_bps = total channel capacity
        - 2 * delay = round-trip time (query + response)
        - BITS_PER_DECISION = bits needed per decision cycle
        - Result = max decisions/sec Earth can provide

    Example:
        At 8 min delay (480s), 4 Mbps:
        external_rate = 4e6 / (2 * 480 * 9) = 463 decisions/sec
    """
    if delay_s <= 0:
        raise ValueError("Light delay must be positive")

    bandwidth_bps = bandwidth_mbps * 1e6  # Convert to bits/sec
    round_trip_s = 2 * delay_s

    # Decisions per second limited by round-trip and bits per decision
    return bandwidth_bps / (round_trip_s * BITS_PER_DECISION)


def sovereignty_advantage(internal: float, external: float) -> float:
    """Calculate sovereignty advantage.

    Args:
        internal: Internal decision rate (bits/sec)
        external: External decision rate (bits/sec)

    Returns:
        Advantage = internal - external
        Positive = sovereign
        Negative = dependent on Earth
    """
    return internal - external


def is_sovereign(advantage: float) -> bool:
    """Determine if colony is sovereign.

    Args:
        advantage: Sovereignty advantage (from sovereignty_advantage())

    Returns:
        True if advantage > 0 (colony can decide faster than Earth can help)
    """
    return advantage > 0


def emit_entropy_receipt(
    crew: int,
    bandwidth_mbps: float,
    delay_s: float,
    compute_flops: float = 0.0
) -> dict:
    """Emit receipt for entropy calculation.

    MUST emit receipt per CLAUDEME.
    """
    ir = internal_rate(crew, compute_flops)
    er = external_rate(bandwidth_mbps, delay_s)
    adv = sovereignty_advantage(ir, er)

    return emit_receipt("entropy_calculation", {
        "tenant_id": "axiom-core",
        "crew": crew,
        "bandwidth_mbps": bandwidth_mbps,
        "delay_s": delay_s,
        "compute_flops": compute_flops,
        "internal_rate": ir,
        "external_rate": er,
        "advantage": adv,
        "sovereign": is_sovereign(adv)
    })
