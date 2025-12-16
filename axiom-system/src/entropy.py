"""AXIOM-SYSTEM v2 Entropy Module - Compression rates with Neuralink constants.

Status: UPDATED from v3.1
Purpose: Add Neuralink constants, update MDL beta, system entropy functions

NEW CONSTANTS (from Grok):
  - NEURALINK_MULTIPLIER = 1e5 (effective 100,000x baseline)
  - HUMAN_BASELINE_BPS = 10 (10 bps/person baseline)
  - MDL_BETA = 0.09 (tuned for 96% compression)
  - KESSLER_THRESHOLD = 0.73 (73% ESA 2025)
  - DEBRIS_COUNT_2025 = 100000 (10^5 objects >10cm)
  - CME_PROBABILITY_PER_DAY = 0.02 (NOAA Cycle 25)
  - MOON_RELAY_BOOST = 0.40 (+40% Mars external rate)
  - QUEUE_DELAY_SOLS = 7 (7 sols avg competing missions)
  - SOVEREIGNTY_THRESHOLD_NEURALINK = 5 (5 crew with Neuralink)
  - SOVEREIGNTY_THRESHOLD_BASELINE = 25 (baseline without Neuralink)
"""

import math
import numpy as np

from .core import emit_receipt, dual_hash

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFIED CONSTANTS (NASA/Physics) - KEPT from v3.1
# ═══════════════════════════════════════════════════════════════════════════════
HUMAN_METABOLIC_W = 100                 # Physiology
MOXIE_O2_G_PER_HR = 5.5                # NASA Perseverance
ISS_WATER_RECOVERY = 0.98              # NASA ECLSS 2023
ISS_O2_CLOSURE = 0.875                 # NASA
MARS_RELAY_MBPS = 2.0                  # NASA MRO
LIGHT_DELAY_MIN = 3                    # Physics (min)
LIGHT_DELAY_MAX = 22                   # Physics (max)
SOLAR_FLUX_MAX = 590                   # NASA Viking
SOLAR_FLUX_DUST = 6                    # NASA
KILOPOWER_KW = 10                      # NASA KRUSTY

# ═══════════════════════════════════════════════════════════════════════════════
# NEW CONSTANTS (from Grok v2 response)
# ═══════════════════════════════════════════════════════════════════════════════
# Neuralink/Decision constants
NEURALINK_MULTIPLIER = 1e5             # "effective 100,000x baseline" (Grok)
HUMAN_BASELINE_BPS = 10                # "10 bps/person" voice/gesture baseline (Grok)
MDL_BETA = 0.09                        # "MDL beta=0.09 tuned for 96% compression" (Grok)

# Orbital/Debris constants
KESSLER_THRESHOLD = 0.73               # "73% threshold" (ESA 2025)
DEBRIS_COUNT_2025 = 100000             # "10^5 objects >10cm" (ESA 2025)

# Solar constants
CME_PROBABILITY_PER_DAY = 0.02         # "P(CME)=0.02/day" (NOAA Cycle 25)

# Network constants
MOON_RELAY_BOOST = 0.40                # "Moon relay -> Mars external_rate +40%" (Grok)
QUEUE_DELAY_SOLS = 7                   # "7 sols avg" competing missions (Grok)

# Sovereignty thresholds
SOVEREIGNTY_THRESHOLD_NEURALINK = 5    # "5 with Neuralink (1 Mbps/person)" (Grok)
SOVEREIGNTY_THRESHOLD_BASELINE = 25    # v3.1 baseline without Neuralink

# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED CONSTANTS (kept from v3.1)
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DECISIONS_PER_PERSON_PER_SEC = 0.1  # Tesla FSD proxy
MEANING_FRACTION = 0.001                  # Protocol overhead
LATENCY_DECAY_TAU = 600                   # 10 min Shannon decay
CONJUNCTION_BLACKOUT_DAYS = 14            # Solar conjunction
AI_LOG_FACTOR = 0.3                       # xAI scaling


# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY FUNCTIONS (kept from v3.1)
# ═══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(dist: np.ndarray) -> float:
    """H = -sum p(x) log2 p(x). Skip zeros."""
    dist = np.asarray(dist, dtype=float)
    dist = dist[dist > 0]
    if len(dist) == 0:
        return 0.0
    dist = dist / dist.sum()
    return -np.sum(dist * np.log2(dist))


def subsystem_entropy(state: dict, subsystem: str) -> float:
    """Entropy for one subsystem."""
    if subsystem not in state:
        return 0.0
    sub_state = state[subsystem]
    if isinstance(sub_state, dict):
        values = list(sub_state.values())
        numeric = [v for v in values if isinstance(v, (int, float)) and v > 0]
        if numeric:
            return shannon_entropy(np.array(numeric))
    return 0.0


def total_colony_entropy(state: dict) -> float:
    """Sum of 4 subsystems."""
    subsystems = ['atmosphere', 'thermal', 'resource', 'decision']
    return sum(subsystem_entropy(state, s) for s in subsystems)


def entropy_rate(states: list) -> float:
    """dH/dt in bits/day."""
    if len(states) < 2:
        return 0.0
    entropies = [total_colony_entropy(s) if isinstance(s, dict) else
                 total_colony_entropy(s.__dict__) if hasattr(s, '__dict__') else 0.0
                 for s in states]
    return (entropies[-1] - entropies[0]) / len(states)


def entropy_status(rate: float) -> str:
    """'stable'/'accumulating'/'critical'."""
    if rate <= 0:
        return "stable"
    elif rate < 0.1:
        return "accumulating"
    else:
        return "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSION RATE FUNCTIONS (MODIFIED for Neuralink)
# ═══════════════════════════════════════════════════════════════════════════════

def human_compression_rate(crew: int, expertise: float = 0.8) -> float:
    """crew x BASE x expertise. bits/sec."""
    return crew * BASE_DECISIONS_PER_PERSON_PER_SEC * expertise


def ai_compression_rate(compute_flops: float) -> float:
    """AI_LOG_FACTOR x log(flops/1e15)."""
    if compute_flops <= 0:
        return 0.0
    return AI_LOG_FACTOR * math.log(compute_flops / 1e15 + 1)


def neuralink_effective_rate(crew: int, mbps_per_person: float) -> float:
    """Returns crew x mbps_per_person x 1e6 x NEURALINK_MULTIPLIER / 1e6.

    NEW: Grok says "1 Mbps = 10^6 bps -> effective 100,000x baseline 10 bps"

    Args:
        crew: Number of Neuralink-equipped crew
        mbps_per_person: Bandwidth per person in Mbps (default 1.0)

    Returns:
        Effective decision rate in bits/sec
    """
    if mbps_per_person <= 0 or crew <= 0:
        return 0.0
    # 1 Mbps = 1e6 bps, NEURALINK_MULTIPLIER = 1e5 (100,000x)
    # Effective rate = crew * mbps * 1e6 * (MULTIPLIER / baseline) / 1e6
    # Simplifies to: crew * mbps * NEURALINK_MULTIPLIER
    return crew * mbps_per_person * NEURALINK_MULTIPLIER


def internal_compression_rate(crew: int, expertise: float,
                              compute_flops: float, neuralink_bandwidth_mbps: float = 0.0) -> float:
    """Internal compression rate with Neuralink support.

    MODIFIED: Add neuralink_bandwidth_mbps parameter.
    If neuralink_bandwidth_mbps > 0, multiply by NEURALINK_MULTIPLIER.
    """
    human = human_compression_rate(crew, expertise)
    ai = ai_compression_rate(compute_flops)
    base = human + ai

    if neuralink_bandwidth_mbps > 0:
        # Neuralink provides massive multiplier
        neuralink = neuralink_effective_rate(crew, neuralink_bandwidth_mbps)
        return base + neuralink
    return base


def effective_bandwidth(raw_mbps: float, latency_sec: float) -> float:
    """raw x MEANING x exp(-latency/TAU).

    Returns decision-equivalent bits/sec, not raw bits.
    2 Mbps Mars relay -> ~0.0015 effective decision bits/sec at 180s latency.
    """
    decay = math.exp(-latency_sec / LATENCY_DECAY_TAU)
    return raw_mbps * MEANING_FRACTION * decay


def conjunction_mask(sol: int) -> float:
    """0.0 during 14-day blackout else 1.0."""
    # Mars year ~668 sols, conjunction roughly every 780 Earth days (~760 sols)
    sol_in_cycle = sol % 760
    if 373 <= sol_in_cycle <= 373 + CONJUNCTION_BLACKOUT_DAYS:
        return 0.0
    return 1.0


def external_compression_rate(bandwidth_mbps: float, latency_sec: float, sol: int,
                              relay_efficiency: float = 1.0,
                              congestion_factor: float = 0.0,
                              solar_factor: float = 1.0) -> float:
    """External compression rate with network adjustments.

    MODIFIED: Add relay_efficiency, congestion_factor, solar_factor parameters.

    Args:
        bandwidth_mbps: Raw bandwidth in Mbps
        latency_sec: Light delay in seconds
        sol: Current sol (for conjunction mask)
        relay_efficiency: Relay path efficiency (0-1), default 1.0
        congestion_factor: Network congestion (0-1), 0=no congestion
        solar_factor: Solar activity factor (0-1), 1=nominal

    Returns:
        Effective external compression rate in bits/sec
    """
    eff = effective_bandwidth(bandwidth_mbps, latency_sec)
    mask = conjunction_mask(sol)
    # Apply network adjustments
    adjusted = eff * mask * relay_efficiency * (1 - congestion_factor) * solar_factor
    return adjusted


def compression_advantage(internal: float, external: float) -> float:
    """internal - external. THE KEY."""
    return internal - external


def sovereignty_threshold(neuralink: bool = False) -> int:
    """Return crew threshold for sovereignty.

    MODIFIED: Return SOVEREIGNTY_THRESHOLD_NEURALINK if neuralink else SOVEREIGNTY_THRESHOLD_BASELINE.

    Grok: "Threshold drops 25->5 crew" with Neuralink
    """
    if neuralink:
        return SOVEREIGNTY_THRESHOLD_NEURALINK
    return SOVEREIGNTY_THRESHOLD_BASELINE


def is_sovereign(advantage: float) -> bool:
    """Check if advantage indicates sovereignty."""
    return advantage > 0


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: SYSTEM-LEVEL ENTROPY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def system_entropy_budget(body_entropies: dict) -> float:
    """Sum of all body entropies.

    Args:
        body_entropies: dict[str, float] mapping body_id to entropy

    Returns:
        Total system entropy
    """
    return sum(body_entropies.values())


def entropy_conservation_check(generated: float, exported: float, stored: float) -> bool:
    """Check entropy conservation law.

    Grok: "sum generated = sum exported + sum stored"

    Returns True if |generated - exported - stored| < 0.001
    """
    return abs(generated - exported - stored) < 0.001


def mdl_compression(decision_bits: float) -> float:
    """Apply MDL compression with tuned beta.

    Grok: "MDL beta=0.09 tuned for 96% compression"

    Returns compressed bits = decision_bits * (1 - MDL_BETA * 0.96)
    """
    return decision_bits * (1 - MDL_BETA * 0.96)


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIPT EMISSION
# ═══════════════════════════════════════════════════════════════════════════════

def emit_entropy_receipt(state: dict, states: list = None) -> dict:
    """Emit entropy receipt with v2 compression metrics."""
    H_total = total_colony_entropy(state)
    rate = entropy_rate(states) if states else 0.0

    decision = state.get('decision', {}) if isinstance(state, dict) else {}
    internal = decision.get('internal_rate', 0.0)
    external = decision.get('external_rate', 0.0)
    advantage = decision.get('advantage', compression_advantage(internal, external))
    sovereign = decision.get('sovereign', is_sovereign(advantage))

    data = {
        "H_total": H_total,
        "entropy_rate": rate,
        "entropy_status": entropy_status(rate),
        "internal_compression_rate": internal,
        "external_compression_rate": external,
        "compression_advantage": advantage,
        "sovereignty": sovereign,
        "neuralink_multiplier": NEURALINK_MULTIPLIER,
        "mdl_beta": MDL_BETA,
    }
    return emit_receipt("entropy", data)
