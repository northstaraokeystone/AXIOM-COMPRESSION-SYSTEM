"""AXIOM-COLONY v3.1 Entropy Module - Compression rates engine.

Paradigm (v3.1):
  NEW: internal_compression_rate vs external_compression_rate
       compression_advantage = internal - external
       advantage > 0 → SOVEREIGN
"""

import math
import numpy as np

from src.core import emit_receipt, dual_hash

# Verified Constants (NASA/Physics)
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

# Derived Constants (v3.1 NEW)
BASE_DECISIONS_PER_PERSON_PER_SEC = 0.1  # Tesla FSD proxy
MEANING_FRACTION = 0.001                  # Protocol overhead
LATENCY_DECAY_TAU = 600                   # 10 min Shannon decay
CONJUNCTION_BLACKOUT_DAYS = 14            # Solar conjunction
AI_LOG_FACTOR = 0.3                       # xAI scaling
NEURALINK_MULTIPLIER_MAX = 3.0            # Augmentation ceiling


def shannon_entropy(dist: np.ndarray) -> float:
    """H = -Σ p(x) log₂ p(x). Skip zeros."""
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
    """"stable"/"accumulating"/"critical"."""
    if rate <= 0:
        return "stable"
    elif rate < 0.1:
        return "accumulating"
    else:
        return "critical"


# v3.1 NEW: Compression Rate Functions

def human_compression_rate(crew: int, expertise: float = 0.8) -> float:
    """crew × BASE × expertise. bits/sec. NEW."""
    return crew * BASE_DECISIONS_PER_PERSON_PER_SEC * expertise


def ai_compression_rate(compute_flops: float) -> float:
    """AI_LOG_FACTOR × log(flops/1e15). NEW."""
    if compute_flops <= 0:
        return 0.0
    return AI_LOG_FACTOR * math.log(compute_flops / 1e15 + 1)


def neuralink_compression_rate(crew: int, neuralink_frac: float) -> float:
    """crew × BASE × frac × MAX. NEW."""
    return crew * BASE_DECISIONS_PER_PERSON_PER_SEC * neuralink_frac * NEURALINK_MULTIPLIER_MAX


def internal_compression_rate(crew: int, expertise: float,
                              compute_flops: float, neuralink_frac: float) -> float:
    """Piecewise: human + ai, neuralink cap. NEW."""
    human = human_compression_rate(crew, expertise)
    ai = ai_compression_rate(compute_flops)
    neuralink = neuralink_compression_rate(crew, neuralink_frac)
    # Neuralink provides multiplier, not additive
    base = human + ai
    augmented = base * (1 + neuralink_frac * (NEURALINK_MULTIPLIER_MAX - 1))
    return augmented


def effective_bandwidth(raw_mbps: float, latency_sec: float) -> float:
    """raw × MEANING × exp(-latency/TAU). NEW.

    Returns decision-equivalent bits/sec, not raw bits.
    2 Mbps Mars relay → ~0.0015 effective decision bits/sec at 180s latency.
    """
    decay = math.exp(-latency_sec / LATENCY_DECAY_TAU)
    return raw_mbps * MEANING_FRACTION * decay


def conjunction_mask(sol: int) -> float:
    """0.0 during 14-day blackout else 1.0. NEW."""
    # Mars year ~668 sols, conjunction roughly every 780 Earth days (~760 sols)
    # Simplified: blackout at certain sols
    sol_in_cycle = sol % 760
    if 373 <= sol_in_cycle <= 373 + CONJUNCTION_BLACKOUT_DAYS:
        return 0.0
    return 1.0


def external_compression_rate(bandwidth_mbps: float, latency_sec: float, sol: int) -> float:
    """effective × mask. bits/sec. NEW."""
    eff = effective_bandwidth(bandwidth_mbps, latency_sec)
    mask = conjunction_mask(sol)
    return eff * mask


def compression_advantage(internal: float, external: float) -> float:
    """internal - external. THE KEY. NEW."""
    return internal - external


def sovereignty_threshold(internal: float, external: float) -> bool:
    """internal > external. RENAMED from old decision_sovereignty."""
    return internal > external


def uncertainty_overhead(confidences: dict) -> float:
    """Σ(1/conf - 1). NEW."""
    total = 0.0
    for conf in confidences.values():
        if isinstance(conf, (int, float)) and conf > 0:
            total += (1.0 / conf) - 1.0
    return total


def emit_entropy_receipt(state: dict, states: list = None) -> dict:
    """Emit entropy receipt with v3.1 compression metrics."""
    H_total = total_colony_entropy(state)
    rate = entropy_rate(states) if states else 0.0

    # Extract compression data from state if available
    decision = state.get('decision', {}) if isinstance(state, dict) else {}
    internal = decision.get('internal_rate', 0.0)
    external = decision.get('external_rate', 0.0)
    advantage = decision.get('advantage', compression_advantage(internal, external))
    sovereign = decision.get('sovereign', sovereignty_threshold(internal, external))

    data = {
        "H_total": H_total,
        "entropy_rate": rate,
        "entropy_status": entropy_status(rate),
        "internal_compression_rate": internal,
        "external_compression_rate": external,
        "compression_advantage": advantage,
        "sovereignty": sovereign
    }
    return emit_receipt("entropy", data)
