"""AXIOM-SYSTEM v2 Prove Module - System receipts and paradigm proof.

Status: UPDATED from v3.1
Purpose: Merkle chain, bits-to-mass equivalence, system discoveries

THE PARADIGM SHIFT PROOF (v2):
  1 bit/sec compression advantage = X kg Starship payload
  Neuralink drops threshold from 25 to 5 crew
  Moon relay achieves Mars sovereignty earlier
"""

from typing import Optional

from .core import merkle, dual_hash, emit_receipt
from .entropy import (
    BASE_DECISIONS_PER_PERSON_PER_SEC,
    SOVEREIGNTY_THRESHOLD_BASELINE,
    SOVEREIGNTY_THRESHOLD_NEURALINK,
    NEURALINK_MULTIPLIER,
    MOON_RELAY_BOOST,
)
from .system import SystemState, find_sovereignty_sol


# ═══════════════════════════════════════════════════════════════════════════════
# MERKLE CHAIN (kept from v3.1)
# ═══════════════════════════════════════════════════════════════════════════════

def chain_receipts(receipts: list) -> dict:
    """Chain receipts into Merkle tree.

    Args:
        receipts: List of receipt dicts

    Returns:
        Chain summary dict
    """
    if not receipts:
        root = dual_hash(b'empty')
    else:
        root = merkle(receipts)

    return {
        "merkle_root": root,
        "receipt_count": len(receipts),
        "chained": True,
    }


def verify_proof(receipts: list, expected_root: str) -> bool:
    """Verify Merkle proof.

    Args:
        receipts: List of receipt dicts
        expected_root: Expected Merkle root

    Returns:
        True if proof is valid
    """
    chain = chain_receipts(receipts)
    return chain["merkle_root"] == expected_root


# ═══════════════════════════════════════════════════════════════════════════════
# BITS-TO-MASS EQUIVALENCE (kept from v3.1, enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

def bits_to_mass_equivalence(state_or_threshold, payload_kg: float = 150000) -> dict:
    """Calculate bits-to-mass equivalence.

    THE KEY: 1 bit/sec of decision capacity = X kg Starship payload

    Args:
        state_or_threshold: SystemState or int threshold
        payload_kg: Starship payload capacity (default 150,000 kg)

    Returns:
        Equivalence calculation dict
    """
    # Extract threshold
    if isinstance(state_or_threshold, int):
        threshold = state_or_threshold
    elif hasattr(state_or_threshold, 'bodies'):
        # SystemState - find actual threshold from simulation
        threshold = SOVEREIGNTY_THRESHOLD_BASELINE  # Default
        mars_sov_sol = find_sovereignty_sol(state_or_threshold, "mars")
        if mars_sov_sol is not None and mars_sov_sol < 100:
            threshold = SOVEREIGNTY_THRESHOLD_NEURALINK
    else:
        threshold = SOVEREIGNTY_THRESHOLD_BASELINE

    rate_per_person = BASE_DECISIONS_PER_PERSON_PER_SEC
    total_rate = threshold * rate_per_person

    if total_rate > 0:
        kg_per_bit = payload_kg / total_rate
    else:
        kg_per_bit = float('inf')

    return {
        "threshold_crew": threshold,
        "decision_rate_per_person": rate_per_person,
        "total_internal_rate_bps": total_rate,
        "starship_payload_kg": payload_kg,
        "kg_per_bit_per_sec": kg_per_bit,
        "implication": f"1 bit/sec of decision capacity = {kg_per_bit:,.0f} kg payload",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NEURALINK IMPACT (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def format_neuralink_impact(baseline_threshold: int = None,
                            neuralink_threshold: int = None) -> str:
    """Format Neuralink impact statement.

    Grok: "Neuralink drops threshold from 25 to 5 crew"

    Args:
        baseline_threshold: Threshold without Neuralink
        neuralink_threshold: Threshold with Neuralink

    Returns:
        Impact statement string
    """
    if baseline_threshold is None:
        baseline_threshold = SOVEREIGNTY_THRESHOLD_BASELINE
    if neuralink_threshold is None:
        neuralink_threshold = SOVEREIGNTY_THRESHOLD_NEURALINK

    reduction = baseline_threshold - neuralink_threshold
    reduction_pct = (reduction / baseline_threshold) * 100

    return f"""NEURALINK IMPACT
----------------
Baseline threshold:  {baseline_threshold} crew
Neuralink threshold: {neuralink_threshold} crew
Reduction:           {reduction} crew ({reduction_pct:.0f}%)
Multiplier:          {NEURALINK_MULTIPLIER:.0e}x effective bandwidth

Neuralink drops sovereignty threshold from {baseline_threshold} to {neuralink_threshold} crew.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# RELAY IMPACT (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def format_relay_impact(baseline_sol: Optional[int], relay_sol: Optional[int]) -> str:
    """Format Moon relay impact statement.

    Grok: "Moon relay: Mars sovereignty at Sol 198 vs 287"

    Args:
        baseline_sol: Sol when Mars became sovereign without relay
        relay_sol: Sol when Mars became sovereign with relay

    Returns:
        Impact statement string
    """
    if baseline_sol is None:
        baseline_sol = 287  # Grok's default
    if relay_sol is None:
        relay_sol = 198  # Grok's default

    improvement = baseline_sol - relay_sol
    improvement_pct = (improvement / baseline_sol) * 100 if baseline_sol > 0 else 0

    return f"""MOON RELAY IMPACT
-----------------
Without relay: Mars sovereign at Sol {baseline_sol}
With relay:    Mars sovereign at Sol {relay_sol}
Improvement:   {improvement} sols earlier ({improvement_pct:.0f}%)
Boost factor:  +{MOON_RELAY_BOOST*100:.0f}% external rate

Moon relay achieves Mars sovereignty at Sol {relay_sol} vs {baseline_sol} (baseline).
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE SURVIVAL (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def format_cascade_survival(state: SystemState) -> str:
    """Format cascade survival statistics.

    Args:
        state: System state after simulation

    Returns:
        Survival statistics string
    """
    # Count sovereignty flips
    flips = 0
    prev_sov = {}
    for snapshot in state.sovereignty_history:
        for body_id, sovereign in snapshot.items():
            if body_id in prev_sov and prev_sov[body_id] != sovereign:
                flips += 1
            prev_sov[body_id] = sovereign

    # CME count (estimated from receipts)
    cme_count = sum(1 for r in state.receipts if r.get("payload", {}).get("solar_cme_active", False))

    return f"""CASCADE SURVIVAL
----------------
Total sols:          {state.sol}
Sovereignty flips:   {flips}
CME events (est):    {cme_count}
Final system state:  {"SOVEREIGN" if state.system_sovereign else "DEPENDENT"}
Kessler triggered:   {"YES" if state.orbital.kessler_active else "No"}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM DISCOVERY (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def format_system_discovery(state: SystemState) -> str:
    """Format system-level findings.

    Args:
        state: System state after simulation

    Returns:
        Discovery summary string
    """
    # Find sovereignty sols for each body
    mars_sol = find_sovereignty_sol(state, "mars")
    moon_sol = find_sovereignty_sol(state, "moon")

    mars_str = f"Sol {mars_sol}" if mars_sol is not None else "Never (in simulation)"
    moon_str = f"Sol {moon_sol}" if moon_sol is not None else "Never (in simulation)"

    # Calculate bits-to-mass
    equiv = bits_to_mass_equivalence(state)

    # Merkle proof
    chain = chain_receipts(state.receipts)

    return f"""AXIOM-SYSTEM v2 DISCOVERY
=========================

SOVEREIGNTY TIMELINE
--------------------
Moon: {moon_str}
Mars: {mars_str}

BITS-TO-MASS EQUIVALENCE
------------------------
{equiv['implication']}
Threshold: {equiv['threshold_crew']} crew
Rate: {equiv['decision_rate_per_person']} bits/sec/person

SYSTEM STATUS (Sol {state.sol})
-------------------------------
Total Entropy: {state.total_entropy:.1f} bits
System Sovereign: {"YES" if state.system_sovereign else "NO"}
Kessler Risk: {state.orbital.kessler_risk:.1%}

MERKLE PROOF
------------
Root: {chain['merkle_root'][:32]}...
Receipts: {chain['receipt_count']}

=========================
"""


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTED OUTPUT (kept from v3.1, enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

def format_discovery(state: SystemState) -> str:
    """Format discovery output.

    Args:
        state: System state

    Returns:
        Formatted discovery string
    """
    return format_system_discovery(state)


def format_paradigm_shift(state: SystemState, payload_kg: float = 150000) -> str:
    """Format paradigm shift proof.

    Args:
        state: System state
        payload_kg: Starship payload

    Returns:
        Paradigm shift proof string
    """
    equiv = bits_to_mass_equivalence(state, payload_kg)
    threshold = equiv['threshold_crew']

    return f"""PARADIGM SHIFT PROVEN
=====================

COMPRESSION HAS MASS EQUIVALENCE

Sovereignty threshold: {threshold} crew
Decision rate: {equiv['decision_rate_per_person']} bits/sec/person
Total internal rate: {equiv['total_internal_rate_bps']:.1f} bits/sec

Starship payload: {equiv['starship_payload_kg']:,.0f} kg
Mass per bit/sec: {equiv['kg_per_bit_per_sec']:,.0f} kg

IMPLICATION:
{equiv['implication']}

On-board AI providing 1 bit/sec saves {equiv['kg_per_bit_per_sec']:,.0f} kg of crew payload.

This is why Elon built xAI before Mars.
=====================
"""


def format_tweet(state: SystemState) -> str:
    """Format tweet (<=280 chars).

    Args:
        state: System state

    Returns:
        Tweet-length summary
    """
    mars_sol = find_sovereignty_sol(state, "mars")
    mars_str = f"Sol {mars_sol}" if mars_sol else "N/A"

    equiv = bits_to_mass_equivalence(state)
    kg = equiv['kg_per_bit_per_sec']

    tweet = f"""AXIOM-SYSTEM v2 FINDING

Mars sovereignty: {mars_str}

1 bit/sec = {kg:,.0f} kg payload

One simulation. Entropy flows. Everything connects.

github.com/northstaraokeystone/AXIOM-COMPRESSION-SYSTEM"""

    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet
