"""AXIOM-SYSTEM v2 Observe Module - Three views of same state.

Status: NEW
Purpose: micro/macro/meta views - same state, different lenses

Grok: "Three Views as Observation Filters" - same state, three lenses
"""

from typing import Optional

from .core import merkle
from .system import SystemState
from .entropy import (
    SOVEREIGNTY_THRESHOLD_BASELINE,
    SOVEREIGNTY_THRESHOLD_NEURALINK,
    NEURALINK_MULTIPLIER,
)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def observe(state: SystemState, view: str, target: Optional[str] = None) -> str:
    """Route to appropriate view function.

    Args:
        state: System state to observe
        view: "micro", "macro", or "meta"
        target: Body ID for micro view

    Returns:
        Formatted observation string
    """
    if view == "micro":
        if target is None:
            target = "mars"  # Default to Mars
        return observe_micro(state, target)
    elif view == "macro":
        return observe_macro(state)
    elif view == "meta":
        return observe_meta(state)
    else:
        return f"Unknown view: {view}. Use 'micro', 'macro', or 'meta'."


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_bandwidth_bar(share: float, width: int = 12) -> str:
    """Format bandwidth share as Unicode bar chart.

    Args:
        share: Bandwidth share (0-1)
        width: Bar width in characters

    Returns:
        Unicode bar string
    """
    filled = int(share * width)
    empty = width - filled
    return '\u2588' * filled + '\u2591' * empty


def format_status_indicator(sovereign: bool) -> str:
    """Format sovereignty status indicator.

    Args:
        sovereign: Whether body is sovereign

    Returns:
        Status string
    """
    if sovereign:
        return "\u2713 SOVEREIGN"
    return "\u2717 DEPENDENT"


def format_rate_direction(rate: float) -> str:
    """Format entropy rate with direction indicator.

    Args:
        rate: Entropy rate

    Returns:
        Formatted rate string
    """
    if rate < 0:
        return f"{rate:.1f} bits/sol \u2014 EXPORTING"
    elif rate > 0:
        return f"+{rate:.1f} bits/sol \u2014 ACCUMULATING"
    else:
        return "0.0 bits/sol \u2014 STABLE"


# ═══════════════════════════════════════════════════════════════════════════════
# MICRO VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def observe_micro(state: SystemState, body_id: str) -> str:
    """Local subsystem states, compression rates, sovereignty.

    Args:
        state: System state
        body_id: Body to observe

    Returns:
        Formatted micro view
    """
    if body_id not in state.bodies:
        return f"Body '{body_id}' not found in system."

    body = state.bodies[body_id]

    # Determine crew info (Mars/Moon have crew)
    crew_info = ""
    neuralink_pct = 0
    compute_flops = 0

    if body_id == "mars":
        # Would need to store config in state, using defaults
        crew_info = f"Crew: ~10"
        compute_flops = 1e15
    elif body_id == "moon":
        crew_info = f"Crew: ~4"
        compute_flops = 1e14

    # Calculate to sovereignty
    if body.sovereign:
        to_sovereignty = "Already sovereign"
    else:
        # Estimate crew needed
        deficit = abs(body.advantage)
        crew_per_bit = 0.1  # BASE_DECISIONS_PER_PERSON_PER_SEC
        crew_needed = int(deficit / crew_per_bit) + 1
        to_sovereignty = f"+{crew_needed} crew OR enable Neuralink"

    output = f"""
MICRO VIEW ({body_id.upper()})
{'=' * 50}

{crew_info} | Neuralink: {neuralink_pct}% | Compute: {compute_flops:.0e} FLOPS

Light Delay: {body.delay_s:.1f}s
Bandwidth Share: {body.bandwidth_share:.1%}
Relay Path: {' -> '.join(body.relay_path)}

COMPRESSION RATES
-----------------
Internal Rate:  {body.internal_rate:.3f} bits/sec
External Rate:  {body.external_rate:.3f} bits/sec
Advantage:      {body.advantage:+.3f} bits/sec

Status: {format_status_indicator(body.sovereign)}

ENTROPY
-------
Current:   {body.entropy:.1f} bits
Rate:      {format_rate_direction(body.entropy_rate)}
Generated: {body.entropy_generated:.2f} bits/sol
Exported:  {body.entropy_exported:.2f} bits/sol

TO SOVEREIGNTY
--------------
{to_sovereignty}

{'=' * 50}
"""
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# MACRO VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def observe_macro(state: SystemState) -> str:
    """All bodies summarized, bandwidth pie, relay paths, queue.

    Args:
        state: System state

    Returns:
        Formatted macro view
    """
    # Build body table
    body_rows = []
    for body_id, body in state.bodies.items():
        if body_id == "earth":
            bw_bar = format_bandwidth_bar(1.0)
            sov = "N/A"
            rate = f"{-12.4:.1f} bits/sol"  # Earth exports
        else:
            bw_bar = format_bandwidth_bar(body.bandwidth_share)
            sov = format_status_indicator(body.sovereign)
            rate = f"{body.entropy_rate:+.1f} bits/sol"

        delay_str = f"({body.delay_s:.1f}s)" if body.delay_s > 0 else "(anchor)"

        body_rows.append(
            f"{body_id.capitalize():10} {delay_str:10} {bw_bar} {sov:15} {rate}"
        )

    body_table = "\n".join(body_rows)

    # Relay info
    relay_info = []
    for body_id, body in state.bodies.items():
        if body_id != "earth" and len(body.relay_path) > 2:
            path_str = " -> ".join(body.relay_path)
            relay_info.append(f"  {body_id}: {path_str}")

    relay_section = "\n".join(relay_info) if relay_info else "  All bodies direct to Earth"

    # Queue info
    queue_section = "  (empty)" if not state.starship_queue else "\n".join(
        f"  {m.destination}-{m.type} (Sol {m.scheduled_sol})"
        for m in state.starship_queue[:5]
    )

    output = f"""
MACRO VIEW
{'=' * 70}
                        AXIOM-SYSTEM v2 \u2014 Sol {state.sol}
{'=' * 70}

BODY STATUS
-----------
{'Body':10} {'Delay':10} {'Bandwidth':12}    {'Sovereignty':15} {'Entropy Rate'}
{'-' * 70}
{body_table}

RELAY TOPOLOGY
--------------
{relay_section}

STARSHIP QUEUE
--------------
{queue_section}

{'=' * 70}
"""
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# META VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def observe_meta(state: SystemState) -> str:
    """System entropy budget, cross-body flows, cascade risk.

    Args:
        state: System state

    Returns:
        Formatted meta view
    """
    # System sovereignty status
    if state.system_sovereign:
        sov_status = "YES (total advantage > 0)"
    else:
        total_adv = sum(b.advantage for b in state.bodies.values() if b.id != "earth")
        sov_status = f"NO (deficit: {abs(total_adv):.2f} bits/sec)"

    # Cascade risk summary
    risk_items = []
    for risk_type, risk_value in state.cascade_risk.items():
        if risk_type == "solar":
            risk_items.append(f"Solar {risk_value*100:.1f}%/day")
        elif risk_type == "kessler":
            risk_items.append(f"Kessler {risk_value*100:.0f}%")
        elif risk_type == "network":
            level = "HIGH" if risk_value > 0.2 else "MODERATE" if risk_value > 0.1 else "LOW"
            risk_items.append(f"Network {level}")

    cascade_risk_str = " | ".join(risk_items)

    # Entropy conservation check
    stored_delta = state.entropy_rate
    conservation_ok = abs(state.entropy_generated - state.entropy_exported - stored_delta) < 0.1
    conservation_status = "\u2713 CONSERVED" if conservation_ok else "\u2717 VIOLATION"

    # Merkle proof
    if state.receipts:
        proof = merkle(state.receipts)
        proof_str = f"{proof[:16]}...{proof[-16:]}"
    else:
        proof_str = "(no receipts)"

    output = f"""
{'=' * 70}
                        AXIOM-SYSTEM v2 \u2014 Sol {state.sol}
{'=' * 70}

META VIEW
---------

SYSTEM ENTROPY
--------------
Total Entropy:     {state.total_entropy:.1f} bits
Rate:              {format_rate_direction(state.entropy_rate)}
Generated:         {state.entropy_generated:.2f} bits/sol
Exported:          {state.entropy_exported:.2f} bits/sol
Conservation:      {conservation_status}

SYSTEM SOVEREIGNTY
------------------
Status:            {sov_status}

CASCADE RISK
------------
{cascade_risk_str}

ORBITAL STATUS
--------------
Debris Ratio:      {state.orbital.debris_ratio:.1%}
Kessler Risk:      {state.orbital.kessler_risk:.1%}
Kessler Active:    {'YES - LAUNCH BLACKOUT' if state.orbital.kessler_active else 'No'}

SOLAR STATUS
------------
CME Active:        {'YES' if state.solar.cme_active else 'No'}
CME Intensity:     {state.solar.cme_intensity:.1f}

{'=' * 70}
Merkle proof: {proof_str}
{'=' * 70}
"""
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def observe_all(state: SystemState, micro_target: str = "mars") -> str:
    """Combined view showing all three perspectives.

    Args:
        state: System state
        micro_target: Body for micro view

    Returns:
        Combined formatted view
    """
    meta = observe_meta(state)
    macro = observe_macro(state)
    micro = observe_micro(state, micro_target)

    return f"{meta}\n{macro}\n{micro}"
