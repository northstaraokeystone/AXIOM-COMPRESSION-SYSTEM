"""Decision Capacity Calculator.

Purpose: Calculate information-theoretic decision capacity in bits/second.

THE NOVEL CLAIM:
    This is the first tool to calculate sovereignty as information capacity
    rather than resource capacity. Everyone models mass and energy.
    Nobody models bits.

THE PHYSICS:
    Sovereignty = internal decision capacity > external input capacity.
    Latency (3-22 min) is irreducible physics.
    Conjunction (14 days blackout) is deterministic orbital mechanics.
    Decision paralysis during emergencies is fatal.
"""

import math

from spaceproof.core import emit_receipt

from .constants import (
    DECISION_BIT_COMPLEXITY_CRITICAL,
    DECISION_BIT_COMPLEXITY_HIGH,
    DECISION_BIT_COMPLEXITY_LOW,
    DECISION_BIT_COMPLEXITY_MEDIUM,
    DECISIONS_PER_DAY_CRITICAL,
    DECISIONS_PER_DAY_HIGH,
    DECISIONS_PER_DAY_LOW,
    DECISIONS_PER_DAY_MEDIUM,
    MARS_CONJUNCTION_BLACKOUT_DAYS,
    MARS_LIGHT_DELAY_AVG_SEC,
    TENANT_ID,
)


def calculate_internal_capacity(
    crew: list[dict],
    expertise: dict | None = None,
) -> float:
    """Calculate bits/sec colony can process internally.

    Based on crew expertise levels x decision complexity x decisions per hour.
    NOVEL - no existing research. Estimated from ISS operations data.

    Args:
        crew: List of crew dicts with 'skills' and 'expertise_level' keys
        expertise: Optional expertise multipliers per skill

    Returns:
        float: Internal decision capacity in bits/second.
    """
    if not crew:
        return 0.0

    if expertise is None:
        expertise = {}

    # Base capacity from crew count and skills
    total_capacity_bps = 0.0

    for member in crew:
        # Get expertise level (EXPERT=1.0, COMPETENT=0.7, NOVICE=0.3)
        level = member.get("expertise_level", 0.7)
        skills = member.get("skills", {})

        # Each skilled crew member can handle decisions in their domain
        # Capacity = skill_coverage * level * base_rate

        skill_coverage = sum(skills.values()) / max(len(skills), 1) if skills else 0.5

        # Base decision rate per crew member (bits/day)
        # CRITICAL decisions are complex, HIGH are simpler, etc.
        critical_bits = DECISIONS_PER_DAY_CRITICAL * DECISION_BIT_COMPLEXITY_CRITICAL
        high_bits = DECISIONS_PER_DAY_HIGH * DECISION_BIT_COMPLEXITY_HIGH
        medium_bits = DECISIONS_PER_DAY_MEDIUM * DECISION_BIT_COMPLEXITY_MEDIUM
        low_bits = DECISIONS_PER_DAY_LOW * DECISION_BIT_COMPLEXITY_LOW

        total_bits_day = (critical_bits + high_bits + medium_bits + low_bits) * level * skill_coverage

        # Convert to bits/second
        capacity_bps = total_bits_day / 86400.0

        total_capacity_bps += capacity_bps

    return total_capacity_bps


def calculate_earth_input_rate(
    bandwidth_mbps: float = 2.0,
    latency_sec: float = MARS_LIGHT_DELAY_AVG_SEC,
    conjunction_blackout: bool = False,
) -> float:
    """Calculate effective decision support rate from Earth.

    KEY INSIGHT: Raw bandwidth is irrelevant for decisions.
    What matters is how quickly Earth can respond to queries.

    Earth's effective capacity is LIMITED BY:
    1. Round-trip latency (3-44 min depending on orbital position)
    2. Human expert availability on Earth (assume 1 expert can process same as 1 crew)
    3. Queue delays for complex decisions

    This models Earth as providing decision SUPPORT, not raw bandwidth.

    Args:
        bandwidth_mbps: Bandwidth in Mbps (affects queue depth, not raw rate)
        latency_sec: One-way latency in seconds
        conjunction_blackout: If True, returns 0 (blackout period)

    Returns:
        float: Earth decision support capacity in bits/second.
    """
    if conjunction_blackout:
        return 0.0

    # Round-trip latency (for question and answer)
    rtt_sec = 2 * latency_sec

    # Model Earth support as "number of equivalent experts" limited by latency
    # With 11 min average one-way (22 min RTT), can process ~3 decisions/hour
    # vs. local crew who can process decisions immediately

    # Decisions per hour Earth can support (limited by RTT)
    # Each decision requires at least one round trip
    decisions_per_hour_per_expert = 3600.0 / rtt_sec

    # Assume 10 Earth experts supporting Mars (can be adjusted)
    earth_experts = 10

    # Total decisions per day Earth can support
    decisions_per_day = decisions_per_hour_per_expert * 24 * earth_experts

    # Convert to bits/sec using average decision complexity
    avg_bits_per_decision = (DECISION_BIT_COMPLEXITY_CRITICAL + DECISION_BIT_COMPLEXITY_HIGH) / 2

    earth_bps = decisions_per_day * avg_bits_per_decision / 86400.0

    # High bandwidth allows slightly higher throughput (parallel consultations)
    bandwidth_factor = min(1.5, 0.5 + bandwidth_mbps / 20.0)

    return max(earth_bps * bandwidth_factor, 0.0)


def calculate_decision_latency_cost(
    latency_sec: float,
    decision_urgency: str = "MEDIUM",
) -> float:
    """Calculate cost function for delayed decisions.

    CRITICAL decisions (life support) have exponential cost.
    ROUTINE decisions (science) have linear cost.

    Args:
        latency_sec: Decision delay in seconds
        decision_urgency: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"

    Returns:
        float: Cost factor (1.0 = baseline, higher = worse).
    """
    # Time thresholds by urgency (seconds)
    thresholds = {
        "CRITICAL": 60,  # 1 minute
        "HIGH": 3600,  # 1 hour
        "MEDIUM": 86400,  # 1 day
        "LOW": 604800,  # 1 week
    }

    threshold = thresholds.get(decision_urgency, 86400)

    if latency_sec <= 0:
        return 1.0

    ratio = latency_sec / threshold

    if decision_urgency == "CRITICAL":
        # Exponential cost for critical decisions
        cost = math.exp(ratio) if ratio < 10 else float("inf")
    elif decision_urgency == "HIGH":
        # Quadratic cost for high priority
        cost = 1.0 + ratio**2
    elif decision_urgency == "MEDIUM":
        # Linear cost for medium
        cost = 1.0 + ratio
    else:
        # Logarithmic cost for low priority
        cost = 1.0 + math.log1p(ratio)

    return cost


def compute_sovereignty_threshold(
    internal_bps: float,
    earth_bps: float,
) -> bool:
    """Determine if colony has achieved sovereignty.

    Returns True when internal > external.
    This IS the sovereignty threshold.
    Mathematical definition of "free planet."

    Args:
        internal_bps: Internal decision capacity (bits/sec)
        earth_bps: Earth input capacity (bits/sec)

    Returns:
        bool: True if sovereign (internal > external).
    """
    return internal_bps > earth_bps


def calculate_conjunction_survival(
    internal_capacity: float,
    decisions_per_day: int = DECISIONS_PER_DAY_CRITICAL + DECISIONS_PER_DAY_HIGH,
    conjunction_days: int = MARS_CONJUNCTION_BLACKOUT_DAYS,
) -> float:
    """Calculate probability colony survives conjunction blackout.

    Requires 100% internal capacity for critical decisions.
    During 14-day blackout, Earth input = 0.

    Args:
        internal_capacity: Internal decision capacity (bits/sec)
        decisions_per_day: Required decisions per day
        conjunction_days: Blackout duration

    Returns:
        float: Survival probability (0-1).
    """
    # Calculate required capacity for critical+high decisions
    required_bits_day = (
        DECISIONS_PER_DAY_CRITICAL * DECISION_BIT_COMPLEXITY_CRITICAL
        + DECISIONS_PER_DAY_HIGH * DECISION_BIT_COMPLEXITY_HIGH
    )

    required_bps = required_bits_day / 86400.0

    # Capacity ratio
    if required_bps <= 0:
        return 1.0

    capacity_ratio = internal_capacity / required_bps

    if capacity_ratio >= 1.0:
        # Full capacity: high survival probability
        # Still not 100% due to random failures
        return 0.99
    elif capacity_ratio >= 0.8:
        # 80-100% capacity: moderate risk
        return 0.80 + 0.19 * (capacity_ratio - 0.8) / 0.2
    elif capacity_ratio >= 0.5:
        # 50-80% capacity: significant risk
        return 0.50 + 0.30 * (capacity_ratio - 0.5) / 0.3
    else:
        # Below 50%: critical risk
        return capacity_ratio  # Linear decrease to 0


def calculate_decision_capacity_score(
    crew: list[dict],
    bandwidth_mbps: float = 2.0,
    latency_sec: float = MARS_LIGHT_DELAY_AVG_SEC,
) -> dict:
    """Calculate comprehensive decision capacity metrics.

    Args:
        crew: Crew configuration
        bandwidth_mbps: Earth bandwidth
        latency_sec: Earth latency

    Returns:
        dict: Decision capacity metrics including sovereignty status.
    """
    internal = calculate_internal_capacity(crew)
    earth = calculate_earth_input_rate(bandwidth_mbps, latency_sec)
    earth_blackout = calculate_earth_input_rate(bandwidth_mbps, latency_sec, conjunction_blackout=True)

    sovereign = compute_sovereignty_threshold(internal, earth)
    conjunction_survival = calculate_conjunction_survival(internal)

    # Calculate latency costs for different decision types
    latency_costs = {
        "CRITICAL": calculate_decision_latency_cost(latency_sec, "CRITICAL"),
        "HIGH": calculate_decision_latency_cost(latency_sec, "HIGH"),
        "MEDIUM": calculate_decision_latency_cost(latency_sec, "MEDIUM"),
        "LOW": calculate_decision_latency_cost(latency_sec, "LOW"),
    }

    return {
        "internal_capacity_bps": internal,
        "earth_capacity_bps": earth,
        "earth_capacity_blackout_bps": earth_blackout,
        "sovereign": sovereign,
        "advantage_bps": internal - earth,
        "advantage_ratio": internal / earth if earth > 0 else float("inf"),
        "conjunction_survival_probability": conjunction_survival,
        "latency_costs": latency_costs,
    }


def emit_decision_capacity_receipt(
    crew: list[dict],
    metrics: dict,
) -> dict:
    """Emit decision capacity receipt.

    Args:
        crew: Crew configuration
        metrics: Decision capacity metrics

    Returns:
        dict: Emitted receipt.
    """
    return emit_receipt(
        "decision_capacity",
        {
            "tenant_id": TENANT_ID,
            "crew_count": len(crew),
            "internal_capacity_bps": metrics["internal_capacity_bps"],
            "earth_capacity_bps": metrics["earth_capacity_bps"],
            "sovereign": metrics["sovereign"],
            "advantage_bps": metrics["advantage_bps"],
            "conjunction_survival": metrics["conjunction_survival_probability"],
        },
    )
