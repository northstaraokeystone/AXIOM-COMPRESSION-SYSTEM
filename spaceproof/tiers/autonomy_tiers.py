"""autonomy_tiers.py - LEO → Mars → Deep-space Autonomy Tier Framework

THE TIER INSIGHT:
    Light-speed delay FORCES computational sovereignty.
    This isn't a bug—it's the DESIGN.

    When colonies are isolated by light-speed, they MUST compress
    reality into receipts to survive. The colonies that compress
    best EVOLVE faster.

    Deep-space voids aren't empty—they're COMPUTATIONAL SUBSTRATE
    for autonomous evolution.

Source: SpaceProof v3.0 Multi-Tier Autonomy Network Evolution
Grok: "Interstellar Starships 2040s"
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum
import math

from ..core import emit_receipt

# === CONSTANTS ===

TENANT_ID = "spaceproof-tiers"

# Light delay constants (seconds)
LIGHT_DELAY_LEO_SEC = 0.0  # Instant Earth communication
LIGHT_DELAY_MARS_SEC = 180.0  # 3 min median (3-22 min range)
LIGHT_DELAY_DEEP_SPACE_YEARS = 4.3  # Alpha Centauri (minimum interstellar)
LIGHT_DELAY_DEEP_SPACE_SEC = 4.3 * 365.25 * 24 * 3600  # ~135,792,000 seconds

# Tier transition thresholds
MARS_TIER_THRESHOLD_SEC = 60.0  # > 1 minute = Mars tier
DEEP_SPACE_THRESHOLD_SEC = 3600.0 * 24  # > 1 day = Deep space tier

# Decision rate constants
HUMAN_DECISION_RATE_BPS = 10  # bits/sec per person
LOOP_FREQUENCY_LEO_SEC = 1.0  # 1 Hz loop in LEO
LOOP_FREQUENCY_MARS_SEC = 60.0  # 60-second SENSE→ACTUATE
LOOP_FREQUENCY_DEEP_SPACE_SEC = 3600.0 * 24  # Daily loop in deep space


class AutonomyTier(Enum):
    """Autonomy tiers based on light-delay isolation.

    Each tier represents a different level of forced autonomy
    based on communication latency with Earth.
    """

    LEO = ("leo", LIGHT_DELAY_LEO_SEC)
    MARS = ("mars", LIGHT_DELAY_MARS_SEC)
    DEEP_SPACE = ("deep_space", LIGHT_DELAY_DEEP_SPACE_SEC)

    def __init__(self, tier_name: str, light_delay_sec: float):
        self.tier_name = tier_name
        self.light_delay_sec = light_delay_sec


@dataclass
class TierConfig:
    """Configuration for an autonomy tier.

    Attributes:
        tier: The autonomy tier
        loop_frequency_sec: SENSE→ACTUATE cycle time
        earth_bandwidth_mbps: Available Earth bandwidth
        autonomy_requirement: Required autonomy level (0-1)
        compression_threshold: Minimum compression ratio
    """

    tier: AutonomyTier
    loop_frequency_sec: float
    earth_bandwidth_mbps: float
    autonomy_requirement: float
    compression_threshold: float


# Default tier configurations
TIER_CONFIGS: Dict[AutonomyTier, TierConfig] = {
    AutonomyTier.LEO: TierConfig(
        tier=AutonomyTier.LEO,
        loop_frequency_sec=LOOP_FREQUENCY_LEO_SEC,
        earth_bandwidth_mbps=1000.0,  # High bandwidth in LEO
        autonomy_requirement=0.0,  # No autonomy required
        compression_threshold=0.5,
    ),
    AutonomyTier.MARS: TierConfig(
        tier=AutonomyTier.MARS,
        loop_frequency_sec=LOOP_FREQUENCY_MARS_SEC,
        earth_bandwidth_mbps=2.0,  # Mars relay bandwidth
        autonomy_requirement=0.5,  # Partial autonomy forced
        compression_threshold=0.84,  # Higher compression needed
    ),
    AutonomyTier.DEEP_SPACE: TierConfig(
        tier=AutonomyTier.DEEP_SPACE,
        loop_frequency_sec=LOOP_FREQUENCY_DEEP_SPACE_SEC,
        earth_bandwidth_mbps=0.001,  # Minimal deep space comms
        autonomy_requirement=1.0,  # Full autonomy required
        compression_threshold=0.95,  # Maximum compression
    ),
}


@dataclass
class TierTransitionResult:
    """Result of a tier transition.

    Attributes:
        from_tier: Source tier
        to_tier: Destination tier
        light_delay_change: Change in light delay
        decision_capacity_before: Capacity before transition
        decision_capacity_after: Capacity after transition
        adjustment_needed: Adjustments required for new tier
        receipts_emitted: Number of receipts emitted
        success: Whether transition succeeded
    """

    from_tier: AutonomyTier
    to_tier: AutonomyTier
    light_delay_change: float
    decision_capacity_before: float
    decision_capacity_after: float
    adjustment_needed: Dict[str, Any]
    receipts_emitted: int
    success: bool


def get_tier_from_delay(light_delay_sec: float) -> AutonomyTier:
    """Determine autonomy tier from light delay.

    Args:
        light_delay_sec: One-way light delay in seconds

    Returns:
        Appropriate AutonomyTier
    """
    if light_delay_sec <= MARS_TIER_THRESHOLD_SEC:
        return AutonomyTier.LEO
    elif light_delay_sec <= DEEP_SPACE_THRESHOLD_SEC:
        return AutonomyTier.MARS
    else:
        return AutonomyTier.DEEP_SPACE


def calculate_tier_decision_capacity(
    tier: AutonomyTier,
    crew: int,
    bandwidth_mbps: float,
    augmentation_factor: float = 1.0,
) -> float:
    """Calculate decision capacity for a tier.

    Capacity is adjusted for light-delay constraints.
    Higher tiers have lower external input but must have
    higher internal capacity.

    Args:
        tier: Autonomy tier
        crew: Number of crew members
        bandwidth_mbps: Available bandwidth
        augmentation_factor: AI/Neuralink augmentation

    Returns:
        Decision capacity in bits/sec
    """
    config = TIER_CONFIGS[tier]

    # Internal capacity: crew * human rate * augmentation
    internal_capacity = crew * HUMAN_DECISION_RATE_BPS * augmentation_factor

    # External capacity: bandwidth / (2 * delay)
    # But loop frequency limits how often we can use it
    if tier.light_delay_sec > 0:
        external_rate = (bandwidth_mbps * 1e6) / (2 * tier.light_delay_sec)
        # Further limited by loop frequency
        external_rate *= min(1.0, config.loop_frequency_sec / tier.light_delay_sec)
    else:
        external_rate = bandwidth_mbps * 1e6  # Full bandwidth in LEO

    # Total effective capacity
    # In higher tiers, internal dominates; external approaches zero
    autonomy_weight = config.autonomy_requirement
    effective_capacity = internal_capacity * (1 + autonomy_weight) + external_rate * (1 - autonomy_weight)

    return effective_capacity


def earth_input_by_tier(tier: AutonomyTier, bandwidth_mbps: float) -> float:
    """Calculate Earth input rate for a tier.

    Rate drops to effectively zero for DEEP_SPACE.

    Args:
        tier: Autonomy tier
        bandwidth_mbps: Nominal bandwidth

    Returns:
        Earth input rate in bits/sec
    """
    if tier == AutonomyTier.DEEP_SPACE:
        # Years of delay = effectively zero
        return 0.0

    if tier.light_delay_sec <= 0:
        return bandwidth_mbps * 1e6  # Full rate

    # Rate penalized by round-trip delay
    return (bandwidth_mbps * 1e6) / (2 * tier.light_delay_sec)


def tier_transition(
    from_tier: AutonomyTier,
    to_tier: AutonomyTier,
    crew: int,
    bandwidth_mbps: float = 2.0,
    augmentation_factor: float = 1.0,
) -> TierTransitionResult:
    """Simulate transition between tiers.

    Moving to higher tier requires increased autonomy.
    System must adjust loop frequency, compression, and
    decision-making patterns.

    Args:
        from_tier: Current tier
        to_tier: Target tier
        crew: Number of crew
        bandwidth_mbps: Available bandwidth
        augmentation_factor: AI/Neuralink factor

    Returns:
        TierTransitionResult with transition details
    """
    # Calculate capacities
    capacity_before = calculate_tier_decision_capacity(from_tier, crew, bandwidth_mbps, augmentation_factor)
    capacity_after = calculate_tier_decision_capacity(to_tier, crew, bandwidth_mbps, augmentation_factor)

    # Light delay change
    delay_change = to_tier.light_delay_sec - from_tier.light_delay_sec

    # Determine adjustments needed
    from_config = TIER_CONFIGS[from_tier]
    to_config = TIER_CONFIGS[to_tier]

    adjustments = {}

    # Loop frequency adjustment
    if to_config.loop_frequency_sec != from_config.loop_frequency_sec:
        adjustments["loop_frequency"] = {
            "from": from_config.loop_frequency_sec,
            "to": to_config.loop_frequency_sec,
            "change_ratio": to_config.loop_frequency_sec / from_config.loop_frequency_sec,
        }

    # Compression threshold adjustment
    if to_config.compression_threshold != from_config.compression_threshold:
        adjustments["compression_threshold"] = {
            "from": from_config.compression_threshold,
            "to": to_config.compression_threshold,
        }

    # Autonomy requirement adjustment
    if to_config.autonomy_requirement != from_config.autonomy_requirement:
        adjustments["autonomy_requirement"] = {
            "from": from_config.autonomy_requirement,
            "to": to_config.autonomy_requirement,
        }

    # Check if transition is valid
    # Moving to higher tier requires sufficient internal capacity
    success = True
    if to_config.autonomy_requirement > from_config.autonomy_requirement:
        # Need internal capacity > required rate
        internal_only = crew * HUMAN_DECISION_RATE_BPS * augmentation_factor
        required_rate = capacity_after * to_config.autonomy_requirement
        if internal_only < required_rate:
            success = False
            adjustments["insufficient_autonomy"] = {
                "internal_capacity": internal_only,
                "required_capacity": required_rate,
            }

    result = TierTransitionResult(
        from_tier=from_tier,
        to_tier=to_tier,
        light_delay_change=delay_change,
        decision_capacity_before=capacity_before,
        decision_capacity_after=capacity_after,
        adjustment_needed=adjustments,
        receipts_emitted=1,
        success=success,
    )

    # Emit tier transition receipt
    emit_receipt(
        "tier_transition_receipt",
        {
            "tenant_id": TENANT_ID,
            "from_tier": from_tier.tier_name,
            "to_tier": to_tier.tier_name,
            "light_delay_sec": to_tier.light_delay_sec,
            "decision_capacity_change": capacity_after - capacity_before,
            "adjustments": adjustments,
            "success": success,
        },
    )

    return result


def calculate_evolution_rate(tier: AutonomyTier) -> float:
    """Calculate evolutionary rate for a tier.

    Isolated systems evolve DIFFERENT compression strategies.
    When they reconnect, BEST strategy wins (natural selection).

    Higher isolation = faster local evolution.

    Args:
        tier: Autonomy tier

    Returns:
        Evolution rate (arbitrary units, higher = faster)
    """
    config = TIER_CONFIGS[tier]

    # Evolution rate scales with:
    # 1. Isolation (autonomy requirement)
    # 2. Loop frequency (more cycles = more evolution)
    # 3. Compression pressure (harder constraints = faster adaptation)

    isolation_factor = 1 + config.autonomy_requirement * 10
    loop_factor = 1 + math.log10(1 + config.loop_frequency_sec)
    compression_factor = 1 + config.compression_threshold * 2

    return isolation_factor * loop_factor * compression_factor


def multi_tier_loop_config(tier: AutonomyTier) -> Dict[str, Any]:
    """Get loop configuration for a tier.

    Each tier has different loop parameters optimized
    for its light-delay constraints.

    Args:
        tier: Autonomy tier

    Returns:
        Dict with loop configuration
    """
    config = TIER_CONFIGS[tier]

    return {
        "tier": tier.tier_name,
        "cycle_time_sec": config.loop_frequency_sec,
        "max_pending_actions": int(10 / config.autonomy_requirement) if config.autonomy_requirement > 0 else 100,
        "auto_approve_threshold": 0.5 - 0.3 * config.autonomy_requirement,  # Lower threshold for higher tiers
        "compression_target": config.compression_threshold,
        "earth_sync_interval_sec": (
            max(config.loop_frequency_sec * 10, tier.light_delay_sec * 2) if tier.light_delay_sec > 0 else 60.0
        ),
    }


def validate_tier_readiness(
    tier: AutonomyTier,
    crew: int,
    compute_mass_kg: float,
    power_available_w: float,
    current_compression_ratio: float,
) -> Dict:
    """Validate if a system is ready for a tier.

    Args:
        tier: Target tier
        crew: Available crew
        compute_mass_kg: Available compute
        power_available_w: Available power
        current_compression_ratio: Current compression performance

    Returns:
        Readiness assessment dict
    """
    config = TIER_CONFIGS[tier]

    # Check compression readiness
    compression_ready = current_compression_ratio >= config.compression_threshold

    # Estimate augmentation factor from compute mass
    # (Simplified from decision_augmented.py)
    if compute_mass_kg > 0:
        augmentation = 1 + math.log2(1 + compute_mass_kg) * 0.5
    else:
        augmentation = 1.0

    # Check decision capacity
    capacity = calculate_tier_decision_capacity(tier, crew, 2.0, augmentation)
    required_capacity = crew * HUMAN_DECISION_RATE_BPS * (1 + config.autonomy_requirement)
    capacity_ready = capacity >= required_capacity

    # Check power (rough estimate: 100W per crew + 100W per factor point)
    power_needed = crew * 100 + (augmentation - 1) * 100
    power_ready = power_available_w >= power_needed

    readiness = {
        "tier": tier.tier_name,
        "compression_ready": compression_ready,
        "compression_current": current_compression_ratio,
        "compression_required": config.compression_threshold,
        "capacity_ready": capacity_ready,
        "capacity_current": capacity,
        "capacity_required": required_capacity,
        "power_ready": power_ready,
        "power_current": power_available_w,
        "power_required": power_needed,
        "overall_ready": compression_ready and capacity_ready and power_ready,
    }

    emit_receipt(
        "tier_readiness_receipt",
        {
            "tenant_id": TENANT_ID,
            **readiness,
        },
    )

    return readiness
