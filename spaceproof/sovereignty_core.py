"""sovereignty.py - The Core Equation

THE PEARL:
    sovereignty = internal_rate > external_rate

One equation. One curve. One number.

This module implements the sovereignty equation and threshold finding.

Source: SpaceProof D20 Production Evolution
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

from .core import emit_receipt

# === PHYSICS CONSTANTS ===

# Human decision rate (bits per second per person)
HUMAN_DECISION_RATE_BPS = 10

# Starlink Mars bandwidth range (Mbps)
STARLINK_MARS_BANDWIDTH_MIN_MBPS = 2.0
STARLINK_MARS_BANDWIDTH_MAX_MBPS = 100.0
STARLINK_MARS_BANDWIDTH_EXPECTED_MBPS = 10.0

# Mars light delay range (seconds)
MARS_LIGHT_DELAY_MIN_S = 180  # 3 min opposition
MARS_LIGHT_DELAY_MAX_S = 1320  # 22 min conjunction
MARS_LIGHT_DELAY_AVG_S = 480  # 8 min average

# === AUGMENTATION CONSTANTS (v3.0) ===
# Grok: "xAI autonomy for deep-space lacks policy framework"

AI_AUGMENTATION_FACTOR = 5.0  # xAI multiplier
NEURALINK_AUGMENTATION_FACTOR = 20.0  # Neural interface (speculative)
HUMAN_ONLY_FACTOR = 1.0  # Baseline

TENANT_ID = "spaceproof-sovereignty"


@dataclass
class SovereigntyConfig:
    """Configuration for sovereignty calculation.

    Attributes:
        crew: Number of crew members
        compute_flops: Compute capacity in FLOPS (default 0)
        bandwidth_mbps: Communication bandwidth (default 2.0 Mbps minimum)
        delay_s: One-way light delay (default 480s = 8 min average)
        augmentation_factor: AI/Neuralink multiplier (v3.0)
    """

    crew: int
    compute_flops: float = 0.0
    bandwidth_mbps: float = STARLINK_MARS_BANDWIDTH_MIN_MBPS
    delay_s: float = MARS_LIGHT_DELAY_AVG_S
    augmentation_factor: float = HUMAN_ONLY_FACTOR  # v3.0: AI (5.0), Neuralink (20.0)


@dataclass
class SovereigntyResult:
    """Result of sovereignty calculation.

    Attributes:
        internal_rate: Internal decision rate (bits/sec)
        external_rate: External decision rate (bits/sec)
        advantage: internal - external
        sovereign: True if advantage > 0
        threshold_crew: Crew where advantage crosses zero (if computed)
    """

    internal_rate: float
    external_rate: float
    advantage: float
    sovereign: bool
    threshold_crew: Optional[int] = None


def internal_rate(
    crew: int, compute_flops: float = 0.0, augmentation_factor: float = 1.0
) -> float:
    """Calculate internal decision rate.

    Internal = log2(1 + crew * human_rate * augmentation + compute_flops * 1e-15)

    v3.0: Added augmentation_factor for AI/Neuralink support.

    Args:
        crew: Number of crew members
        compute_flops: Compute capacity in FLOPS
        augmentation_factor: AI/Neuralink multiplier (1.0=human, 5.0=AI, 20.0=Neuralink)

    Returns:
        Internal decision rate in bits/sec
    """
    effective_crew = crew * augmentation_factor
    return math.log2(1 + effective_crew * HUMAN_DECISION_RATE_BPS + compute_flops * 1e-15)


def external_rate(bandwidth_mbps: float, delay_s: float) -> float:
    """Calculate external decision rate.

    External = (bandwidth * 1e6) / (2 * delay)

    Args:
        bandwidth_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds

    Returns:
        External decision rate in bits/sec
    """
    return (bandwidth_mbps * 1e6) / (2 * delay_s)


def sovereignty_advantage(ir: float, er: float) -> float:
    """Calculate sovereignty advantage.

    Args:
        ir: Internal rate
        er: External rate

    Returns:
        Advantage (positive = sovereign)
    """
    return ir - er


def is_sovereign(advantage: float) -> bool:
    """Check if advantage indicates sovereignty.

    Args:
        advantage: Sovereignty advantage

    Returns:
        True if sovereign (advantage > 0)
    """
    return advantage > 0


def compute_sovereignty(config: SovereigntyConfig) -> SovereigntyResult:
    """THE core equation. Compute sovereignty for given configuration.

    sovereignty = internal_rate > external_rate

    v3.0: Now includes augmentation_factor in calculation.

    Args:
        config: SovereigntyConfig with crew, compute, bandwidth, delay, augmentation

    Returns:
        SovereigntyResult with rates, advantage, and sovereignty status
    """
    ir = internal_rate(config.crew, config.compute_flops, config.augmentation_factor)
    er = external_rate(config.bandwidth_mbps, config.delay_s)
    adv = sovereignty_advantage(ir, er)
    sov = is_sovereign(adv)

    result = SovereigntyResult(
        internal_rate=ir,
        external_rate=er,
        advantage=adv,
        sovereign=sov,
    )

    emit_receipt(
        "sovereignty",
        {
            "tenant_id": TENANT_ID,
            "crew": config.crew,
            "compute_flops": config.compute_flops,
            "bandwidth_mbps": config.bandwidth_mbps,
            "delay_s": config.delay_s,
            "augmentation_factor": config.augmentation_factor,
            "effective_crew": config.crew * config.augmentation_factor,
            "internal_rate": ir,
            "external_rate": er,
            "advantage": adv,
            "sovereign": sov,
        },
    )

    return result


def find_threshold(
    compute_flops: float = 0.0,
    bandwidth_mbps: float = STARLINK_MARS_BANDWIDTH_MIN_MBPS,
    delay_s: float = MARS_LIGHT_DELAY_AVG_S,
    max_crew: int = 1000,
    augmentation_factor: float = 1.0,
) -> int:
    """Find the crew size where sovereignty is achieved.

    Binary search for the threshold where internal > external.

    v3.0: Added augmentation_factor support.

    Args:
        compute_flops: Compute capacity
        bandwidth_mbps: Bandwidth
        delay_s: Light delay
        max_crew: Maximum crew to search
        augmentation_factor: AI/Neuralink multiplier (v3.0)

    Returns:
        Minimum crew for sovereignty (or max_crew if not achievable)
    """
    er = external_rate(bandwidth_mbps, delay_s)

    low, high = 1, max_crew
    while low < high:
        mid = (low + high) // 2
        ir = internal_rate(mid, compute_flops, augmentation_factor)
        if ir > er:
            high = mid
        else:
            low = mid + 1

    return low


def calculate_sovereignty_threshold(
    crew: int,
    augmentation_factor: float,
    bandwidth_mbps: float,
    delay_s: float,
) -> Dict:
    """Calculate sovereignty threshold with augmentation.

    v3.0 API: Validates 4 crew + AI = 20 crew human-only.

    Args:
        crew: Physical crew count
        augmentation_factor: AI/Neuralink multiplier
        bandwidth_mbps: Earth bandwidth
        delay_s: Light delay

    Returns:
        Dict with threshold analysis
    """
    config = SovereigntyConfig(
        crew=crew,
        augmentation_factor=augmentation_factor,
        bandwidth_mbps=bandwidth_mbps,
        delay_s=delay_s,
    )

    result = compute_sovereignty(config)

    # Find equivalent unaugmented crew
    equivalent_crew = find_threshold(
        bandwidth_mbps=bandwidth_mbps,
        delay_s=delay_s,
        augmentation_factor=1.0,
    )

    return {
        "crew": crew,
        "augmentation_factor": augmentation_factor,
        "effective_crew": crew * augmentation_factor,
        "sovereign": result.sovereign,
        "threshold_crew": equivalent_crew,
        "internal_rate": result.internal_rate,
        "external_rate": result.external_rate,
    }


def sensitivity_analysis(
    config: SovereigntyConfig,
    bandwidth_range: Tuple[float, float] = (2.0, 100.0),
    delay_range: Tuple[float, float] = (180.0, 1320.0),
    steps: int = 10,
) -> List[dict]:
    """Analyze sovereignty sensitivity to bandwidth and delay.

    Args:
        config: Base configuration
        bandwidth_range: (min, max) bandwidth in Mbps
        delay_range: (min, max) delay in seconds
        steps: Number of steps for each dimension

    Returns:
        List of analysis results
    """
    results = []

    bw_step = (bandwidth_range[1] - bandwidth_range[0]) / steps
    delay_step = (delay_range[1] - delay_range[0]) / steps

    for i in range(steps + 1):
        bw = bandwidth_range[0] + i * bw_step
        for j in range(steps + 1):
            delay = delay_range[0] + j * delay_step

            test_config = SovereigntyConfig(
                crew=config.crew,
                compute_flops=config.compute_flops,
                bandwidth_mbps=bw,
                delay_s=delay,
            )

            result = compute_sovereignty(test_config)

            results.append({
                "bandwidth_mbps": bw,
                "delay_s": delay,
                "internal_rate": result.internal_rate,
                "external_rate": result.external_rate,
                "advantage": result.advantage,
                "sovereign": result.sovereign,
            })

    return results
