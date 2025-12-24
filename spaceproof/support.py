"""support.py - L0-L4 Receipt Level Infrastructure

THE SUPPORT INSIGHT:
    "Helpers and support throughout the universe" =
    The same pattern at every scale:
    - L0: Telemetry gaps spawn sensor helpers
    - L1: Decision gaps spawn agent helpers
    - L2: Deployment gaps spawn process helpers
    - L3: Quality gaps spawn optimization helpers
    - L4: Meta gaps spawn audit helpers

    When L4 insights improve L0 processing, the system achieves SELF-VERIFICATION.
    Not consciousness. Not AGI. Just receipts that spawn receipts that improve receipts.

Source Pattern: ProofPack v3 §3.1 - Five Receipt Levels
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set

from .core import emit_receipt


# === CONSTANTS ===

TENANT_ID = "spaceproof-autonomy"
"""Tenant for support receipts."""


# === ENUMS ===


class SupportLevel(Enum):
    """Five levels of receipt infrastructure.

    L0_TELEMETRY: Raw events (autonomy state, propulsion state, latency)
    L1_AGENTS: Decision receipts (optimization, helper actions)
    L2_CHANGES: Deployment receipts (helper deployments, config changes)
    L3_QUALITY: Effectiveness receipts (helper performance, timeline accuracy)
    L4_META: Receipts about receipts (coverage, completeness)
    """

    L0_TELEMETRY = "L0"
    L1_AGENTS = "L1"
    L2_CHANGES = "L2"
    L3_QUALITY = "L3"
    L4_META = "L4"


# === RECEIPT TYPE MAPPINGS ===

LEVEL_RECEIPT_TYPES: Dict[SupportLevel, Set[str]] = {
    SupportLevel.L0_TELEMETRY: {
        "autonomy_state",
        "propulsion_state",
        "latency",
        "bandwidth",
        "telemetry",
        "sensor",
        "raw_event",
        "heartbeat",
    },
    SupportLevel.L1_AGENTS: {
        "optimization",
        "decision",
        "selection",
        "agent_action",
        "gate_decision",
        "stage_gate",
        "helper_action",
    },
    SupportLevel.L2_CHANGES: {
        "helper_deployment",
        "helper_blueprint",
        "config_change",
        "deployment",
        "migration",
        "schema_change",
    },
    SupportLevel.L3_QUALITY: {
        "helper_effectiveness",
        "timeline_accuracy",
        "quality",
        "validation",
        "chain",
        "sovereignty",
        "threshold",
    },
    SupportLevel.L4_META: {
        "support_level",
        "coverage",
        "completeness",
        "audit",
        "meta",
        "harvest",
        "helper_retirement",
    },
}

# All expected receipt types per level (for coverage calculation)
EXPECTED_TYPES: Dict[SupportLevel, Set[str]] = {
    SupportLevel.L0_TELEMETRY: {"autonomy_state", "propulsion_state", "latency"},
    SupportLevel.L1_AGENTS: {"optimization", "decision", "gate_decision"},
    SupportLevel.L2_CHANGES: {"helper_deployment", "helper_blueprint", "config_change"},
    SupportLevel.L3_QUALITY: {"helper_effectiveness", "validation", "chain"},
    SupportLevel.L4_META: {"support_level", "coverage", "harvest"},
}


# === DATACLASSES ===


@dataclass
class SupportCoverage:
    """Coverage metrics for a support level.

    Attributes:
        level: SupportLevel enum value
        receipt_count: Number of receipts at this level
        coverage_ratio: Ratio of observed types to expected types (0-1)
        gaps: List of missing receipt types
    """

    level: SupportLevel
    receipt_count: int
    coverage_ratio: float
    gaps: List[str]


# === FUNCTIONS ===


def classify_receipt(receipt: Dict) -> SupportLevel:
    """Determine which level a receipt belongs to.

    Args:
        receipt: Receipt dict with receipt_type field

    Returns:
        SupportLevel enum value
    """
    receipt_type = receipt.get("receipt_type", "").lower()

    for level, types in LEVEL_RECEIPT_TYPES.items():
        if receipt_type in types:
            return level

    # Default to L0 for unknown types (treat as raw telemetry)
    return SupportLevel.L0_TELEMETRY


def measure_coverage(receipts: List[Dict]) -> Dict[SupportLevel, SupportCoverage]:
    """Calculate coverage per level.

    Args:
        receipts: List of receipt dicts

    Returns:
        Dict mapping SupportLevel to SupportCoverage

    Emits: support_level_receipt for each level
    """
    # Group receipts by level
    by_level: Dict[SupportLevel, List[Dict]] = {level: [] for level in SupportLevel}
    types_seen: Dict[SupportLevel, Set[str]] = {level: set() for level in SupportLevel}

    for receipt in receipts:
        level = classify_receipt(receipt)
        by_level[level].append(receipt)
        types_seen[level].add(receipt.get("receipt_type", ""))

    # Calculate coverage for each level
    coverage = {}

    for level in SupportLevel:
        receipts_at_level = by_level[level]
        seen = types_seen[level]
        expected = EXPECTED_TYPES.get(level, set())

        # Coverage = types seen / types expected
        if expected:
            covered_types = seen & expected
            coverage_ratio = len(covered_types) / len(expected)
            gaps = list(expected - seen)
        else:
            coverage_ratio = 1.0 if seen else 0.0
            gaps = []

        support_coverage = SupportCoverage(
            level=level,
            receipt_count=len(receipts_at_level),
            coverage_ratio=coverage_ratio,
            gaps=gaps,
        )
        coverage[level] = support_coverage

        # Emit receipt for this level
        emit_receipt(
            "support_level",
            {
                "tenant_id": TENANT_ID,
                "level": level.value,
                "receipt_count": len(receipts_at_level),
                "coverage_ratio": round(coverage_ratio, 4),
                "gaps": gaps,
                "types_seen": list(seen),
                "self_verifying": _is_self_verifying(level, coverage_ratio),
            },
        )

    return coverage


def _is_self_verifying(level: SupportLevel, coverage_ratio: float) -> bool:
    """Check if level qualifies as self-verifying.

    L4 with high coverage = self-verifying (meta receipts tracking receipts).
    """
    return level == SupportLevel.L4_META and coverage_ratio >= 0.8


def check_completeness(coverage: Dict[SupportLevel, SupportCoverage]) -> bool:
    """Check if all levels have ≥95% coverage.

    Args:
        coverage: Dict from measure_coverage()

    Returns:
        True if all levels ≥0.95 coverage
    """
    for level, cov in coverage.items():
        if cov.coverage_ratio < 0.95:
            return False
    return True


def detect_gaps(coverage: Dict[SupportLevel, SupportCoverage]) -> List[str]:
    """Return missing receipt types per level.

    Args:
        coverage: Dict from measure_coverage()

    Returns:
        List of "LEVEL: missing_type" strings
    """
    all_gaps = []

    for level, cov in coverage.items():
        for gap_type in cov.gaps:
            all_gaps.append(f"{level.value}: {gap_type}")

    return all_gaps


def l4_feedback(coverage: Dict[SupportLevel, SupportCoverage], l0_params: Dict) -> Dict:
    """Use L4 insights to suggest L0 improvements.

    This IS self-verification: meta-level analysis improving base-level config.

    Args:
        coverage: Dict from measure_coverage()
        l0_params: Current L0 telemetry parameters

    Returns:
        Updated L0 params with suggested improvements
    """
    suggestions = dict(l0_params)

    # Check L0 coverage
    l0_cov = coverage.get(SupportLevel.L0_TELEMETRY)
    if l0_cov and l0_cov.coverage_ratio < 0.8:
        # L0 coverage low - suggest enabling more telemetry
        suggestions["telemetry_level"] = "verbose"
        suggestions["sample_rate_increase"] = 2.0

    # Check if we're missing key L0 types
    if l0_cov and l0_cov.gaps:
        suggestions["enable_types"] = l0_cov.gaps

    # Check L3 quality level
    l3_cov = coverage.get(SupportLevel.L3_QUALITY)
    if l3_cov and l3_cov.coverage_ratio < 0.7:
        # Quality receipts low - suggest more validation
        suggestions["validation_frequency"] = "every_cycle"

    # Mark as self-verifying feedback
    suggestions["l4_feedback_applied"] = True
    suggestions["feedback_source"] = "support.l4_feedback"

    # Emit meta receipt about the feedback
    emit_receipt(
        "support_level",
        {
            "tenant_id": TENANT_ID,
            "level": "L4",
            "receipt_count": 1,
            "coverage_ratio": coverage.get(
                SupportLevel.L4_META, SupportCoverage(SupportLevel.L4_META, 0, 0.0, [])
            ).coverage_ratio,
            "gaps": [],
            "self_verifying": True,
            "feedback_applied": True,
            "l0_improvements": list(suggestions.keys()),
        },
    )

    return suggestions


def get_level_summary(coverage: Dict[SupportLevel, SupportCoverage]) -> str:
    """Generate human-readable coverage summary.

    Args:
        coverage: Dict from measure_coverage()

    Returns:
        Formatted summary string
    """
    lines = ["SUPPORT LEVEL COVERAGE", "=" * 40]

    for level in SupportLevel:
        cov = coverage.get(level)
        if cov:
            status = "OK" if cov.coverage_ratio >= 0.95 else "GAP"
            lines.append(
                f"{level.value}: {cov.receipt_count:4d} receipts | "
                f"{cov.coverage_ratio:5.1%} coverage | {status}"
            )
            if cov.gaps:
                lines.append(f"       Missing: {', '.join(cov.gaps)}")

    complete = check_completeness(coverage)
    lines.append("=" * 40)
    lines.append(f"COMPLETE: {'YES' if complete else 'NO'}")

    return "\n".join(lines)


def initialize_coverage() -> Dict[SupportLevel, SupportCoverage]:
    """Create initial empty coverage dict.

    Returns:
        Dict with zero coverage for all levels
    """
    return {
        level: SupportCoverage(
            level=level,
            receipt_count=0,
            coverage_ratio=0.0,
            gaps=list(EXPECTED_TYPES.get(level, set())),
        )
        for level in SupportLevel
    }
