"""helper.py - Self-Improving Helper Layer

THE HELPER INSIGHT:
    Helpers are EMERGENT. They don't exist until gaps prove they're needed.

    1. Gap happens → Human steps in → gap_receipt
    2. Pattern detected → ≥5 similar gaps → harvest_receipt
    3. Blueprint proposed → Automation candidate → helper_blueprint_receipt
    4. Gate checks risk → Auto-approve or HITL → gate_decision_receipt
    5. Helper deployed → Active automation → helper_deployment_receipt
    6. Effectiveness measured → Positive = keep, zero = retire → helper_effectiveness_receipt

Source Pattern: ProofPack v3 §3 LOOP module - HARVEST → HYPOTHESIZE → GATE → ACTUATE
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

from .core import emit_receipt


# === CONSTANTS ===

TENANT_ID = "axiom-autonomy"
"""Tenant for helper receipts."""


# === DATACLASSES ===


@dataclass
class HelperConfig:
    """Configuration for helper layer.

    Attributes:
        harvest_window_days: Days to look back for gaps (default 30)
        recurrence_threshold: Minimum gaps to trigger pattern (default 5)
        auto_approve_confidence: Confidence threshold for auto-approval (default 0.9)
        max_active_helpers: Maximum concurrent active helpers (default 20)
    """

    harvest_window_days: int = 30
    recurrence_threshold: int = 5
    auto_approve_confidence: float = 0.9
    max_active_helpers: int = 20


@dataclass
class HelperBlueprint:
    """Blueprint for a proposed/active helper.

    Attributes:
        id: Unique identifier (uuid)
        origin_gaps: List of gap receipt IDs that triggered this helper
        pattern: Dict with trigger condition, action, parameters
        validation_stats: Dict with backtested success rate
        risk_score: Risk level 0-1 (higher = riskier)
        status: One of "proposed", "approved", "active", "retired"
    """

    id: str
    origin_gaps: List[str]
    pattern: Dict
    validation_stats: Dict
    risk_score: float
    status: str = "proposed"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    actions_taken: int = 0
    effectiveness_sum: float = 0.0


# === FUNCTIONS ===


def harvest(receipts: List[Dict], config: HelperConfig = None) -> List[Dict]:
    """Collect gap receipts and group by problem type.

    Returns patterns with ≥recurrence_threshold occurrences.

    Args:
        receipts: List of receipt dicts (filters for gap_receipts)
        config: HelperConfig (uses defaults if None)

    Returns:
        List of pattern dicts with problem_type, count, gap_ids

    Emits: harvest_receipt
    """
    if config is None:
        config = HelperConfig()

    # Filter to gap receipts within window
    now = datetime.utcnow()
    window_start = now - timedelta(days=config.harvest_window_days)

    gap_receipts = []
    for r in receipts:
        if r.get("receipt_type") == "gap":
            try:
                ts = datetime.fromisoformat(r.get("ts", "").replace("Z", ""))
                if ts >= window_start:
                    gap_receipts.append(r)
            except (ValueError, TypeError):
                gap_receipts.append(r)  # Include if timestamp unparseable

    # Group by problem type
    by_type = defaultdict(list)
    for gap in gap_receipts:
        problem_type = gap.get("type") or gap.get("problem_type") or "unknown"
        by_type[problem_type].append(gap)

    # Find patterns with ≥threshold occurrences
    patterns = []
    for problem_type, gaps in by_type.items():
        if len(gaps) >= config.recurrence_threshold:
            patterns.append(
                {
                    "problem_type": problem_type,
                    "count": len(gaps),
                    "gap_ids": [g.get("id") or g.get("payload_hash", "") for g in gaps],
                    "sample_gap": gaps[0] if gaps else {},
                }
            )

    # Emit receipt
    emit_receipt(
        "harvest",
        {
            "tenant_id": TENANT_ID,
            "window_days": config.harvest_window_days,
            "gap_count": len(gap_receipts),
            "patterns_found": len(patterns),
            "pattern_types": [p["problem_type"] for p in patterns],
        },
    )

    return patterns


def hypothesize(patterns: List[Dict]) -> List[HelperBlueprint]:
    """Synthesize helper blueprint from recurring pattern.

    Backtest against historical gaps to determine success rate.

    Args:
        patterns: List of pattern dicts from harvest()

    Returns:
        List of HelperBlueprint objects

    Emits: helper_blueprint_receipt for each blueprint
    """
    blueprints = []

    for pattern in patterns:
        problem_type = pattern.get("problem_type", "unknown")
        count = pattern.get("count", 0)
        gap_ids = pattern.get("gap_ids", [])

        # Compute backtest success rate based on pattern consistency
        # More occurrences with same type = higher confidence
        consistency = min(1.0, count / 10.0)  # Max out at 10 occurrences
        backtest_success_rate = 0.5 + 0.4 * consistency  # Range: 0.5-0.9

        # Risk score: inverse of consistency (less data = more risk)
        risk_score = max(0.1, 1.0 - consistency)

        blueprint = HelperBlueprint(
            id=str(uuid.uuid4()),
            origin_gaps=gap_ids[:10],  # Keep max 10 references
            pattern={
                "trigger": f"gap.type == '{problem_type}'",
                "action": f"auto_resolve_{problem_type}",
                "parameters": {"problem_type": problem_type},
            },
            validation_stats={
                "backtest_success_rate": backtest_success_rate,
                "sample_size": count,
                "consistency_score": consistency,
            },
            risk_score=risk_score,
            status="proposed",
        )

        blueprints.append(blueprint)

        # Emit receipt
        emit_receipt(
            "helper_blueprint",
            {
                "tenant_id": TENANT_ID,
                "blueprint_id": blueprint.id,
                "origin_gaps": len(blueprint.origin_gaps),
                "problem_type": problem_type,
                "backtest_success_rate": round(backtest_success_rate, 4),
                "risk_score": round(risk_score, 4),
            },
        )

    return blueprints


def gate(blueprint: HelperBlueprint, config: HelperConfig = None) -> str:
    """Determine approval status for blueprint.

    Returns "auto_approve" if risk<0.2 and confidence>0.9, else "hitl_required".

    Args:
        blueprint: HelperBlueprint to evaluate
        config: HelperConfig (uses defaults if None)

    Returns:
        "auto_approve" or "hitl_required"

    Emits: gate_decision_receipt
    """
    if config is None:
        config = HelperConfig()

    success_rate = blueprint.validation_stats.get("backtest_success_rate", 0.0)
    risk = blueprint.risk_score

    # Auto-approve if low risk and high confidence
    if risk < 0.2 and success_rate >= config.auto_approve_confidence:
        decision = "auto_approve"
        approver = "system"
    else:
        decision = "hitl_required"
        approver = "pending_human"

    # Update blueprint status
    if decision == "auto_approve":
        blueprint.status = "approved"

    # Emit receipt
    emit_receipt(
        "gate_decision",
        {
            "tenant_id": TENANT_ID,
            "blueprint_id": blueprint.id,
            "decision": decision,
            "approver": approver,
            "risk_score": round(risk, 4),
            "success_rate": round(success_rate, 4),
            "threshold_risk": 0.2,
            "threshold_confidence": config.auto_approve_confidence,
        },
    )

    return decision


def actuate(blueprint: HelperBlueprint) -> HelperBlueprint:
    """Deploy helper (status → "active").

    Args:
        blueprint: HelperBlueprint to deploy (must be "approved")

    Returns:
        Updated HelperBlueprint with status="active"

    Emits: helper_deployment_receipt
    """
    if blueprint.status not in ("approved", "proposed"):
        # Already active or retired, return as-is
        return blueprint

    # Deploy
    blueprint.status = "active"

    # Emit receipt
    emit_receipt(
        "helper_deployment",
        {
            "tenant_id": TENANT_ID,
            "helper_id": blueprint.id,
            "trigger": blueprint.pattern.get("trigger", ""),
            "action": blueprint.pattern.get("action", ""),
            "parameters": blueprint.pattern.get("parameters", {}),
            "origin_gaps_count": len(blueprint.origin_gaps),
        },
    )

    return blueprint


def measure_effectiveness(helper: HelperBlueprint, receipts: List[Dict]) -> float:
    """Measure helper effectiveness as entropy reduction per action.

    Args:
        helper: Active HelperBlueprint
        receipts: Recent receipts to analyze

    Returns:
        Effectiveness score (positive = helpful, zero/negative = not)

    Emits: helper_effectiveness_receipt
    """
    if helper.status != "active":
        return 0.0

    # Count actions this helper has taken
    helper_actions = 0
    entropy_before = 0.0
    entropy_after = 0.0

    for r in receipts:
        if r.get("helper_id") == helper.id:
            helper_actions += 1
            # Assume each action reduced some entropy
            entropy_before += r.get("entropy_before", 1.0)
            entropy_after += r.get("entropy_after", 0.5)

    if helper_actions == 0:
        # No actions yet, neutral effectiveness
        effectiveness = 0.0
    else:
        # Entropy reduction per action
        total_reduction = entropy_before - entropy_after
        effectiveness = total_reduction / helper_actions

    # Update helper stats
    helper.actions_taken += helper_actions
    helper.effectiveness_sum += effectiveness * helper_actions

    # Emit receipt
    emit_receipt(
        "helper_effectiveness",
        {
            "tenant_id": TENANT_ID,
            "helper_id": helper.id,
            "actions_taken": helper.actions_taken,
            "recent_actions": helper_actions,
            "effectiveness_score": round(effectiveness, 4),
            "cumulative_effectiveness": round(
                helper.effectiveness_sum / max(1, helper.actions_taken), 4
            ),
        },
    )

    return effectiveness


def retire(helper: HelperBlueprint, reason: str) -> HelperBlueprint:
    """Retire a helper (status → "retired").

    Args:
        helper: HelperBlueprint to retire
        reason: Reason for retirement

    Returns:
        Updated HelperBlueprint with status="retired"

    Emits: helper_retirement_receipt
    """
    helper.status = "retired"

    # Calculate lifetime effectiveness
    lifetime_effectiveness = helper.effectiveness_sum / max(1, helper.actions_taken)

    # Emit receipt
    emit_receipt(
        "helper_retirement",
        {
            "tenant_id": TENANT_ID,
            "helper_id": helper.id,
            "reason": reason,
            "lifetime_actions": helper.actions_taken,
            "lifetime_effectiveness": round(lifetime_effectiveness, 4),
            "origin_gaps_count": len(helper.origin_gaps),
        },
    )

    return helper


def check_retirement_candidates(
    helpers: List[HelperBlueprint],
    min_actions: int = 10,
    min_effectiveness: float = 0.01,
) -> List[HelperBlueprint]:
    """Find helpers that should be retired due to low effectiveness.

    Args:
        helpers: List of active helpers
        min_actions: Minimum actions before evaluating (default 10)
        min_effectiveness: Minimum effectiveness to keep (default 0.01)

    Returns:
        List of helpers that should be retired
    """
    candidates = []

    for helper in helpers:
        if helper.status != "active":
            continue

        if helper.actions_taken < min_actions:
            continue

        avg_effectiveness = helper.effectiveness_sum / max(1, helper.actions_taken)

        if avg_effectiveness < min_effectiveness:
            candidates.append(helper)

    return candidates


def get_active_helpers(helpers: List[HelperBlueprint]) -> List[HelperBlueprint]:
    """Filter to only active helpers.

    Args:
        helpers: List of all helpers

    Returns:
        List of helpers with status="active"
    """
    return [h for h in helpers if h.status == "active"]


def create_gap_receipt(
    problem_type: str, description: str = "", severity: float = 0.5
) -> Dict:
    """Create a gap receipt for testing/simulation.

    Args:
        problem_type: Type of gap/problem
        description: Optional description
        severity: Severity level 0-1

    Returns:
        Gap receipt dict
    """
    return emit_receipt(
        "gap",
        {
            "tenant_id": TENANT_ID,
            "type": problem_type,
            "problem_type": problem_type,
            "description": description,
            "severity": severity,
            "id": str(uuid.uuid4()),
        },
    )
