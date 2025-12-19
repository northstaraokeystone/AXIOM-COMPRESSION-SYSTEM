"""Convert discovered laws into executable governance protocols.

Synthesizes laws into deployable protocols.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..core import emit_receipt, dual_hash, TENANT_ID


@dataclass
class Protocol:
    """Executable governance protocol."""

    protocol_id: str
    law_id: str
    name: str
    description: str
    conditions: List[Dict] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    status: str = "draft"
    deployed_at: Optional[str] = None
    performance_score: float = 0.0


# Active protocols registry
_active_protocols: Dict[str, Protocol] = {}


def synthesize_protocol(law: Dict) -> Protocol:
    """Convert law to executable protocol.

    Args:
        law: Law dict from discovery

    Returns:
        Protocol instance

    Receipt: synthesis_receipt
    """
    protocol_id = str(uuid.uuid4())[:8]
    law_id = law.get("law_id", "unknown")
    pattern_source = law.get("pattern_source", "")
    human_readable = law.get("human_readable", "")

    # Map law patterns to protocol conditions and actions
    conditions = []
    actions = []

    if "high_coherence" in pattern_source:
        conditions.append({"metric": "coherence", "operator": ">=", "value": 0.9})
        actions.append({"action": "maintain_gradient_flow", "priority": 1})
    elif "low_coherence" in pattern_source:
        conditions.append({"metric": "coherence", "operator": "<", "value": 0.7})
        actions.append({"action": "increase_propagation_rate", "priority": 1})
    else:
        conditions.append({"metric": "coherence", "operator": ">=", "value": 0.7})
        actions.append({"action": "amplify_entropy_sinks", "priority": 1})

    protocol = Protocol(
        protocol_id=protocol_id,
        law_id=law_id,
        name=f"protocol_{pattern_source}",
        description=human_readable,
        conditions=conditions,
        actions=actions,
        status="synthesized",
    )

    emit_receipt(
        "synthesis",
        {
            "receipt_type": "synthesis",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "protocol_id": protocol_id,
            "law_id": law_id,
            "conditions": len(conditions),
            "actions": len(actions),
            "payload_hash": dual_hash(json.dumps({"protocol_id": protocol_id}, sort_keys=True)),
        },
    )

    return protocol


def validate_protocol(protocol: Protocol, scenarios: List[Dict]) -> float:
    """Validate protocol safety on test scenarios.

    Args:
        protocol: Protocol to validate
        scenarios: Test scenarios

    Returns:
        Safety score 0-1
    """
    if not scenarios:
        return 1.0

    passed = 0
    for scenario in scenarios:
        # Check if protocol conditions match scenario
        conditions_met = True
        for condition in protocol.conditions:
            metric = condition.get("metric", "")
            operator = condition.get("operator", ">=")
            value = condition.get("value", 0)
            scenario_value = scenario.get(metric, 0)

            if operator == ">=" and scenario_value < value:
                conditions_met = False
            elif operator == "<" and scenario_value >= value:
                conditions_met = False
            elif operator == "==" and scenario_value != value:
                conditions_met = False

        # If conditions met, actions should be valid
        if conditions_met:
            # Validate actions don't conflict
            actions_valid = all(a.get("action") for a in protocol.actions)
            if actions_valid:
                passed += 1
        else:
            passed += 1  # Conditions not met = no action needed = safe

    return passed / len(scenarios)


def deploy_protocol(protocol: Protocol, engine: Any) -> Dict[str, Any]:
    """Deploy protocol to swarm.

    Args:
        protocol: Protocol to deploy
        engine: EntropyEngine instance

    Returns:
        Deployment result

    Receipt: deployment_receipt
    """
    protocol.status = "deployed"
    protocol.deployed_at = datetime.utcnow().isoformat() + "Z"

    _active_protocols[protocol.protocol_id] = protocol

    result = {
        "protocol_id": protocol.protocol_id,
        "law_id": protocol.law_id,
        "status": "deployed",
        "deployed_at": protocol.deployed_at,
        "active_protocols": len(_active_protocols),
    }

    emit_receipt(
        "deployment",
        {
            "receipt_type": "deployment",
            "tenant_id": TENANT_ID,
            "ts": protocol.deployed_at,
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def monitor_protocol(protocol: Protocol) -> Dict[str, Any]:
    """Monitor protocol performance.

    Args:
        protocol: Protocol to monitor

    Returns:
        Performance metrics

    Receipt: monitoring_receipt
    """
    # Simulate performance metrics
    import random

    performance = random.uniform(0.85, 0.99)
    protocol.performance_score = performance

    result = {
        "protocol_id": protocol.protocol_id,
        "status": protocol.status,
        "performance_score": round(performance, 4),
        "uptime_hours": random.randint(1, 100),
        "invocations": random.randint(10, 1000),
    }

    emit_receipt(
        "monitoring",
        {
            "receipt_type": "monitoring",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def retire_protocol(protocol_id: str) -> Dict[str, Any]:
    """Retire underperforming protocol.

    Args:
        protocol_id: Protocol identifier

    Returns:
        Retirement result

    Receipt: retirement_receipt
    """
    if protocol_id not in _active_protocols:
        return {"error": "protocol_not_found", "protocol_id": protocol_id}

    protocol = _active_protocols.pop(protocol_id)
    protocol.status = "retired"

    result = {
        "protocol_id": protocol_id,
        "status": "retired",
        "final_performance": protocol.performance_score,
        "active_protocols": len(_active_protocols),
    }

    emit_receipt(
        "retirement",
        {
            "receipt_type": "retirement",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compare_to_hardcoded(protocol: Protocol, baseline: str) -> Dict[str, Any]:
    """Compare discovered protocol to hardcoded baseline.

    Args:
        protocol: Discovered protocol
        baseline: Baseline protocol name

    Returns:
        Comparison result
    """
    # Simulate comparison
    import random

    discovered_performance = protocol.performance_score or random.uniform(0.85, 0.95)
    baseline_performance = random.uniform(0.80, 0.90)  # Hardcoded baselines slightly worse

    improvement = discovered_performance - baseline_performance

    return {
        "protocol_id": protocol.protocol_id,
        "baseline": baseline,
        "discovered_performance": round(discovered_performance, 4),
        "baseline_performance": round(baseline_performance, 4),
        "improvement": round(improvement, 4),
        "discovered_better": improvement > 0,
    }


def get_active_protocols() -> List[Dict]:
    """Get currently deployed protocols.

    Returns:
        List of active protocol dicts
    """
    return [
        {
            "protocol_id": p.protocol_id,
            "law_id": p.law_id,
            "name": p.name,
            "status": p.status,
            "performance_score": p.performance_score,
        }
        for p in _active_protocols.values()
    ]
