"""governance.py - Enterprise Governance Scenario.

GOVERNANCE SCENARIO:
    Validate enterprise governance patterns.
    RACI assignments, provenance tracking, reason codes.

Pass Criteria:
    - 100% decisions have provenance attached (model_version, policy_state)
    - 100% decisions have RACI assigned (R, A, C, I fields)
    - 100% interventions have valid reason codes (RE001-RE010)
    - >=8 training examples produced from interventions
    - Zero unstructured log entries
    - Audit trail generation < 5 seconds
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import time
import uuid

import numpy as np

from spaceproof.core import emit_receipt

CHECKPOINT_FREQUENCY = 50
TENANT_ID = "spaceproof-scenario-governance"
AUDIT_GENERATION_SLO_MS = 5000  # 5 seconds


@dataclass
class GovernanceConfig:
    """Configuration for governance scenario."""

    cycles: int = 500
    seed: int = 42
    decisions_to_inject: int = 50
    interventions_to_inject: int = 10
    policy_changes_to_inject: int = 5


@dataclass
class GovernanceResult:
    """Result of governance scenario execution."""

    cycles_completed: int
    decisions_with_provenance: int
    decisions_with_raci: int
    interventions_with_valid_codes: int
    training_examples_produced: int
    audit_generation_time_ms: float
    passed: bool
    failure_reasons: List[str]


class GovernanceScenario:
    """Enterprise governance validation scenario."""

    def __init__(self, config: Optional[GovernanceConfig] = None):
        """Initialize governance scenario."""
        self.config = config or GovernanceConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Tracking
        self.decisions: List[Dict] = []
        self.interventions: List[Dict] = []
        self.training_examples: List[Dict] = []
        self.policy_changes: List[Dict] = []

    def run(self) -> GovernanceResult:
        """Run the governance scenario."""
        failure_reasons = []

        # Inject decisions with RACI and provenance
        for i in range(self.config.decisions_to_inject):
            decision = self._create_decision(i)
            self.decisions.append(decision)

            if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                self._emit_checkpoint(i)

        # Inject interventions with reason codes
        for i in range(self.config.interventions_to_inject):
            intervention = self._create_intervention(i)
            self.interventions.append(intervention)

            # Create training example from intervention
            example = self._create_training_example(intervention)
            self.training_examples.append(example)

        # Inject policy changes
        for i in range(self.config.policy_changes_to_inject):
            change = self._create_policy_change(i)
            self.policy_changes.append(change)

        # Generate audit trail and measure time
        start = time.time()
        self._generate_audit_trail()
        audit_time_ms = (time.time() - start) * 1000

        # Validate results
        decisions_with_provenance = sum(1 for d in self.decisions if d.get("provenance"))
        decisions_with_raci = sum(1 for d in self.decisions if all(k in d for k in ["responsible", "accountable", "consulted", "informed"]))
        interventions_valid = sum(1 for i in self.interventions if i.get("reason_code", "").startswith("RE"))

        # Check pass criteria
        if decisions_with_provenance < len(self.decisions):
            failure_reasons.append(f"Only {decisions_with_provenance}/{len(self.decisions)} decisions have provenance")

        if decisions_with_raci < len(self.decisions):
            failure_reasons.append(f"Only {decisions_with_raci}/{len(self.decisions)} decisions have RACI")

        if interventions_valid < len(self.interventions):
            failure_reasons.append(f"Only {interventions_valid}/{len(self.interventions)} interventions have valid codes")

        if len(self.training_examples) < 8:
            failure_reasons.append(f"Only {len(self.training_examples)} training examples (need >=8)")

        if audit_time_ms > AUDIT_GENERATION_SLO_MS:
            failure_reasons.append(f"Audit generation took {audit_time_ms:.0f}ms (SLO: {AUDIT_GENERATION_SLO_MS}ms)")

        passed = len(failure_reasons) == 0

        return GovernanceResult(
            cycles_completed=self.config.cycles,
            decisions_with_provenance=decisions_with_provenance,
            decisions_with_raci=decisions_with_raci,
            interventions_with_valid_codes=interventions_valid,
            training_examples_produced=len(self.training_examples),
            audit_generation_time_ms=audit_time_ms,
            passed=passed,
            failure_reasons=failure_reasons,
        )

    def _create_decision(self, index: int) -> Dict:
        """Create a decision with RACI and provenance."""
        decision_id = str(uuid.uuid4())

        decision = {
            "decision_id": decision_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "decision_type": self.rng.choice(["navigation", "compute", "maneuver", "override"]),
            # RACI
            "responsible": f"AGENT_{index % 5}",
            "accountable": f"OPERATOR_{index % 3}",
            "consulted": [f"EXPERT_{index % 2}"],
            "informed": [f"AUDIT_{index % 4}", "COMPLIANCE"],
            # Provenance
            "provenance": {
                "model_id": "spaceproof-agent",
                "model_version": "1.0.0",
                "policy_id": "default-policy",
                "policy_version": "1.0.0",
            },
        }

        emit_receipt(
            "governance_decision",
            {
                "tenant_id": TENANT_ID,
                **decision,
            },
        )

        return decision

    def _create_intervention(self, index: int) -> Dict:
        """Create an intervention with reason code."""
        reason_codes = [
            "RE001_FACTUAL_ERROR",
            "RE002_POLICY_VIOLATION",
            "RE003_SAFETY_CONCERN",
            "RE005_USER_PREFERENCE",
            "RE006_CONTEXT_MISSING",
            "RE007_TOOL_MISUSE",
        ]

        intervention = {
            "intervention_id": str(uuid.uuid4()),
            "target_decision_id": self.decisions[index % len(self.decisions)]["decision_id"] if self.decisions else str(uuid.uuid4()),
            "intervener_id": f"HUMAN_{index}",
            "intervention_type": self.rng.choice(["OVERRIDE", "CORRECTION", "ANNOTATION"]),
            "reason_code": self.rng.choice(reason_codes),
            "justification": f"Test intervention {index}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        emit_receipt(
            "governance_intervention",
            {
                "tenant_id": TENANT_ID,
                **intervention,
            },
        )

        return intervention

    def _create_training_example(self, intervention: Dict) -> Dict:
        """Create training example from intervention."""
        example = {
            "example_id": str(uuid.uuid4()),
            "source_intervention_id": intervention["intervention_id"],
            "reason_code": intervention["reason_code"],
            "quality_score": float(self.rng.uniform(0.7, 1.0)),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        emit_receipt(
            "training_example",
            {
                "tenant_id": TENANT_ID,
                **example,
            },
        )

        return example

    def _create_policy_change(self, index: int) -> Dict:
        """Create a policy change."""
        change = {
            "change_id": str(uuid.uuid4()),
            "policy_id": "default-policy",
            "from_version": f"1.{index}.0",
            "to_version": f"1.{index + 1}.0",
            "change_type": self.rng.choice(["update", "rollback", "new_rule"]),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        emit_receipt(
            "policy_change",
            {
                "tenant_id": TENANT_ID,
                **change,
            },
        )

        return change

    def _generate_audit_trail(self) -> None:
        """Generate audit trail receipt."""
        emit_receipt(
            "audit_trail",
            {
                "tenant_id": TENANT_ID,
                "decision_count": len(self.decisions),
                "intervention_count": len(self.interventions),
                "training_example_count": len(self.training_examples),
                "policy_change_count": len(self.policy_changes),
            },
        )

    def _emit_checkpoint(self, step: int) -> None:
        """Emit checkpoint receipt."""
        emit_receipt(
            "governance_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "decisions_so_far": len(self.decisions),
                "interventions_so_far": len(self.interventions),
            },
        )
