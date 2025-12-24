"""loop.py - 60-Second SENSE→ACTUATE Cycle

D20 Production Evolution: The brain that improves 1-8.

THE LOOP INSIGHT:
    Autonomy = fast iteration.
    Every 60 seconds: sense, analyze, hypothesize, gate, actuate.
    The system that iterates fastest wins.

Source: SpaceProof D20 Production Evolution

SLOs:
    - Cycle time: <= 60 seconds
    - High-risk actions (risk > 0.5): require HITL approval
    - Receipt emission: every cycle, even if no actions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import time
import uuid

from .core import emit_receipt

# === CONSTANTS ===

TENANT_ID = "spaceproof-loop"

# Cycle configuration
CYCLE_TIME_LIMIT_SEC = 60
HIGH_RISK_THRESHOLD = 0.5


@dataclass
class Action:
    """An action proposal."""

    id: str
    action_type: str
    description: str
    risk: float
    payload: Dict = field(default_factory=dict)
    approved: bool = False
    executed: bool = False


@dataclass
class CycleResult:
    """Result of a complete loop cycle."""

    cycle_id: str
    phase_timings: Dict[str, float]
    actions_proposed: int
    actions_approved: int
    actions_executed: int
    receipts_sensed: int
    anomalies_detected: int
    cycle_time_sec: float
    completed: bool


class Loop:
    """60-second SENSE→ANALYZE→HYPOTHESIZE→GATE→ACTUATE→EMIT cycle."""

    def __init__(
        self,
        sources: Optional[List[Callable]] = None,
        hitl_callback: Optional[Callable] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize the loop.

        Args:
            sources: List of callable receipt sources
            hitl_callback: Human-in-the-loop approval callback
            config: Loop configuration
        """
        self.sources = sources or []
        self.hitl_callback = hitl_callback
        self.config = config or {}
        self.cycle_count = 0
        self.last_cycle_result: Optional[CycleResult] = None

    def run_cycle(self) -> CycleResult:
        """Run a complete SENSE→ACTUATE cycle.

        Returns:
            CycleResult with timings and action counts
        """
        cycle_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        phase_timings: Dict[str, float] = {}

        # Phase 1: SENSE
        phase_start = time.time()
        receipts = self.sense()
        phase_timings["sense"] = time.time() - phase_start

        # Phase 2: ANALYZE
        phase_start = time.time()
        analysis = self.analyze(receipts)
        phase_timings["analyze"] = time.time() - phase_start

        # Phase 3: HYPOTHESIZE
        phase_start = time.time()
        proposals = self.hypothesize(analysis)
        phase_timings["hypothesize"] = time.time() - phase_start

        # Phase 4: GATE
        phase_start = time.time()
        approved = self.gate(proposals)
        phase_timings["gate"] = time.time() - phase_start

        # Phase 5: ACTUATE
        phase_start = time.time()
        executed = self.actuate(approved)
        phase_timings["actuate"] = time.time() - phase_start

        # Phase 6: EMIT
        phase_start = time.time()
        cycle_time = time.time() - start_time

        result = CycleResult(
            cycle_id=cycle_id,
            phase_timings=phase_timings,
            actions_proposed=len(proposals),
            actions_approved=len(approved),
            actions_executed=len(executed),
            receipts_sensed=len(receipts),
            anomalies_detected=analysis.get("anomaly_count", 0),
            cycle_time_sec=cycle_time,
            completed=cycle_time <= CYCLE_TIME_LIMIT_SEC,
        )

        # Emit loop receipt
        emit_receipt(
            "loop_receipt",
            {
                "tenant_id": TENANT_ID,
                "cycle_id": cycle_id,
                "phase_timings": phase_timings,
                "actions_proposed": result.actions_proposed,
                "actions_approved": result.actions_approved,
                "actions_executed": result.actions_executed,
                "cycle_time_sec": cycle_time,
                "completed_in_time": result.completed,
            },
        )

        phase_timings["emit"] = time.time() - phase_start

        self.cycle_count += 1
        self.last_cycle_result = result

        return result

    def sense(self) -> List[Dict]:
        """Query all receipt sources.

        Returns:
            List of receipts from all sources
        """
        receipts = []

        for source in self.sources:
            try:
                source_receipts = source()
                if isinstance(source_receipts, list):
                    receipts.extend(source_receipts)
                elif isinstance(source_receipts, dict):
                    receipts.append(source_receipts)
            except Exception as e:
                # Log but don't fail
                emit_receipt(
                    "sense_error",
                    {
                        "tenant_id": TENANT_ID,
                        "source": str(source),
                        "error": str(e),
                    },
                )

        return receipts

    def analyze(self, receipts: List[Dict]) -> Dict:
        """Analyze receipts for patterns and anomalies.

        Args:
            receipts: List of receipts to analyze

        Returns:
            Analysis dict with patterns and anomalies
        """
        analysis = {
            "receipt_count": len(receipts),
            "anomaly_count": 0,
            "anomalies": [],
            "patterns": [],
            "metrics": {},
        }

        # Group by receipt type
        by_type: Dict[str, List[Dict]] = {}
        for receipt in receipts:
            rtype = receipt.get("receipt_type", "unknown")
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(receipt)

        analysis["receipt_types"] = {k: len(v) for k, v in by_type.items()}

        # Look for anomaly receipts
        for receipt in receipts:
            if receipt.get("receipt_type") == "anomaly":
                analysis["anomaly_count"] += 1
                analysis["anomalies"].append(
                    {
                        "metric": receipt.get("metric"),
                        "classification": receipt.get("classification"),
                        "delta": receipt.get("delta"),
                    }
                )

        return analysis

    def hypothesize(self, analysis: Dict) -> List[Action]:
        """Generate action proposals based on analysis.

        Args:
            analysis: Analysis dict from analyze()

        Returns:
            List of Action proposals
        """
        proposals = []

        # If anomalies detected, propose investigation
        for anomaly in analysis.get("anomalies", []):
            classification = anomaly.get("classification", "unknown")

            # Higher risk for more severe classifications
            risk_map = {
                "deviation": 0.2,
                "drift": 0.3,
                "degradation": 0.4,
                "violation": 0.6,
                "fraud": 0.8,
                "anti_pattern": 0.9,
            }
            risk = risk_map.get(classification, 0.5)

            proposals.append(
                Action(
                    id=str(uuid.uuid4())[:8],
                    action_type="investigate",
                    description=f"Investigate {classification}: {anomaly.get('metric')}",
                    risk=risk,
                    payload={"anomaly": anomaly},
                )
            )

        # Add monitoring action if no anomalies
        if not proposals:
            proposals.append(
                Action(
                    id=str(uuid.uuid4())[:8],
                    action_type="monitor",
                    description="Continue monitoring",
                    risk=0.0,
                    payload={},
                )
            )

        return proposals

    def gate(self, proposals: List[Action]) -> List[Action]:
        """Filter by risk threshold. HITL gate for high-risk.

        Args:
            proposals: List of Action proposals

        Returns:
            List of approved Actions
        """
        approved = []

        for action in proposals:
            if action.risk <= HIGH_RISK_THRESHOLD:
                # Auto-approve low-risk actions
                action.approved = True
                approved.append(action)
            elif self.hitl_callback:
                # HITL approval required
                if self.hitl_callback(action):
                    action.approved = True
                    approved.append(action)
                # else: rejected, not added
            else:
                # No HITL callback, skip high-risk
                emit_receipt(
                    "hitl_required",
                    {
                        "tenant_id": TENANT_ID,
                        "action_id": action.id,
                        "action_type": action.action_type,
                        "risk": action.risk,
                    },
                )

        return approved

    def actuate(self, approved: List[Action]) -> List[Action]:
        """Execute approved actions.

        Args:
            approved: List of approved Actions

        Returns:
            List of executed Actions
        """
        executed = []

        for action in approved:
            if not action.approved:
                continue

            try:
                # Execute based on action type
                if action.action_type == "investigate":
                    # Emit investigation receipt
                    emit_receipt(
                        "investigation",
                        {
                            "tenant_id": TENANT_ID,
                            "action_id": action.id,
                            "target": action.payload.get("anomaly", {}),
                        },
                    )
                elif action.action_type == "monitor":
                    # Just acknowledge
                    pass

                action.executed = True
                executed.append(action)

            except Exception as e:
                emit_receipt(
                    "actuate_error",
                    {
                        "tenant_id": TENANT_ID,
                        "action_id": action.id,
                        "error": str(e),
                    },
                )

        return executed


def run_loop_once(
    sources: Optional[List[Callable]] = None, config: Optional[Dict] = None
) -> CycleResult:
    """Convenience function to run a single loop cycle.

    Args:
        sources: Receipt sources
        config: Loop config

    Returns:
        CycleResult
    """
    loop = Loop(sources=sources, config=config)
    return loop.run_cycle()


def run_loop_continuous(
    sources: Optional[List[Callable]] = None,
    config: Optional[Dict] = None,
    max_cycles: int = 0,
    interval_sec: float = 60.0,
) -> List[CycleResult]:
    """Run loop continuously.

    Args:
        sources: Receipt sources
        config: Loop config
        max_cycles: Maximum cycles (0 = infinite)
        interval_sec: Seconds between cycles

    Returns:
        List of CycleResults
    """
    loop = Loop(sources=sources, config=config)
    results = []
    cycle = 0

    while max_cycles == 0 or cycle < max_cycles:
        cycle_start = time.time()
        result = loop.run_cycle()
        results.append(result)
        cycle += 1

        # Wait for next cycle
        elapsed = time.time() - cycle_start
        if elapsed < interval_sec:
            time.sleep(interval_sec - elapsed)

    return results
