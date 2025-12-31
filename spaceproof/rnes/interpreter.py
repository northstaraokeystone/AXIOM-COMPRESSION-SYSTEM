"""interpreter.py - Receipt-driven execution.

Execute operations based on receipt content and history.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

RNES_TENANT = "spaceproof-rnes"


@dataclass
class ExecutionStep:
    """Single step in execution plan."""

    step_id: str
    operation: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_cost: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "estimated_cost": self.estimated_cost,
        }


@dataclass
class ExecutionPlan:
    """Plan for receipt-driven execution."""

    plan_id: str
    source_receipt_id: str
    steps: List[ExecutionStep]
    total_estimated_cost: float
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "source_receipt_id": self.source_receipt_id,
            "steps": [s.to_dict() for s in self.steps],
            "step_count": len(self.steps),
            "total_estimated_cost": self.total_estimated_cost,
            "created_at": self.created_at,
        }


@dataclass
class ExecutionResult:
    """Result of receipt-driven execution."""

    execution_id: str
    plan_id: str
    steps_executed: int
    steps_successful: int
    total_actual_cost: float
    outputs: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "plan_id": self.plan_id,
            "steps_executed": self.steps_executed,
            "steps_successful": self.steps_successful,
            "total_actual_cost": self.total_actual_cost,
            "success": self.success,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# Operation registry
_operations: Dict[str, Callable] = {}


def register_operation(name: str, func: Callable) -> None:
    """Register an operation for receipt-driven execution.

    Args:
        name: Operation name
        func: Operation function
    """
    _operations[name] = func


def interpret_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Interpret a receipt to extract execution intent.

    Args:
        receipt: Receipt to interpret

    Returns:
        Execution intent
    """
    receipt_type = receipt.get("receipt_type", "unknown")

    intent = {
        "receipt_id": receipt.get("receipt_id", receipt.get("payload_hash", "")),
        "receipt_type": receipt_type,
        "operation": None,
        "parameters": {},
        "can_execute": False,
    }

    # Map receipt types to operations
    operation_map = {
        "data_ingest": "ingest",
        "compute_inference": "inference",
        "maneuver_audit": "audit",
        "human_intervention": "apply_correction",
        "training_example": "add_to_training",
        "redaction": "apply_redaction",
        "sync": "synchronize",
    }

    if receipt_type in operation_map:
        intent["operation"] = operation_map[receipt_type]
        intent["parameters"] = {k: v for k, v in receipt.items() if k not in ["receipt_type", "ts", "tenant_id", "payload_hash"]}
        intent["can_execute"] = intent["operation"] in _operations

    return intent


def build_execution_plan(
    receipts: List[Dict[str, Any]],
) -> ExecutionPlan:
    """Build execution plan from receipts.

    Args:
        receipts: List of receipts to plan from

    Returns:
        ExecutionPlan
    """
    steps = []
    total_cost = 0.0

    for i, receipt in enumerate(receipts):
        intent = interpret_receipt(receipt)

        if intent["can_execute"]:
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                operation=intent["operation"],
                parameters=intent["parameters"],
                dependencies=[steps[i - 1].step_id] if i > 0 else [],
                estimated_cost=0.01,  # Default cost estimate
            )
            steps.append(step)
            total_cost += step.estimated_cost

    return ExecutionPlan(
        plan_id=str(uuid.uuid4()),
        source_receipt_id=receipts[0].get("receipt_id", "") if receipts else "",
        steps=steps,
        total_estimated_cost=total_cost,
    )


def execute_from_receipt(
    receipt: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> ExecutionResult:
    """Execute operation from a single receipt.

    Args:
        receipt: Receipt to execute from
        context: Optional execution context

    Returns:
        ExecutionResult
    """
    execution_id = str(uuid.uuid4())
    intent = interpret_receipt(receipt)

    if not intent["can_execute"]:
        return ExecutionResult(
            execution_id=execution_id,
            plan_id="",
            steps_executed=0,
            steps_successful=0,
            total_actual_cost=0.0,
            outputs={},
            success=False,
            error=f"Operation '{intent['operation']}' not registered",
            completed_at=datetime.utcnow().isoformat() + "Z",
        )

    # Execute operation
    operation = _operations[intent["operation"]]
    try:
        result = operation(intent["parameters"], context or {})
        outputs = {"result": result}
        success = True
        error = None
    except Exception as e:
        outputs = {}
        success = False
        error = str(e)

    return ExecutionResult(
        execution_id=execution_id,
        plan_id="",
        steps_executed=1,
        steps_successful=1 if success else 0,
        total_actual_cost=0.01,
        outputs=outputs,
        success=success,
        error=error,
        completed_at=datetime.utcnow().isoformat() + "Z",
    )


def emit_execution_receipt(result: ExecutionResult) -> Dict[str, Any]:
    """Emit execution receipt.

    Args:
        result: ExecutionResult to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "execution",
        {
            "tenant_id": RNES_TENANT,
            **result.to_dict(),
        },
    )


def clear_operations() -> None:
    """Clear registered operations (for testing)."""
    global _operations
    _operations = {}


# Register default operations
def _default_ingest(params: Dict, context: Dict) -> Dict:
    """Default ingest operation."""
    return {"ingested": True, "params": params}


def _default_inference(params: Dict, context: Dict) -> Dict:
    """Default inference operation."""
    return {"inferred": True, "params": params}


register_operation("ingest", _default_ingest)
register_operation("inference", _default_inference)
