"""saga.py - Saga Pattern for Distributed Transaction Compensation.

THE SAGA PARADIGM:
    Each step in a validation sequence has a compensating action.
    If any step fails, compensation runs in reverse order.
    This ensures idempotent, retryable transactions.

    The Saga orchestrator:
    1. Executes steps forward
    2. On failure, executes compensations backward
    3. Ensures eventual consistency

Source: SpaceProof D20 Production Evolution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
import time
import logging

from spaceproof.core import emit_receipt

# === CONSTANTS ===

TENANT_ID = "spaceproof-saga"

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_MS = 100
DEFAULT_RETRY_BACKOFF = 2.0

logger = logging.getLogger(__name__)


class SagaStatus(Enum):
    """Status of a saga execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


@dataclass
class StepResult:
    """Result of a single saga step."""

    step_id: str
    status: SagaStatus
    duration_ms: float
    data: Any = None
    error: Optional[str] = None
    retries: int = 0


T = TypeVar("T")


class SagaStep(ABC, Generic[T]):
    """Abstract base class for saga steps."""

    def __init__(
        self,
        step_id: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_ms: float = DEFAULT_RETRY_DELAY_MS,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ):
        """Initialize saga step.

        Args:
            step_id: Unique identifier for this step
            max_retries: Maximum retry attempts
            retry_delay_ms: Initial retry delay in milliseconds
            retry_backoff: Backoff multiplier for retries
        """
        self.step_id = step_id
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.retry_backoff = retry_backoff
        self.status = SagaStatus.PENDING

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> T:
        """Execute the step forward.

        Args:
            context: Shared saga context

        Returns:
            Step result data

        Raises:
            Exception: If step fails
        """
        pass

    @abstractmethod
    def compensate(self, context: Dict[str, Any], result: T) -> bool:
        """Execute compensating transaction.

        Args:
            context: Shared saga context
            result: Result from execute()

        Returns:
            True if compensation succeeded
        """
        pass

    def execute_with_retry(self, context: Dict[str, Any]) -> StepResult:
        """Execute step with retry logic.

        Args:
            context: Shared saga context

        Returns:
            StepResult with execution details
        """
        start = time.time()
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                self.status = SagaStatus.RUNNING
                result = self.execute(context)
                self.status = SagaStatus.COMPLETED

                duration = (time.time() - start) * 1000
                return StepResult(
                    step_id=self.step_id,
                    status=SagaStatus.COMPLETED,
                    duration_ms=duration,
                    data=result,
                    retries=retries,
                )

            except Exception as e:
                last_error = str(e)
                retries += 1

                if retries <= self.max_retries:
                    delay = self.retry_delay_ms * (self.retry_backoff ** (retries - 1))
                    time.sleep(delay / 1000)

        self.status = SagaStatus.FAILED
        duration = (time.time() - start) * 1000

        return StepResult(
            step_id=self.step_id,
            status=SagaStatus.FAILED,
            duration_ms=duration,
            error=last_error,
            retries=retries,
        )


class FunctionStep(SagaStep[Any]):
    """Saga step implemented via functions."""

    def __init__(
        self,
        step_id: str,
        execute_fn: Callable[[Dict[str, Any]], Any],
        compensate_fn: Callable[[Dict[str, Any], Any], bool],
        **kwargs,
    ):
        super().__init__(step_id, **kwargs)
        self.execute_fn = execute_fn
        self.compensate_fn = compensate_fn

    def execute(self, context: Dict[str, Any]) -> Any:
        return self.execute_fn(context)

    def compensate(self, context: Dict[str, Any], result: Any) -> bool:
        return self.compensate_fn(context, result)


@dataclass
class SagaResult:
    """Result of saga execution."""

    saga_id: str
    status: SagaStatus
    total_duration_ms: float
    step_results: List[StepResult]
    failed_step: Optional[str] = None
    compensation_results: List[StepResult] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class Saga:
    """Saga orchestrator for distributed transaction management."""

    def __init__(self, saga_id: str, steps: List[SagaStep]):
        """Initialize saga.

        Args:
            saga_id: Unique identifier for this saga
            steps: Ordered list of saga steps
        """
        self.saga_id = saga_id
        self.steps = steps
        self.status = SagaStatus.PENDING
        self.executed_steps: List[tuple[SagaStep, StepResult]] = []

    def execute(self, initial_context: Optional[Dict[str, Any]] = None) -> SagaResult:
        """Execute saga with automatic compensation on failure.

        Args:
            initial_context: Initial saga context

        Returns:
            SagaResult with execution details
        """
        context = initial_context or {}
        step_results = []
        start_time = time.time()
        failed_step = None

        self.status = SagaStatus.RUNNING

        # Execute steps forward
        for step in self.steps:
            result = step.execute_with_retry(context)
            step_results.append(result)
            self.executed_steps.append((step, result))

            if result.status == SagaStatus.COMPLETED:
                # Store result in context for subsequent steps
                context[f"step_{step.step_id}_result"] = result.data
            else:
                failed_step = step.step_id
                break

        total_duration = (time.time() - start_time) * 1000

        # If failed, run compensation
        if failed_step:
            self.status = SagaStatus.COMPENSATING
            compensation_results = self._compensate(context)

            if all(r.status == SagaStatus.COMPENSATED for r in compensation_results):
                self.status = SagaStatus.COMPENSATED
            else:
                self.status = SagaStatus.COMPENSATION_FAILED

            # Emit saga receipt
            self._emit_saga_receipt(
                step_results,
                compensation_results,
                failed_step,
                total_duration,
            )

            return SagaResult(
                saga_id=self.saga_id,
                status=self.status,
                total_duration_ms=total_duration,
                step_results=step_results,
                failed_step=failed_step,
                compensation_results=compensation_results,
                context=context,
            )

        self.status = SagaStatus.COMPLETED

        # Emit success receipt
        self._emit_saga_receipt(step_results, [], None, total_duration)

        return SagaResult(
            saga_id=self.saga_id,
            status=self.status,
            total_duration_ms=total_duration,
            step_results=step_results,
            context=context,
        )

    def _compensate(self, context: Dict[str, Any]) -> List[StepResult]:
        """Run compensation in reverse order.

        Args:
            context: Current saga context

        Returns:
            List of compensation results
        """
        compensation_results = []

        # Compensate in reverse order
        for step, original_result in reversed(self.executed_steps):
            if original_result.status == SagaStatus.COMPLETED:
                start = time.time()
                try:
                    success = step.compensate(context, original_result.data)
                    duration = (time.time() - start) * 1000

                    compensation_results.append(
                        StepResult(
                            step_id=f"compensate_{step.step_id}",
                            status=SagaStatus.COMPENSATED if success else SagaStatus.COMPENSATION_FAILED,
                            duration_ms=duration,
                            data={"original_step": step.step_id},
                        )
                    )

                except Exception as e:
                    duration = (time.time() - start) * 1000
                    compensation_results.append(
                        StepResult(
                            step_id=f"compensate_{step.step_id}",
                            status=SagaStatus.COMPENSATION_FAILED,
                            duration_ms=duration,
                            error=str(e),
                        )
                    )

        return compensation_results

    def _emit_saga_receipt(
        self,
        step_results: List[StepResult],
        compensation_results: List[StepResult],
        failed_step: Optional[str],
        total_duration: float,
    ) -> None:
        """Emit receipt for saga execution.

        Args:
            step_results: Results from forward execution
            compensation_results: Results from compensation
            failed_step: ID of failed step (if any)
            total_duration: Total execution time
        """
        emit_receipt(
            "saga",
            {
                "tenant_id": TENANT_ID,
                "saga_id": self.saga_id,
                "status": self.status.value,
                "total_duration_ms": total_duration,
                "steps_executed": len(step_results),
                "steps_completed": sum(1 for r in step_results if r.status == SagaStatus.COMPLETED),
                "failed_step": failed_step,
                "compensations_executed": len(compensation_results),
                "compensations_succeeded": sum(1 for r in compensation_results if r.status == SagaStatus.COMPENSATED),
            },
        )


class SagaBuilder:
    """Builder for constructing sagas with fluent API."""

    def __init__(self, saga_id: str):
        """Initialize saga builder.

        Args:
            saga_id: Unique identifier for the saga
        """
        self.saga_id = saga_id
        self.steps: List[SagaStep] = []

    def add_step(
        self,
        step_id: str,
        execute_fn: Callable[[Dict[str, Any]], Any],
        compensate_fn: Callable[[Dict[str, Any], Any], bool],
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> "SagaBuilder":
        """Add a step to the saga.

        Args:
            step_id: Unique identifier for the step
            execute_fn: Function to execute step
            compensate_fn: Function to compensate step
            max_retries: Maximum retry attempts

        Returns:
            Self for chaining
        """
        self.steps.append(
            FunctionStep(
                step_id=step_id,
                execute_fn=execute_fn,
                compensate_fn=compensate_fn,
                max_retries=max_retries,
            )
        )
        return self

    def build(self) -> Saga:
        """Build the saga.

        Returns:
            Configured Saga instance
        """
        return Saga(self.saga_id, self.steps)


# === VALIDATION SAGA IMPLEMENTATIONS ===


def create_validation_saga(
    saga_id: str,
    module_configs: List[Dict[str, Any]],
) -> Saga:
    """Create a validation saga for module execution.

    Args:
        saga_id: Saga identifier
        module_configs: List of module configurations with:
            - module_id: str
            - validate_fn: Callable
            - compensate_fn: Callable

    Returns:
        Configured Saga for validation
    """
    builder = SagaBuilder(saga_id)

    for config in module_configs:
        module_id = config["module_id"]

        def make_execute(cfg):
            def execute(ctx):
                result = cfg["validate_fn"](ctx.get("input_data"))
                ctx[f"{cfg['module_id']}_result"] = result
                return result

            return execute

        def make_compensate(cfg):
            def compensate(ctx, result):
                if cfg.get("compensate_fn"):
                    return cfg["compensate_fn"](ctx, result)
                return True

            return compensate

        builder.add_step(
            step_id=module_id,
            execute_fn=make_execute(config),
            compensate_fn=make_compensate(config),
        )

    return builder.build()


def create_entropy_pump_saga(saga_id: str) -> Saga:
    """Create saga for entropy pump validation cycle.

    Implements the full entropy reduction pipeline:
    1. Measure initial entropy
    2. Execute compression
    3. Verify entropy reduction
    4. Check coherence
    5. Generate receipt

    Args:
        saga_id: Saga identifier

    Returns:
        Configured Saga for entropy pump cycle
    """
    from spaceproof.engine.entropy import shannon_entropy, coherence_score
    import numpy as np

    def measure_entropy(ctx):
        data = ctx.get("input_data", b"")
        if isinstance(data, bytes):
            measurement = shannon_entropy(data)
        else:
            import json

            data_bytes = json.dumps(data, sort_keys=True, default=str).encode()
            measurement = shannon_entropy(data_bytes)
        ctx["entropy_before"] = measurement.normalized
        return measurement

    def compensate_entropy(ctx, result):
        ctx.pop("entropy_before", None)
        return True

    def execute_compression(ctx):
        import zlib

        data = ctx.get("input_data", b"")
        if not isinstance(data, bytes):
            import json

            data = json.dumps(data, sort_keys=True, default=str).encode()
        compressed = zlib.compress(data, level=9)
        ctx["compressed_data"] = compressed
        ctx["compression_ratio"] = len(data) / len(compressed)
        return {"original_size": len(data), "compressed_size": len(compressed)}

    def compensate_compression(ctx, result):
        ctx.pop("compressed_data", None)
        ctx.pop("compression_ratio", None)
        return True

    def verify_reduction(ctx):
        compressed = ctx.get("compressed_data", b"")
        measurement = shannon_entropy(compressed)
        ctx["entropy_after"] = measurement.normalized
        delta = ctx.get("entropy_before", 0) - measurement.normalized
        ctx["entropy_delta"] = delta
        if delta < 0:
            raise ValueError(f"Negative entropy delta: {delta}")
        return {"delta": delta, "is_pumping": delta > 0}

    def compensate_reduction(ctx, result):
        ctx.pop("entropy_after", None)
        ctx.pop("entropy_delta", None)
        return True

    def check_coherence(ctx):
        values = [
            ctx.get("entropy_before", 0),
            ctx.get("entropy_after", 0),
            ctx.get("compression_ratio", 1),
        ]
        pattern = np.array(values)
        result = coherence_score(pattern)
        ctx["coherence"] = result.score
        ctx["is_alive"] = result.is_alive
        return {"coherence": result.score, "is_alive": result.is_alive}

    def compensate_coherence(ctx, result):
        ctx.pop("coherence", None)
        ctx.pop("is_alive", None)
        return True

    def generate_receipt(ctx):
        from spaceproof.core import emit_receipt

        receipt = emit_receipt(
            "entropy_pump_cycle",
            {
                "tenant_id": TENANT_ID,
                "entropy_before": ctx.get("entropy_before"),
                "entropy_after": ctx.get("entropy_after"),
                "entropy_delta": ctx.get("entropy_delta"),
                "compression_ratio": ctx.get("compression_ratio"),
                "coherence": ctx.get("coherence"),
                "is_alive": ctx.get("is_alive"),
            },
        )
        ctx["receipt"] = receipt
        return receipt

    def compensate_receipt(ctx, result):
        # Receipts are immutable, but we can mark it as reverted
        ctx["receipt_reverted"] = True
        return True

    return (
        SagaBuilder(saga_id)
        .add_step("measure_entropy", measure_entropy, compensate_entropy)
        .add_step("compress", execute_compression, compensate_compression)
        .add_step("verify_reduction", verify_reduction, compensate_reduction)
        .add_step("check_coherence", check_coherence, compensate_coherence)
        .add_step("generate_receipt", generate_receipt, compensate_receipt)
        .build()
    )
