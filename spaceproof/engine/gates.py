"""gates.py - Validation Gate Orchestration.

THE GATE PARADIGM:
    Gates execute in defined order mirroring entropy reduction pipeline.
    Gates are composable using AND/OR logic.
    Saga-pattern rollback enables recovery.

Gate sequence follows entropy flow:
    1. PRE-VALIDATION: Infrastructure health, resource availability
    2. ENTROPY MEASUREMENT: Compute H(input), validate budget
    3. MODULE GATES: Each of 3 modules executes validation
    4. COMPRESSION GATE: Verify entropy reduction (delta > 0)
    5. COHERENCE GATE: Check autocatalytic patterns (tau >= 0.7)
    6. MERKLE CONSTRUCTION: Build proof tree
    7. SIGNATURE GATE: Sign with domain key
    8. POST-VALIDATION: Audit logging, compliance

Source: SpaceProof D20 Production Evolution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

from spaceproof.core import emit_receipt

from spaceproof.engine.entropy import (
    shannon_entropy,
    coherence_score,
    COHERENCE_THRESHOLD,
)

# === CONSTANTS ===

TENANT_ID = "spaceproof-gates"


class GateStatus(Enum):
    """Status of a gate execution."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    COMPENSATED = "compensated"


class GateType(Enum):
    """Type of gate in the sequence."""

    PRE_VALIDATION = "pre_validation"
    ENTROPY_MEASUREMENT = "entropy_measurement"
    MODULE = "module"
    COMPRESSION = "compression"
    COHERENCE = "coherence"
    MERKLE = "merkle"
    SIGNATURE = "signature"
    POST_VALIDATION = "post_validation"


@dataclass
class GateResult:
    """Result of executing a single gate."""

    gate_id: str
    gate_type: GateType
    status: GateStatus
    duration_ms: float
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class GateContext:
    """Context passed through gate sequence."""

    input_data: Any
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    module_results: Dict[str, Any] = field(default_factory=dict)
    merkle_root: Optional[str] = None
    signature: Optional[bytes] = None
    receipts: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Gate(ABC):
    """Abstract base class for validation gates."""

    def __init__(self, gate_id: str, gate_type: GateType, blocking: bool = True):
        """Initialize gate.

        Args:
            gate_id: Unique identifier for this gate
            gate_type: Type of gate
            blocking: If True, failure stops the sequence
        """
        self.gate_id = gate_id
        self.gate_type = gate_type
        self.blocking = blocking
        self.status = GateStatus.PENDING

    @abstractmethod
    def execute(self, context: GateContext) -> GateResult:
        """Execute the gate validation.

        Args:
            context: Current gate context

        Returns:
            GateResult with status and data
        """
        pass

    @abstractmethod
    def compensate(self, context: GateContext) -> bool:
        """Compensating transaction for rollback.

        Args:
            context: Current gate context

        Returns:
            True if compensation succeeded
        """
        pass


class PreValidationGate(Gate):
    """Infrastructure health checks and resource availability."""

    def __init__(
        self,
        gate_id: str = "pre_validation",
        health_checks: Optional[List[Callable[[], bool]]] = None,
    ):
        super().__init__(gate_id, GateType.PRE_VALIDATION)
        self.health_checks = health_checks or []

    def execute(self, context: GateContext) -> GateResult:
        start = time.time()

        # Run health checks
        failed_checks = []
        for i, check in enumerate(self.health_checks):
            try:
                if not check():
                    failed_checks.append(f"check_{i}")
            except Exception as e:
                failed_checks.append(f"check_{i}: {str(e)}")

        duration = (time.time() - start) * 1000

        if failed_checks:
            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.FAILED,
                duration_ms=duration,
                message=f"Health checks failed: {', '.join(failed_checks)}",
                error=str(failed_checks),
            )

        return GateResult(
            gate_id=self.gate_id,
            gate_type=self.gate_type,
            status=GateStatus.PASSED,
            duration_ms=duration,
            message="All health checks passed",
            data={"checks_passed": len(self.health_checks)},
        )

    def compensate(self, context: GateContext) -> bool:
        # No compensation needed for pre-validation
        return True


class EntropyMeasurementGate(Gate):
    """Compute H(input) and validate entropy budget."""

    def __init__(
        self,
        gate_id: str = "entropy_measurement",
        max_entropy: float = 1.0,
    ):
        super().__init__(gate_id, GateType.ENTROPY_MEASUREMENT)
        self.max_entropy = max_entropy

    def execute(self, context: GateContext) -> GateResult:
        start = time.time()

        try:
            # Measure input entropy
            if isinstance(context.input_data, bytes):
                measurement = shannon_entropy(context.input_data)
            elif hasattr(context.input_data, "tobytes"):
                measurement = shannon_entropy(context.input_data.tobytes())
            else:
                # Serialize and measure
                import json

                data_bytes = json.dumps(context.input_data, sort_keys=True, default=str).encode()
                measurement = shannon_entropy(data_bytes)

            context.entropy_before = measurement.normalized

            duration = (time.time() - start) * 1000

            if measurement.normalized > self.max_entropy:
                return GateResult(
                    gate_id=self.gate_id,
                    gate_type=self.gate_type,
                    status=GateStatus.FAILED,
                    duration_ms=duration,
                    message=f"Entropy {measurement.normalized:.3f} exceeds budget {self.max_entropy}",
                    data={"entropy": measurement.normalized},
                    error="entropy_budget_exceeded",
                )

            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.PASSED,
                duration_ms=duration,
                message=f"Entropy measured: {measurement.normalized:.3f}",
                data={
                    "entropy": measurement.normalized,
                    "entropy_bits": measurement.bits,
                    "source_size": measurement.source_size,
                },
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.FAILED,
                duration_ms=duration,
                message=f"Entropy measurement failed: {str(e)}",
                error=str(e),
            )

    def compensate(self, context: GateContext) -> bool:
        context.entropy_before = 0.0
        return True


class ModuleGate(Gate):
    """Execute a validation module."""

    def __init__(
        self,
        gate_id: str,
        module_id: str,
        validate_fn: Callable[[Any], Dict],
        compensate_fn: Optional[Callable[[Dict], bool]] = None,
    ):
        super().__init__(gate_id, GateType.MODULE)
        self.module_id = module_id
        self.validate_fn = validate_fn
        self.compensate_fn = compensate_fn

    def execute(self, context: GateContext) -> GateResult:
        start = time.time()

        try:
            result = self.validate_fn(context.input_data)
            context.module_results[self.module_id] = result

            duration = (time.time() - start) * 1000

            passed = result.get("passed", result.get("passed_slo", True))

            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                duration_ms=duration,
                message=f"Module {self.module_id}: {'passed' if passed else 'failed'}",
                data={"module_id": self.module_id, "result": result},
                error=None if passed else f"Module {self.module_id} validation failed",
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.FAILED,
                duration_ms=duration,
                message=f"Module {self.module_id} error: {str(e)}",
                error=str(e),
            )

    def compensate(self, context: GateContext) -> bool:
        if self.compensate_fn and self.module_id in context.module_results:
            return self.compensate_fn(context.module_results[self.module_id])
        return True


class CompressionGate(Gate):
    """Verify entropy reduction occurred (delta > 0)."""

    def __init__(
        self,
        gate_id: str = "compression",
        min_delta: float = 0.0,
    ):
        super().__init__(gate_id, GateType.COMPRESSION)
        self.min_delta = min_delta

    def execute(self, context: GateContext) -> GateResult:
        start = time.time()

        try:
            # Measure output entropy
            output_data = context.module_results
            import json

            data_bytes = json.dumps(output_data, sort_keys=True, default=str).encode()
            measurement = shannon_entropy(data_bytes)

            context.entropy_after = measurement.normalized
            delta = context.entropy_before - context.entropy_after

            duration = (time.time() - start) * 1000

            if delta < self.min_delta:
                return GateResult(
                    gate_id=self.gate_id,
                    gate_type=self.gate_type,
                    status=GateStatus.FAILED,
                    duration_ms=duration,
                    message=f"Entropy delta {delta:.3f} < minimum {self.min_delta}",
                    data={"delta": delta, "before": context.entropy_before, "after": context.entropy_after},
                    error="insufficient_compression",
                )

            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.PASSED,
                duration_ms=duration,
                message=f"Entropy reduction achieved: delta={delta:.3f}",
                data={
                    "delta": delta,
                    "before": context.entropy_before,
                    "after": context.entropy_after,
                    "is_pumping": delta > 0,
                },
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.FAILED,
                duration_ms=duration,
                message=f"Compression check failed: {str(e)}",
                error=str(e),
            )

    def compensate(self, context: GateContext) -> bool:
        return True


class CoherenceGate(Gate):
    """Check autocatalytic patterns (tau >= 0.7)."""

    def __init__(
        self,
        gate_id: str = "coherence",
        threshold: float = COHERENCE_THRESHOLD,
    ):
        super().__init__(gate_id, GateType.COHERENCE)
        self.threshold = threshold

    def execute(self, context: GateContext) -> GateResult:
        start = time.time()

        try:
            import numpy as np

            # Extract pattern from module results
            values = []
            for module_id, result in context.module_results.items():
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, (int, float)):
                            values.append(v)

            if len(values) < 2:
                duration = (time.time() - start) * 1000
                return GateResult(
                    gate_id=self.gate_id,
                    gate_type=self.gate_type,
                    status=GateStatus.SKIPPED,
                    duration_ms=duration,
                    message="Insufficient data for coherence check",
                    data={"values_count": len(values)},
                )

            pattern = np.array(values)
            result = coherence_score(pattern)

            duration = (time.time() - start) * 1000

            if result.score < self.threshold:
                return GateResult(
                    gate_id=self.gate_id,
                    gate_type=self.gate_type,
                    status=GateStatus.FAILED,
                    duration_ms=duration,
                    message=f"Coherence {result.score:.3f} < threshold {self.threshold}",
                    data={
                        "coherence": result.score,
                        "is_alive": result.is_alive,
                        "autocatalytic": result.autocatalytic,
                    },
                    error="low_coherence",
                )

            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.PASSED,
                duration_ms=duration,
                message=f"Coherence check passed: tau={result.score:.3f}",
                data={
                    "coherence": result.score,
                    "is_alive": result.is_alive,
                    "autocatalytic": result.autocatalytic,
                    "pattern_strength": result.pattern_strength,
                },
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.FAILED,
                duration_ms=duration,
                message=f"Coherence check error: {str(e)}",
                error=str(e),
            )

    def compensate(self, context: GateContext) -> bool:
        return True


class MerkleGate(Gate):
    """Build proof tree from module outputs."""

    def __init__(self, gate_id: str = "merkle"):
        super().__init__(gate_id, GateType.MERKLE)

    def execute(self, context: GateContext) -> GateResult:
        start = time.time()

        try:
            from spaceproof.core import merkle

            # Build Merkle tree from module results
            items = list(context.module_results.values())
            root = merkle(items)

            context.merkle_root = root

            duration = (time.time() - start) * 1000

            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.PASSED,
                duration_ms=duration,
                message=f"Merkle root computed: {root[:16]}...",
                data={"merkle_root": root, "item_count": len(items)},
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.FAILED,
                duration_ms=duration,
                message=f"Merkle construction failed: {str(e)}",
                error=str(e),
            )

    def compensate(self, context: GateContext) -> bool:
        context.merkle_root = None
        return True


class SignatureGate(Gate):
    """Sign receipt with domain-specific key."""

    def __init__(
        self,
        gate_id: str = "signature",
        sign_fn: Optional[Callable[[bytes], bytes]] = None,
    ):
        super().__init__(gate_id, GateType.SIGNATURE)
        self.sign_fn = sign_fn

    def execute(self, context: GateContext) -> GateResult:
        start = time.time()

        try:
            import hashlib
            import json

            # Create payload to sign
            payload = {
                "merkle_root": context.merkle_root,
                "entropy_delta": context.entropy_before - context.entropy_after,
                "module_count": len(context.module_results),
            }
            payload_bytes = json.dumps(payload, sort_keys=True).encode()

            if self.sign_fn:
                signature = self.sign_fn(payload_bytes)
            else:
                # Default: use SHA256 as placeholder signature
                signature = hashlib.sha256(payload_bytes).digest()

            context.signature = signature

            duration = (time.time() - start) * 1000

            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.PASSED,
                duration_ms=duration,
                message=f"Signed: {signature.hex()[:16]}...",
                data={"signature": signature.hex()},
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.FAILED,
                duration_ms=duration,
                message=f"Signing failed: {str(e)}",
                error=str(e),
            )

    def compensate(self, context: GateContext) -> bool:
        context.signature = None
        return True


class PostValidationGate(Gate):
    """Audit logging and compliance checks."""

    def __init__(
        self,
        gate_id: str = "post_validation",
        audit_fn: Optional[Callable[[GateContext], None]] = None,
    ):
        super().__init__(gate_id, GateType.POST_VALIDATION, blocking=False)
        self.audit_fn = audit_fn

    def execute(self, context: GateContext) -> GateResult:
        start = time.time()

        try:
            # Emit audit receipt
            audit_data = {
                "tenant_id": TENANT_ID,
                "entropy_before": context.entropy_before,
                "entropy_after": context.entropy_after,
                "entropy_delta": context.entropy_before - context.entropy_after,
                "modules_executed": list(context.module_results.keys()),
                "merkle_root": context.merkle_root,
                "has_signature": context.signature is not None,
            }

            emit_receipt("gate_audit", audit_data)

            if self.audit_fn:
                self.audit_fn(context)

            duration = (time.time() - start) * 1000

            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.PASSED,
                duration_ms=duration,
                message="Audit logging complete",
                data=audit_data,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return GateResult(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                status=GateStatus.FAILED,
                duration_ms=duration,
                message=f"Audit logging failed: {str(e)}",
                error=str(e),
            )

    def compensate(self, context: GateContext) -> bool:
        return True


@dataclass
class GateSequenceResult:
    """Result of executing a full gate sequence."""

    passed: bool
    results: List[GateResult]
    total_duration_ms: float
    failed_gate: Optional[str] = None
    compensated: bool = False
    context: Optional[GateContext] = None


class GateOrchestrator:
    """Orchestrates execution of gate sequence with rollback support."""

    def __init__(self, gates: List[Gate]):
        """Initialize orchestrator with gate sequence.

        Args:
            gates: Ordered list of gates to execute
        """
        self.gates = gates
        self.executed_gates: List[Tuple[Gate, GateResult]] = []

    def execute(self, input_data: Any) -> GateSequenceResult:
        """Execute full gate sequence.

        Args:
            input_data: Input data for validation

        Returns:
            GateSequenceResult with all results
        """
        context = GateContext(input_data=input_data)
        results = []
        start_time = time.time()
        failed_gate = None

        for gate in self.gates:
            result = gate.execute(context)
            results.append(result)
            self.executed_gates.append((gate, result))

            if result.status == GateStatus.FAILED and gate.blocking:
                failed_gate = gate.gate_id
                break

        total_duration = (time.time() - start_time) * 1000

        if failed_gate:
            # Perform compensation
            compensated = self._compensate(context)
            return GateSequenceResult(
                passed=False,
                results=results,
                total_duration_ms=total_duration,
                failed_gate=failed_gate,
                compensated=compensated,
                context=context,
            )

        return GateSequenceResult(
            passed=True,
            results=results,
            total_duration_ms=total_duration,
            context=context,
        )

    def _compensate(self, context: GateContext) -> bool:
        """Run compensation in reverse order.

        Args:
            context: Current gate context

        Returns:
            True if all compensations succeeded
        """
        all_compensated = True

        # Compensate in reverse order
        for gate, result in reversed(self.executed_gates):
            if result.status in [GateStatus.PASSED, GateStatus.FAILED]:
                try:
                    if not gate.compensate(context):
                        all_compensated = False
                except Exception:
                    all_compensated = False

        return all_compensated


def create_standard_gate_sequence(
    module_gates: List[Tuple[str, Callable[[Any], Dict]]],
    health_checks: Optional[List[Callable[[], bool]]] = None,
    sign_fn: Optional[Callable[[bytes], bytes]] = None,
    audit_fn: Optional[Callable[[GateContext], None]] = None,
) -> List[Gate]:
    """Create standard 8-gate sequence.

    Args:
        module_gates: List of (module_id, validate_fn) tuples
        health_checks: Optional health check functions
        sign_fn: Optional signing function
        audit_fn: Optional audit function

    Returns:
        Ordered list of Gate instances
    """
    gates = [
        PreValidationGate(health_checks=health_checks or []),
        EntropyMeasurementGate(),
    ]

    # Add module gates
    for i, (module_id, validate_fn) in enumerate(module_gates):
        gates.append(
            ModuleGate(
                gate_id=f"module_{i}_{module_id}",
                module_id=module_id,
                validate_fn=validate_fn,
            )
        )

    gates.extend(
        [
            CompressionGate(),
            CoherenceGate(),
            MerkleGate(),
            SignatureGate(sign_fn=sign_fn),
            PostValidationGate(audit_fn=audit_fn),
        ]
    )

    return gates
