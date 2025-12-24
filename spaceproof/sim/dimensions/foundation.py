"""foundation.py - D1-D5 Foundation Dimensions.

Foundation dimensions handle basic validation:
    D1 - Schema Validation: Receipt structure correctness
    D2 - Bounds Check: Value ranges and limits
    D3 - Entropy Measurement: Basic Shannon entropy
    D4 - Type Validation: Data type correctness
    D5 - Format Validation: String/date/hash formats

These emerge naturally from implementing ultimate dimensions,
following the "receipts all the way down" principle.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
from datetime import datetime

from spaceproof.engine.entropy import shannon_entropy
from spaceproof.core import emit_receipt


@dataclass
class DimensionResult:
    """Result of dimension validation."""

    dimension: str
    passed: bool
    message: str
    details: Dict[str, Any]


class BaseDimension(ABC):
    """Abstract base class for dimension validators."""

    dimension_id: str
    dimension_name: str

    @abstractmethod
    def validate(self, data: Any) -> DimensionResult:
        """Validate data against this dimension.

        Args:
            data: Data to validate

        Returns:
            DimensionResult with pass/fail and details
        """
        pass


class D1_SchemaValidation(BaseDimension):
    """D1: Receipt structure correctness."""

    dimension_id = "D1"
    dimension_name = "Schema Validation"

    REQUIRED_FIELDS = ["receipt_type", "ts", "tenant_id", "payload_hash"]

    def validate(self, data: Dict) -> DimensionResult:
        """Validate receipt schema.

        Args:
            data: Receipt dict

        Returns:
            DimensionResult
        """
        if not isinstance(data, dict):
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message="Data must be a dictionary",
                details={"type": type(data).__name__},
            )

        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in data]

        if missing_fields:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message=f"Missing required fields: {missing_fields}",
                details={"missing": missing_fields, "present": list(data.keys())},
            )

        return DimensionResult(
            dimension=self.dimension_id,
            passed=True,
            message="Schema valid",
            details={"fields": list(data.keys())},
        )


class D2_BoundsCheck(BaseDimension):
    """D2: Value ranges and limits."""

    dimension_id = "D2"
    dimension_name = "Bounds Check"

    def __init__(
        self,
        bounds: Optional[Dict[str, tuple]] = None,
    ):
        """Initialize bounds checker.

        Args:
            bounds: Dict of field_name -> (min, max) tuples
        """
        self.bounds = bounds or {
            "entropy": (0.0, 1.0),
            "compression_ratio": (0.0, 1000.0),
            "coherence": (0.0, 1.0),
            "delta": (-1.0, 1.0),
        }

    def validate(self, data: Dict) -> DimensionResult:
        """Check values are within bounds.

        Args:
            data: Data dict with numeric fields

        Returns:
            DimensionResult
        """
        violations = []

        for field, (min_val, max_val) in self.bounds.items():
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        violations.append(
                            {
                                "field": field,
                                "value": value,
                                "min": min_val,
                                "max": max_val,
                            }
                        )

        if violations:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message=f"{len(violations)} bounds violations",
                details={"violations": violations},
            )

        return DimensionResult(
            dimension=self.dimension_id,
            passed=True,
            message="All values within bounds",
            details={"checked_fields": list(self.bounds.keys())},
        )


class D3_EntropyMeasurement(BaseDimension):
    """D3: Basic Shannon entropy measurement."""

    dimension_id = "D3"
    dimension_name = "Entropy Measurement"

    def __init__(self, expected_range: tuple = (0.0, 1.0)):
        """Initialize entropy checker.

        Args:
            expected_range: Expected entropy range (min, max)
        """
        self.expected_range = expected_range

    def validate(self, data: bytes) -> DimensionResult:
        """Measure entropy of data.

        Args:
            data: Bytes to measure

        Returns:
            DimensionResult with entropy measurement
        """
        if not isinstance(data, bytes):
            try:
                import json

                data = json.dumps(data, sort_keys=True, default=str).encode()
            except Exception as e:
                return DimensionResult(
                    dimension=self.dimension_id,
                    passed=False,
                    message=f"Cannot convert to bytes: {e}",
                    details={},
                )

        measurement = shannon_entropy(data)

        in_range = self.expected_range[0] <= measurement.normalized <= self.expected_range[1]

        return DimensionResult(
            dimension=self.dimension_id,
            passed=in_range,
            message=f"Entropy: {measurement.normalized:.4f}",
            details={
                "entropy": measurement.normalized,
                "entropy_bits": measurement.bits,
                "source_size": measurement.source_size,
                "in_expected_range": in_range,
            },
        )


class D4_TypeValidation(BaseDimension):
    """D4: Data type correctness."""

    dimension_id = "D4"
    dimension_name = "Type Validation"

    def __init__(self, type_spec: Optional[Dict[str, type]] = None):
        """Initialize type validator.

        Args:
            type_spec: Dict of field_name -> expected_type
        """
        self.type_spec = type_spec or {
            "receipt_type": str,
            "ts": str,
            "tenant_id": str,
            "payload_hash": str,
        }

    def validate(self, data: Dict) -> DimensionResult:
        """Validate field types.

        Args:
            data: Data dict

        Returns:
            DimensionResult
        """
        if not isinstance(data, dict):
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message="Data must be a dictionary",
                details={},
            )

        type_errors = []

        for field, expected_type in self.type_spec.items():
            if field in data:
                actual_type = type(data[field])
                if not isinstance(data[field], expected_type):
                    type_errors.append(
                        {
                            "field": field,
                            "expected": expected_type.__name__,
                            "actual": actual_type.__name__,
                        }
                    )

        if type_errors:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message=f"{len(type_errors)} type errors",
                details={"errors": type_errors},
            )

        return DimensionResult(
            dimension=self.dimension_id,
            passed=True,
            message="All types correct",
            details={"validated_fields": list(self.type_spec.keys())},
        )


class D5_FormatValidation(BaseDimension):
    """D5: String/date/hash format validation."""

    dimension_id = "D5"
    dimension_name = "Format Validation"

    # Format patterns
    HASH_PATTERN = re.compile(r"^[a-f0-9]{64}:[a-f0-9]{64}$")
    ISO_DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")

    def validate(self, data: Dict) -> DimensionResult:
        """Validate field formats.

        Args:
            data: Data dict

        Returns:
            DimensionResult
        """
        format_errors = []

        # Check payload_hash format
        if "payload_hash" in data:
            if not self.HASH_PATTERN.match(str(data["payload_hash"])):
                format_errors.append(
                    {
                        "field": "payload_hash",
                        "error": "Invalid dual-hash format (expected SHA256:BLAKE3)",
                    }
                )

        # Check timestamp format
        if "ts" in data:
            if not self.ISO_DATETIME_PATTERN.match(str(data["ts"])):
                format_errors.append(
                    {
                        "field": "ts",
                        "error": "Invalid ISO8601 datetime format",
                    }
                )

        if format_errors:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message=f"{len(format_errors)} format errors",
                details={"errors": format_errors},
            )

        return DimensionResult(
            dimension=self.dimension_id,
            passed=True,
            message="All formats valid",
            details={},
        )
