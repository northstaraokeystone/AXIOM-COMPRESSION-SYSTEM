"""intermediate.py - D6-D10 Intermediate Dimensions.

Intermediate dimensions handle cross-cutting concerns:
    D6 - Cross-Module Validation: Consistency between modules
    D7 - Pattern Recognition: Identify recurring patterns
    D8 - Temporal Consistency: Time-ordered validation
    D9 - Basic Coherence: Pattern coherence checks
    D10 - Dependency Validation: Module dependency satisfaction
"""

from typing import Dict, List
import numpy as np

from spaceproof.engine.entropy import coherence_score, COHERENCE_THRESHOLD
from spaceproof.sim.dimensions.foundation import BaseDimension, DimensionResult


class D6_CrossModuleValidation(BaseDimension):
    """D6: Consistency between modules."""

    dimension_id = "D6"
    dimension_name = "Cross-Module Validation"

    def __init__(self, expected_modules: List[str] = None):
        """Initialize cross-module validator.

        Args:
            expected_modules: List of expected module IDs
        """
        self.expected_modules = expected_modules or []

    def validate(self, data: Dict) -> DimensionResult:
        """Validate cross-module consistency.

        Args:
            data: Dict with module_results

        Returns:
            DimensionResult
        """
        module_results = data.get("module_results", data)

        if not isinstance(module_results, dict):
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message="No module results found",
                details={},
            )

        # Check expected modules present
        missing_modules = [m for m in self.expected_modules if m not in module_results]

        # Check consistency: all modules should have passed or all failed
        pass_states = []
        for module_id, result in module_results.items():
            if isinstance(result, dict):
                passed = result.get("passed", result.get("passed_slo", True))
                pass_states.append(passed)

        # Consistency: all same state
        consistent = len(set(pass_states)) <= 1 if pass_states else True

        if missing_modules:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message=f"Missing modules: {missing_modules}",
                details={"missing": missing_modules, "present": list(module_results.keys())},
            )

        return DimensionResult(
            dimension=self.dimension_id,
            passed=consistent,
            message="Modules consistent" if consistent else "Module results inconsistent",
            details={
                "modules": list(module_results.keys()),
                "consistent": consistent,
                "pass_states": pass_states,
            },
        )


class D7_PatternRecognition(BaseDimension):
    """D7: Identify recurring patterns."""

    dimension_id = "D7"
    dimension_name = "Pattern Recognition"

    def __init__(self, min_pattern_strength: float = 0.5):
        """Initialize pattern recognizer.

        Args:
            min_pattern_strength: Minimum strength to recognize pattern
        """
        self.min_pattern_strength = min_pattern_strength

    def validate(self, data: np.ndarray) -> DimensionResult:
        """Recognize patterns in data.

        Args:
            data: Numeric array

        Returns:
            DimensionResult with pattern info
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if len(data) < 10:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,  # Not enough data to recognize patterns
                message="Insufficient data for pattern recognition",
                details={"data_length": len(data)},
            )

        # FFT-based pattern detection
        fft = np.fft.fft(data - np.mean(data))
        power = np.abs(fft) ** 2
        power = power[: len(power) // 2]  # Only positive frequencies

        # Find dominant frequency
        dominant_idx = np.argmax(power[1:]) + 1  # Skip DC
        total_power = np.sum(power[1:])
        dominant_power = power[dominant_idx]

        pattern_strength = dominant_power / total_power if total_power > 0 else 0

        pattern_found = pattern_strength >= self.min_pattern_strength

        return DimensionResult(
            dimension=self.dimension_id,
            passed=True,  # Pattern recognition is informational
            message=f"Pattern strength: {pattern_strength:.3f}",
            details={
                "pattern_found": pattern_found,
                "pattern_strength": float(pattern_strength),
                "dominant_frequency": int(dominant_idx),
            },
        )


class D8_TemporalConsistency(BaseDimension):
    """D8: Time-ordered validation."""

    dimension_id = "D8"
    dimension_name = "Temporal Consistency"

    def validate(self, data: List[Dict]) -> DimensionResult:
        """Validate temporal ordering.

        Args:
            data: List of receipts with timestamps

        Returns:
            DimensionResult
        """
        if not isinstance(data, list) or len(data) < 2:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="Insufficient data for temporal validation",
                details={"count": len(data) if isinstance(data, list) else 0},
            )

        timestamps = []
        for item in data:
            if isinstance(item, dict) and "ts" in item:
                timestamps.append(item["ts"])

        if len(timestamps) < 2:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="Insufficient timestamps",
                details={"timestamp_count": len(timestamps)},
            )

        # Check monotonic ordering
        is_ordered = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))

        return DimensionResult(
            dimension=self.dimension_id,
            passed=is_ordered,
            message="Timestamps ordered" if is_ordered else "Timestamps out of order",
            details={
                "timestamp_count": len(timestamps),
                "first": timestamps[0] if timestamps else None,
                "last": timestamps[-1] if timestamps else None,
            },
        )


class D9_BasicCoherence(BaseDimension):
    """D9: Pattern coherence checks."""

    dimension_id = "D9"
    dimension_name = "Basic Coherence"

    def __init__(self, threshold: float = COHERENCE_THRESHOLD):
        """Initialize coherence checker.

        Args:
            threshold: Minimum coherence threshold
        """
        self.threshold = threshold

    def validate(self, data: np.ndarray) -> DimensionResult:
        """Check pattern coherence.

        Args:
            data: Numeric array

        Returns:
            DimensionResult with coherence info
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        result = coherence_score(data)

        return DimensionResult(
            dimension=self.dimension_id,
            passed=result.score >= self.threshold,
            message=f"Coherence: {result.score:.3f} (threshold: {self.threshold})",
            details={
                "coherence": result.score,
                "is_alive": result.is_alive,
                "autocatalytic": result.autocatalytic,
                "pattern_strength": result.pattern_strength,
            },
        )


class D10_DependencyValidation(BaseDimension):
    """D10: Module dependency satisfaction."""

    dimension_id = "D10"
    dimension_name = "Dependency Validation"

    # Module dependency graph
    DEPENDENCIES = {
        "witness": ["compress"],
        "sovereignty": ["compress", "witness"],
        "anchor": ["ledger"],
        "loop": ["compress", "sovereignty"],
    }

    def validate(self, data: Dict) -> DimensionResult:
        """Validate module dependencies are satisfied.

        Args:
            data: Dict with executed modules

        Returns:
            DimensionResult
        """
        executed_modules = set(data.get("modules", []))

        if not executed_modules:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="No modules to validate",
                details={},
            )

        missing_deps = []
        for module in executed_modules:
            if module in self.DEPENDENCIES:
                for dep in self.DEPENDENCIES[module]:
                    if dep not in executed_modules:
                        missing_deps.append({"module": module, "missing": dep})

        if missing_deps:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=False,
                message=f"Missing dependencies: {len(missing_deps)}",
                details={"missing": missing_deps},
            )

        return DimensionResult(
            dimension=self.dimension_id,
            passed=True,
            message="All dependencies satisfied",
            details={"modules": list(executed_modules)},
        )
