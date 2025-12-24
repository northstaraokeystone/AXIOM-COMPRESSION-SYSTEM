"""advanced.py - D11-D14 Advanced Dimensions.

Advanced dimensions handle complex validation:
    D11 - Complex Patterns: Multi-variable pattern detection
    D12 - Multi-Step Verification: Chained validation steps
    D13 - Anomaly Detection: Statistical outlier detection
    D14 - Statistical Validation: Distribution testing
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from spaceproof.engine.entropy import shannon_entropy, FRAUD_SIGNAL_THRESHOLD, RANDOM_SIGNAL_THRESHOLD
from spaceproof.sim.dimensions.foundation import BaseDimension, DimensionResult


class D11_ComplexPatterns(BaseDimension):
    """D11: Multi-variable pattern detection."""

    dimension_id = "D11"
    dimension_name = "Complex Patterns"

    def __init__(self, correlation_threshold: float = 0.7):
        """Initialize complex pattern detector.

        Args:
            correlation_threshold: Threshold for significant correlation
        """
        self.correlation_threshold = correlation_threshold

    def validate(self, data: Dict[str, np.ndarray]) -> DimensionResult:
        """Detect complex multi-variable patterns.

        Args:
            data: Dict of variable_name -> array

        Returns:
            DimensionResult with pattern info
        """
        if not isinstance(data, dict) or len(data) < 2:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="Insufficient variables for complex pattern detection",
                details={"variable_count": len(data) if isinstance(data, dict) else 0},
            )

        # Convert to arrays
        arrays = {}
        for name, arr in data.items():
            if isinstance(arr, np.ndarray):
                arrays[name] = arr
            elif isinstance(arr, (list, tuple)):
                arrays[name] = np.array(arr)

        if len(arrays) < 2:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="Insufficient numeric variables",
                details={},
            )

        # Compute correlation matrix
        names = list(arrays.keys())
        n_vars = len(names)
        min_len = min(len(arr) for arr in arrays.values())

        # Truncate to same length
        matrix = np.column_stack([arrays[name][:min_len] for name in names])
        correlations = np.corrcoef(matrix.T)

        # Find significant correlations
        significant_pairs = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr = correlations[i, j]
                if abs(corr) >= self.correlation_threshold:
                    significant_pairs.append(
                        {
                            "var1": names[i],
                            "var2": names[j],
                            "correlation": float(corr),
                        }
                    )

        complex_pattern_found = len(significant_pairs) > 0

        return DimensionResult(
            dimension=self.dimension_id,
            passed=True,  # Informational
            message=f"Found {len(significant_pairs)} significant correlations",
            details={
                "pattern_found": complex_pattern_found,
                "correlations": significant_pairs,
                "variables": names,
            },
        )


class D12_MultiStepVerification(BaseDimension):
    """D12: Chained validation steps."""

    dimension_id = "D12"
    dimension_name = "Multi-Step Verification"

    def __init__(self, steps: List[callable] = None):
        """Initialize multi-step verifier.

        Args:
            steps: List of validation functions
        """
        self.steps = steps or []

    def validate(self, data: Any) -> DimensionResult:
        """Run chained validation steps.

        Args:
            data: Data to validate

        Returns:
            DimensionResult with step results
        """
        if not self.steps:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="No validation steps configured",
                details={},
            )

        step_results = []
        current_data = data
        all_passed = True

        for i, step in enumerate(self.steps):
            try:
                result = step(current_data)
                passed = result.get("passed", True) if isinstance(result, dict) else bool(result)
                step_results.append(
                    {
                        "step": i,
                        "passed": passed,
                        "result": result if isinstance(result, dict) else {"value": result},
                    }
                )
                if not passed:
                    all_passed = False
                    break  # Stop on first failure
                # Chain: output becomes next input
                if isinstance(result, dict) and "output" in result:
                    current_data = result["output"]
            except Exception as e:
                step_results.append(
                    {
                        "step": i,
                        "passed": False,
                        "error": str(e),
                    }
                )
                all_passed = False
                break

        return DimensionResult(
            dimension=self.dimension_id,
            passed=all_passed,
            message=f"{len(step_results)}/{len(self.steps)} steps completed",
            details={
                "steps_completed": len(step_results),
                "total_steps": len(self.steps),
                "step_results": step_results,
            },
        )


class D13_AnomalyDetection(BaseDimension):
    """D13: Statistical outlier detection."""

    dimension_id = "D13"
    dimension_name = "Anomaly Detection"

    def __init__(self, sigma_threshold: float = 3.0):
        """Initialize anomaly detector.

        Args:
            sigma_threshold: Standard deviations for anomaly detection
        """
        self.sigma_threshold = sigma_threshold

    def validate(self, data: np.ndarray) -> DimensionResult:
        """Detect statistical anomalies.

        Args:
            data: Numeric array

        Returns:
            DimensionResult with anomaly info
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if len(data) < 5:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="Insufficient data for anomaly detection",
                details={},
            )

        mean = np.mean(data)
        std = np.std(data)

        if std < 1e-10:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="Data has no variance",
                details={"mean": float(mean)},
            )

        # Find anomalies
        z_scores = np.abs((data - mean) / std)
        anomaly_indices = np.where(z_scores > self.sigma_threshold)[0]
        anomaly_count = len(anomaly_indices)

        # Also check entropy for fraud signal
        h = shannon_entropy(data.tobytes())
        fraud_signal = h.normalized < FRAUD_SIGNAL_THRESHOLD
        random_signal = h.normalized > RANDOM_SIGNAL_THRESHOLD

        return DimensionResult(
            dimension=self.dimension_id,
            passed=anomaly_count == 0,
            message=f"Found {anomaly_count} anomalies",
            details={
                "anomaly_count": anomaly_count,
                "anomaly_indices": anomaly_indices.tolist()[:10],  # First 10
                "mean": float(mean),
                "std": float(std),
                "max_z_score": float(np.max(z_scores)),
                "entropy": h.normalized,
                "fraud_signal": fraud_signal,
                "random_signal": random_signal,
            },
        )


class D14_StatisticalValidation(BaseDimension):
    """D14: Distribution testing."""

    dimension_id = "D14"
    dimension_name = "Statistical Validation"

    def __init__(
        self,
        expected_distribution: str = "normal",
        p_threshold: float = 0.05,
    ):
        """Initialize statistical validator.

        Args:
            expected_distribution: Expected distribution type
            p_threshold: P-value threshold for test
        """
        self.expected_distribution = expected_distribution
        self.p_threshold = p_threshold

    def validate(self, data: np.ndarray) -> DimensionResult:
        """Test distribution fit.

        Args:
            data: Numeric array

        Returns:
            DimensionResult with statistical test results
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if len(data) < 20:
            return DimensionResult(
                dimension=self.dimension_id,
                passed=True,
                message="Insufficient data for statistical testing",
                details={},
            )

        # Basic statistics
        mean = np.mean(data)
        std = np.std(data)
        skewness = self._compute_skewness(data)
        kurtosis = self._compute_kurtosis(data)

        # Simple normality check using skewness and kurtosis
        # Normal: skewness ~ 0, kurtosis ~ 3
        if self.expected_distribution == "normal":
            is_normal = abs(skewness) < 2 and abs(kurtosis - 3) < 7
            passed = is_normal
            test_name = "Normality (skewness/kurtosis)"
        else:
            # Default: just compute stats
            passed = True
            test_name = "Descriptive statistics"
            is_normal = None

        return DimensionResult(
            dimension=self.dimension_id,
            passed=passed,
            message=f"{test_name}: {'passed' if passed else 'failed'}",
            details={
                "mean": float(mean),
                "std": float(std),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "is_normal": is_normal,
                "sample_size": len(data),
            },
        )

    @staticmethod
    def _compute_skewness(data: np.ndarray) -> float:
        """Compute skewness."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.sum(((data - mean) / std) ** 3) / n)

    @staticmethod
    def _compute_kurtosis(data: np.ndarray) -> float:
        """Compute kurtosis."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 3.0
        return float(np.sum(((data - mean) / std) ** 4) / n)
