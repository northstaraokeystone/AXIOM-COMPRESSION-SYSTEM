"""stress.py - Edge Case Stress Testing Scenario.

STRESS SCENARIO:
    Edge cases at 3-5x normal intensity.
    Heavy-tail distributions.
    Extreme value sampling.
    Checkpoint frequency: 10 steps (more monitoring under stress).

This scenario tests system resilience under extreme conditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from spaceproof.engine.entropy import (
    shannon_entropy,
    coherence_score,
    ENTROPY_DELTA_CRITICAL,
)
from spaceproof.core import emit_receipt


CHECKPOINT_FREQUENCY = 10  # More frequent monitoring under stress
TENANT_ID = "spaceproof-scenario-stress"


@dataclass
class StressConfig:
    """Configuration for stress scenario."""

    steps: int = 1000
    seed: int = 42
    intensity_multiplier: float = 3.0  # 3-5x normal
    sample_size: int = 1000
    spike_probability: float = 0.1  # Probability of extreme spike
    max_spike_magnitude: float = 100.0


@dataclass
class StressResult:
    """Result of stress scenario execution."""

    steps_completed: int
    avg_entropy_delta: float
    min_entropy_delta: float
    critical_count: int
    recovery_rate: float
    stress_survived: bool


class StressScenario:
    """Edge case stress testing scenario."""

    def __init__(self, config: Optional[StressConfig] = None):
        """Initialize stress scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or StressConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.results: List[Dict] = []
        self.critical_events: List[int] = []
        self.recoveries: List[int] = []

    def generate_input(self, step: int) -> np.ndarray:
        """Generate stress input with heavy tails.

        Uses Cauchy distribution for heavy tails and random spikes.

        Args:
            step: Current step number

        Returns:
            Input data array
        """
        # Base: Cauchy (heavy tails)
        data = self.rng.standard_cauchy(self.config.sample_size)
        data = data * self.config.intensity_multiplier

        # Add spikes
        if self.rng.random() < self.config.spike_probability:
            n_spikes = self.rng.integers(1, 10)
            spike_indices = self.rng.choice(self.config.sample_size, n_spikes, replace=False)
            spike_values = self.rng.uniform(-self.config.max_spike_magnitude, self.config.max_spike_magnitude, n_spikes)
            data[spike_indices] = spike_values

        # Clip to prevent numerical issues
        data = np.clip(data, -1e6, 1e6)

        return data

    def validate_step(self, step: int, input_data: np.ndarray, output_data: np.ndarray) -> Dict:
        """Validate a single stress step.

        Args:
            step: Current step number
            input_data: Input array
            output_data: Output array

        Returns:
            Step validation result
        """
        h_before = shannon_entropy(input_data.tobytes())
        h_after = shannon_entropy(output_data.tobytes())
        delta = h_before.normalized - h_after.normalized

        coh = coherence_score(output_data)

        is_critical = delta < ENTROPY_DELTA_CRITICAL

        # Track critical events and recoveries
        if is_critical:
            self.critical_events.append(step)
        elif self.critical_events and step == self.critical_events[-1] + 1:
            # Recovered from previous critical
            self.recoveries.append(step)

        result = {
            "step": step,
            "entropy_before": h_before.normalized,
            "entropy_after": h_after.normalized,
            "entropy_delta": delta,
            "coherence": coh.score,
            "is_critical": is_critical,
            "input_range": (float(np.min(input_data)), float(np.max(input_data))),
        }

        self.results.append(result)

        # Emit checkpoint (more frequent under stress)
        if (step + 1) % CHECKPOINT_FREQUENCY == 0:
            self._emit_checkpoint(step)

        return result

    def _emit_checkpoint(self, step: int) -> None:
        """Emit stress checkpoint receipt."""
        recent = self.results[-CHECKPOINT_FREQUENCY:]

        emit_receipt(
            "stress_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "avg_delta": np.mean([r["entropy_delta"] for r in recent]),
                "min_delta": min(r["entropy_delta"] for r in recent),
                "critical_count": sum(1 for r in recent if r["is_critical"]),
                "intensity": self.config.intensity_multiplier,
            },
        )

    def evaluate(self) -> StressResult:
        """Evaluate stress test results.

        Returns:
            StressResult with stress test metrics
        """
        if not self.results:
            return StressResult(
                steps_completed=0,
                avg_entropy_delta=0.0,
                min_entropy_delta=0.0,
                critical_count=0,
                recovery_rate=0.0,
                stress_survived=False,
            )

        deltas = [r["entropy_delta"] for r in self.results]

        critical_count = len(self.critical_events)
        recovery_count = len(self.recoveries)
        recovery_rate = recovery_count / critical_count if critical_count > 0 else 1.0

        # Survived stress if:
        # - Less than 20% critical events
        # - Recovery rate > 50%
        # - Average delta still positive
        avg_delta = np.mean(deltas)
        stress_survived = (
            critical_count / len(self.results) < 0.2 and recovery_rate >= 0.5 and avg_delta > ENTROPY_DELTA_CRITICAL
        )

        return StressResult(
            steps_completed=len(self.results),
            avg_entropy_delta=avg_delta,
            min_entropy_delta=min(deltas),
            critical_count=critical_count,
            recovery_rate=recovery_rate,
            stress_survived=stress_survived,
        )
