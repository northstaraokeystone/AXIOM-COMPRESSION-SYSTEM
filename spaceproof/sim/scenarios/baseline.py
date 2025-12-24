"""baseline.py - Normal Operation Scenario.

BASELINE SCENARIO:
    Standard probability distributions.
    Expected load patterns.
    Checkpoint frequency: 100 steps.

This is the reference scenario against which all others are compared.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from spaceproof.engine.entropy import (
    shannon_entropy,
    coherence_score,
    COHERENCE_THRESHOLD,
    ENTROPY_DELTA_HEALTHY,
)
from spaceproof.core import emit_receipt


CHECKPOINT_FREQUENCY = 100
TENANT_ID = "spaceproof-scenario-baseline"


@dataclass
class BaselineConfig:
    """Configuration for baseline scenario."""

    steps: int = 1000
    seed: int = 42
    distribution: str = "normal"  # normal, uniform, exponential
    mean: float = 0.0
    std: float = 1.0
    sample_size: int = 1000


@dataclass
class BaselineResult:
    """Result of baseline scenario execution."""

    steps_completed: int
    avg_entropy_delta: float
    avg_coherence: float
    alive_ratio: float
    within_normal_bounds: bool
    deviation_count: int


class BaselineScenario:
    """Normal operation scenario for baseline validation."""

    def __init__(self, config: Optional[BaselineConfig] = None):
        """Initialize baseline scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or BaselineConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.results: List[Dict] = []

    def generate_input(self, step: int) -> np.ndarray:
        """Generate input data from standard distribution.

        Args:
            step: Current step number

        Returns:
            Input data array
        """
        if self.config.distribution == "uniform":
            return self.rng.uniform(-1, 1, self.config.sample_size)
        elif self.config.distribution == "exponential":
            return self.rng.exponential(1.0, self.config.sample_size)
        else:  # normal
            return self.rng.normal(self.config.mean, self.config.std, self.config.sample_size)

    def validate_step(self, step: int, input_data: np.ndarray, output_data: np.ndarray) -> Dict:
        """Validate a single step.

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

        result = {
            "step": step,
            "entropy_before": h_before.normalized,
            "entropy_after": h_after.normalized,
            "entropy_delta": delta,
            "coherence": coh.score,
            "is_alive": coh.is_alive,
            "within_bounds": delta >= 0 and coh.score >= 0.5,
        }

        self.results.append(result)

        # Emit checkpoint
        if (step + 1) % CHECKPOINT_FREQUENCY == 0:
            self._emit_checkpoint(step)

        return result

    def _emit_checkpoint(self, step: int) -> None:
        """Emit checkpoint receipt."""
        recent = self.results[-CHECKPOINT_FREQUENCY:]

        emit_receipt(
            "baseline_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "avg_delta": np.mean([r["entropy_delta"] for r in recent]),
                "avg_coherence": np.mean([r["coherence"] for r in recent]),
                "alive_ratio": sum(1 for r in recent if r["is_alive"]) / len(recent),
            },
        )

    def evaluate(self) -> BaselineResult:
        """Evaluate overall scenario results.

        Returns:
            BaselineResult with summary metrics
        """
        if not self.results:
            return BaselineResult(
                steps_completed=0,
                avg_entropy_delta=0.0,
                avg_coherence=0.0,
                alive_ratio=0.0,
                within_normal_bounds=False,
                deviation_count=0,
            )

        deltas = [r["entropy_delta"] for r in self.results]
        coherences = [r["coherence"] for r in self.results]
        alive_count = sum(1 for r in self.results if r["is_alive"])
        deviation_count = sum(1 for r in self.results if not r["within_bounds"])

        avg_delta = np.mean(deltas)
        avg_coherence = np.mean(coherences)

        # Within normal bounds if:
        # - Average delta is positive (entropy reduction)
        # - Average coherence above 0.5
        # - Less than 10% deviations
        within_bounds = avg_delta >= 0 and avg_coherence >= 0.5 and deviation_count / len(self.results) < 0.1

        return BaselineResult(
            steps_completed=len(self.results),
            avg_entropy_delta=avg_delta,
            avg_coherence=avg_coherence,
            alive_ratio=alive_count / len(self.results),
            within_normal_bounds=within_bounds,
            deviation_count=deviation_count,
        )
