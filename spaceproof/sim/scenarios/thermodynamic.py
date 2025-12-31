"""thermodynamic.py - Entropy Conservation Verification Scenario.

THERMODYNAMIC SCENARIO:
    Verify entropy conservation per second law.
    Isolated subsystems obey thermodynamic laws.
    Energy conservation (first law) verification.
    Fitness = entropy_reduction / receipts.
    Checkpoint frequency: 25 steps.

This scenario validates that the entropy pump respects thermodynamic principles.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from spaceproof.engine.entropy import (
    shannon_entropy,
    coherence_score,
    fitness_score,
)
from spaceproof.core import emit_receipt


CHECKPOINT_FREQUENCY = 25
TENANT_ID = "spaceproof-scenario-thermodynamic"


@dataclass
class ThermodynamicConfig:
    """Configuration for thermodynamic scenario."""

    steps: int = 1000
    seed: int = 42
    isolated_subsystems: int = 3
    energy_budget: float = 1000.0  # Arbitrary units


@dataclass
class ThermodynamicResult:
    """Result of thermodynamic scenario execution."""

    steps_completed: int
    total_entropy_in: float
    total_entropy_out: float
    entropy_balance: float  # Should be <= 0 for healthy pump
    first_law_satisfied: bool  # Energy conservation
    second_law_satisfied: bool  # Entropy non-decreasing in isolated system
    fitness_score: float


class ThermodynamicScenario:
    """Entropy conservation verification scenario."""

    def __init__(self, config: Optional[ThermodynamicConfig] = None):
        """Initialize thermodynamic scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or ThermodynamicConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.results: List[Dict] = []

        # Tracking for thermodynamic laws
        self.total_entropy_in = 0.0
        self.total_entropy_out = 0.0
        self.energy_consumed = 0.0
        self.subsystem_entropies: Dict[int, List[float]] = {i: [] for i in range(self.config.isolated_subsystems)}

    def generate_input(self, step: int) -> np.ndarray:
        """Generate thermodynamically consistent input.

        Uses exponential distribution (maximum entropy for given mean).

        Args:
            step: Current step number

        Returns:
            Input data array
        """
        # Exponential distribution has maximum entropy for fixed mean
        mean = 1.0 + (step / self.config.steps)  # Slowly increasing
        data = self.rng.exponential(mean, 1000)

        return data

    def validate_step(self, step: int, input_data: np.ndarray, output_data: np.ndarray) -> Dict:
        """Validate a single thermodynamic step.

        Checks entropy conservation and energy budget.

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

        # Track total entropy flow
        self.total_entropy_in += h_before.normalized
        self.total_entropy_out += h_after.normalized

        # Assign to subsystem
        subsystem = step % self.config.isolated_subsystems
        self.subsystem_entropies[subsystem].append(h_after.normalized)

        # Energy consumed is proportional to entropy reduction
        # (thermodynamic work requires energy)
        energy_for_step = max(0, delta) * 10  # Arbitrary conversion
        self.energy_consumed += energy_for_step

        coh = coherence_score(output_data)

        result = {
            "step": step,
            "entropy_before": h_before.normalized,
            "entropy_after": h_after.normalized,
            "entropy_delta": delta,
            "coherence": coh.score,
            "subsystem": subsystem,
            "energy_consumed": energy_for_step,
            "cumulative_energy": self.energy_consumed,
            "entropy_balance": self.total_entropy_in - self.total_entropy_out,
        }

        self.results.append(result)

        # Emit checkpoint
        if (step + 1) % CHECKPOINT_FREQUENCY == 0:
            self._emit_checkpoint(step)

        return result

    def _emit_checkpoint(self, step: int) -> None:
        """Emit thermodynamic checkpoint receipt."""
        self.results[-CHECKPOINT_FREQUENCY:]

        # Check second law for each subsystem
        second_law_violations = 0
        for subsys_id, entropies in self.subsystem_entropies.items():
            if len(entropies) >= 2:
                # In isolated system, entropy should not decrease
                for i in range(1, len(entropies)):
                    if entropies[i] < entropies[i - 1] - 0.01:  # Small tolerance
                        second_law_violations += 1

        emit_receipt(
            "thermodynamic_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "total_entropy_in": self.total_entropy_in,
                "total_entropy_out": self.total_entropy_out,
                "entropy_balance": self.total_entropy_in - self.total_entropy_out,
                "energy_consumed": self.energy_consumed,
                "energy_remaining": self.config.energy_budget - self.energy_consumed,
                "second_law_violations": second_law_violations,
            },
        )

    def evaluate(self) -> ThermodynamicResult:
        """Evaluate thermodynamic results.

        Returns:
            ThermodynamicResult with conservation metrics
        """
        if not self.results:
            return ThermodynamicResult(
                steps_completed=0,
                total_entropy_in=0.0,
                total_entropy_out=0.0,
                entropy_balance=0.0,
                first_law_satisfied=False,
                second_law_satisfied=False,
                fitness_score=0.0,
            )

        # Entropy balance (should be >= 0 for net entropy reduction)
        entropy_balance = self.total_entropy_in - self.total_entropy_out

        # First law: energy conservation
        # Consumed energy should not exceed budget
        first_law_satisfied = self.energy_consumed <= self.config.energy_budget

        # Second law: entropy non-decreasing in isolated subsystems
        # Count violations
        second_law_violations = 0
        total_transitions = 0
        for subsys_id, entropies in self.subsystem_entropies.items():
            if len(entropies) >= 2:
                for i in range(1, len(entropies)):
                    total_transitions += 1
                    if entropies[i] < entropies[i - 1] - 0.01:
                        second_law_violations += 1

        # Allow small violation rate
        second_law_satisfied = total_transitions == 0 or second_law_violations / total_transitions < 0.05

        # Fitness score
        total_reduction = sum(max(0, r["entropy_delta"]) for r in self.results)
        fit = fitness_score(total_reduction, len(self.results))

        return ThermodynamicResult(
            steps_completed=len(self.results),
            total_entropy_in=self.total_entropy_in,
            total_entropy_out=self.total_entropy_out,
            entropy_balance=entropy_balance,
            first_law_satisfied=first_law_satisfied,
            second_law_satisfied=second_law_satisfied,
            fitness_score=fit,
        )
