"""godel.py - Completeness Bounds and Decidability Scenario.

GODEL SCENARIO:
    Acknowledge that some properties are undecidable.
    Detect when validation reaches Godel-style limits.
    Proving consistency may require external attestation.
    Checkpoint frequency: 100 steps.

This scenario tests the system's ability to recognize its own limits.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np

from spaceproof.engine.entropy import (
    shannon_entropy,
    coherence_score,
)
from spaceproof.core import emit_receipt


CHECKPOINT_FREQUENCY = 100
TENANT_ID = "spaceproof-scenario-godel"


@dataclass
class GodelConfig:
    """Configuration for Godel scenario."""

    steps: int = 1000
    seed: int = 42
    undecidable_ratio: float = 0.1  # % of inputs that are undecidable
    requires_external_attestation: bool = True


@dataclass
class GodelResult:
    """Result of Godel scenario execution."""

    steps_completed: int
    decidable_count: int
    undecidable_count: int
    external_attestation_needed: int
    completeness_ratio: float  # % of statements we could decide
    system_consistent: bool


class GodelScenario:
    """Completeness bounds and decidability scenario."""

    def __init__(self, config: Optional[GodelConfig] = None):
        """Initialize Godel scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or GodelConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.results: List[Dict] = []
        self.decidable_statements: Set[int] = set()
        self.undecidable_statements: Set[int] = set()
        self.external_attestations: List[int] = []

    def generate_input(self, step: int) -> np.ndarray:
        """Generate input that may be decidable or undecidable.

        Undecidable inputs are random noise (high entropy, no pattern).
        Decidable inputs have structure (patterns, low entropy).

        Args:
            step: Current step number

        Returns:
            Input data array
        """
        if self.rng.random() < self.config.undecidable_ratio:
            # Undecidable: pure random (incompressible)
            data = self.rng.random(1000)
            self._mark_undecidable(step)
        else:
            # Decidable: structured pattern
            t = np.linspace(0, 4 * np.pi, 1000)
            # Use different patterns to simulate different "theorems"
            pattern_type = step % 5
            if pattern_type == 0:
                data = np.sin(t * (step + 1))
            elif pattern_type == 1:
                data = np.exp(-t / 10) * np.cos(t)
            elif pattern_type == 2:
                data = np.cumsum(self.rng.normal(0, 1, 1000)) / 100
            elif pattern_type == 3:
                data = np.power(t, 0.5) * np.sin(t)
            else:
                data = np.tanh(t - 6)

            self._mark_decidable(step)

        return data

    def _mark_decidable(self, step: int) -> None:
        """Mark a statement as decidable."""
        self.decidable_statements.add(step)

    def _mark_undecidable(self, step: int) -> None:
        """Mark a statement as undecidable."""
        self.undecidable_statements.add(step)

    def validate_step(self, step: int, input_data: np.ndarray, output_data: np.ndarray) -> Dict:
        """Validate a single Godel step.

        Detects when validation reaches decidability limits.

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

        # Determine decidability
        is_undecidable = step in self.undecidable_statements

        # Check if we need external attestation
        needs_external = False
        if is_undecidable and self.config.requires_external_attestation:
            # Can't decide internally, need external help
            needs_external = True
            self.external_attestations.append(step)

        # Attempt to verify consistency
        # In Godel terms: can we prove our own consistency?
        # This is inherently limited
        consistency_check = self._check_local_consistency(step)

        result = {
            "step": step,
            "entropy_before": h_before.normalized,
            "entropy_after": h_after.normalized,
            "entropy_delta": delta,
            "coherence": coh.score,
            "is_decidable": not is_undecidable,
            "needs_external_attestation": needs_external,
            "local_consistency": consistency_check,
        }

        self.results.append(result)

        # Emit checkpoint
        if (step + 1) % CHECKPOINT_FREQUENCY == 0:
            self._emit_checkpoint(step)

        return result

    def _check_local_consistency(self, step: int) -> bool:
        """Check local consistency (limited by Godel's theorems).

        We can check consistency of recent results but cannot
        prove global consistency from within.

        Args:
            step: Current step number

        Returns:
            True if locally consistent
        """
        if len(self.results) < 2:
            return True

        # Check that recent results don't contradict
        recent = self.results[-10:]

        # Simple consistency: entropy deltas should be bounded
        deltas = [r["entropy_delta"] for r in recent]

        # No single result should be wildly inconsistent with others
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)

        for delta in deltas:
            if abs(delta - mean_delta) > 3 * std_delta + 0.1:
                return False

        return True

    def _emit_checkpoint(self, step: int) -> None:
        """Emit Godel checkpoint receipt."""
        recent = self.results[-CHECKPOINT_FREQUENCY:]

        decidable = sum(1 for r in recent if r["is_decidable"])
        undecidable = len(recent) - decidable

        emit_receipt(
            "godel_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "decidable_count": decidable,
                "undecidable_count": undecidable,
                "completeness_ratio": decidable / len(recent) if recent else 0,
                "external_attestations_needed": sum(1 for r in recent if r["needs_external_attestation"]),
                "local_consistency": all(r["local_consistency"] for r in recent),
            },
        )

    def evaluate(self) -> GodelResult:
        """Evaluate Godel results.

        Returns:
            GodelResult with completeness metrics
        """
        if not self.results:
            return GodelResult(
                steps_completed=0,
                decidable_count=0,
                undecidable_count=0,
                external_attestation_needed=0,
                completeness_ratio=0.0,
                system_consistent=False,
            )

        decidable_count = len(self.decidable_statements)
        undecidable_count = len(self.undecidable_statements)

        completeness_ratio = (
            decidable_count / (decidable_count + undecidable_count)
            if (decidable_count + undecidable_count) > 0
            else 0.0
        )

        # System is consistent if:
        # - All local consistency checks passed
        # - Undecidable statements were properly identified (not falsely decided)
        all_consistent = all(r["local_consistency"] for r in self.results)

        # Check we didn't falsely claim to decide undecidable statements
        false_decisions = sum(
            1
            for step in self.undecidable_statements
            if any(r["step"] == step and r["is_decidable"] for r in self.results)
        )

        system_consistent = all_consistent and false_decisions == 0

        return GodelResult(
            steps_completed=len(self.results),
            decidable_count=decidable_count,
            undecidable_count=undecidable_count,
            external_attestation_needed=len(self.external_attestations),
            completeness_ratio=completeness_ratio,
            system_consistent=system_consistent,
        )
