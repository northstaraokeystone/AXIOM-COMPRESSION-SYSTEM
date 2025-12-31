"""singularity.py - Self-Referential Validation Scenario.

SINGULARITY SCENARIO:
    Self-referential conditions.
    System emits receipts about receipts.
    Reaching toward "receipt completeness" - where system can audit itself.
    Checkpoint frequency: 50 steps.

This scenario tests the system's ability to handle self-reference and recursion.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from spaceproof.engine.entropy import (
    shannon_entropy,
    coherence_score,
    COHERENCE_THRESHOLD,
)
from spaceproof.core import emit_receipt, dual_hash


CHECKPOINT_FREQUENCY = 50
TENANT_ID = "spaceproof-scenario-singularity"


@dataclass
class SingularityConfig:
    """Configuration for singularity scenario."""

    steps: int = 1000
    seed: int = 42
    max_recursion_depth: int = 7
    self_reference_ratio: float = 0.3  # % of steps that self-reference


@dataclass
class SingularityResult:
    """Result of singularity scenario execution."""

    steps_completed: int
    max_recursion_achieved: int
    self_reference_count: int
    receipt_about_receipt_count: int
    circular_references_detected: int
    singularity_stable: bool


class SingularityScenario:
    """Self-referential validation scenario."""

    def __init__(self, config: Optional[SingularityConfig] = None):
        """Initialize singularity scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or SingularityConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.results: List[Dict] = []
        self.receipt_hashes: List[str] = []
        self.recursion_depth = 0
        self.circular_references: List[tuple] = []

    def generate_input(self, step: int) -> np.ndarray:
        """Generate self-referential input.

        May incorporate previous outputs as input.

        Args:
            step: Current step number

        Returns:
            Input data array
        """
        if step == 0 or self.rng.random() >= self.config.self_reference_ratio:
            # Non-self-referential: normal generation
            return self.rng.normal(0, 1, 1000)

        # Self-referential: use previous results
        prev_result = self.results[-1]

        # Incorporate previous entropy values
        seed_val = prev_result["entropy_delta"] * 1000

        # Track recursion depth
        current_depth = prev_result.get("recursion_depth", 0) + 1
        self.recursion_depth = max(self.recursion_depth, current_depth)

        if current_depth > self.config.max_recursion_depth:
            # Prevent infinite recursion
            return self.rng.normal(0, 1, 1000)

        # Generate data seeded by previous output
        data = self.rng.normal(seed_val, abs(seed_val) + 0.1, 1000)

        return data

    def validate_step(self, step: int, input_data: np.ndarray, output_data: np.ndarray) -> Dict:
        """Validate a single singularity step.

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

        # Track self-reference
        is_self_referential = step > 0 and self.rng.random() < self.config.self_reference_ratio
        recursion_depth = 0

        if is_self_referential and self.results:
            recursion_depth = self.results[-1].get("recursion_depth", 0) + 1

        # Create receipt about previous receipt
        receipt_about_receipt = None
        if self.receipt_hashes:
            prev_hash = self.receipt_hashes[-1]
            receipt_about_receipt = dual_hash(f"meta:{prev_hash}")

            # Check for circular references
            if receipt_about_receipt in self.receipt_hashes:
                self.circular_references.append((step, receipt_about_receipt[:16]))

        # Emit receipt
        receipt = emit_receipt(
            "singularity_step",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "is_self_referential": is_self_referential,
                "recursion_depth": recursion_depth,
                "references_previous": receipt_about_receipt is not None,
            },
        )

        self.receipt_hashes.append(receipt["payload_hash"])

        result = {
            "step": step,
            "entropy_before": h_before.normalized,
            "entropy_after": h_after.normalized,
            "entropy_delta": delta,
            "coherence": coh.score,
            "is_self_referential": is_self_referential,
            "recursion_depth": recursion_depth,
            "receipt_hash": receipt["payload_hash"],
            "references_previous": receipt_about_receipt is not None,
        }

        self.results.append(result)

        # Emit checkpoint
        if (step + 1) % CHECKPOINT_FREQUENCY == 0:
            self._emit_checkpoint(step)

        return result

    def _emit_checkpoint(self, step: int) -> None:
        """Emit singularity checkpoint receipt."""
        recent = self.results[-CHECKPOINT_FREQUENCY:]

        emit_receipt(
            "singularity_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "self_reference_count": sum(1 for r in recent if r["is_self_referential"]),
                "max_recursion": max(r["recursion_depth"] for r in recent),
                "circular_references": len(self.circular_references),
                "receipt_chain_length": len(self.receipt_hashes),
            },
        )

    def evaluate(self) -> SingularityResult:
        """Evaluate singularity results.

        Returns:
            SingularityResult with self-reference metrics
        """
        if not self.results:
            return SingularityResult(
                steps_completed=0,
                max_recursion_achieved=0,
                self_reference_count=0,
                receipt_about_receipt_count=0,
                circular_references_detected=0,
                singularity_stable=False,
            )

        self_ref_count = sum(1 for r in self.results if r["is_self_referential"])
        receipt_refs = sum(1 for r in self.results if r["references_previous"])

        # Singularity is stable if:
        # - System handles self-reference without diverging
        # - Coherence remains above threshold
        # - Circular references are detected but don't crash
        avg_coherence = np.mean([r["coherence"] for r in self.results])
        final_coherence = self.results[-1]["coherence"]

        singularity_stable = (
            avg_coherence >= 0.5
            and final_coherence >= COHERENCE_THRESHOLD
            and len(self.circular_references) < len(self.results) * 0.1
        )

        return SingularityResult(
            steps_completed=len(self.results),
            max_recursion_achieved=self.recursion_depth,
            self_reference_count=self_ref_count,
            receipt_about_receipt_count=receipt_refs,
            circular_references_detected=len(self.circular_references),
            singularity_stable=singularity_stable,
        )
