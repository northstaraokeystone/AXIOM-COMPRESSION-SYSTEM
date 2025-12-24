"""genesis.py - System Initialization Scenario.

GENESIS SCENARIO:
    Bootstrap validation.
    Seed-dependent activation follows hierarchical ordering.
    Checkpoint frequency: 1 step (every action during startup).

This scenario validates system initialization and bootstrap procedures.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from spaceproof.engine.entropy import (
    shannon_entropy,
    coherence_score,
    COHERENCE_THRESHOLD,
)
from spaceproof.core import emit_receipt


CHECKPOINT_FREQUENCY = 1  # Every step during genesis
TENANT_ID = "spaceproof-scenario-genesis"


@dataclass
class GenesisConfig:
    """Configuration for genesis scenario."""

    steps: int = 100  # Shorter for initialization
    seed: int = 42
    hierarchy_levels: int = 5
    bootstrap_phases: List[str] = None

    def __post_init__(self):
        if self.bootstrap_phases is None:
            self.bootstrap_phases = [
                "entropy_init",
                "module_load",
                "connection_establish",
                "validation_start",
                "steady_state",
            ]


@dataclass
class GenesisResult:
    """Result of genesis scenario execution."""

    steps_completed: int
    phases_completed: List[str]
    bootstrap_time_steps: int
    hierarchy_established: bool
    initial_entropy: float
    final_entropy: float
    genesis_successful: bool


class GenesisScenario:
    """System initialization and bootstrap scenario."""

    def __init__(self, config: Optional[GenesisConfig] = None):
        """Initialize genesis scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or GenesisConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.results: List[Dict] = []
        self.current_phase = 0
        self.hierarchy_order: List[int] = []

    def generate_input(self, step: int) -> np.ndarray:
        """Generate structured bootstrap input.

        Input grows in size and structure as system initializes.

        Args:
            step: Current step number

        Returns:
            Input data array
        """
        # Size grows during bootstrap
        size = 100 + step * 10

        # Structure increases with hierarchy
        level = min(step // 20, self.config.hierarchy_levels - 1)
        self.hierarchy_order.append(level)

        # Generate structured data based on phase
        if self.current_phase == 0:  # entropy_init
            # Pure random for initial entropy measurement
            data = self.rng.random(size)
        elif self.current_phase == 1:  # module_load
            # Patterned data for module initialization
            data = np.sin(np.linspace(0, 2 * np.pi, size))
            data += self.rng.normal(0, 0.1, size)
        elif self.current_phase == 2:  # connection_establish
            # Correlated data for connection patterns
            data = np.cumsum(self.rng.normal(0, 1, size))
            data = data / np.max(np.abs(data) + 1e-10)
        elif self.current_phase == 3:  # validation_start
            # Mixed patterns for validation
            t = np.linspace(0, 4 * np.pi, size)
            data = np.sin(t) + 0.5 * np.sin(2 * t) + 0.25 * np.sin(3 * t)
        else:  # steady_state
            # Normal operation data
            data = self.rng.normal(0, 1, size)

        return data

    def validate_step(self, step: int, input_data: np.ndarray, output_data: np.ndarray) -> Dict:
        """Validate a single genesis step.

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

        # Determine current phase
        steps_per_phase = self.config.steps // len(self.config.bootstrap_phases)
        self.current_phase = min(step // steps_per_phase, len(self.config.bootstrap_phases) - 1)

        result = {
            "step": step,
            "phase": self.config.bootstrap_phases[self.current_phase],
            "phase_index": self.current_phase,
            "entropy_before": h_before.normalized,
            "entropy_after": h_after.normalized,
            "entropy_delta": delta,
            "coherence": coh.score,
            "hierarchy_level": self.hierarchy_order[-1] if self.hierarchy_order else 0,
        }

        self.results.append(result)

        # Emit checkpoint every step during genesis
        self._emit_checkpoint(step)

        return result

    def _emit_checkpoint(self, step: int) -> None:
        """Emit genesis checkpoint receipt."""
        result = self.results[-1]

        emit_receipt(
            "genesis_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "step": step,
                "phase": result["phase"],
                "entropy_delta": result["entropy_delta"],
                "coherence": result["coherence"],
                "hierarchy_level": result["hierarchy_level"],
            },
        )

    def evaluate(self) -> GenesisResult:
        """Evaluate genesis results.

        Returns:
            GenesisResult with initialization metrics
        """
        if not self.results:
            return GenesisResult(
                steps_completed=0,
                phases_completed=[],
                bootstrap_time_steps=0,
                hierarchy_established=False,
                initial_entropy=0.0,
                final_entropy=0.0,
                genesis_successful=False,
            )

        # Phases completed
        phases_seen = set(r["phase"] for r in self.results)
        phases_completed = [p for p in self.config.bootstrap_phases if p in phases_seen]

        # Find when system reached steady state
        steady_state_steps = [
            r["step"] for r in self.results if r["phase"] == "steady_state" and r["coherence"] >= COHERENCE_THRESHOLD
        ]
        bootstrap_time = steady_state_steps[0] if steady_state_steps else self.config.steps

        # Check hierarchy ordering
        hierarchy_established = self._check_hierarchy_order()

        # Genesis successful if:
        # - All phases completed
        # - Hierarchy properly established
        # - Final state is coherent
        final_coherence = self.results[-1]["coherence"]
        genesis_successful = (
            len(phases_completed) == len(self.config.bootstrap_phases)
            and hierarchy_established
            and final_coherence >= COHERENCE_THRESHOLD
        )

        return GenesisResult(
            steps_completed=len(self.results),
            phases_completed=phases_completed,
            bootstrap_time_steps=bootstrap_time,
            hierarchy_established=hierarchy_established,
            initial_entropy=self.results[0]["entropy_before"],
            final_entropy=self.results[-1]["entropy_after"],
            genesis_successful=genesis_successful,
        )

    def _check_hierarchy_order(self) -> bool:
        """Check if hierarchy was established in proper order.

        Returns:
            True if hierarchy levels activated in order
        """
        if not self.hierarchy_order:
            return False

        # Should see levels in increasing order (with possible repeats)
        max_seen = -1
        for level in self.hierarchy_order:
            if level < max_seen - 1:  # Allow for some flexibility
                return False
            max_seen = max(max_seen, level)

        return max_seen == self.config.hierarchy_levels - 1
