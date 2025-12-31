"""monte_carlo.py - Core Monte Carlo Simulation Engine.

THE MONTE CARLO PARADIGM:
    Isolated simulations with shared entropy-based validation core.
    Each domain configuration runs independently, producing domain-specific receipts.
    The entropy pump measures health: delta = H(before) - H(after).

Module Composition Matrix (3-of-7 selection):
    xAI:     compress + witness + sovereignty  -> 10x telemetry reduction
    DOGE:    ledger + detect + anchor          -> $162B fraud prevention
    NASA:    compress + sovereignty + loop     -> 1.3s latency resilience
    Defense: compress + ledger + anchor        -> Cryptographic provenance
    DOT:     compress + ledger + detect        -> Continuous validation

Source: SpaceProof D20 Production Evolution + xAI collaboration
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import numpy as np

from spaceproof.core import emit_receipt, merkle

from spaceproof.engine.entropy import (
    shannon_entropy,
    coherence_score,
    fitness_score,
    ThompsonState,
    COHERENCE_THRESHOLD,
    ENTROPY_DELTA_CRITICAL,
)

from spaceproof.engine.receipts import (
    DomainConfig,
    SpaceProofReceipt,
    build_xai_receipt,
    build_doge_receipt,
    build_nasa_receipt,
    build_defense_receipt,
    build_dot_receipt,
)


# === CONSTANTS ===

TENANT_ID = "spaceproof-sim"

# Default simulation parameters
DEFAULT_STEPS = 1000
DEFAULT_SEED = 42


class Scenario(Enum):
    """Eight-scenario validation framework.

    v3.0: Added NETWORK and ADVERSARIAL scenarios for multi-tier autonomy.
    """

    BASELINE = "baseline"  # Normal operation, standard distributions
    STRESS = "stress"  # Edge cases at 3-5x intensity
    GENESIS = "genesis"  # System initialization, bootstrap validation
    SINGULARITY = "singularity"  # Self-referential conditions
    THERMODYNAMIC = "thermodynamic"  # Entropy conservation verification
    GODEL = "godel"  # Completeness bounds, decidability limits
    # v3.0: Multi-tier autonomy scenarios
    NETWORK = "network"  # 1000 colonies, 1M colonists network validation
    ADVERSARIAL = "adversarial"  # DoD hostile audit under combat conditions


@dataclass
class CheckpointConfig:
    """Checkpoint frequency per scenario."""

    scenario: Scenario
    frequency: int  # Steps between checkpoints

    @classmethod
    def for_scenario(cls, scenario: Scenario) -> "CheckpointConfig":
        """Get checkpoint config for scenario."""
        frequencies = {
            Scenario.BASELINE: 100,  # Normal monitoring
            Scenario.STRESS: 10,  # More monitoring under stress
            Scenario.GENESIS: 1,  # Every action during startup
            Scenario.SINGULARITY: 50,  # Moderate for self-reference
            Scenario.THERMODYNAMIC: 25,  # Frequent for entropy tracking
            Scenario.GODEL: 100,  # Standard for completeness checks
            # v3.0: Multi-tier autonomy scenarios
            Scenario.NETWORK: 30,  # Daily checkpoints for network validation
            Scenario.ADVERSARIAL: 10,  # Frequent for attack detection
        }
        return cls(scenario=scenario, frequency=frequencies.get(scenario, 100))


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""

    domain: DomainConfig
    scenario: Scenario
    steps: int = DEFAULT_STEPS
    seed: int = DEFAULT_SEED
    checkpoint_frequency: Optional[int] = None
    stress_multiplier: float = 1.0  # For STRESS scenario
    modules: List[str] = field(default_factory=list)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.checkpoint_frequency:
            self.checkpoint_frequency = CheckpointConfig.for_scenario(self.scenario).frequency

        if not self.modules:
            # Default module selection per domain
            module_map = {
                DomainConfig.XAI: ["compress", "witness", "sovereignty"],
                DomainConfig.DOGE: ["ledger", "detect", "anchor"],
                DomainConfig.NASA: ["compress", "sovereignty", "loop"],
                DomainConfig.DEFENSE: ["compress", "ledger", "anchor"],
                DomainConfig.DOT: ["compress", "ledger", "detect"],
            }
            self.modules = module_map.get(self.domain, ["compress", "ledger", "anchor"])


@dataclass
class StepResult:
    """Result of a single simulation step."""

    step: int
    entropy_before: float
    entropy_after: float
    entropy_delta: float
    coherence: float
    is_alive: bool
    module_results: Dict[str, Any]
    receipt: Optional[SpaceProofReceipt] = None
    duration_ms: float = 0.0


@dataclass
class SimulationResult:
    """Result of complete simulation run."""

    config: SimulationConfig
    steps_completed: int
    total_duration_ms: float
    entropy_deltas: List[float]
    coherence_scores: List[float]
    alive_ratio: float
    fitness_score: float
    checkpoints: List[Dict[str, Any]]
    final_merkle_root: str
    receipts: List[SpaceProofReceipt]
    passed: bool
    failure_reason: Optional[str] = None


class MonteCarloEngine:
    """Core Monte Carlo simulation engine for SpaceProof validation."""

    def __init__(self, config: SimulationConfig):
        """Initialize simulation engine.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.step_results: List[StepResult] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.receipts: List[SpaceProofReceipt] = []

        # Thompson sampling states for module selection
        self.module_states = {module: ThompsonState.new() for module in config.modules}

    def run(self) -> SimulationResult:
        """Run the full simulation.

        Returns:
            SimulationResult with all metrics
        """
        start_time = time.time()

        # Initialize based on scenario
        self._initialize_scenario()

        # Run simulation steps
        for step in range(self.config.steps):
            result = self._run_step(step)
            self.step_results.append(result)

            # Checkpoint if needed
            if (step + 1) % self.config.checkpoint_frequency == 0:
                self._create_checkpoint(step)

            # Early termination for critical failures
            if result.entropy_delta < ENTROPY_DELTA_CRITICAL:
                return self._create_result(
                    steps_completed=step + 1,
                    passed=False,
                    failure_reason=f"Critical entropy delta at step {step}: {result.entropy_delta:.3f}",
                )

        total_duration = (time.time() - start_time) * 1000
        return self._create_result(
            steps_completed=self.config.steps,
            passed=True,
            total_duration=total_duration,
        )

    def _initialize_scenario(self) -> None:
        """Initialize based on scenario type."""
        if self.config.scenario == Scenario.STRESS:
            # Apply stress multiplier to distributions
            self.stress_multiplier = self.config.stress_multiplier or 3.0
        elif self.config.scenario == Scenario.GENESIS:
            # Bootstrap initialization
            self._emit_genesis_receipt()
        elif self.config.scenario == Scenario.SINGULARITY:
            # Prepare for self-referential validation
            pass
        elif self.config.scenario == Scenario.THERMODYNAMIC:
            # Initialize entropy tracking
            self.initial_entropy = 0.0
            self.total_entropy_generated = 0.0
        elif self.config.scenario == Scenario.GODEL:
            # Track decidability bounds
            self.undecidable_count = 0
        # v3.0: Multi-tier autonomy scenarios
        elif self.config.scenario == Scenario.NETWORK:
            # Initialize network-scale tracking
            self.network_colonies = self.config.custom_params.get("n_colonies", 100)
            self.network_population = self.config.custom_params.get("population", 100000)
        elif self.config.scenario == Scenario.ADVERSARIAL:
            # Initialize adversarial attack tracking
            self.attacks_attempted = 0
            self.attacks_blocked = 0

    def _run_step(self, step: int) -> StepResult:
        """Run a single simulation step.

        Args:
            step: Current step number

        Returns:
            StepResult with step metrics
        """
        start = time.time()

        # Generate input data based on scenario
        input_data = self._generate_input(step)

        # Measure initial entropy
        h_before = shannon_entropy(input_data)

        # Execute modules using Thompson sampling for selection order
        module_results = self._execute_modules(input_data)

        # Measure final entropy
        import json

        output_bytes = json.dumps(module_results, sort_keys=True, default=str).encode()
        h_after = shannon_entropy(output_bytes)

        # Compute delta (positive = compression/order)
        delta = h_before.normalized - h_after.normalized

        # Compute coherence
        values = []
        for result in module_results.values():
            if isinstance(result, dict):
                for v in result.values():
                    if isinstance(v, (int, float)):
                        values.append(v)
        pattern = np.array(values) if values else np.array([0.0])
        coh = coherence_score(pattern)

        # Update Thompson states based on success
        success = delta > 0 and coh.score >= COHERENCE_THRESHOLD
        for module in self.config.modules:
            self.module_states[module] = self.module_states[module].update(success)

        # Build domain-specific receipt
        receipt = self._build_receipt(module_results)
        if receipt:
            self.receipts.append(receipt)

        duration = (time.time() - start) * 1000

        return StepResult(
            step=step,
            entropy_before=h_before.normalized,
            entropy_after=h_after.normalized,
            entropy_delta=delta,
            coherence=coh.score,
            is_alive=coh.is_alive,
            module_results=module_results,
            receipt=receipt,
            duration_ms=duration,
        )

    def _generate_input(self, step: int) -> bytes:
        """Generate input data based on scenario.

        Args:
            step: Current step number

        Returns:
            Input data as bytes
        """
        if self.config.scenario == Scenario.STRESS:
            # Heavy-tail distribution, extreme values
            size = int(1000 * self.config.stress_multiplier)
            data = self.rng.standard_cauchy(size) * 1000
            data = np.clip(data, -1e6, 1e6)
        elif self.config.scenario == Scenario.GENESIS:
            # Structured bootstrap data
            size = 100 + step * 10
            data = self.rng.normal(0, 1, size)
        elif self.config.scenario == Scenario.SINGULARITY:
            # Self-referential: include previous step info
            if self.step_results:
                prev = self.step_results[-1]
                seed_val = prev.entropy_delta * 1000
                data = self.rng.normal(seed_val, abs(seed_val) + 0.1, 1000)
            else:
                data = self.rng.normal(0, 1, 1000)
        elif self.config.scenario == Scenario.THERMODYNAMIC:
            # Conserved quantities
            data = self.rng.exponential(1.0, 1000)
        elif self.config.scenario == Scenario.GODEL:
            # Mix of decidable and undecidable patterns
            if self.rng.random() < 0.1:
                # Undecidable: random noise
                data = self.rng.random(1000)
            else:
                # Decidable: structured pattern
                data = np.sin(np.linspace(0, 4 * np.pi, 1000) * (step + 1))
        # v3.0: Multi-tier autonomy scenarios
        elif self.config.scenario == Scenario.NETWORK:
            # Network-scale data: multi-colony telemetry
            n_colonies = getattr(self, "network_colonies", 100)
            data = self.rng.normal(0, 1, n_colonies * 10)  # 10 metrics per colony
        elif self.config.scenario == Scenario.ADVERSARIAL:
            # Adversarial scenario: mix of legitimate and attack data
            if self.rng.random() < 0.1:  # 10% attack attempts
                # Corrupted data pattern
                data = np.concatenate(
                    [
                        self.rng.normal(0, 1, 500),
                        self.rng.normal(100, 0.1, 500),  # Anomalous spike
                    ]
                )
            else:
                data = self.rng.normal(0, 1, 1000)
        else:  # BASELINE
            # Standard normal distribution
            data = self.rng.normal(0, 1, 1000)

        return data.astype(np.float64).tobytes()

    def _execute_modules(self, input_data: bytes) -> Dict[str, Any]:
        """Execute configured modules.

        Args:
            input_data: Input data bytes

        Returns:
            Dict of module results
        """
        results = {}

        for module_id in self.config.modules:
            if module_id == "compress":
                results["compress"] = self._run_compress(input_data)
            elif module_id == "witness":
                results["witness"] = self._run_witness(input_data)
            elif module_id == "sovereignty":
                results["sovereignty"] = self._run_sovereignty()
            elif module_id == "ledger":
                results["ledger"] = self._run_ledger(input_data)
            elif module_id == "detect":
                results["detect"] = self._run_detect(input_data)
            elif module_id == "anchor":
                results["anchor"] = self._run_anchor(list(results.values()))
            elif module_id == "loop":
                results["loop"] = self._run_loop()

        return results

    def _run_compress(self, data: bytes) -> Dict:
        """Run compress module simulation."""
        import zlib

        compressed = zlib.compress(data, level=9)
        ratio = len(data) / len(compressed)
        return {
            "compression_ratio": ratio,
            "input_size": len(data),
            "output_size": len(compressed),
            "passed_slo": ratio >= 10,
            "recall": 0.999,
            "algorithm": "hybrid",
        }

    def _run_witness(self, data: bytes) -> Dict:
        """Run witness module simulation."""
        # Simulate KAN compression analysis
        return {
            "compression": float(self.rng.uniform(0.8, 1.2)),
            "r_squared": float(self.rng.uniform(0.7, 0.99)),
            "equation": "a*x + b",
            "epochs": 100,
        }

    def _run_sovereignty(self) -> Dict:
        """Run sovereignty module simulation."""
        crew = int(self.rng.integers(10, 100))
        internal = np.log2(1 + crew * 10)
        external = (2.0 * 1e6) / (2 * 480)
        advantage = internal - external
        return {
            "crew": crew,
            "internal_rate": internal,
            "external_rate": external,
            "advantage": advantage,
            "sovereign": advantage > 0,
            "threshold_crew": 50,
        }

    def _run_ledger(self, data: bytes) -> Dict:
        """Run ledger module simulation."""
        from spaceproof.core import dual_hash

        return {
            "entry_count": int(self.rng.integers(1, 100)),
            "merkle_root": dual_hash(data),
            "valid": True,
        }

    def _run_detect(self, data: bytes) -> Dict:
        """Run detect module simulation."""
        h = shannon_entropy(data)
        delta = float(self.rng.normal(0, 0.5))

        if abs(delta) > 3.0:
            classification = "fraud"
        elif abs(delta) > 2.0:
            classification = "violation"
        elif abs(delta) > 1.0:
            classification = "drift"
        else:
            classification = "normal"

        return {
            "entropy": h.normalized,
            "delta": delta,
            "delta_sigma": abs(delta),
            "classification": classification,
            "severity": "low" if classification == "normal" else "high",
            "confidence": 0.95,
        }

    def _run_anchor(self, items: List) -> Dict:
        """Run anchor module simulation."""
        root = merkle(items)
        return {
            "root": root,
            "item_count": len(items),
            "algorithm": "dual_hash_merkle",
        }

    def _run_loop(self) -> Dict:
        """Run loop module simulation."""
        cycle_time = float(self.rng.uniform(10, 50))
        return {
            "cycle_id": str(self.rng.integers(0, 10000)),
            "cycle_time_sec": cycle_time,
            "completed": cycle_time <= 60,
            "actions_proposed": int(self.rng.integers(1, 10)),
            "actions_executed": int(self.rng.integers(0, 5)),
            "phase_timings": {"sense": 1.0, "analyze": 2.0, "actuate": 3.0},
        }

    def _build_receipt(self, module_results: Dict) -> Optional[SpaceProofReceipt]:
        """Build domain-specific receipt.

        Args:
            module_results: Results from module execution

        Returns:
            SpaceProofReceipt or None
        """
        domain = self.config.domain
        prev_proof = self.receipts[-1].hash() if self.receipts else None

        try:
            if domain == DomainConfig.XAI:
                return build_xai_receipt(
                    module_results.get("compress", {}),
                    module_results.get("witness", {}),
                    module_results.get("sovereignty", {}),
                    previous_proof=prev_proof,
                )
            elif domain == DomainConfig.DOGE:
                return build_doge_receipt(
                    module_results.get("ledger", {}),
                    module_results.get("detect", {}),
                    module_results.get("anchor", {}),
                    previous_proof=prev_proof,
                )
            elif domain == DomainConfig.NASA:
                return build_nasa_receipt(
                    module_results.get("compress", {}),
                    module_results.get("sovereignty", {}),
                    module_results.get("loop", {}),
                    previous_proof=prev_proof,
                )
            elif domain == DomainConfig.DEFENSE:
                return build_defense_receipt(
                    module_results.get("compress", {}),
                    module_results.get("ledger", {}),
                    module_results.get("anchor", {}),
                    classification="unclassified",
                    chain_position=len(self.receipts),
                    previous_proof=prev_proof,
                )
            elif domain == DomainConfig.DOT:
                return build_dot_receipt(
                    module_results.get("compress", {}),
                    module_results.get("ledger", {}),
                    module_results.get("detect", {}),
                    previous_proof=prev_proof,
                )
        except Exception:
            return None

        return None

    def _create_checkpoint(self, step: int) -> None:
        """Create checkpoint at current step.

        Args:
            step: Current step number
        """
        recent = self.step_results[-self.config.checkpoint_frequency :]
        avg_delta = np.mean([r.entropy_delta for r in recent])
        avg_coherence = np.mean([r.coherence for r in recent])
        alive_count = sum(1 for r in recent if r.is_alive)

        checkpoint = {
            "step": step,
            "avg_entropy_delta": avg_delta,
            "avg_coherence": avg_coherence,
            "alive_ratio": alive_count / len(recent),
            "receipts_count": len(self.receipts),
            "timestamp": time.time(),
        }
        self.checkpoints.append(checkpoint)

        emit_receipt(
            "simulation_checkpoint",
            {
                "tenant_id": TENANT_ID,
                "domain": self.config.domain.value,
                "scenario": self.config.scenario.value,
                **checkpoint,
            },
        )

    def _emit_genesis_receipt(self) -> None:
        """Emit genesis receipt for GENESIS scenario."""
        emit_receipt(
            "simulation_genesis",
            {
                "tenant_id": TENANT_ID,
                "domain": self.config.domain.value,
                "modules": self.config.modules,
                "seed": self.config.seed,
                "steps": self.config.steps,
            },
        )

    def _create_result(
        self,
        steps_completed: int,
        passed: bool,
        total_duration: float = 0.0,
        failure_reason: Optional[str] = None,
    ) -> SimulationResult:
        """Create final simulation result.

        Args:
            steps_completed: Number of steps completed
            passed: Whether simulation passed
            total_duration: Total duration in ms
            failure_reason: Reason for failure if any

        Returns:
            SimulationResult
        """
        if not self.step_results:
            return SimulationResult(
                config=self.config,
                steps_completed=0,
                total_duration_ms=total_duration,
                entropy_deltas=[],
                coherence_scores=[],
                alive_ratio=0.0,
                fitness_score=0.0,
                checkpoints=self.checkpoints,
                final_merkle_root="",
                receipts=self.receipts,
                passed=False,
                failure_reason=failure_reason or "No steps completed",
            )

        deltas = [r.entropy_delta for r in self.step_results]
        coherences = [r.coherence for r in self.step_results]
        alive_count = sum(1 for r in self.step_results if r.is_alive)

        # Compute fitness: total entropy reduction / receipts
        total_reduction = sum(max(0, d) for d in deltas)
        fit = fitness_score(total_reduction, len(self.receipts)) if self.receipts else 0.0

        # Build final Merkle root
        receipt_dicts = [r.to_dict() for r in self.receipts] if self.receipts else []
        final_root = merkle(receipt_dicts) if receipt_dicts else ""

        return SimulationResult(
            config=self.config,
            steps_completed=steps_completed,
            total_duration_ms=total_duration,
            entropy_deltas=deltas,
            coherence_scores=coherences,
            alive_ratio=alive_count / len(self.step_results),
            fitness_score=fit,
            checkpoints=self.checkpoints,
            final_merkle_root=final_root,
            receipts=self.receipts,
            passed=passed,
            failure_reason=failure_reason,
        )


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run a single simulation with given config.

    Args:
        config: Simulation configuration

    Returns:
        SimulationResult
    """
    engine = MonteCarloEngine(config)
    return engine.run()


def run_domain_simulation(
    domain: DomainConfig,
    scenario: Scenario = Scenario.BASELINE,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
) -> SimulationResult:
    """Convenience function to run domain simulation.

    Args:
        domain: Domain configuration
        scenario: Validation scenario
        steps: Number of steps
        seed: Random seed

    Returns:
        SimulationResult
    """
    config = SimulationConfig(
        domain=domain,
        scenario=scenario,
        steps=steps,
        seed=seed,
    )
    return run_simulation(config)
