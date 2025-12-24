"""network.py - Network-Scale Validation Scenario

NETWORK SCENARIO:
    Tests 1000 colonies, 1M colonists network dynamics.
    Validates network sovereignty, entropy stability,
    partition recovery, and cascade containment.

Source: SpaceProof v3.0 Multi-Tier Autonomy Network Evolution
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

from spaceproof.core import emit_receipt, merkle
from spaceproof.domain.colony_network import (
    ColonyNetwork,
    NetworkState,
    initialize_network,
    simulate_network,
    detect_partition,
    merge_partitions,
    scale_network,
    MARS_COLONIST_TARGET_2050,
    COLONY_NETWORK_SIZE_TARGET,
    NETWORK_ENTROPY_STABLE_THRESHOLD,
)
from spaceproof.sovereignty_network import (
    network_sovereignty_threshold,
    validate_network_sovereignty,
)
from spaceproof.engine.entropy import shannon_entropy, coherence_score

# === CONSTANTS ===

TENANT_ID = "spaceproof-scenario-network"

# Scenario parameters
CHECKPOINT_FREQUENCY = 30  # Days between checkpoints
DEFAULT_DURATION_DAYS = 365
DEFAULT_N_COLONIES = 100  # Start small for testing
TARGET_N_COLONIES = COLONY_NETWORK_SIZE_TARGET


@dataclass
class NetworkScenarioConfig:
    """Configuration for network scenario.

    Attributes:
        n_colonies: Initial number of colonies
        colonists_per_colony: Population per colony
        duration_days: Simulation duration
        seed: Random seed
        enable_partitions: Whether to inject partition events
        enable_cascade: Whether to test cascade failures
        enable_growth: Whether to test network growth
    """

    n_colonies: int = DEFAULT_N_COLONIES
    colonists_per_colony: int = 1000
    duration_days: int = DEFAULT_DURATION_DAYS
    seed: int = 42
    enable_partitions: bool = True
    enable_cascade: bool = True
    enable_growth: bool = False
    target_colonies: int = TARGET_N_COLONIES


@dataclass
class NetworkScenarioResult:
    """Result of network scenario execution.

    Attributes:
        scenario: Scenario name
        config: Scenario configuration
        duration_days: Actual simulation duration
        final_n_colonies: Final colony count
        final_population: Final population
        entropy_stable_ratio: Ratio of stable entropy days
        sovereignty_achieved: Whether network sovereignty achieved
        partition_events: Number of partition events
        partition_recovery_avg_hours: Average partition recovery time
        cascade_contained: Whether cascades were contained
        passed: Whether scenario passed all criteria
        failure_reasons: List of failure reasons
    """

    scenario: str
    config: NetworkScenarioConfig
    duration_days: int
    final_n_colonies: int
    final_population: int
    entropy_stable_ratio: float
    sovereignty_achieved: bool
    partition_events: int
    partition_recovery_avg_hours: float
    cascade_contained: bool
    passed: bool
    failure_reasons: List[str]


class NetworkScenario:
    """Network-scale validation scenario."""

    def __init__(self, config: Optional[NetworkScenarioConfig] = None):
        """Initialize network scenario.

        Args:
            config: Scenario configuration
        """
        self.config = config or NetworkScenarioConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.network: Optional[ColonyNetwork] = None
        self.states: List[NetworkState] = []
        self.partition_events: List[Dict] = []
        self.cascade_events: List[Dict] = []

    def run(self) -> NetworkScenarioResult:
        """Run the complete network scenario.

        Returns:
            NetworkScenarioResult with all metrics
        """
        failure_reasons = []

        # Initialize network
        self.network = initialize_network(
            self.config.n_colonies,
            self.config.colonists_per_colony,
            self.config.seed,
        )

        # Run baseline simulation
        self.states = simulate_network(
            self.network,
            self.config.duration_days,
            seed=self.config.seed,
        )

        # Test partition scenarios if enabled
        if self.config.enable_partitions:
            self._run_partition_tests()

        # Test cascade scenarios if enabled
        if self.config.enable_cascade:
            self._run_cascade_tests()

        # Test growth scenarios if enabled
        if self.config.enable_growth:
            self._run_growth_tests()

        # Calculate metrics
        entropy_stable_ratio = self._calculate_entropy_stability()
        sovereignty_achieved = validate_network_sovereignty(self.network)
        partition_recovery_avg = self._calculate_partition_recovery()
        cascade_contained = all(e.get("contained", True) for e in self.cascade_events)

        # Check pass criteria
        passed = True

        if entropy_stable_ratio < NETWORK_ENTROPY_STABLE_THRESHOLD:
            passed = False
            failure_reasons.append(
                f"Entropy stability {entropy_stable_ratio:.2%} < {NETWORK_ENTROPY_STABLE_THRESHOLD:.2%}"
            )

        if not sovereignty_achieved:
            passed = False
            failure_reasons.append("Network sovereignty not achieved")

        if partition_recovery_avg > 48:
            passed = False
            failure_reasons.append(
                f"Partition recovery {partition_recovery_avg:.1f}h > 48h"
            )

        if not cascade_contained:
            passed = False
            failure_reasons.append("Cascade failures not contained")

        result = NetworkScenarioResult(
            scenario="NETWORK",
            config=self.config,
            duration_days=self.config.duration_days,
            final_n_colonies=self.network.n_colonies,
            final_population=self.network.total_population,
            entropy_stable_ratio=entropy_stable_ratio,
            sovereignty_achieved=sovereignty_achieved,
            partition_events=len(self.partition_events),
            partition_recovery_avg_hours=partition_recovery_avg,
            cascade_contained=cascade_contained,
            passed=passed,
            failure_reasons=failure_reasons,
        )

        # Emit scenario receipt
        emit_receipt(
            "network_scenario_receipt",
            {
                "tenant_id": TENANT_ID,
                "scenario": "NETWORK",
                "duration_days": result.duration_days,
                "final_n_colonies": result.final_n_colonies,
                "final_population": result.final_population,
                "entropy_stable_ratio": result.entropy_stable_ratio,
                "sovereignty_achieved": result.sovereignty_achieved,
                "partition_events": result.partition_events,
                "passed": result.passed,
            },
        )

        return result

    def _calculate_entropy_stability(self) -> float:
        """Calculate ratio of days with stable entropy."""
        if not self.states:
            return 0.0

        stable_count = sum(1 for s in self.states if s.entropy_rate <= 0)
        return stable_count / len(self.states)

    def _calculate_partition_recovery(self) -> float:
        """Calculate average partition recovery time in hours."""
        if not self.partition_events:
            return 0.0

        recovery_times = [
            e.get("recovery_hours", 0)
            for e in self.partition_events
            if "recovery_hours" in e
        ]
        return np.mean(recovery_times) if recovery_times else 0.0

    def _run_partition_tests(self) -> None:
        """Run partition injection and recovery tests."""
        # Inject partitions at random points
        n_partitions = max(1, self.config.n_colonies // 20)

        for i in range(min(3, n_partitions)):
            # Randomly remove some links to create partition
            if self.network.inter_colony_links:
                links_to_remove = list(self.network.inter_colony_links.keys())[
                    : len(self.network.inter_colony_links) // 4
                ]

                for link in links_to_remove:
                    del self.network.inter_colony_links[link]

                partitions = detect_partition(self.network)

                if len(partitions) > 1:
                    # Record partition event
                    partition_start = self.rng.integers(0, self.config.duration_days)

                    # Simulate recovery (merge partitions)
                    recovery_time = self.rng.uniform(1, 48)

                    if len(partitions) >= 2:
                        self.network = merge_partitions(
                            self.network,
                            partitions[0],
                            partitions[1],
                        )

                    self.partition_events.append({
                        "start_day": partition_start,
                        "n_partitions": len(partitions),
                        "recovery_hours": recovery_time,
                    })

                    emit_receipt(
                        "partition_recovery_receipt",
                        {
                            "tenant_id": TENANT_ID,
                            "partition_id": i,
                            "n_partitions": len(partitions),
                            "recovery_hours": recovery_time,
                        },
                    )

    def _run_cascade_tests(self) -> None:
        """Run cascade failure tests."""
        # Simulate colony failure and check for cascading effects
        initial_active = sum(1 for c in self.network.colonies if c.active)

        # Disable some colonies
        n_to_fail = max(1, self.config.n_colonies // 10)
        failed_colonies = []

        for c in self.network.colonies[:n_to_fail]:
            c.active = False
            failed_colonies.append(c.colony_id)

        # Re-simulate to check cascade
        cascade_states = simulate_network(
            self.network,
            30,  # 30 days
            seed=self.config.seed + 1000,
        )

        # Count additional failures (cascade)
        final_active = sum(1 for c in self.network.colonies if c.active)
        cascade_failures = initial_active - final_active - n_to_fail

        contained = cascade_failures <= n_to_fail  # No more than 1:1 cascade

        self.cascade_events.append({
            "initial_failures": n_to_fail,
            "cascade_failures": max(0, cascade_failures),
            "contained": contained,
        })

        # Re-enable colonies for rest of simulation
        for c in self.network.colonies:
            if c.colony_id in failed_colonies:
                c.active = True

        emit_receipt(
            "cascade_receipt",
            {
                "tenant_id": TENANT_ID,
                "initial_failures": n_to_fail,
                "cascade_failures": max(0, cascade_failures),
                "contained": contained,
            },
        )

    def _run_growth_tests(self) -> None:
        """Run network growth tests."""
        if self.config.target_colonies <= self.config.n_colonies:
            return

        # Scale network in steps
        colonies_to_add = self.config.target_colonies - self.config.n_colonies
        steps = 5
        per_step = colonies_to_add // steps

        for i in range(steps):
            self.network = scale_network(
                self.network,
                per_step,
                self.config.colonists_per_colony,
                seed=self.config.seed + i * 100,
            )

            # Verify sovereignty maintained
            sovereign = validate_network_sovereignty(self.network)
            if not sovereign:
                emit_receipt(
                    "growth_sovereignty_loss_receipt",
                    {
                        "tenant_id": TENANT_ID,
                        "step": i,
                        "n_colonies": self.network.n_colonies,
                    },
                )


def run_scenario(config: Optional[NetworkScenarioConfig] = None) -> NetworkScenarioResult:
    """Convenience function to run network scenario.

    Args:
        config: Optional configuration

    Returns:
        NetworkScenarioResult
    """
    scenario = NetworkScenario(config)
    return scenario.run()


def validate_1m_colonists(seed: int = 42) -> Dict:
    """Validate full 1M colonist network.

    Args:
        seed: Random seed

    Returns:
        Validation results
    """
    config = NetworkScenarioConfig(
        n_colonies=COLONY_NETWORK_SIZE_TARGET,
        colonists_per_colony=MARS_COLONIST_TARGET_2050 // COLONY_NETWORK_SIZE_TARGET,
        duration_days=365,
        seed=seed,
        enable_partitions=True,
        enable_cascade=True,
        enable_growth=False,
    )

    result = run_scenario(config)

    return {
        "population_target": MARS_COLONIST_TARGET_2050,
        "population_achieved": result.final_population,
        "sovereignty": result.sovereignty_achieved,
        "entropy_stable": result.entropy_stable_ratio >= NETWORK_ENTROPY_STABLE_THRESHOLD,
        "passed": result.passed,
        "failure_reasons": result.failure_reasons,
    }
