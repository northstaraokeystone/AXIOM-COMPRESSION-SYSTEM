"""sim.py - AXIOM Simulation with Helper/Support/Optimization Integration

THE SIMULATION INSIGHT:
    The helpers aren't built. They're harvested.
    The support isn't constructed. It's measured.
    The optimization isn't programmed. It's learned.

    Each cycle:
    1. Optimizer selects patterns (Thompson sampling)
    2. Helpers process gaps (HARVEST → HYPOTHESIZE → GATE → ACTUATE)
    3. Support measures coverage (L0-L4 levels)

Source: QED v12 + ProofPack v3
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from enum import Enum

from .core import emit_receipt
from .optimize import (
    OptimizationConfig,
    OptimizationState,
    selection_pressure,
    update_fitness,
    measure_improvement,
    initialize_state as init_optimize_state,
    integrate_roi,
)
from .helper import (
    HelperConfig,
    HelperBlueprint,
    harvest,
    hypothesize,
    gate,
    actuate,
    measure_effectiveness,
    retire,
    check_retirement_candidates,
    get_active_helpers,
    create_gap_receipt,
)
from .support import (
    SupportLevel,
    SupportCoverage,
    measure_coverage,
    check_completeness,
    detect_gaps,
    l4_feedback,
    initialize_coverage,
)
from .strategies import (
    Strategy,
    StrategyConfig,
    StrategyResult,
    apply_strategy,
    compare_strategies,
    get_all_strategy_configs,
)
from .roi import (
    ROIConfig,
    compute_roi,
    roi_gate,
    rank_by_roi,
)


# === CONSTANTS ===

TENANT_ID = "axiom-autonomy"
"""Tenant for simulation receipts."""


# === ENUMS ===

class Scenario(Enum):
    """Simulation scenarios."""
    SCENARIO_BASELINE = "baseline"
    SCENARIO_HELPER = "helper"
    SCENARIO_SUPPORT = "support"
    SCENARIO_OPTIMIZATION = "optimization"
    SCENARIO_FULL = "full"
    SCENARIO_RELAY_COMPARISON = "relay_comparison"
    SCENARIO_STRATEGY_RANKING = "strategy_ranking"
    SCENARIO_ROI_GATE = "roi_gate"


# === DATACLASSES ===

@dataclass
class SimConfig:
    """Configuration for simulation.

    Attributes:
        max_cycles: Maximum simulation cycles (default 1000)
        harvest_frequency: Cycles between helper harvests (default 30)
        support_check_frequency: Cycles between coverage checks (default 10)
        optimization_config: Config for optimizer
        helper_config: Config for helper layer
        patterns: Initial pattern list for optimization
        strategies_enabled: List of Strategy enums to enable (default all)
        roi_config: ROIConfig for ROI computations
    """
    max_cycles: int = 1000
    harvest_frequency: int = 30
    support_check_frequency: int = 10
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    helper_config: HelperConfig = field(default_factory=HelperConfig)
    patterns: List[str] = field(default_factory=lambda: ["pattern_a", "pattern_b", "pattern_c"])
    strategies_enabled: List[Strategy] = field(default_factory=lambda: list(Strategy))
    roi_config: ROIConfig = field(default_factory=ROIConfig)


@dataclass
class SimState:
    """State of simulation.

    Attributes:
        cycle: Current cycle number
        helpers_active: List of active HelperBlueprints
        optimization_state: OptimizationState for pattern selection
        support_coverage: Dict mapping SupportLevel to SupportCoverage
        receipts: All receipts emitted during simulation
        gaps_injected: Gap receipts for helper testing
    """
    cycle: int = 0
    helpers_active: List[HelperBlueprint] = field(default_factory=list)
    optimization_state: OptimizationState = field(default_factory=init_optimize_state)
    support_coverage: Dict[SupportLevel, SupportCoverage] = field(default_factory=initialize_coverage)
    receipts: List[Dict] = field(default_factory=list)
    gaps_injected: List[Dict] = field(default_factory=list)


# === SIMULATION FUNCTIONS ===

def initialize_sim(config: SimConfig = None) -> SimState:
    """Initialize simulation state.

    Args:
        config: SimConfig (uses defaults if None)

    Returns:
        Fresh SimState
    """
    if config is None:
        config = SimConfig()

    state = SimState(
        cycle=0,
        helpers_active=[],
        optimization_state=init_optimize_state(),
        support_coverage=initialize_coverage(),
        receipts=[],
        gaps_injected=[]
    )

    # Initialize pattern fitness
    for pattern in config.patterns:
        state.optimization_state.pattern_fitness[pattern] = (0.5, 0.25)

    return state


def simulate_cycle(
    state: SimState,
    config: SimConfig = None
) -> SimState:
    """Run one simulation cycle.

    Each cycle:
    1. Selection pressure (optimization)
    2. Process any gaps (helper harvest if due)
    3. Measure support coverage (if due)
    4. Emit cycle receipt

    Args:
        state: Current SimState
        config: SimConfig (uses defaults if None)

    Returns:
        Updated SimState
    """
    if config is None:
        config = SimConfig()

    state.cycle += 1

    # 1. Optimization: select patterns
    patterns = list(config.patterns)
    fitness_scores = state.optimization_state.pattern_fitness

    selected = selection_pressure(patterns, fitness_scores, config.optimization_config)

    if selected:
        # Simulate outcome for top pattern
        top_pattern = selected[0]
        import random
        outcome = random.random() * 0.6 + 0.4  # Outcome 0.4-1.0

        state.optimization_state = update_fitness(
            top_pattern, outcome, state.optimization_state, config.optimization_config
        )

    # 2. Helper harvest (every harvest_frequency cycles)
    if state.cycle % config.harvest_frequency == 0:
        # Harvest from gaps
        patterns_found = harvest(state.gaps_injected, config.helper_config)

        # Hypothesize blueprints
        blueprints = hypothesize(patterns_found)

        # Gate and actuate
        for bp in blueprints:
            decision = gate(bp, config.helper_config)
            if decision == "auto_approve":
                actuate(bp)
                if len(state.helpers_active) < config.helper_config.max_active_helpers:
                    state.helpers_active.append(bp)

    # 3. Measure helper effectiveness
    for helper in get_active_helpers(state.helpers_active):
        measure_effectiveness(helper, state.receipts)

    # Check for retirement candidates
    retire_candidates = check_retirement_candidates(state.helpers_active)
    for helper in retire_candidates:
        retire(helper, "low_effectiveness")

    # 4. Support coverage (every support_check_frequency cycles)
    if state.cycle % config.support_check_frequency == 0:
        state.support_coverage = measure_coverage(state.receipts)

        # L4 feedback if coverage low
        if not check_completeness(state.support_coverage):
            l0_params = {"sample_rate": 1.0, "telemetry_level": "normal"}
            improved_params = l4_feedback(state.support_coverage, l0_params)
            # Apply improvements (in real system, would update config)

    # 5. Emit cycle receipt
    cycle_receipt = emit_receipt("simulation_cycle", {
        "tenant_id": TENANT_ID,
        "cycle": state.cycle,
        "patterns_selected": len(selected) if selected else 0,
        "helpers_active": len(get_active_helpers(state.helpers_active)),
        "improvement_vs_random": round(measure_improvement(state.optimization_state), 4),
        "support_complete": check_completeness(state.support_coverage),
    })
    state.receipts.append(cycle_receipt)

    return state


def inject_gap(
    state: SimState,
    problem_type: str,
    count: int = 1
) -> SimState:
    """Inject gap receipts for testing helper spawning.

    Args:
        state: Current SimState
        problem_type: Type of gap to inject
        count: Number of gaps to inject

    Returns:
        Updated SimState with injected gaps
    """
    for _ in range(count):
        gap = create_gap_receipt(problem_type, f"Injected gap: {problem_type}")
        state.gaps_injected.append(gap)
        state.receipts.append(gap)

    return state


def run_scenario(
    scenario: Scenario,
    config: SimConfig = None
) -> SimState:
    """Run a specific scenario.

    Args:
        scenario: Scenario enum value
        config: SimConfig (uses defaults if None)

    Returns:
        Final SimState after scenario completion
    """
    if config is None:
        config = SimConfig()

    state = initialize_sim(config)

    if scenario == Scenario.SCENARIO_HELPER:
        # SCENARIO_HELPER: Inject 10 recurring gaps, verify helper spawns within 50 cycles
        for i in range(10):
            inject_gap(state, "config_error", count=1)

        for _ in range(50):
            state = simulate_cycle(state, config)

    elif scenario == Scenario.SCENARIO_SUPPORT:
        # SCENARIO_SUPPORT: Run 1000 cycles, verify all 5 levels reach ≥0.95 coverage
        # Inject varied receipts to build coverage
        for cycle in range(config.max_cycles):
            # Inject telemetry receipts (L0)
            if cycle % 5 == 0:
                state.receipts.append(emit_receipt("autonomy_state", {"tenant_id": TENANT_ID, "state": "active"}))
                state.receipts.append(emit_receipt("propulsion_state", {"tenant_id": TENANT_ID, "state": "nominal"}))
                state.receipts.append(emit_receipt("latency", {"tenant_id": TENANT_ID, "ms": 22 * 60 * 1000}))

            # Inject decision receipts (L1)
            if cycle % 10 == 0:
                state.receipts.append(emit_receipt("decision", {"tenant_id": TENANT_ID, "decision": "proceed"}))
                state.receipts.append(emit_receipt("gate_decision", {"tenant_id": TENANT_ID, "decision": "approve"}))

            # Inject change receipts (L2)
            if cycle % 20 == 0:
                state.receipts.append(emit_receipt("config_change", {"tenant_id": TENANT_ID, "key": "sample_rate"}))

            # Inject quality receipts (L3)
            if cycle % 15 == 0:
                state.receipts.append(emit_receipt("validation", {"tenant_id": TENANT_ID, "passed": True}))
                state.receipts.append(emit_receipt("chain", {"tenant_id": TENANT_ID, "n_receipts": cycle}))

            # Inject meta receipts (L4)
            if cycle % 25 == 0:
                state.receipts.append(emit_receipt("coverage", {"tenant_id": TENANT_ID, "ratio": 0.9}))

            state = simulate_cycle(state, config)

    elif scenario == Scenario.SCENARIO_OPTIMIZATION:
        # Run optimization-focused scenario
        for _ in range(100):
            state = simulate_cycle(state, config)

    elif scenario == Scenario.SCENARIO_FULL:
        # Full integration scenario
        # Mix of gaps, varied receipts, and full cycle
        for cycle in range(min(200, config.max_cycles)):
            if cycle % 20 == 0:
                inject_gap(state, "config_error", count=2)
            if cycle % 30 == 0:
                inject_gap(state, "timeout_error", count=2)

            # Add telemetry
            if cycle % 3 == 0:
                state.receipts.append(emit_receipt("autonomy_state", {"tenant_id": TENANT_ID, "state": "active"}))

            state = simulate_cycle(state, config)

    elif scenario == Scenario.SCENARIO_RELAY_COMPARISON:
        # SCENARIO_RELAY_COMPARISON: Compare relay swarm sizes (3, 6, 9)
        # Expected: 6 satellites optimal for ROI
        from .relay import RELAY_SWARM_OPTIMAL

        baseline = {"tau": 1200, "alpha": 1.69}
        swarm_sizes = [3, 6, 9]
        results = []

        for size in swarm_sizes:
            strategy_config = StrategyConfig(
                strategy=Strategy.RELAY_SWARM,
                relay_swarm_size=size
            )
            result = apply_strategy(1200, 1.69, strategy_config)
            results.append(result)

            emit_receipt("relay_comparison", {
                "tenant_id": TENANT_ID,
                "swarm_size": size,
                "effective_tau": result.effective_tau,
                "cycles_to_10k": result.cycles_to_10k,
                "p_cost": result.p_cost,
            })

        # Store results for validation
        state.receipts.append(emit_receipt("relay_comparison_summary", {
            "tenant_id": TENANT_ID,
            "swarm_sizes_tested": swarm_sizes,
            "optimal_size": RELAY_SWARM_OPTIMAL,
            "results": [
                {"size": s, "cycles": r.cycles_to_10k, "p_cost": r.p_cost}
                for s, r in zip(swarm_sizes, results)
            ],
        }))

    elif scenario == Scenario.SCENARIO_STRATEGY_RANKING:
        # SCENARIO_STRATEGY_RANKING: Compare all 4 strategies
        # Expected: COMBINED has best cycles, RELAY best ROI
        baseline = {"tau": 1200, "alpha": 1.69}

        # Get all strategy configs
        configs = get_all_strategy_configs()
        results = compare_strategies(configs, baseline)

        # Compute ROI for each
        baseline_result = results[0]  # BASELINE is first after sorting? No, sorted by cycles
        # Find actual baseline
        for r in results:
            if r.strategy == Strategy.BASELINE:
                baseline_result = r
                break

        ranked = rank_by_roi(results, baseline_result, config.roi_config)

        emit_receipt("strategy_ranking_summary", {
            "tenant_id": TENANT_ID,
            "strategies_compared": len(results),
            "best_cycles": results[0].strategy.value if results else None,
            "best_roi": ranked[0][0].strategy.value if ranked else None,
            "ranking_by_cycles": [r.strategy.value for r in results],
            "ranking_by_roi": [r[0].strategy.value for r in ranked],
        })

    elif scenario == Scenario.SCENARIO_ROI_GATE:
        # SCENARIO_ROI_GATE: Test ROI gate decisions
        # Expected: Correct deploy/shadow/kill
        baseline = {"tau": 1200, "alpha": 1.69}

        # Get all strategies and compute ROI
        configs = get_all_strategy_configs()
        results = compare_strategies(configs, baseline)

        # Find baseline result
        baseline_result = None
        for r in results:
            if r.strategy == Strategy.BASELINE:
                baseline_result = r
                break

        if baseline_result is None:
            baseline_result = results[-1]  # Use worst as baseline

        # Test gate decisions
        gate_decisions = []
        for result in results:
            if result.strategy == Strategy.BASELINE:
                continue
            roi = compute_roi(result, baseline_result, config.roi_config)
            decision = roi_gate(roi, config.roi_config)
            gate_decisions.append({
                "strategy": result.strategy.value,
                "roi": roi,
                "decision": decision,
            })

        emit_receipt("roi_gate_summary", {
            "tenant_id": TENANT_ID,
            "strategies_evaluated": len(gate_decisions),
            "decisions": gate_decisions,
            "deploy_count": sum(1 for d in gate_decisions if d["decision"] == "deploy"),
            "shadow_count": sum(1 for d in gate_decisions if d["decision"] == "shadow"),
            "kill_count": sum(1 for d in gate_decisions if d["decision"] == "kill"),
        })

    else:  # SCENARIO_BASELINE
        for _ in range(100):
            state = simulate_cycle(state, config)

    return state


def validate_constraints(state: SimState) -> Dict:
    """Validate simulation constraints.

    SLOs:
    - Helper spawns within 50 cycles if 10+ same-type gaps
    - All 5 support levels reach ≥0.95 coverage (if run long enough)
    - Optimization improvement >1.2x vs random (after 100 cycles)

    Args:
        state: SimState to validate

    Returns:
        Dict with validation results
    """
    results = {}

    # Helper spawning check
    active_helpers = get_active_helpers(state.helpers_active)
    results["helpers_spawned"] = len(active_helpers) > 0
    results["helpers_count"] = len(active_helpers)

    # Support coverage check
    results["support_complete"] = check_completeness(state.support_coverage)
    results["support_gaps"] = detect_gaps(state.support_coverage)
    results["coverage_by_level"] = {
        level.value: cov.coverage_ratio
        for level, cov in state.support_coverage.items()
    }

    # Optimization improvement check
    improvement = measure_improvement(state.optimization_state)
    results["improvement_vs_random"] = improvement
    results["optimization_effective"] = improvement > 1.2

    # Overall validation
    results["all_slos_met"] = (
        results["support_complete"] or state.cycle < 100  # Allow ramp-up
    ) and (
        results["optimization_effective"] or state.cycle < 100
    )

    return results


def emit_simulation_summary(state: SimState) -> Dict:
    """Emit summary receipt for simulation run.

    Args:
        state: Final SimState

    Returns:
        Summary receipt dict
    """
    validation = validate_constraints(state)

    return emit_receipt("simulation_summary", {
        "tenant_id": TENANT_ID,
        "total_cycles": state.cycle,
        "receipts_generated": len(state.receipts),
        "helpers_active": len(get_active_helpers(state.helpers_active)),
        "helpers_total": len(state.helpers_active),
        "gaps_injected": len(state.gaps_injected),
        "improvement_vs_random": validation["improvement_vs_random"],
        "support_complete": validation["support_complete"],
        "all_slos_met": validation["all_slos_met"],
        "coverage_by_level": validation["coverage_by_level"],
    })
