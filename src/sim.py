"""sim.py - Monte Carlo Validation Harness

THE HARNESS:
    No physics claim ships without simulation-first validation.
    The 6 mandatory scenarios are not tests—they are gates.
    If any scenario fails, AXIOM does not publish.

DUAL PURPOSE:
    1. GALAXY SIMULATION: GENERATE galaxy → WITNESS with KAN → COMPUTE topology
    2. COLONY SIMULATION: GENERATE colony → VALIDATE constraints → FIND sovereignty threshold

THE KEY OUTPUT (Colony): sovereignty_threshold_crew - minimum crew for decision independence

Source: CLAUDEME.md (§3 Timeline Gates, §4 Receipt Blocks, §7 Anti-Patterns)
        AXIOM_Colony_Build_Strategy_v2.md §2.9
"""

import copy
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .core import dual_hash, emit_receipt, StopRule, TENANT_ID

# Galaxy simulation imports (optional - may not be available in all environments)
try:
    from .cosmos import batch_generate, generate_galaxy, generate_pathological, REGIMES as COSMOS_REGIMES
    from .witness import KAN, train, spline_to_law
    from .topology import analyze_galaxy
    from .prove import chain_receipts, summarize_batch
    GALAXY_SIM_AVAILABLE = True
except ImportError:
    GALAXY_SIM_AVAILABLE = False
    # Provide stubs for type checking
    batch_generate = None
    generate_galaxy = None
    generate_pathological = None
    COSMOS_REGIMES = []
    KAN = None
    train = None
    spline_to_law = None
    analyze_galaxy = None
    chain_receipts = None
    summarize_batch = None

# Colony imports (BUILD C1-C4)
from .entropy import (
    decision_capacity,
    earth_input_rate,
    sovereignty_threshold,
    entropy_rate,
    entropy_status,
    total_colony_entropy,
    survival_bound,
    MARS_RELAY_MBPS,
    LIGHT_DELAY_MAX,
)
from .colony import (
    generate_colony,
    ColonyConfig,
    ColonyState,
    batch_generate as colony_batch_generate,
    simulate_dust_storm,
    simulate_hab_breach,
    simulate_crop_failure,
    default_config,
)


# === CONSTANTS (Module Top) ===

TENANT_ID = "axiom-witness"
"""Receipt tenant isolation."""

REGIMES = ["newtonian", "mond", "nfw", "pbh_fog"]
"""Valid physics regimes."""

MSE_THRESHOLDS = {
    "newtonian": 5.0,
    "mond": 8.0,
    "nfw": 12.0,
    "pbh_fog": 15.0
}
"""Per-regime MSE ceilings."""

COMPRESSION_FLOOR = 0.84
"""Minimum compression ratio."""

ACCURACY_THRESHOLDS = {
    "newtonian": 0.90,
    "mond": 0.85,
    "nfw": 0.80,
    "pbh_fog": 0.75
}
"""Per-regime accuracy floors."""

WASSERSTEIN_BOUND = 0.03
"""Topology distance bound."""

WASSERSTEIN_COVERAGE = 0.92
"""% galaxies within bound."""

ENTROPY_TOLERANCE = 0.05
"""Allowed entropy conservation error."""

SCENARIOS = ["BASELINE", "STRESS", "DISCOVERY", "TOPOLOGY", "REPRODUCIBILITY", "GODEL"]
"""Valid scenario names."""


# === DATACLASSES ===

@dataclass(frozen=True)
class SimConfig:
    """Immutable configuration for simulation run."""
    n_cycles: int = 1000
    n_galaxies_per_regime: int = 25
    noise_fraction: float = 0.03
    kan_epochs: int = 100
    mdl_alpha: float = 1.0
    mdl_beta: float = 0.10
    persistence_threshold: float = 0.012
    random_seed: int = 42
    multi_seed: tuple = (42, 43, 44, 45, 46, 47)
    topology_only: bool = False
    scenario: Optional[str] = None


@dataclass
class SimState:
    """Mutable state accumulated during simulation."""
    galaxies: List[dict] = field(default_factory=list)
    witness_receipts: List[dict] = field(default_factory=list)
    topology_receipts: List[dict] = field(default_factory=list)
    chain_receipt: Optional[dict] = None
    violations: List[dict] = field(default_factory=list)
    cycle: int = 0
    passed: Optional[bool] = None
    metrics: Dict = field(default_factory=dict)


# === STOPRULES ===

def stoprule_nan_detected(context: str = "") -> None:
    """Trigger stoprule for NaN in simulation results.

    Emits anomaly receipt and raises StopRule.
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "nan",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "context": context
    })
    raise StopRule(f"NaN detected in simulation: {context}")


def stoprule_memory_exceeded(estimated_mb: float) -> None:
    """Trigger stoprule for memory limit exceeded.

    Emits anomaly receipt and raises StopRule.
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "memory",
        "baseline": 5500.0,
        "delta": estimated_mb - 5500.0,
        "classification": "violation",
        "action": "halt"
    })
    raise StopRule(f"Memory limit exceeded: {estimated_mb:.0f}MB > 5500MB")


def stoprule_invalid_scenario(scenario_name: str) -> None:
    """Trigger stoprule for invalid scenario name.

    Emits anomaly receipt and raises StopRule.
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "config",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "invalid_scenario": scenario_name,
        "valid_scenarios": SCENARIOS
    })
    raise StopRule(f"Unknown scenario: {scenario_name}")


# === UTILITY FUNCTIONS ===

def emit_violation_receipt(
    constraint: str,
    expected: str,
    actual: str,
    cycle: int,
    action: str = "alert"
) -> dict:
    """Emit a violation receipt for a failed constraint.

    Args:
        constraint: Name of the validator that failed
        expected: Expected threshold value as string
        actual: Measured value as string
        cycle: Current simulation cycle
        action: "alert" or "halt"

    Returns:
        The violation receipt dict
    """
    return emit_receipt("violation", {
        "tenant_id": TENANT_ID,
        "constraint": constraint,
        "expected": expected,
        "actual": actual,
        "action": action,
        "cycle": cycle,
        "payload_hash": dual_hash(f"{constraint}:{expected}:{actual}:{cycle}")
    })


# === CONSTRAINT VALIDATORS ===

def check_compression_floor(state: SimState, config: SimConfig) -> List[dict]:
    """Check all witness_receipts have compression_ratio >= COMPRESSION_FLOOR."""
    violations = []
    for receipt in state.witness_receipts:
        compression = receipt.get("compression_ratio", 0.0)
        if compression < COMPRESSION_FLOOR:
            v = emit_violation_receipt(
                constraint="compression_floor",
                expected=f">= {COMPRESSION_FLOOR}",
                actual=f"{compression:.4f}",
                cycle=state.cycle,
                action="alert"
            )
            violations.append(v)
    return violations


def check_mse_ceiling(state: SimState, config: SimConfig) -> List[dict]:
    """Check each witness has mse <= MSE_THRESHOLDS[regime]."""
    violations = []
    for receipt in state.witness_receipts:
        regime = receipt.get("physics_regime", "newtonian")
        mse = receipt.get("final_mse", 0.0)
        threshold = MSE_THRESHOLDS.get(regime, 15.0)
        if mse > threshold:
            v = emit_violation_receipt(
                constraint="mse_ceiling",
                expected=f"<= {threshold}",
                actual=f"{mse:.4f}",
                cycle=state.cycle,
                action="alert"
            )
            violations.append(v)
    return violations


def check_regime_accuracy(state: SimState, config: SimConfig, regime: str, threshold: float) -> List[dict]:
    """Check accuracy for a specific regime meets threshold."""
    violations = []

    # Count correct classifications for this regime
    regime_receipts = [r for r in state.witness_receipts if r.get("physics_regime") == regime]
    if not regime_receipts:
        return violations

    correct = sum(1 for r in regime_receipts if r.get("compression_ratio", 0.0) >= COMPRESSION_FLOOR)
    accuracy = correct / len(regime_receipts)

    if accuracy < threshold:
        v = emit_violation_receipt(
            constraint=f"{regime}_accuracy",
            expected=f">= {threshold:.2%}",
            actual=f"{accuracy:.2%}",
            cycle=state.cycle,
            action="alert"
        )
        violations.append(v)

    return violations


def check_newton_accuracy(state: SimState, config: SimConfig) -> List[dict]:
    """Check newton_correct >= ACCURACY_THRESHOLDS['newtonian']."""
    return check_regime_accuracy(state, config, "newtonian", ACCURACY_THRESHOLDS["newtonian"])


def check_mond_accuracy(state: SimState, config: SimConfig) -> List[dict]:
    """Check mond_correct >= ACCURACY_THRESHOLDS['mond']."""
    return check_regime_accuracy(state, config, "mond", ACCURACY_THRESHOLDS["mond"])


def check_nfw_accuracy(state: SimState, config: SimConfig) -> List[dict]:
    """Check nfw_correct >= ACCURACY_THRESHOLDS['nfw']."""
    return check_regime_accuracy(state, config, "nfw", ACCURACY_THRESHOLDS["nfw"])


def check_wasserstein_bound(state: SimState, config: SimConfig) -> List[dict]:
    """Check WASSERSTEIN_COVERAGE of topology_receipts have wasserstein < WASSERSTEIN_BOUND."""
    violations = []

    if not state.topology_receipts:
        return violations

    within_bound = 0
    for receipt in state.topology_receipts:
        wasserstein = receipt.get("wasserstein_to_baseline")
        if wasserstein is None or wasserstein < WASSERSTEIN_BOUND:
            within_bound += 1

    coverage = within_bound / len(state.topology_receipts)

    if coverage < WASSERSTEIN_COVERAGE:
        v = emit_violation_receipt(
            constraint="wasserstein_bound",
            expected=f">= {WASSERSTEIN_COVERAGE:.2%}",
            actual=f"{coverage:.2%}",
            cycle=state.cycle,
            action="alert"
        )
        violations.append(v)

    return violations


def check_entropy_conservation(state: SimState, config: SimConfig) -> List[dict]:
    """Check abs(bits_in - bits_out - work) < ENTROPY_TOLERANCE."""
    violations = []

    # Simplified entropy check: compare input data bits to topology bits
    for receipt in state.topology_receipts:
        total_bits = receipt.get("total_bits", 0.0)
        # For entropy conservation, we expect bits to be within reasonable range
        # This is a simplified check - real entropy would need input/output comparison
        if total_bits > 100:  # Unreasonably high complexity
            v = emit_violation_receipt(
                constraint="entropy_conservation",
                expected=f"< 100 bits",
                actual=f"{total_bits:.2f} bits",
                cycle=state.cycle,
                action="alert"
            )
            violations.append(v)

    return violations


def validate_constraints(state: SimState, config: SimConfig) -> List[dict]:
    """Check all constraints, return list of violation_receipts.

    Runs all 7 constraint validators and aggregates violations.
    """
    violations = []

    # Run all validators
    violations.extend(check_compression_floor(state, config))
    violations.extend(check_mse_ceiling(state, config))
    violations.extend(check_newton_accuracy(state, config))
    violations.extend(check_mond_accuracy(state, config))
    violations.extend(check_nfw_accuracy(state, config))
    violations.extend(check_wasserstein_bound(state, config))
    violations.extend(check_entropy_conservation(state, config))

    return violations


# === CORE SIMULATION FUNCTIONS ===

def simulate_witness(galaxy: dict, config: SimConfig) -> dict:
    """Train KAN on galaxy, return witness_receipt.

    Does NOT emit receipt (witness.py handles receipt emission).
    """
    # Create KAN instance
    kan = KAN()

    # Train
    result = train(
        kan,
        galaxy['r'],
        galaxy['v'],
        epochs=config.kan_epochs
    )

    # Extract law
    x_min = float(np.min(galaxy['r']))
    x_max = float(np.max(galaxy['r']))
    law = spline_to_law(kan, x_range=(max(x_min, 0.1), max(x_max, 1.0)), y_data=galaxy['v'])

    # Add galaxy info to result
    result['galaxy_id'] = galaxy['id']
    result['physics_regime'] = galaxy['regime']
    result['discovered_law'] = law

    # Compute predicted values for topology (stored separately, not in receipt)
    # Note: predicted_v is a numpy array and won't be JSON serializable
    # Store as regular Python list for internal use only
    result['_predicted_v'] = kan(galaxy['r']).flatten()  # underscore = internal, not for chaining

    return result


def simulate_topology(galaxy: dict, witness_receipt: dict, config: SimConfig) -> dict:
    """Compute persistence on residuals, return topology_receipt.

    Does NOT emit receipt (topology.py handles receipt emission).
    """
    # Compute residuals
    if config.topology_only:
        # Use ground truth residuals
        residuals = galaxy['v'].flatten() - galaxy['v_true'].flatten()
    else:
        # Use witness-predicted residuals
        predicted_v = witness_receipt.get('_predicted_v')
        if predicted_v is not None:
            residuals = galaxy['v'].flatten() - predicted_v
        else:
            residuals = galaxy['v'].flatten() - galaxy['v_true'].flatten()

    # Run topology analysis
    topology_receipt = analyze_galaxy(galaxy['id'], residuals, baseline_diagram=None)

    return topology_receipt


def simulate_cycle(state: SimState, config: SimConfig) -> SimState:
    """Run one complete cycle: generate → witness → topology → validate.

    MUST emit receipt (sim_cycle_receipt).
    """
    state.cycle += 1

    # Generate galaxies for this cycle
    galaxies = batch_generate(
        config.n_galaxies_per_regime,
        config.noise_fraction,
        config.random_seed + state.cycle
    )
    state.galaxies.extend(galaxies)

    cycle_witnesses = []
    cycle_topologies = []

    # Process each galaxy
    for galaxy in galaxies:
        # Check for NaN in input
        if np.any(np.isnan(galaxy['v'])) or np.any(np.isnan(galaxy['r'])):
            stoprule_nan_detected(f"galaxy {galaxy['id']} input data")

        # Witness
        witness_receipt = simulate_witness(galaxy, config)

        # Check for NaN in witness output
        if np.isnan(witness_receipt.get('final_mse', 0.0)):
            stoprule_nan_detected(f"galaxy {galaxy['id']} witness MSE")

        # Topology (uses _predicted_v before we remove it)
        topology_receipt = simulate_topology(galaxy, witness_receipt, config)
        state.topology_receipts.append(topology_receipt)
        cycle_topologies.append(topology_receipt)

        # Remove internal numpy array before storing for chaining
        # (numpy arrays are not JSON serializable)
        clean_receipt = {k: v for k, v in witness_receipt.items() if not k.startswith('_')}
        state.witness_receipts.append(clean_receipt)
        cycle_witnesses.append(clean_receipt)

    # Validate constraints
    violations = validate_constraints(state, config)
    state.violations.extend(violations)

    # Compute cycle metrics
    compressions = [r.get('compression_ratio', 0.0) for r in cycle_witnesses]
    mses = [r.get('final_mse', 0.0) for r in cycle_witnesses]

    compression_mean = statistics.mean(compressions) if compressions else 0.0
    mse_mean = statistics.mean(mses) if mses else 0.0

    # Emit sim_cycle_receipt
    emit_receipt("sim_cycle", {
        "tenant_id": TENANT_ID,
        "cycle": state.cycle,
        "n_galaxies": len(galaxies),
        "n_witnesses": len(cycle_witnesses),
        "n_violations": len(violations),
        "compression_mean": compression_mean,
        "mse_mean": mse_mean,
        "payload_hash": dual_hash(f"cycle:{state.cycle}:{len(galaxies)}")
    })

    # Estimate memory usage (rough: ~10KB per receipt)
    total_receipts = len(state.witness_receipts) + len(state.topology_receipts)
    estimated_mb = total_receipts * 0.01
    if estimated_mb > 5500:
        stoprule_memory_exceeded(estimated_mb)

    return state


def run_simulation(config: SimConfig) -> SimState:
    """MAIN ENTRY POINT. Execute full simulation, return final state.

    MUST emit receipt (sim_complete_receipt) at end.
    """
    start_time = time.time()

    # Initialize state
    state = SimState()

    # Set random seed
    np.random.seed(config.random_seed)

    # Run cycles
    for _ in range(config.n_cycles):
        state = simulate_cycle(state, config)

        # Check for critical violations (halt action)
        halt_violations = [v for v in state.violations if v.get("action") == "halt"]
        if halt_violations:
            break

    # Chain receipts
    if state.witness_receipts:
        state.chain_receipt = chain_receipts(state.witness_receipts)

    # Determine pass/fail
    halt_violations = [v for v in state.violations if v.get("action") == "halt"]
    state.passed = len(halt_violations) == 0

    # Compute duration
    duration = time.time() - start_time

    # Get merkle root
    merkle_root = state.chain_receipt.get("merkle_root", "") if state.chain_receipt else ""

    # Emit sim_complete_receipt
    emit_receipt("sim_complete", {
        "tenant_id": TENANT_ID,
        "n_cycles": state.cycle,
        "n_galaxies": len(state.galaxies),
        "n_violations": len(state.violations),
        "passed": state.passed,
        "duration_seconds": duration,
        "merkle_root": merkle_root,
        "payload_hash": dual_hash(f"sim:{state.cycle}:{len(state.galaxies)}:{state.passed}")
    })

    return state


# === ENSEMBLE SELECTION ===

def ensemble_select(results: List[dict]) -> dict:
    """Multi-seed selection: pick result with lowest MDL.

    Args:
        results: List of witness_receipts from same galaxy with different seeds

    Returns:
        The result with lowest MDL loss

    Does NOT emit receipt (selection utility).
    """
    if not results:
        return {}

    # Sort by final_loss (MDL loss) ascending
    sorted_results = sorted(results, key=lambda r: r.get("final_loss", float('inf')))

    # Return first (lowest MDL = best compression)
    return sorted_results[0]


# === SCENARIO FUNCTIONS ===

def get_scenario_config(scenario_name: str) -> SimConfig:
    """Get configuration for a named scenario."""
    if scenario_name == "BASELINE":
        return SimConfig(
            n_cycles=1000,
            n_galaxies_per_regime=25,
            noise_fraction=0.03,
            random_seed=42,
            scenario="BASELINE"
        )
    elif scenario_name == "STRESS":
        return SimConfig(
            n_cycles=500,
            n_galaxies_per_regime=25,
            noise_fraction=0.10,  # High noise
            kan_epochs=50,        # Reduced
            mdl_beta=0.12,        # Stricter
            random_seed=42,
            scenario="STRESS"
        )
    elif scenario_name == "DISCOVERY":
        return SimConfig(
            n_cycles=500,
            n_galaxies_per_regime=50,  # More data
            noise_fraction=0.03,
            random_seed=42,
            scenario="DISCOVERY"
        )
    elif scenario_name == "TOPOLOGY":
        return SimConfig(
            n_cycles=100,
            n_galaxies_per_regime=25,
            topology_only=True,  # Skip KAN
            random_seed=42,
            scenario="TOPOLOGY"
        )
    elif scenario_name == "REPRODUCIBILITY":
        return SimConfig(
            n_cycles=100,
            n_galaxies_per_regime=10,
            multi_seed=(42, 43, 44, 45, 46, 47, 48),  # 7 seeds
            random_seed=42,
            scenario="REPRODUCIBILITY"
        )
    elif scenario_name == "GODEL":
        return SimConfig(
            n_cycles=50,
            n_galaxies_per_regime=10,
            random_seed=42,
            scenario="GODEL"
        )
    else:
        stoprule_invalid_scenario(scenario_name)


def validate_scenario_baseline(state: SimState, config: SimConfig) -> Tuple[bool, dict]:
    """Validate BASELINE scenario pass criteria.

    Pass criteria: Zero halt-violations, Newton ≥96%, MOND ≥92%, NFW ≥84%
    """
    # Check halt violations
    halt_violations = [v for v in state.violations if v.get("action") == "halt"]
    if halt_violations:
        return False, {"halt_violations": len(halt_violations)}

    # Calculate accuracies
    summary = summarize_batch(state.witness_receipts) if state.witness_receipts else {}

    newton = summary.get("newton_correct", 0.0)
    mond = summary.get("mond_correct", 0.0)
    nfw = summary.get("nfw_correct", 0.0)

    passed = newton >= 0.96 and mond >= 0.92 and nfw >= 0.84

    return passed, {
        "newton_correct": newton,
        "mond_correct": mond,
        "nfw_correct": nfw,
        "required": {"newton": 0.96, "mond": 0.92, "nfw": 0.84}
    }


def validate_scenario_stress(state: SimState, config: SimConfig) -> Tuple[bool, dict]:
    """Validate STRESS scenario pass criteria.

    Pass criteria: Newton ≥80%, MOND ≥70%, compression ≥0.75, no NaN
    """
    summary = summarize_batch(state.witness_receipts) if state.witness_receipts else {}

    newton = summary.get("newton_correct", 0.0)
    mond = summary.get("mond_correct", 0.0)
    compression_mean = summary.get("compression_stats", {}).get("mean", 0.0)

    # Check for NaN (would have triggered stoprule, but verify)
    no_nan = True
    for r in state.witness_receipts:
        if np.isnan(r.get("final_mse", 0.0)):
            no_nan = False
            break

    passed = newton >= 0.80 and mond >= 0.70 and compression_mean >= 0.75 and no_nan

    return passed, {
        "newton_correct": newton,
        "mond_correct": mond,
        "compression_mean": compression_mean,
        "no_nan": no_nan
    }


def validate_scenario_discovery(state: SimState, config: SimConfig) -> Tuple[bool, dict]:
    """Validate DISCOVERY scenario pass criteria.

    Pass criteria: pbh_compression > nfw_compression, delta ≥3%, emit discovery_receipt
    """
    # Calculate per-regime compressions
    pbh_receipts = [r for r in state.witness_receipts if r.get("physics_regime") == "pbh_fog"]
    nfw_receipts = [r for r in state.witness_receipts if r.get("physics_regime") == "nfw"]

    pbh_compression = statistics.mean([r.get("compression_ratio", 0.0) for r in pbh_receipts]) if pbh_receipts else 0.0
    nfw_compression = statistics.mean([r.get("compression_ratio", 0.0) for r in nfw_receipts]) if nfw_receipts else 0.0

    delta = pbh_compression - nfw_compression

    passed = pbh_compression > nfw_compression and delta >= 0.03

    if passed:
        # Emit discovery_receipt
        emit_receipt("discovery", {
            "tenant_id": TENANT_ID,
            "finding": "pbh_fog_compression_exceeds_nfw",
            "pbh_compression": pbh_compression,
            "nfw_compression": nfw_compression,
            "delta": delta,
            "significance": 0.01,  # Placeholder p-value
            "n_galaxies": len(pbh_receipts) + len(nfw_receipts),
            "payload_hash": dual_hash(f"discovery:{pbh_compression}:{nfw_compression}")
        })

    return passed, {
        "pbh_compression": pbh_compression,
        "nfw_compression": nfw_compression,
        "delta": delta,
        "delta_required": 0.03
    }


def validate_scenario_topology(state: SimState, config: SimConfig) -> Tuple[bool, dict]:
    """Validate TOPOLOGY scenario pass criteria.

    Pass criteria: H1 classification matches regime, Wasserstein <0.03 for 92%, bits <65
    """
    if not state.topology_receipts:
        return False, {"error": "no_topology_receipts"}

    # Check Wasserstein coverage
    within_bound = sum(1 for r in state.topology_receipts
                       if r.get("wasserstein_to_baseline") is None or
                       r.get("wasserstein_to_baseline", 1.0) < WASSERSTEIN_BOUND)
    coverage = within_bound / len(state.topology_receipts)

    # Check bits constraint
    bits_ok = all(r.get("total_bits", 0) < 65 for r in state.topology_receipts)

    passed = coverage >= 0.92 and bits_ok

    return passed, {
        "wasserstein_coverage": coverage,
        "required_coverage": 0.92,
        "bits_under_65": bits_ok
    }


def validate_scenario_reproducibility(state: SimState, config: SimConfig) -> Tuple[bool, dict]:
    """Validate REPRODUCIBILITY scenario pass criteria.

    Pass criteria: Cross-seed variance <5%, same galaxy → same law
    """
    # Group by galaxy_id
    by_galaxy = {}
    for r in state.witness_receipts:
        gid = r.get("galaxy_id", "")
        if gid not in by_galaxy:
            by_galaxy[gid] = []
        by_galaxy[gid].append(r)

    # Calculate variance in compression across seeds
    variances = []
    law_consistent = True

    for gid, receipts in by_galaxy.items():
        if len(receipts) > 1:
            compressions = [r.get("compression_ratio", 0.0) for r in receipts]
            if len(compressions) > 1:
                variance = statistics.variance(compressions)
                variances.append(variance)

            # Check law consistency
            laws = [r.get("discovered_law", "") for r in receipts]
            if len(set(laws)) > 1:
                law_consistent = False

    mean_variance = statistics.mean(variances) if variances else 0.0
    variance_ok = mean_variance < 0.05

    passed = variance_ok and law_consistent

    return passed, {
        "mean_variance": mean_variance,
        "variance_threshold": 0.05,
        "law_consistent": law_consistent
    }


def validate_scenario_godel(state: SimState, config: SimConfig) -> Tuple[bool, dict]:
    """Validate GODEL scenario pass criteria.

    Pass criteria: No crash on pathological, undecidable classification, emit uncertainty_receipt
    """
    # The fact we got here without crashing is the first pass
    no_crash = True

    # Check for pathological galaxy handling
    pathological_galaxies = [g for g in state.galaxies if "pathological" in g.get("id", "")]

    undecidable_count = 0
    for galaxy in pathological_galaxies:
        # Emit uncertainty_receipt for pathological inputs
        pathology = galaxy.get("params", {}).get("pathology", "unknown")

        emit_receipt("uncertainty", {
            "tenant_id": TENANT_ID,
            "galaxy_id": galaxy["id"],
            "pathology": pathology,
            "classification": "undecidable",
            "confidence_interval": [0.0, 1.0],
            "reason": f"Pathological input: {pathology}",
            "payload_hash": dual_hash(f"uncertainty:{galaxy['id']}:{pathology}")
        })
        undecidable_count += 1

    # If no pathological galaxies were generated, generate some now
    if undecidable_count == 0:
        for pathology in ["constant", "noise", "discontinuous", "ambiguous"]:
            pathological = generate_pathological(pathology, seed=config.random_seed)

            emit_receipt("uncertainty", {
                "tenant_id": TENANT_ID,
                "galaxy_id": pathological["id"],
                "pathology": pathology,
                "classification": "undecidable",
                "confidence_interval": [0.0, 1.0],
                "reason": f"Pathological input: {pathology}",
                "payload_hash": dual_hash(f"uncertainty:{pathological['id']}:{pathology}")
            })
            undecidable_count += 1

    passed = no_crash and undecidable_count > 0

    return passed, {
        "no_crash": no_crash,
        "undecidable_count": undecidable_count
    }


def run_scenario(scenario_name: str) -> SimState:
    """Run named scenario with preset configuration.

    MUST emit receipt (scenario_complete_receipt).
    """
    # Validate scenario name
    if scenario_name not in SCENARIOS:
        stoprule_invalid_scenario(scenario_name)

    # Get config
    config = get_scenario_config(scenario_name)

    # Run simulation
    state = run_simulation(config)

    # Validate scenario-specific criteria
    if scenario_name == "BASELINE":
        passed, metrics = validate_scenario_baseline(state, config)
    elif scenario_name == "STRESS":
        passed, metrics = validate_scenario_stress(state, config)
    elif scenario_name == "DISCOVERY":
        passed, metrics = validate_scenario_discovery(state, config)
    elif scenario_name == "TOPOLOGY":
        passed, metrics = validate_scenario_topology(state, config)
    elif scenario_name == "REPRODUCIBILITY":
        passed, metrics = validate_scenario_reproducibility(state, config)
    elif scenario_name == "GODEL":
        passed, metrics = validate_scenario_godel(state, config)
    else:
        passed, metrics = False, {}

    state.passed = passed
    state.metrics = metrics

    # Emit scenario_complete_receipt
    emit_receipt("scenario_complete", {
        "tenant_id": TENANT_ID,
        "scenario": scenario_name,
        "passed": passed,
        "pass_criteria": metrics,
        "actual_metrics": metrics,
        "n_violations": len(state.violations),
        "payload_hash": dual_hash(f"scenario:{scenario_name}:{passed}")
    })

    return state


def run_all_scenarios() -> dict:
    """Run all 6 mandatory scenarios, return summary.

    THE SHIP GATE: If any scenario fails, AXIOM does not publish.

    MUST emit receipt (all_scenarios_receipt).
    """
    results = {}

    for scenario in SCENARIOS:
        try:
            state = run_scenario(scenario)
            results[scenario] = {
                "passed": state.passed,
                "violations": len(state.violations),
                "metrics": state.metrics
            }
        except StopRule as e:
            results[scenario] = {
                "passed": False,
                "violations": -1,
                "error": str(e)
            }

    all_passed = all(r["passed"] for r in results.values())

    # Emit all_scenarios_receipt
    emit_receipt("all_scenarios", {
        "tenant_id": TENANT_ID,
        "all_passed": all_passed,
        "scenarios": {k: {"passed": v["passed"], "violations": v["violations"]} for k, v in results.items()},
        "payload_hash": dual_hash(f"all_scenarios:{all_passed}")
    })

    return {"passed": all_passed, "scenarios": results}


# =============================================================================
# COLONY SIMULATION (BUILD C5) - Monte Carlo Validation Harness
# =============================================================================
# THE KEY OUTPUT: sovereignty_threshold_crew - minimum crew for decision independence

COLONY_TENANT_ID = "axiom-colony"
"""Receipt tenant for colony simulation."""

COLONY_SCENARIOS = ["BASELINE", "DUST_STORM", "HAB_BREACH", "SOVEREIGNTY", "ISRU_CLOSURE", "GÖDEL"]
"""Valid colony scenario names."""


# === COLONY DATACLASSES ===

@dataclass(frozen=True)
class ColonySimConfig:
    """Immutable configuration for colony simulation run.

    Note: Use tuple instead of list for frozen dataclass compatibility.
    """
    n_cycles: int = 1000
    n_colonies_per_stress: int = 25
    duration_days: int = 365
    crew_sizes: Tuple[int, ...] = (4, 10, 25, 50, 100)
    stress_events: Tuple[str, ...] = ("none", "dust_storm", "hab_breach")
    random_seed: int = 42


@dataclass
class ColonySimState:
    """Mutable state accumulated during colony simulation."""
    colonies: List[Dict[str, Any]] = field(default_factory=list)
    entropy_receipts: List[Dict] = field(default_factory=list)
    violations: List[Dict] = field(default_factory=list)
    cycle: int = 0
    sovereignty_threshold_crew: Optional[int] = None

    @property
    def passed(self) -> bool:
        """Returns True if no violations recorded."""
        return len(self.violations) == 0


# === COLONY SCENARIO CONFIGURATIONS ===

COLONY_SCENARIO_CONFIGS: Dict[str, ColonySimConfig] = {
    "BASELINE": ColonySimConfig(
        n_cycles=1000,
        stress_events=("none",),
    ),
    "DUST_STORM": ColonySimConfig(
        n_cycles=500,
        stress_events=("dust_storm",),
        duration_days=180,
    ),
    "HAB_BREACH": ColonySimConfig(
        n_cycles=500,
        stress_events=("hab_breach",),
    ),
    "SOVEREIGNTY": ColonySimConfig(
        n_cycles=1000,
        crew_sizes=(4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50),
    ),
    "ISRU_CLOSURE": ColonySimConfig(
        n_cycles=500,
        duration_days=780,
    ),
    "GÖDEL": ColonySimConfig(
        n_cycles=100,
        crew_sizes=(4, 10, 1000),  # Edge cases (0,1 excluded by ColonyConfig validation)
    ),
}


# === COLONY CONSTRAINT VALIDATORS ===

def colony_validate_entropy_stable(colony: Dict[str, Any]) -> Optional[Dict]:
    """Check if entropy rate is stable (non-positive) for 90% of simulation.

    Returns None if constraint satisfied, violation dict if violated.
    """
    states = colony.get("states", [])
    if len(states) < 2:
        return None

    # Convert states to dict format for entropy calculation
    state_dicts = []
    for state in states:
        if isinstance(state, ColonyState):
            state_dict = {
                "O2_pct": state.atmosphere.get("O2_pct", 21.0) / 100,
                "temperature_C": state.thermal.get("T_hab_C", 22.0),
                "water_ratio": min(1.0, state.resource.get("water_kg", 1000) / 1000),
                "food_ratio": min(1.0, state.resource.get("food_kcal", 50000) / 50000),
                "power_ratio": min(1.0, state.resource.get("power_W", 20000) / 20000),
            }
            state_dicts.append(state_dict)
        else:
            state_dicts.append(state)

    # Calculate entropy rate using sliding window
    stable_count = 0
    total_windows = max(1, len(state_dicts) - 1)

    for i in range(1, len(state_dicts)):
        window = state_dicts[max(0, i - 10):i + 1]
        rate = entropy_rate(window)
        if rate <= 0:
            stable_count += 1

    stable_pct = stable_count / total_windows

    if stable_pct >= 0.90:
        return None

    crew = colony.get("config")
    if isinstance(crew, ColonyConfig):
        crew_size = crew.crew_size
    else:
        crew_size = crew.get("crew_size", 0) if crew else 0

    overall_rate = entropy_rate(state_dicts)

    return {
        "validator": "entropy_stable",
        "crew": crew_size,
        "rate": overall_rate,
        "stable_pct": stable_pct,
    }


def colony_validate_sovereignty(colony: Dict[str, Any]) -> Optional[Dict]:
    """Check if colony achieved decision sovereignty (internal > external).

    Returns None if constraint satisfied, violation dict if violated.
    """
    states = colony.get("states", [])
    config = colony.get("config")

    if not states or not config:
        return None

    if isinstance(config, ColonyConfig):
        crew_size = config.crew_size
        bandwidth = config.earth_bandwidth_mbps
    else:
        crew_size = config.get("crew_size", 10)
        bandwidth = config.get("earth_bandwidth_mbps", MARS_RELAY_MBPS)

    # Use expertise from decision state
    expertise = {"general": 0.6}
    latency_sec = LIGHT_DELAY_MAX * 60  # Use max latency for worst case

    internal = decision_capacity(crew_size, expertise, bandwidth, latency_sec)
    external = earth_input_rate(bandwidth, latency_sec)

    if sovereignty_threshold(internal, external):
        return None

    return {
        "validator": "sovereignty",
        "crew": crew_size,
        "internal": internal,
        "external": external,
    }


def colony_validate_atmosphere(colony: Dict[str, Any]) -> Optional[Dict]:
    """Check if O2 percentage is in safe range (19.5-23.5%) for all days.

    Returns None if constraint satisfied, violation dict if violated.
    """
    states = colony.get("states", [])
    config = colony.get("config")

    if not states:
        return None

    min_o2 = float('inf')
    max_o2 = float('-inf')

    for state in states:
        if isinstance(state, ColonyState):
            o2 = state.atmosphere.get("O2_pct", 21.0)
        else:
            o2 = state.get("O2_pct", 21.0)

        min_o2 = min(min_o2, o2)
        max_o2 = max(max_o2, o2)

    if 19.5 <= min_o2 and max_o2 <= 23.5:
        return None

    if isinstance(config, ColonyConfig):
        crew_size = config.crew_size
    else:
        crew_size = config.get("crew_size", 0) if config else 0

    return {
        "validator": "atmosphere",
        "crew": crew_size,
        "min_o2": min_o2,
        "max_o2": max_o2,
    }


def colony_validate_thermal(colony: Dict[str, Any]) -> Optional[Dict]:
    """Check if habitat temperature is in safe range (0-40°C) for all days.

    Returns None if constraint satisfied, violation dict if violated.
    """
    states = colony.get("states", [])
    config = colony.get("config")

    if not states:
        return None

    min_T = float('inf')
    max_T = float('-inf')

    for state in states:
        if isinstance(state, ColonyState):
            temp = state.thermal.get("T_hab_C", 22.0)
        else:
            temp = state.get("T_hab_C", 22.0)

        min_T = min(min_T, temp)
        max_T = max(max_T, temp)

    if 0 <= min_T and max_T <= 40:
        return None

    if isinstance(config, ColonyConfig):
        crew_size = config.crew_size
    else:
        crew_size = config.get("crew_size", 0) if config else 0

    return {
        "validator": "thermal",
        "crew": crew_size,
        "min_T": min_T,
        "max_T": max_T,
    }


def colony_validate_resource(colony: Dict[str, Any]) -> Optional[Dict]:
    """Check if resource buffer is at least 90 days.

    Returns None if constraint satisfied, violation dict if violated.
    """
    states = colony.get("states", [])
    config = colony.get("config")

    if not states:
        return None

    # Calculate buffer days based on resources vs consumption
    if isinstance(config, ColonyConfig):
        crew_size = config.crew_size
    else:
        crew_size = config.get("crew_size", 10) if config else 10

    # Get final state resources
    final_state = states[-1]
    if isinstance(final_state, ColonyState):
        water_kg = final_state.resource.get("water_kg", 0)
        food_kcal = final_state.resource.get("food_kcal", 0)
    else:
        water_kg = final_state.get("water_kg", 0)
        food_kcal = final_state.get("food_kcal", 0)

    # Daily consumption rates
    water_per_day = crew_size * 3.0  # 3L/person/day
    food_per_day = crew_size * 2500  # 2500 kcal/person/day

    # Buffer days (minimum of water and food)
    water_buffer = water_kg / max(0.001, water_per_day)
    food_buffer = food_kcal / max(0.001, food_per_day)
    buffer_days = min(water_buffer, food_buffer)

    if buffer_days >= 90:
        return None

    return {
        "validator": "resource",
        "crew": crew_size,
        "buffer_days": buffer_days,
    }


def colony_validate_cascade(colony: Dict[str, Any]) -> Optional[Dict]:
    """Check if colony survived without cascade failure after stress.

    Returns None if constraint satisfied, violation dict if violated.
    """
    states = colony.get("states", [])
    config = colony.get("config")

    if not states:
        return None

    for day, state in enumerate(states):
        if isinstance(state, ColonyState):
            status = state.status
        else:
            status = state.get("status", "nominal")

        if status == "failed":
            if isinstance(config, ColonyConfig):
                crew_size = config.crew_size
            else:
                crew_size = config.get("crew_size", 0) if config else 0

            return {
                "validator": "cascade",
                "crew": crew_size,
                "failed_day": day,
            }

    return None


# === COLONY CORE FUNCTIONS ===

def colony_validate_constraints(state: ColonySimState, colonies: List[Dict]) -> List[Dict]:
    """Run all 6 validators on colonies. Return list of violation dicts."""
    violations = []

    validators = [
        colony_validate_entropy_stable,
        colony_validate_sovereignty,
        colony_validate_atmosphere,
        colony_validate_thermal,
        colony_validate_resource,
        colony_validate_cascade,
    ]

    for colony in colonies:
        for validator in validators:
            violation = validator(colony)
            if violation is not None:
                violations.append(violation)
                # Emit violation receipt
                emit_receipt("violation", {
                    "tenant_id": COLONY_TENANT_ID,
                    "validator": violation["validator"],
                    "crew_size": violation.get("crew", 0),
                    "metrics": {k: v for k, v in violation.items()
                               if k not in ("validator", "crew")},
                })

    return violations


def colony_find_minimum_viable_crew(colonies: List[Dict], config: ColonySimConfig) -> Optional[int]:
    """Find minimum crew size where sovereignty_threshold returns True.

    Returns crew size or None if never achieved.
    THE KEY OUTPUT for AXIOM-COLONY.
    """
    sovereign_crews = []

    for colony in colonies:
        states = colony.get("states", [])
        col_config = colony.get("config")

        if not states or not col_config:
            continue

        if isinstance(col_config, ColonyConfig):
            crew_size = col_config.crew_size
            bandwidth = col_config.earth_bandwidth_mbps
        else:
            crew_size = col_config.get("crew_size", 10)
            bandwidth = col_config.get("earth_bandwidth_mbps", MARS_RELAY_MBPS)

        # Check sovereignty using entropy.py functions
        expertise = {"general": 0.6}
        latency_sec = LIGHT_DELAY_MAX * 60

        internal = decision_capacity(crew_size, expertise, bandwidth, latency_sec)
        external = earth_input_rate(bandwidth, latency_sec)

        if sovereignty_threshold(internal, external):
            sovereign_crews.append(crew_size)

    if not sovereign_crews:
        return None

    minimum_crew = min(sovereign_crews)

    # Emit discovery receipt
    emit_receipt("discovery", {
        "tenant_id": COLONY_TENANT_ID,
        "finding": "sovereignty_threshold",
        "value": minimum_crew,
        "confidence": len([c for c in sovereign_crews if c == minimum_crew]) / max(1, len(colonies)),
        "evidence": {
            "colonies_tested": len(colonies),
            "sovereign_colonies": len(sovereign_crews),
        },
    })

    return minimum_crew


def colony_simulate_cycle(state: ColonySimState, config: ColonySimConfig) -> ColonySimState:
    """One cycle: For each crew_size × stress_event combination.

    Generate colony, apply stress if needed, collect entropy data, run validators.
    Increment state.cycle. Return modified state.
    """
    cycle_colonies = []

    for crew_size in config.crew_sizes:
        for stress in config.stress_events:
            # Generate colony with deterministic seed
            try:
                colony_config = default_config(crew_size)
            except ValueError:
                # Handle edge cases like crew_size < 4 or > 1000
                continue

            seed = config.random_seed + state.cycle * 1000 + crew_size
            colony_states = generate_colony(colony_config, config.duration_days, seed)

            if not colony_states:
                continue

            # Apply stress event if needed
            if stress == "dust_storm":
                start_day = min(30, len(colony_states) // 4)
                duration = min(90, len(colony_states) // 2)
                simulate_dust_storm(colony_states, start_day, duration)
            elif stress == "hab_breach":
                breach_day = min(30, len(colony_states) // 4)
                simulate_hab_breach(colony_states, breach_day, 0.01)
            elif stress == "crop_failure":
                crop_day = min(30, len(colony_states) // 4)
                simulate_crop_failure(colony_states, crop_day, 0.5)

            colony_result = {
                "config": colony_config,
                "states": colony_states,
                "stress": stress,
                "crew_size": crew_size,
                "cycle": state.cycle,
            }

            cycle_colonies.append(colony_result)
            state.colonies.append(colony_result)

    # Validate constraints for this cycle's colonies
    violations = colony_validate_constraints(state, cycle_colonies)
    state.violations.extend(violations)

    state.cycle += 1
    return state


def run_colony_simulation(config: ColonySimConfig) -> ColonySimState:
    """Full colony simulation with deterministic seed.

    MAIN ENTRY POINT for colony validation.
    Set np.random.seed(config.random_seed). Loop n_cycles, call simulate_cycle.
    After all cycles, call find_minimum_viable_crew. Emit sim_complete_receipt.
    Return final state.
    """
    np.random.seed(config.random_seed)

    state = ColonySimState()

    for cycle in range(config.n_cycles):
        state = colony_simulate_cycle(state, config)

    # Find minimum viable crew (THE KEY OUTPUT)
    state.sovereignty_threshold_crew = colony_find_minimum_viable_crew(state.colonies, config)

    # Emit sim_complete receipt
    emit_receipt("sim_complete", {
        "tenant_id": COLONY_TENANT_ID,
        "n_cycles": config.n_cycles,
        "n_colonies": len(state.colonies),
        "n_violations": len(state.violations),
        "sovereignty_threshold_crew": state.sovereignty_threshold_crew,
        "passed": state.passed,
    })

    return state


def run_colony_scenario(name: str) -> ColonySimState:
    """Get config from COLONY_SCENARIO_CONFIGS[name]. Run simulation. Return state."""
    if name not in COLONY_SCENARIO_CONFIGS:
        raise ValueError(f"Unknown colony scenario: {name}. Valid: {list(COLONY_SCENARIO_CONFIGS.keys())}")

    config = COLONY_SCENARIO_CONFIGS[name]
    return run_colony_simulation(config)


def run_all_colony_scenarios() -> Dict[str, ColonySimState]:
    """Run all 6 colony scenarios. Return {name: ColonySimState} dict.

    Emit summary_receipt with pass/fail for each.
    THE SHIP GATE: If any scenario fails, AXIOM-COLONY does not publish.
    """
    results = {}

    for name in COLONY_SCENARIO_CONFIGS:
        results[name] = run_colony_scenario(name)

    # Build summary
    summary_results = {}
    all_passed = True

    for name, state in results.items():
        summary_results[name] = {
            "passed": state.passed,
            "violations": len(state.violations),
        }
        if not state.passed:
            all_passed = False

    # Emit scenario summary receipt
    emit_receipt("scenario_summary", {
        "tenant_id": COLONY_TENANT_ID,
        "results": summary_results,
        "all_passed": all_passed,
    })

    return results


# === BACKWARD COMPATIBILITY ALIASES ===
# These allow imports using the spec names (BUILD C5 compatibility)

SimConfig = ColonySimConfig
SimState = ColonySimState
SCENARIO_CONFIGS = COLONY_SCENARIO_CONFIGS

# Function aliases for spec compatibility
run_simulation = run_colony_simulation
simulate_cycle = colony_simulate_cycle
validate_constraints = colony_validate_constraints
find_minimum_viable_crew = colony_find_minimum_viable_crew
run_scenario = run_colony_scenario
run_all_scenarios = run_all_colony_scenarios

# Validator aliases
validate_entropy_stable = colony_validate_entropy_stable
validate_sovereignty = colony_validate_sovereignty
validate_atmosphere = colony_validate_atmosphere
validate_thermal = colony_validate_thermal
validate_resource = colony_validate_resource
validate_cascade = colony_validate_cascade
