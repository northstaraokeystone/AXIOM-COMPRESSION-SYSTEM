"""cli/d19.py - D19 Swarm Intelligence CLI commands.

Commands for D19 emergent swarm intelligence and law discovery.

D19.1 UPDATE:
  - Added --live_only flag for reality-only mode
  - Added --live_stream_test for live ingest testing
  - Added --alpha_threshold_test for threshold testing
  - Killed synthetic scenarios: "Reality is the only valid scenario"
"""

from argparse import Namespace
from typing import Any, Dict


def cmd_d19_info(args: Namespace) -> Dict[str, Any]:
    """Show D19 configuration.

    Args:
        args: CLI arguments

    Returns:
        Dict with D19 info
    """
    from src.depths.d19_swarm_intelligence import load_d19_config, get_d19_status

    config = load_d19_config()
    status = get_d19_status()

    print("\n=== D19 SWARM INTELLIGENCE CONFIGURATION ===")
    print(f"Version: {config.get('version', '19.0.0')}")
    print(f"Depth: {config.get('depth', 19)}")
    print(f"Scale: {config.get('scale', 'swarm_intelligence')}")
    print(f"Paradigm: {config.get('paradigm', 'compression_as_coordination')}")

    d19_config = config.get("d19_config", {})
    print("\nD19 Recursion:")
    print(f"  Alpha floor: {d19_config.get('alpha_floor', 3.93)}")
    print(f"  Alpha target: {d19_config.get('alpha_target', 3.92)}")
    print(f"  Alpha ceiling: {d19_config.get('alpha_ceiling', 3.98)}")
    print(f"  Uplift: {d19_config.get('uplift', 0.44)}")
    print(f"  Central coordinator: {d19_config.get('central_coordinator', False)}")

    gate_1 = config.get("gate_1_config", {})
    print("\nGate 1 - Swarm Entropy Engine:")
    print(f"  Node count: {gate_1.get('node_count', 100)}")
    print(f"  Coordination mode: {gate_1.get('coordination_mode', 'entropy_gradient')}")
    print(f"  Convergence target: {gate_1.get('convergence_target', 0.95)}")

    gate_2 = config.get("gate_2_config", {})
    print("\nGate 2 - Law Witness Module:")
    print(f"  KAN architecture: {gate_2.get('kan_architecture', [100, 20, 5, 1])}")
    print(f"  Compression target: {gate_2.get('compression_target', 0.90)}")
    print(f"  Law discovery threshold: {gate_2.get('law_discovery_threshold', 0.85)}")

    gate_3 = config.get("gate_3_config", {})
    print("\nGate 3 - Autocatalytic Patterns:")
    print(f"  Self-reference threshold: {gate_3.get('self_reference_threshold', 0.70)}")
    print(f"  Pattern birth fitness: {gate_3.get('pattern_birth_fitness', 0.60)}")
    print(f"  Migration latency tolerance: {gate_3.get('migration_latency_tolerance_ms', 5000)}ms")

    gate_4 = config.get("gate_4_config", {})
    print("\nGate 4 - Multi-Scale Federation:")
    print(f"  Hierarchy levels: {gate_4.get('hierarchy_levels', ['node', 'cluster', 'planet', 'system'])}")
    print(f"  Law composition: {gate_4.get('law_composition_mode', 'bottom_up')}")
    print(f"  Constraint propagation: {gate_4.get('constraint_propagation_mode', 'top_down')}")

    gate_5 = config.get("gate_5_config", {})
    print("\nGate 5 - Quantum Consensus:")
    print(f"  Correlation target: {gate_5.get('correlation_target', 0.9999)}")
    print(f"  Byzantine via decoherence: {gate_5.get('byzantine_detection_via_decoherence', True)}")
    print(f"  State sync mode: {gate_5.get('state_sync_mode', 'correlation_verified')}")

    print("\nInnovation Targets:")
    targets = config.get("innovation_targets", {})
    for key, value in targets.items():
        print(f"  {key}: {value}")

    print("\n" + "-" * 50)
    print("Paradigm: Laws are discovered, not programmed")
    print("The swarm doesn't follow rules - it witnesses them")

    return {"config": config, "status": status}


def cmd_d19_run(args: Namespace) -> Dict[str, Any]:
    """Run full D19 swarm intelligence simulation.

    Args:
        args: CLI arguments

    Returns:
        Dict with D19 results
    """
    from src.depths.d19_swarm_intelligence import run_d19

    print("\n=== D19 SWARM INTELLIGENCE EXECUTION ===")
    print("Running all 5 gates...\n")

    result = run_d19()

    print(f"Depth: {result.get('depth', 19)}")
    print(f"Scale: {result.get('scale', 'swarm_intelligence')}")
    print(f"Paradigm: {result.get('paradigm', 'compression_as_coordination')}")
    print(f"\nEffective alpha: {result.get('eff_alpha', 0)}")
    print(f"Alpha floor: {result.get('alpha_floor', 3.93)}")
    print(f"Alpha target: {result.get('alpha_target', 3.92)}")
    print(f"\nFloor met (>= 3.93): {result.get('floor_met', False)}")
    print(f"Target met (>= 3.92): {result.get('target_met', False)}")
    print(f"All gates passed: {result.get('all_gates_passed', False)}")

    innovation = result.get("innovation", {})
    print(f"\nInnovation targets met: {innovation.get('targets_met', 0)}/{innovation.get('total_targets', 0)}")
    print(f"Success ratio: {innovation.get('success_ratio', 0):.2%}")

    print(f"\nSLO passed: {result.get('slo_passed', False)}")
    print(f"Gate: {result.get('gate', 't48h')}")

    return result


def cmd_d19_gate_1(args: Namespace) -> Dict[str, Any]:
    """Run Gate 1: Swarm entropy engine.

    Args:
        args: CLI arguments

    Returns:
        Dict with Gate 1 results
    """
    from src.depths.d19_swarm_intelligence import load_d19_config, run_gate_1

    config = load_d19_config()
    result = run_gate_1(config)

    print("\n=== GATE 1: SWARM ENTROPY ENGINE ===")
    print(f"Node count: {result.get('node_count', 0)}")
    print(f"Coherence: {result.get('coherence', 0):.4f}")
    print(f"Convergence: {result.get('convergence', 0):.4f}")
    print(f"Target met: {result.get('target_met', False)}")

    return result


def cmd_d19_gate_2(args: Namespace) -> Dict[str, Any]:
    """Run Gate 2: Law witness module.

    Args:
        args: CLI arguments

    Returns:
        Dict with Gate 2 results
    """
    from src.depths.d19_swarm_intelligence import load_d19_config, run_gate_2

    config = load_d19_config()
    result = run_gate_2(config)

    print("\n=== GATE 2: LAW WITNESS MODULE ===")
    print(f"KAN architecture: {result.get('kan_architecture', [])}")
    print(f"Training samples: {result.get('training_samples', 0)}")
    print(f"Average loss: {result.get('avg_loss', 0):.6f}")
    print(f"Law discovered: {result.get('law_discovered', False)}")
    print(f"Law ID: {result.get('law_id', 'N/A')}")
    print(f"Compression ratio: {result.get('compression_ratio', 0):.4f}")
    print(f"Target met: {result.get('target_met', False)}")

    return result


def cmd_d19_gate_1_2(args: Namespace) -> Dict[str, Any]:
    """Run Gates 1+2 in parallel.

    Args:
        args: CLI arguments

    Returns:
        Dict with combined results
    """
    from src.depths.d19_swarm_intelligence import load_d19_config, run_gate_1_2_parallel

    config = load_d19_config()
    result = run_gate_1_2_parallel(config)

    print("\n=== GATES 1+2 PARALLEL EXECUTION ===")
    print(f"Mode: {result.get('mode', 'parallel')}")

    gate_1 = result.get("gate_1", {})
    print("\nGate 1 (Entropy Engine):")
    print(f"  Coherence: {gate_1.get('coherence', 0):.4f}")
    print(f"  Target met: {gate_1.get('target_met', False)}")

    gate_2 = result.get("gate_2", {})
    print("\nGate 2 (Law Witness):")
    print(f"  Law discovered: {gate_2.get('law_discovered', False)}")
    print(f"  Compression: {gate_2.get('compression_ratio', 0):.4f}")
    print(f"  Target met: {gate_2.get('target_met', False)}")

    print(f"\nBoth passed: {result.get('both_passed', False)}")

    return result


def cmd_d19_gate_3(args: Namespace) -> Dict[str, Any]:
    """Run Gate 3: Autocatalytic patterns.

    Args:
        args: CLI arguments

    Returns:
        Dict with Gate 3 results
    """
    from src.depths.d19_swarm_intelligence import load_d19_config, run_gate_3

    config = load_d19_config()
    result = run_gate_3(config)

    print("\n=== GATE 3: AUTOCATALYTIC PATTERNS ===")
    print(f"Patterns detected: {result.get('patterns_detected', 0)}")
    print(f"Autocatalytic count: {result.get('autocatalytic_count', 0)}")
    print(f"Patterns born: {result.get('patterns_born', 0)}")
    print(f"Patterns selected: {result.get('patterns_selected', 0)}")
    print(f"Migration candidates: {result.get('migration_candidates', 0)}")
    print(f"Migration success: {result.get('migration_success', False)}")
    print(f"Target met: {result.get('target_met', False)}")

    return result


def cmd_d19_gate_4(args: Namespace) -> Dict[str, Any]:
    """Run Gate 4: Multi-scale federation.

    Args:
        args: CLI arguments

    Returns:
        Dict with Gate 4 results
    """
    from src.depths.d19_swarm_intelligence import load_d19_config, run_gate_4

    config = load_d19_config()
    result = run_gate_4(config)

    print("\n=== GATE 4: MULTI-SCALE FEDERATION ===")
    print(f"Hierarchy levels: {result.get('hierarchy_levels', [])}")
    print(f"Cluster laws: {result.get('cluster_laws', 0)}")
    print(f"Planet laws: {result.get('planet_laws', 0)}")
    print(f"System law: {result.get('system_law', False)}")
    print(f"Constraints propagated: {result.get('constraints_propagated', 0)}")
    print(f"Dispute detected: {result.get('dispute_detected', False)}")
    print(f"Resolution accuracy: {result.get('resolution_accuracy', 0):.2%}")
    print(f"Target met: {result.get('target_met', False)}")

    return result


def cmd_d19_gate_5(args: Namespace) -> Dict[str, Any]:
    """Run Gate 5: Quantum consensus.

    Args:
        args: CLI arguments

    Returns:
        Dict with Gate 5 results
    """
    from src.depths.d19_swarm_intelligence import load_d19_config, run_gate_5

    config = load_d19_config()
    result = run_gate_5(config)

    print("\n=== GATE 5: QUANTUM-ENTANGLED CONSENSUS ===")
    print(f"Entanglement pairs: {result.get('entanglement_pairs', 0)}")
    print(f"Average correlation: {result.get('avg_correlation', 0):.6f}")
    print(f"Consensus achieved: {result.get('consensus_achieved', False)}")
    print(f"State sync ratio: {result.get('state_sync_ratio', 0):.4f}")
    print(f"Byzantine detection active: {result.get('byzantine_detection_active', False)}")
    print(f"Byzantine detected: {result.get('byzantine_detected', False)}")
    print(f"Target met: {result.get('target_met', False)}")

    return result


def cmd_d19_tweet(args: Namespace) -> Dict[str, Any]:
    """Generate X thread for D19.

    Args:
        args: CLI arguments

    Returns:
        Dict with tweet content
    """
    tweet = """D19 SWARM INTELLIGENCE SHIPPED

100 nodes | No central coordinator | Laws discovered, not coded

Gate 1: Entropy gradient coordination
Gate 2: KAN witnesses coordination laws
Gate 3: Autocatalytic patterns (birth/death/migrate)
Gate 4: Multi-scale federation intelligence
Gate 5: Quantum consensus via 99.99% correlation

The swarm doesn't follow rules-it witnesses them.

D18: Infrastructure. D19: Intelligence.
D20: Self-improving infrastructure?"""

    print("\n=== D19 X THREAD ===")
    print(tweet)
    print(f"\n({len(tweet)} chars)")

    return {"tweet": tweet, "chars": len(tweet)}


# === D19.1 LIVE-ONLY CLI COMMANDS ===


def cmd_d19_run_live_only(args: Namespace) -> Dict[str, Any]:
    """Run D19 in live-only mode (--live_only flag).

    D19.1: Reality is the only valid scenario.
    No synthetic scenarios - live triad entropy only.

    Args:
        args: CLI arguments

    Returns:
        Dict with D19 live-only results
    """
    from src.depths.d19_swarm_intelligence import run_d19_live_only, SYNTHETIC_SCENARIOS_ENABLED

    print("\n=== D19.1 LIVE-ONLY EXECUTION ===")
    print("Mode: Reality-only (synthetic KILLED)")
    print("Entropy source: live_triad (AgentProof + NEURON)")
    print(f"Synthetic enabled: {SYNTHETIC_SCENARIOS_ENABLED}")
    print("\nRunning 4 gates...\n")

    result = run_d19_live_only()

    print(f"Depth: {result.get('depth', 19)}")
    print(f"Scale: {result.get('scale', 'swarm_intelligence')}")
    print(f"Mode: {result.get('mode', 'live_only')}")
    print(f"Entropy source: {result.get('entropy_source', 'live_triad')}")

    print(f"\nEffective alpha: {result.get('eff_alpha', 0)}")
    print(f"Alpha floor: {result.get('alpha_floor', 3.93)}")
    print(f"Alpha target: {result.get('alpha_target', 3.92)}")

    gates = result.get("gates", {})

    print("\n--- Gate 1: Live Triad Ingest ---")
    g1 = gates.get("gate_1", {})
    print(f"  Sources connected: {g1.get('sources_connected', {})}")
    print(f"  Receipts ingested: {g1.get('receipts_ingested', 0)}")
    print(f"  Live entropy: {g1.get('live_entropy', 0):.6f}")
    print(f"  Target met: {g1.get('target_met', False)}")

    print("\n--- Gate 2: Alpha Threshold Law ---")
    g2 = gates.get("gate_2", {})
    print(f"  Current alpha: {g2.get('current_alpha', 0):.4f}")
    print(f"  Threshold: {g2.get('threshold', 1.20)}")
    print(f"  Threshold crossed: {g2.get('threshold_crossed', False)}")
    print(f"  Law triggered: {g2.get('law_triggered', False)}")
    print(f"  Target met: {g2.get('target_met', False)}")

    print("\n--- Gate 3: Receipt-Enforced Law ---")
    g3 = gates.get("gate_3", {})
    print(f"  Chain receipts: {g3.get('chain_receipts', 0)}")
    print(f"  Compression ratio: {g3.get('compression_ratio', 0):.4f}")
    print(f"  Causality verified: {g3.get('causality_verified', False)}")
    print(f"  Law enforced: {g3.get('law_enforced', False)}")
    print(f"  Target met: {g3.get('target_met', False)}")

    print("\n--- Gate 4: Reality-Only Validation ---")
    g4 = gates.get("gate_4", {})
    print(f"  Synthetic enabled: {g4.get('synthetic_enabled', False)}")
    print(f"  Reality only: {g4.get('reality_only', True)}")
    print(f"  Target met: {g4.get('target_met', False)}")

    print(f"\nFloor met (>= 3.93): {result.get('floor_met', False)}")
    print(f"Target met (>= 3.92): {result.get('target_met', False)}")
    print(f"All gates passed: {result.get('all_gates_passed', False)}")
    print(f"SLO passed: {result.get('slo_passed', False)}")
    print(f"Gate: {result.get('gate', 't24h')}")

    print("\n" + "-" * 50)
    print(f"Insight: {result.get('insight', '')}")

    return result


def cmd_d19_live_stream_test(args: Namespace) -> Dict[str, Any]:
    """Test live triad ingest functionality (--live_stream_test flag).

    Args:
        args: CLI arguments

    Returns:
        Dict with test results
    """
    from src.depths.d19_swarm_intelligence import test_live_stream

    print("\n=== LIVE STREAM TEST ===")
    print("Testing AgentProof + NEURON ingest...\n")

    result = test_live_stream()

    print(f"AgentProof connected: {result.get('agentproof_connected', False)}")
    print(f"NEURON connected: {result.get('neuron_connected', False)}")
    print(f"Receipts ingested: {result.get('receipts_ingested', 0)}")
    print(f"Live entropy: {result.get('live_entropy', 0):.6f}")

    status = result.get("status", {})
    print(f"\nEntropy source: {status.get('entropy_source', 'N/A')}")
    print(f"Synthetic enabled: {status.get('synthetic_enabled', False)}")
    print(f"Paradigm: {status.get('paradigm', 'N/A')}")

    print(f"\nTest passed: {result.get('passed', False)}")

    return result


def cmd_d19_alpha_threshold_test(args: Namespace) -> Dict[str, Any]:
    """Test alpha threshold law trigger (--alpha_threshold_test flag).

    Args:
        args: CLI arguments

    Returns:
        Dict with test results
    """
    from src.depths.d19_swarm_intelligence import test_alpha_threshold

    print("\n=== ALPHA THRESHOLD TEST ===")
    print("Testing law discovery trigger on α > 1.20...\n")

    result = test_alpha_threshold()

    print(f"Threshold: {result.get('threshold', 1.20)}")
    print(f"Below threshold check (1.15 < 1.20): {result.get('below_threshold_check', False)}")
    print(f"Above threshold check (1.25 > 1.20): {result.get('above_threshold_check', False)}")
    print(f"Law triggered: {result.get('law_triggered', False)}")
    print(f"Law ID: {result.get('law_id', 'N/A')}")

    status = result.get("status", {})
    print(f"\nTrigger count: {status.get('trigger_count', 0)}")
    print(f"In cooldown: {status.get('in_cooldown', False)}")
    print(f"Enforcement mode: {status.get('enforcement_mode', 'N/A')}")

    print(f"\nTest passed: {result.get('passed', False)}")

    return result


def cmd_d19_1_tweet(args: Namespace) -> Dict[str, Any]:
    """Generate X thread for D19.1.

    Args:
        args: CLI arguments

    Returns:
        Dict with tweet content
    """
    tweet = """D19.1 LIVE TRIAD ENTROPY SHIPPED

KILLED:
- All synthetic scenarios
- Standalone entropy simulation
- "Reality is the only valid scenario"

ADDED:
- Live AgentProof + NEURON ingest
- α > 1.20 triggers law discovery
- Receipt chain = physical law

Laws aren't discovered—they're enforced by the chain.

The swarm becomes the physicist."""

    print("\n=== D19.1 X THREAD ===")
    print(tweet)
    print(f"\n({len(tweet)} chars)")

    return {"tweet": tweet, "chars": len(tweet)}
