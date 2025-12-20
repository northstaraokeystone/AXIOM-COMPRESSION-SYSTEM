"""cli/darwinian.py - Darwinian enforcer CLI commands.

AXIOM v2: The receipt chain is a Darwinian replicator.
"""

import json
from src.darwinian_enforce import (
    load_spec,
    run_selection_cycle,
    get_active_laws,
    get_darwinian_info,
    reset_darwinian_state,
)
from src.entropy import (
    calculate_latency_pressure,
    apply_latency_selection,
    evolve_under_latency,
    MARS_MIN_LATENCY_MS,
    MARS_MAX_LATENCY_MS,
    JUPITER_MAX_LATENCY_MS,
)


def cmd_darwinian_info():
    """Show Darwinian enforcer configuration."""
    info = get_darwinian_info()
    print(json.dumps(info, indent=2))


def cmd_darwinian_simulate(simulate: bool = True):
    """Run selection cycle with sample population."""
    reset_darwinian_state()

    # Create sample population with varying compression
    population = [
        {"id": "high1", "compression": 0.95, "type": "winner"},
        {"id": "high2", "compression": 0.92, "type": "winner"},
        {"id": "neutral", "compression": 0.85, "type": "neutral"},
        {"id": "low1", "compression": 0.6, "type": "loser", "_generation": 3},
        {"id": "low2", "compression": 0.4, "type": "loser", "_generation": 6},
    ]

    print(f"Input population: {len(population)} receipts")
    print("Running Darwinian selection cycle...")

    # Run selection
    survivors = run_selection_cycle(population)

    print(f"Survivors: {len(survivors)} receipts")
    print("Selection cycle complete. Check receipts.jsonl for path_amplification and path_starvation receipts.")


def cmd_latency_selection_test():
    """Test latency selection with Mars parameters."""
    reset_darwinian_state()

    # Create population with varying delay tolerance
    population = [
        {"id": "sensitive1", "tolerance": 0.1},
        {"id": "sensitive2", "tolerance": 0.2},
        {"id": "moderate1", "tolerance": 0.5},
        {"id": "tolerant1", "tolerance": 0.8},
        {"id": "tolerant2", "tolerance": 0.9},
    ]

    print("=== LATENCY SELECTION TEST ===")
    print(f"Input population: {len(population)} receipts")
    print()

    # Test at Mars min latency
    print(f"Mars Opposition (min latency: {MARS_MIN_LATENCY_MS}ms = 3 min)")
    pressure_min = calculate_latency_pressure(MARS_MIN_LATENCY_MS)
    print(f"  Pressure: {pressure_min:.4f}")
    survivors_min = apply_latency_selection(population, MARS_MIN_LATENCY_MS)
    print(f"  Survivors: {len(survivors_min)}/{len(population)}")
    print()

    # Test at Mars max latency
    print(f"Mars Conjunction (max latency: {MARS_MAX_LATENCY_MS}ms = 22 min)")
    pressure_max = calculate_latency_pressure(MARS_MAX_LATENCY_MS)
    print(f"  Pressure: {pressure_max:.4f}")
    survivors_max = apply_latency_selection(population, MARS_MAX_LATENCY_MS)
    print(f"  Survivors: {len(survivors_max)}/{len(population)}")
    print()

    # Test at Jupiter max latency
    print(f"Jupiter (max latency: {JUPITER_MAX_LATENCY_MS}ms = 53 min)")
    pressure_jup = calculate_latency_pressure(JUPITER_MAX_LATENCY_MS)
    print(f"  Pressure: {pressure_jup:.4f}")
    survivors_jup = apply_latency_selection(population, JUPITER_MAX_LATENCY_MS)
    print(f"  Survivors: {len(survivors_jup)}/{len(population)}")
    print()

    print("THE INSIGHT: Distance doesn't degrade - it SELECTS.")
    print("Delay-tolerant primitives survive. The rest die.")


def cmd_show_laws():
    """Display active imposed laws."""
    laws = get_active_laws()

    if not laws:
        print("No active laws. Laws crystallize after 10 generations of Darwinian selection.")
        return

    print(f"=== ACTIVE LAWS ({len(laws)}) ===")
    for i, law in enumerate(laws, 1):
        print(f"\nLaw {i}: {law['id']}")
        print(f"  Pattern: {json.dumps(law.get('pattern', {}))}")
        print(f"  Generation: {law.get('generation', 'unknown')}")


def cmd_evolve(generations: int, latency_ms: int, simulate: bool = True):
    """Run evolution under latency pressure."""
    reset_darwinian_state()

    # Create initial population
    population = [
        {"id": f"r{i}", "tolerance": 0.3 + (i * 0.05)}
        for i in range(10)
    ]

    initial_avg = sum(r.get("tolerance", 0) for r in population) / len(population)

    print(f"=== EVOLUTION UNDER LATENCY ===")
    print(f"Latency: {latency_ms}ms")
    print(f"Generations: {generations}")
    print(f"Initial population: {len(population)} receipts")
    print(f"Initial avg tolerance: {initial_avg:.4f}")
    print()

    # Evolve
    evolved = evolve_under_latency(population, generations, latency_ms)

    if evolved:
        final_avg = sum(r.get("tolerance", 0) for r in evolved) / len(evolved)
        print(f"Final population: {len(evolved)} receipts")
        print(f"Final avg tolerance: {final_avg:.4f}")
        print(f"Tolerance increase: {final_avg - initial_avg:.4f}")
    else:
        print("Population went extinct under latency pressure!")

    print()
    print("THE INSIGHT: Interstellar latency no longer degrades - it EVOLVES.")


def cmd_darwinian_mode(args, simulate: bool = True):
    """Main Darwinian mode entry point."""
    spec = load_spec()
    print("=== DARWINIAN MODE ACTIVE ===")
    print(f"Version: {spec.get('version', 'unknown')}")
    print(f"Paradigm: {spec.get('paradigm', {}).get('new', 'Laws causally enforced')}")
    print()

    if simulate:
        cmd_darwinian_simulate(simulate)
