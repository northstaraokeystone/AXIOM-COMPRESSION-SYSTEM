"""Chaotic n-body simulation CLI commands.

Commands for chaotic gravitational n-body simulation operations.
"""

from argparse import Namespace
from typing import Any, Dict


def cmd_chaos_info(args: Namespace) -> Dict[str, Any]:
    """Show chaos simulation configuration.

    Args:
        args: CLI arguments

    Returns:
        Dict with chaos config
    """
    from src.chaotic_nbody_sim import load_chaos_config

    config = load_chaos_config()

    print("\n=== CHAOTIC N-BODY SIMULATION CONFIGURATION ===")
    print(f"Body count: {config.get('body_count', 7)}")
    print(f"Integration method: {config.get('integration_method', 'symplectic')}")
    print(f"Lyapunov threshold: {config.get('lyapunov_threshold', 0.1)}")
    print(f"Stability target: {config.get('stability_target', 0.95)}")

    print("\nSimulation parameters:")
    print(f"  Time step: {config.get('dt', 0.001)} s")
    print(f"  Max iterations: {config.get('max_iterations', 10000)}")
    print(f"  Gravitational constant: {config.get('G', 6.674e-11)}")

    print("\nBodies:")
    for i, body in enumerate(config.get("bodies", [])):
        print(
            f"  {i + 1}. {body.get('name', f'Body_{i}')}: mass={body.get('mass', 0):.2e} kg"
        )

    return config


def cmd_chaos_simulate(args: Namespace) -> Dict[str, Any]:
    """Run chaotic n-body simulation.

    Args:
        args: CLI arguments

    Returns:
        Dict with simulation results
    """
    from src.chaotic_nbody_sim import simulate_chaos, NBODY_COUNT

    bodies = getattr(args, "chaos_bodies", NBODY_COUNT)
    duration_years = getattr(args, "chaos_duration_years", 1.0)

    result = simulate_chaos(bodies=bodies, duration_years=duration_years)

    print("\n=== CHAOTIC N-BODY SIMULATION ===")
    print(f"Duration: {result.get('duration_years', 0):.2f} years")
    print(f"Time step: {result.get('dt', 0)} s")
    print(f"Total steps: {result.get('total_steps', 0)}")

    print("\nResults:")
    print(f"  Body count: {result.get('body_count', 7)}")
    print(f"  Integration method: {result.get('integration_method', 'symplectic')}")
    print(f"  Final energy: {result.get('final_energy', 0):.6e} J")
    print(f"  Energy drift: {result.get('energy_drift', 0):.6e}")
    print(f"  Energy conserved: {result.get('energy_conserved', False)}")

    print("\nLyapunov Analysis:")
    print(f"  Lyapunov exponent: {result.get('lyapunov_exponent', 0):.6f}")
    print(f"  Threshold: {result.get('lyapunov_threshold', 0.1)}")
    print(f"  Is chaotic: {result.get('is_chaotic', True)}")
    print(f"  Is stable: {result.get('is_stable', False)}")

    return result


def cmd_chaos_stability(args: Namespace) -> Dict[str, Any]:
    """Check simulation stability.

    Args:
        args: CLI arguments

    Returns:
        Dict with stability results
    """
    from src.chaotic_nbody_sim import check_stability

    result = check_stability()

    print("\n=== CHAOS STABILITY CHECK ===")
    print(f"Lyapunov exponent: {result.get('lyapunov_exponent', 0):.6f}")
    print(f"Threshold: {result.get('threshold', 0.1)}")
    print(f"Is stable: {result.get('is_stable', False)}")
    print(f"Stability margin: {result.get('stability_margin', 0):.6f}")

    print("\nEnergy Conservation:")
    print(f"  Initial energy: {result.get('initial_energy', 0):.6e} J")
    print(f"  Final energy: {result.get('final_energy', 0):.6e} J")
    print(f"  Drift: {result.get('energy_drift', 0):.6e}")
    print(f"  Conserved: {result.get('energy_conserved', False)}")

    return result


def cmd_chaos_monte_carlo(args: Namespace) -> Dict[str, Any]:
    """Run Monte Carlo stability analysis.

    Args:
        args: CLI arguments

    Returns:
        Dict with Monte Carlo results
    """
    from src.chaotic_nbody_sim import run_monte_carlo_stability

    runs = getattr(args, "chaos_monte_carlo_runs", 100)
    simulate = getattr(args, "simulate", False)

    result = run_monte_carlo_stability(runs=runs, simulate=simulate)

    print(f"\n=== CHAOS MONTE CARLO STABILITY ({runs} runs) ===")
    print(f"Mode: {result.get('mode', 'execute')}")
    print(f"Runs: {result.get('runs', 0)}")

    print("\nStability Results:")
    print(f"  Stable runs: {result.get('stable_runs', 0)}")
    print(f"  Unstable runs: {result.get('unstable_runs', 0)}")
    print(f"  Stability rate: {result.get('stability_rate', 0):.2%}")
    print(f"  Target: {result.get('stability_target', 0.95):.2%}")
    print(f"  Target met: {result.get('target_met', False)}")

    print("\nLyapunov Statistics:")
    lyapunov = result.get("lyapunov_stats", {})
    print(f"  Min: {lyapunov.get('min', 0):.6f}")
    print(f"  Max: {lyapunov.get('max', 0):.6f}")
    print(f"  Mean: {lyapunov.get('mean', 0):.6f}")
    print(f"  Std: {lyapunov.get('std', 0):.6f}")

    return result


def cmd_chaos_backbone_tolerance(args: Namespace) -> Dict[str, Any]:
    """Compute backbone chaos tolerance.

    Args:
        args: CLI arguments

    Returns:
        Dict with tolerance results
    """
    from src.chaotic_nbody_sim import compute_backbone_chaos_tolerance

    result = compute_backbone_chaos_tolerance()

    print("\n=== BACKBONE CHAOS TOLERANCE ===")
    print(f"Tolerance: {result.get('tolerance', 0):.2%}")
    print(f"Lyapunov exponent: {result.get('lyapunov_exponent', 0):.6f}")
    print(f"Stability: {result.get('stability', 0):.2%}")

    print("\nMetrics:")
    print(f"  Energy conservation: {result.get('energy_conservation', 0):.4f}")
    print(f"  Angular momentum: {result.get('angular_momentum_conservation', 0):.4f}")
    print(f"  Orbital stability: {result.get('orbital_stability', 0):.4f}")

    print(f"\nBackbone compatible: {result.get('backbone_compatible', False)}")

    return result


def cmd_chaos_lyapunov(args: Namespace) -> Dict[str, Any]:
    """Compute Lyapunov exponent.

    Args:
        args: CLI arguments

    Returns:
        Dict with Lyapunov analysis
    """
    from src.chaotic_nbody_sim import compute_lyapunov_exponent

    iterations = getattr(args, "chaos_iterations", 1000)

    result = compute_lyapunov_exponent(iterations=iterations)

    print("\n=== LYAPUNOV EXPONENT ANALYSIS ===")
    print(f"Iterations: {result.get('iterations', 0)}")
    print(f"Lyapunov exponent: {result.get('lyapunov_exponent', 0):.6f}")
    print(f"Threshold: {result.get('threshold', 0.1)}")

    print("\nInterpretation:")
    if result.get("lyapunov_exponent", 0) > 0:
        print("  System is CHAOTIC (positive Lyapunov exponent)")
        print(
            f"  Prediction horizon: ~{1 / result.get('lyapunov_exponent', 1):.1f} time units"
        )
    else:
        print("  System is STABLE (non-positive Lyapunov exponent)")

    print(f"\nIs stable: {result.get('is_stable', False)}")

    return result
