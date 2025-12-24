"""chaotic_nbody_sim.py - Chaotic N-Body Simulations for Backbone Stability

PARADIGM:
    Chaotic n-body simulations validate backbone stability under gravitational
    chaos conditions. 7-body systems (Jovian moons + inner planets) test
    long-term orbital stability using symplectic integration.

THE PHYSICS:
    - Symplectic integration preserves energy (Hamiltonian structure)
    - Lyapunov exponent < 0.1 indicates stable orbits
    - Energy conservation tolerance: 1e-10
    - 100 simulated years of orbital evolution
    - Monte Carlo stability validation (statistical confidence)

CHAOTIC N-BODY CONFIG:
    - body_count: 7 (initial) up to 12 (with dwarf planets)
    - integration_method: symplectic (energy-conserving)
    - timestep_days: 0.1 (high resolution)
    - lyapunov_threshold: 0.1 (chaos measure)
    - stability_target: 0.95 (95% stable orbits)
    - duration: 100 simulated years

Source: Grok - "Chaotic n-body for stability", "Backbone stable in 7-body chaos"
"""

import json
import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from .core import emit_receipt, dual_hash


# === CONSTANTS ===

NBODY_COUNT = 7
"""Initial body count (Jovian + Inner)."""

NBODY_MAX_COUNT = 12
"""Extended body count (add dwarf planets)."""

LYAPUNOV_EXPONENT_THRESHOLD = 0.1
"""Chaos measure threshold. Below this = stable."""

CHAOTIC_STABILITY_TARGET = 0.95
"""Target: 95% stable orbits."""

NBODY_INTEGRATION_METHOD = "symplectic"
"""Energy-conserving integration."""

SYMPLECTIC_INTEGRATION = True
"""Flag indicating symplectic integration is enabled."""

NBODY_TIMESTEP_DAYS = 0.1
"""High resolution timestep (0.1 days)."""

ENERGY_CONSERVATION_TOLERANCE = 1e-10
"""Energy conservation tolerance."""

CHAOS_DURATION_YEARS = 100
"""Simulation duration in years."""

TENANT_ID = "axiom-colony"
"""Tenant ID for receipts."""

# Body orbital parameters (semi-major axis in AU, mass in solar masses)
BODY_PARAMETERS = {
    "sun": {"a": 0.0, "mass": 1.0, "type": "star"},  # Sun at center
    "titan": {"a": 0.00817, "mass": 2.25e-7, "type": "jovian"},
    "europa": {"a": 0.00449, "mass": 8.03e-8, "type": "jovian"},
    "ganymede": {"a": 0.00715, "mass": 2.48e-7, "type": "jovian"},
    "callisto": {"a": 0.01259, "mass": 1.80e-7, "type": "jovian"},
    "venus": {"a": 0.723, "mass": 2.45e-6, "type": "inner"},
    "mercury": {"a": 0.387, "mass": 1.66e-7, "type": "inner"},
    "mars": {"a": 1.524, "mass": 3.23e-7, "type": "inner"},
    "ceres": {"a": 2.77, "mass": 4.73e-10, "type": "dwarf"},
    "pluto": {"a": 39.48, "mass": 2.19e-9, "type": "dwarf"},
    "eris": {"a": 67.78, "mass": 2.78e-9, "type": "dwarf"},
    "haumea": {"a": 43.13, "mass": 6.70e-10, "type": "dwarf"},
    "makemake": {"a": 45.79, "mass": 5.02e-10, "type": "dwarf"},
}


# === CONFIGURATION FUNCTIONS ===


def load_chaos_config() -> Dict[str, Any]:
    """Load chaotic n-body config from d15_chaos_spec.json.

    Returns:
        Dict with chaos configuration

    Receipt: chaotic_nbody_config_receipt
    """
    import os

    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d15_chaos_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("chaotic_nbody_config", {})

    emit_receipt(
        "chaotic_nbody_config",
        {
            "receipt_type": "chaotic_nbody_config",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "body_count": config.get("body_count", NBODY_COUNT),
            "max_body_count": config.get("max_body_count", NBODY_MAX_COUNT),
            "integration_method": config.get(
                "integration_method", NBODY_INTEGRATION_METHOD
            ),
            "timestep_days": config.get("timestep_days", NBODY_TIMESTEP_DAYS),
            "lyapunov_threshold": config.get(
                "lyapunov_threshold", LYAPUNOV_EXPONENT_THRESHOLD
            ),
            "stability_target": config.get(
                "stability_target", CHAOTIC_STABILITY_TARGET
            ),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )

    return config


def initialize_bodies(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Set up initial conditions for n-body simulation.

    Creates body states with positions, velocities, and masses
    based on orbital parameters.

    Args:
        config: Chaos configuration dict (optional, loads from file if not provided)

    Returns:
        List of body state dicts

    Receipt: chaotic_nbody_init_receipt
    """
    if config is None:
        config = load_chaos_config()
    body_count = config.get("body_count", NBODY_COUNT)
    bodies_spec = config.get("bodies", {"jovian": [], "inner": []})

    # Get body names from spec or use defaults
    jovian_bodies = bodies_spec.get(
        "jovian", ["titan", "europa", "ganymede", "callisto"]
    )
    inner_bodies = bodies_spec.get("inner", ["venus", "mercury", "mars"])

    # Always start with Sun, then add other bodies
    all_body_names = ["sun"] + jovian_bodies + inner_bodies

    # Limit to specified count (Sun counts as one of the bodies)
    body_names = all_body_names[:body_count]

    bodies = []
    for name in body_names:
        params = BODY_PARAMETERS.get(name, {"a": 1.0, "mass": 1e-6, "type": "unknown"})

        # Initial position (circular orbit assumption)
        a = params["a"]
        if a == 0:  # Sun at center
            x, y, z = 0.0, 0.0, 0.0
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            theta = random.uniform(0, 2 * math.pi)
            x = a * math.cos(theta)
            y = a * math.sin(theta)
            z = 0.0  # Assume coplanar

            # Circular orbital velocity
            # v = sqrt(G * M / r), normalized to AU/day
            v_orbital = (
                math.sqrt(1.0 / a) * 0.01720209895
            )  # k = Gaussian gravitational constant
            vx = -v_orbital * math.sin(theta)
            vy = v_orbital * math.cos(theta)
            vz = 0.0

        bodies.append(
            {
                "name": name,
                "type": params["type"],
                "mass": params["mass"],
                "position": [x, y, z],
                "velocity": [vx, vy, vz],
                "semi_major_axis": a,
            }
        )

    emit_receipt(
        "chaotic_nbody_init",
        {
            "receipt_type": "chaotic_nbody_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "body_count": len(bodies),
            "body_names": [b["name"] for b in bodies],
            "payload_hash": dual_hash(
                json.dumps({"body_names": [b["name"] for b in bodies]}, sort_keys=True)
            ),
        },
    )

    return bodies


# === PHYSICS FUNCTIONS ===


def compute_gravitational_forces(bodies: List[Dict[str, Any]]) -> List[List[float]]:
    """Compute N-body gravitational forces.

    Uses Newton's law of universal gravitation for all body pairs.
    F = G * m1 * m2 / r^2

    Args:
        bodies: List of body state dicts

    Returns:
        List of force vectors [fx, fy, fz] for each body
    """
    n = len(bodies)
    forces = [[0.0, 0.0, 0.0] for _ in range(n)]

    # Gravitational constant (AU^3 / (solar mass * day^2))
    G = 2.959122082855911e-4

    for i in range(n):
        for j in range(i + 1, n):
            # Position difference
            dx = bodies[j]["position"][0] - bodies[i]["position"][0]
            dy = bodies[j]["position"][1] - bodies[i]["position"][1]
            dz = bodies[j]["position"][2] - bodies[i]["position"][2]

            # Distance
            r_sq = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r_sq)
            r_cubed = r_sq * r

            if r_cubed < 1e-20:
                continue

            # Force magnitude
            mi = bodies[i]["mass"]
            mj = bodies[j]["mass"]
            force_mag = G * mi * mj / r_cubed

            # Force components
            fx = force_mag * dx
            fy = force_mag * dy
            fz = force_mag * dz

            # Newton's third law
            forces[i][0] += fx
            forces[i][1] += fy
            forces[i][2] += fz
            forces[j][0] -= fx
            forces[j][1] -= fy
            forces[j][2] -= fz

    return forces


def symplectic_integrate(
    bodies: List[Dict[str, Any]], dt: float
) -> List[Dict[str, Any]]:
    """Energy-conserving symplectic integration step (leapfrog/Verlet).

    Symplectic integrators preserve the Hamiltonian structure,
    making them ideal for long-term orbital simulations.

    Args:
        bodies: List of body state dicts
        dt: Timestep in days

    Returns:
        Updated list of body states
    """
    # Half-step velocity update
    forces = compute_gravitational_forces(bodies)

    for i, body in enumerate(bodies):
        m = body["mass"]
        if m > 0:
            ax = forces[i][0] / m
            ay = forces[i][1] / m
            az = forces[i][2] / m
        else:
            ax, ay, az = 0.0, 0.0, 0.0

        body["velocity"][0] += 0.5 * ax * dt
        body["velocity"][1] += 0.5 * ay * dt
        body["velocity"][2] += 0.5 * az * dt

    # Full-step position update
    for body in bodies:
        body["position"][0] += body["velocity"][0] * dt
        body["position"][1] += body["velocity"][1] * dt
        body["position"][2] += body["velocity"][2] * dt

    # Half-step velocity update with new positions
    forces = compute_gravitational_forces(bodies)

    for i, body in enumerate(bodies):
        m = body["mass"]
        if m > 0:
            ax = forces[i][0] / m
            ay = forces[i][1] / m
            az = forces[i][2] / m
        else:
            ax, ay, az = 0.0, 0.0, 0.0

        body["velocity"][0] += 0.5 * ax * dt
        body["velocity"][1] += 0.5 * ay * dt
        body["velocity"][2] += 0.5 * az * dt

    return bodies


def compute_total_energy(bodies: List[Dict[str, Any]]) -> float:
    """Compute total mechanical energy (kinetic + potential).

    Args:
        bodies: List of body state dicts

    Returns:
        Total energy in system units
    """
    n = len(bodies)
    G = 2.959122082855911e-4

    # Kinetic energy
    kinetic = 0.0
    for body in bodies:
        m = body["mass"]
        v_sq = sum(v**2 for v in body["velocity"])
        kinetic += 0.5 * m * v_sq

    # Potential energy
    potential = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = bodies[j]["position"][0] - bodies[i]["position"][0]
            dy = bodies[j]["position"][1] - bodies[i]["position"][1]
            dz = bodies[j]["position"][2] - bodies[i]["position"][2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)

            if r > 1e-10:
                potential -= G * bodies[i]["mass"] * bodies[j]["mass"] / r

    return kinetic + potential


def compute_lyapunov_exponent(
    trajectory: Optional[List[List[float]]] = None,
    iterations: int = 100,
) -> Dict[str, Any]:
    """Compute Lyapunov exponent from trajectory divergence.

    The Lyapunov exponent measures the rate of separation of
    infinitesimally close trajectories. Positive = chaotic.

    Args:
        trajectory: List of state vectors over time (optional)
        iterations: Number of iterations for simulation (used if trajectory not provided)

    Returns:
        Dict with lyapunov_exponent, threshold, is_stable
    """
    # If no trajectory provided, simulate one
    if trajectory is None:
        bodies = initialize_bodies()
        trajectory = []
        for _ in range(iterations):
            state = []
            for b in bodies:
                state.extend(b["position"])
                state.extend(b["velocity"])
            trajectory.append(state)
            bodies = symplectic_integrate(bodies, NBODY_TIMESTEP_DAYS)

    if len(trajectory) < 10:
        lyapunov = 0.0
    else:
        # Sample trajectory points
        n_samples = min(100, len(trajectory) - 1)
        step = max(1, len(trajectory) // n_samples)

        divergences = []
        for i in range(0, len(trajectory) - step, step):
            # Compute state difference
            s1 = trajectory[i]
            s2 = trajectory[i + step]

            d = sum((a - b) ** 2 for a, b in zip(s1, s2))
            if d > 1e-20:
                divergences.append(math.log(math.sqrt(d)))

        if not divergences:
            lyapunov = 0.0
        else:
            # Linear fit to log-divergence gives Lyapunov exponent
            n = len(divergences)
            mean_y = sum(divergences) / n
            mean_x = (n - 1) / 2.0

            numerator = sum((i - mean_x) * (divergences[i] - mean_y) for i in range(n))
            denominator = sum((i - mean_x) ** 2 for i in range(n))

            if abs(denominator) < 1e-20:
                lyapunov = 0.0
            else:
                slope = numerator / denominator
                # Normalize by timestep
                lyapunov = abs(slope) / (step * NBODY_TIMESTEP_DAYS)

    lyapunov = round(lyapunov, 6)
    is_stable = lyapunov < LYAPUNOV_EXPONENT_THRESHOLD

    return {
        "lyapunov_exponent": lyapunov,
        "threshold": LYAPUNOV_EXPONENT_THRESHOLD,
        "is_stable": is_stable,
    }


# === SIMULATION FUNCTIONS ===


def simulate_chaos(
    bodies: int = NBODY_COUNT,
    duration_years: float = CHAOS_DURATION_YEARS,
    iterations: Optional[int] = None,
    dt: Optional[float] = None,
    simulate: bool = True,
) -> Dict[str, Any]:
    """Run full chaotic n-body simulation.

    Args:
        bodies: Number of bodies to simulate
        duration_years: Duration in simulated years
        iterations: Number of iterations (overrides duration_years if provided)
        dt: Timestep in days (overrides config if provided)
        simulate: If True, return mode="simulate", else mode="execute"

    Returns:
        Dict with simulation results

    Receipt: chaotic_nbody_stability_receipt
    """
    config = load_chaos_config()
    config["body_count"] = bodies

    # Initialize bodies
    body_states = initialize_bodies(config)

    # Simulation parameters
    if dt is None:
        dt = config.get("timestep_days", NBODY_TIMESTEP_DAYS)

    if iterations is not None:
        n_steps = iterations
    else:
        days_per_year = 365.25
        total_days = duration_years * days_per_year
        n_steps = int(total_days / dt)

    # Record initial energy
    initial_energy = compute_total_energy(body_states)

    # Record trajectory for Lyapunov calculation
    trajectory = []
    sample_interval = max(1, n_steps // 1000)  # Sample ~1000 points

    # Run simulation
    for step in range(n_steps):
        body_states = symplectic_integrate(body_states, dt)

        if step % sample_interval == 0:
            # Flatten state for trajectory
            state = []
            for b in body_states:
                state.extend(b["position"])
                state.extend(b["velocity"])
            trajectory.append(state)

    # Final energy
    final_energy = compute_total_energy(body_states)

    # Compute metrics
    energy_error = abs(final_energy - initial_energy) / abs(initial_energy + 1e-20)
    lyapunov_result = compute_lyapunov_exponent(trajectory)
    lyapunov = lyapunov_result["lyapunov_exponent"]
    energy_conserved = energy_error < ENERGY_CONSERVATION_TOLERANCE

    # Check stability
    lyapunov_threshold = config.get("lyapunov_threshold", LYAPUNOV_EXPONENT_THRESHOLD)
    is_stable = lyapunov < lyapunov_threshold
    stability = 1.0 if is_stable else max(0.0, 1.0 - lyapunov / lyapunov_threshold)

    # Determine mode
    mode = "simulate" if simulate else "execute"

    result = {
        "mode": mode,
        "body_count": len(body_states),
        "iterations": n_steps,
        "duration_years": duration_years if iterations is None else None,
        "timestep_days": dt,
        "n_steps": n_steps,
        "integration_method": "symplectic",
        "initial_energy": round(initial_energy, 12),
        "final_energy": round(final_energy, 12),
        "energy_error": round(energy_error, 15),
        "energy_conserved": energy_conserved,
        "lyapunov_exponent": lyapunov,
        "lyapunov_threshold": lyapunov_threshold,
        "is_stable": is_stable,
        "stability": round(stability, 4),
        "stability_target": CHAOTIC_STABILITY_TARGET,
        "target_met": stability >= CHAOTIC_STABILITY_TARGET,
    }

    emit_receipt(
        "chaotic_nbody_stability",
        {
            "receipt_type": "chaotic_nbody_stability",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "body_count": len(body_states),
            "duration_years": duration_years,
            "lyapunov_exponent": lyapunov,
            "is_stable": is_stable,
            "stability": round(stability, 4),
            "energy_conserved": energy_conserved,
            "target_met": stability >= CHAOTIC_STABILITY_TARGET,
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "body_count": len(body_states),
                        "lyapunov": lyapunov,
                        "stability": round(stability, 4),
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return result


def check_stability(
    lyapunov: Optional[float] = None,
    threshold: float = LYAPUNOV_EXPONENT_THRESHOLD,
) -> Dict[str, Any]:
    """Check if Lyapunov exponent indicates stability.

    Args:
        lyapunov: Computed Lyapunov exponent (optional, computed if not provided)
        threshold: Stability threshold (default: 0.1)

    Returns:
        Dict with is_stable, lyapunov_exponent, stability_margin
    """
    # If no lyapunov provided, compute it
    if lyapunov is None:
        lyap_result = compute_lyapunov_exponent()
        lyapunov = lyap_result["lyapunov_exponent"]

    is_stable = lyapunov < threshold
    stability_margin = threshold - lyapunov if is_stable else 0.0

    return {
        "is_stable": is_stable,
        "lyapunov_exponent": lyapunov,
        "threshold": threshold,
        "stability_margin": round(stability_margin, 6),
    }


def check_energy_conservation(
    initial_E: float, final_E: float, tol: float = ENERGY_CONSERVATION_TOLERANCE
) -> bool:
    """Check if energy is conserved within tolerance.

    Args:
        initial_E: Initial total energy
        final_E: Final total energy
        tol: Tolerance (default: 1e-10)

    Returns:
        True if relative energy error < tolerance
    """
    if abs(initial_E) < 1e-20:
        return True
    error = abs(final_E - initial_E) / abs(initial_E)
    return error < tol


def perturb_body(
    bodies: List[Dict[str, Any]], index: int, magnitude: float
) -> List[Dict[str, Any]]:
    """Apply perturbation to a body for sensitivity analysis.

    Args:
        bodies: List of body states
        index: Index of body to perturb
        magnitude: Perturbation magnitude (relative)

    Returns:
        Perturbed body states
    """
    if 0 <= index < len(bodies):
        body = bodies[index]
        # Perturb velocity
        for i in range(3):
            body["velocity"][i] *= 1.0 + magnitude * (2.0 * random.random() - 1.0)
    return bodies


def run_monte_carlo_stability(runs: int = 100, simulate: bool = True) -> Dict[str, Any]:
    """Run Monte Carlo stability analysis.

    Performs multiple simulations with random initial conditions
    to estimate statistical stability.

    Args:
        runs: Number of Monte Carlo runs
        simulate: If True, run in simulate mode

    Returns:
        Dict with statistical results

    Receipt: monte_carlo_stability_receipt
    """
    config = load_chaos_config()

    stable_count = 0
    lyapunov_values = []
    stability_values = []

    for _ in range(runs):
        # Short simulation for each run
        result = simulate_chaos(
            bodies=config.get("body_count", NBODY_COUNT),
            duration_years=10,  # Shorter for Monte Carlo
            simulate=simulate,
        )

        lyapunov_values.append(result["lyapunov_exponent"])
        stability_values.append(result["stability"])

        if result["is_stable"]:
            stable_count += 1

    # Compute statistics
    mean_lyapunov = sum(lyapunov_values) / len(lyapunov_values)
    mean_stability = sum(stability_values) / len(stability_values)
    stable_fraction = stable_count / runs

    result = {
        "runs": runs,
        "stable_runs": stable_count,
        "unstable_runs": runs - stable_count,
        "stability_rate": round(stable_fraction, 4),
        "mean_lyapunov": round(mean_lyapunov, 6),
        "mean_stability": round(mean_stability, 4),
        "stability_target": CHAOTIC_STABILITY_TARGET,
        "target_met": stable_fraction >= CHAOTIC_STABILITY_TARGET,
    }

    emit_receipt(
        "monte_carlo_stability",
        {
            "receipt_type": "monte_carlo_stability",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "n_runs": runs,
            "stable_fraction": round(stable_fraction, 4),
            "mean_lyapunov": round(mean_lyapunov, 6),
            "target_met": stable_fraction >= CHAOTIC_STABILITY_TARGET,
            "payload_hash": dual_hash(
                json.dumps(
                    {
                        "n_runs": runs,
                        "stable_fraction": round(stable_fraction, 4),
                    },
                    sort_keys=True,
                )
            ),
        },
    )

    return result


def compute_backbone_chaos_tolerance(
    sim_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute backbone chaos tolerance from simulation results.

    Combines stability, energy conservation, and Lyapunov exponent
    into a single tolerance metric.

    Args:
        sim_results: Results from simulate_chaos() (optional, runs sim if not provided)

    Returns:
        Dict with tolerance, lyapunov_exponent, stability, backbone_compatible
    """
    # Run simulation if no results provided
    if sim_results is None:
        sim_results = simulate_chaos(duration_years=1)  # Quick 1-year sim

    # Weight factors
    stability_weight = 0.5
    energy_weight = 0.3
    lyapunov_weight = 0.2

    stability = sim_results.get("stability", 0.0)
    energy_ok = 1.0 if sim_results.get("energy_conserved", False) else 0.5

    lyapunov = sim_results.get("lyapunov_exponent", 1.0)
    lyapunov_threshold = sim_results.get(
        "lyapunov_threshold", LYAPUNOV_EXPONENT_THRESHOLD
    )
    lyapunov_factor = max(0.0, 1.0 - lyapunov / lyapunov_threshold)

    tolerance = (
        stability_weight * stability
        + energy_weight * energy_ok
        + lyapunov_weight * lyapunov_factor
    )

    tolerance = round(tolerance, 4)
    backbone_compatible = tolerance >= CHAOTIC_STABILITY_TARGET

    return {
        "tolerance": tolerance,
        "lyapunov_exponent": lyapunov,
        "stability": stability,
        "backbone_compatible": backbone_compatible,
    }


def get_chaos_info() -> Dict[str, Any]:
    """Get chaotic n-body module configuration.

    Returns:
        Dict with module info
    """
    config = load_chaos_config()

    return {
        "body_count": config.get("body_count", NBODY_COUNT),
        "max_body_count": config.get("max_body_count", NBODY_MAX_COUNT),
        "integration_method": config.get(
            "integration_method", NBODY_INTEGRATION_METHOD
        ),
        "timestep_days": config.get("timestep_days", NBODY_TIMESTEP_DAYS),
        "lyapunov_threshold": config.get(
            "lyapunov_threshold", LYAPUNOV_EXPONENT_THRESHOLD
        ),
        "stability_target": config.get("stability_target", CHAOTIC_STABILITY_TARGET),
        "chaos_duration_years": config.get(
            "chaos_duration_years", CHAOS_DURATION_YEARS
        ),
        "energy_tolerance": ENERGY_CONSERVATION_TOLERANCE,
        "bodies": config.get("bodies", {}),
        "description": "Chaotic n-body simulations for backbone stability validation",
    }
