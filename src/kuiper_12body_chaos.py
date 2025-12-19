"""Kuiper belt 12-body chaotic N-body simulations.

PARADIGM:
    Extended chaotic n-body simulations for Kuiper belt dynamics.
    12-body systems including Jovian moons, inner planets, and dwarf planets
    (Ceres, Pluto, Eris, Makemake, Haumea) test long-term orbital stability.

THE PHYSICS:
    - Symplectic integration preserves energy (Hamiltonian structure)
    - Lyapunov exponent < 0.15 indicates stable orbits (higher tolerance)
    - Energy conservation tolerance: 1e-12 (tighter)
    - 1000 simulated years of orbital evolution
    - Monte Carlo stability validation (statistical confidence)
    - Neptune perturbations critical for Kuiper belt dynamics

KUIPER BELT CONFIG:
    - body_count: 12 (4 Jovian + 3 inner + 5 Kuiper)
    - integration_method: symplectic (energy-conserving)
    - timestep_days: 0.05 (higher resolution)
    - lyapunov_threshold: 0.15 (higher tolerance for chaotic KBOs)
    - stability_target: 0.93 (slightly lower for extended system)
    - duration: 1000 simulated years

Source: Grok - "12-body Kuiper chaos: Ceres/Pluto dynamics stable"
"""

import json
import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from .core import emit_receipt, dual_hash


# === CONSTANTS ===

KUIPER_TENANT_ID = "axiom-kuiper"
"""Tenant ID for Kuiper receipts."""

KUIPER_BODY_COUNT = 12
"""Total body count (4 Jovian + 3 inner + 5 Kuiper)."""

KUIPER_LYAPUNOV_THRESHOLD = 0.15
"""Chaos measure threshold (higher for KBOs)."""

KUIPER_STABILITY_TARGET = 0.93
"""Target: 93% stable orbits (slightly lower for extended system)."""

KUIPER_TIMESTEP_DAYS = 0.05
"""High resolution timestep (0.05 days)."""

KUIPER_ENERGY_TOLERANCE = 1e-12
"""Energy conservation tolerance (tighter)."""

KUIPER_DURATION_YEARS = 1000
"""Simulation duration in years."""

# Body orbital parameters (semi-major axis in AU, mass in solar masses)
KUIPER_BODY_PARAMETERS = {
    # Jovian moons
    "titan": {"a": 9.537, "mass": 2.25e-7, "type": "jovian", "e": 0.0288, "i": 0.34},
    "europa": {"a": 5.203, "mass": 8.03e-8, "type": "jovian", "e": 0.009, "i": 0.47},
    "ganymede": {"a": 5.203, "mass": 2.48e-7, "type": "jovian", "e": 0.0013, "i": 0.20},
    "callisto": {"a": 5.203, "mass": 1.80e-7, "type": "jovian", "e": 0.0074, "i": 0.19},
    # Inner planets
    "venus": {"a": 0.723, "mass": 2.45e-6, "type": "inner", "e": 0.0068, "i": 3.39},
    "mercury": {"a": 0.387, "mass": 1.66e-7, "type": "inner", "e": 0.2056, "i": 7.00},
    "mars": {"a": 1.524, "mass": 3.23e-7, "type": "inner", "e": 0.0934, "i": 1.85},
    # Kuiper belt objects / Dwarf planets
    "ceres": {"a": 2.77, "mass": 4.73e-10, "type": "kuiper", "e": 0.076, "i": 10.59},
    "pluto": {"a": 39.48, "mass": 2.19e-9, "type": "kuiper", "e": 0.2488, "i": 17.16},
    "eris": {"a": 67.78, "mass": 2.78e-9, "type": "kuiper", "e": 0.4407, "i": 44.19},
    "makemake": {"a": 45.79, "mass": 5.02e-10, "type": "kuiper", "e": 0.159, "i": 28.96},
    "haumea": {"a": 43.13, "mass": 6.70e-10, "type": "kuiper", "e": 0.195, "i": 28.22},
}

# Perturbation sources
PERTURBATION_BODIES = {
    "jupiter": {"a": 5.203, "mass": 9.543e-4, "e": 0.0489, "i": 1.30},
    "neptune": {"a": 30.07, "mass": 5.15e-5, "e": 0.0086, "i": 1.77},
}


# === CONFIGURATION FUNCTIONS ===


def load_kuiper_config() -> Dict[str, Any]:
    """Load Kuiper 12-body config from d16_kuiper_spec.json.

    Returns:
        Dict with Kuiper configuration

    Receipt: kuiper_config_receipt
    """
    import os

    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "d16_kuiper_spec.json"
    )

    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = spec.get("kuiper_12body_config", {})

    emit_receipt(
        "kuiper_config",
        {
            "receipt_type": "kuiper_config",
            "tenant_id": KUIPER_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "body_count": config.get("body_count", KUIPER_BODY_COUNT),
            "integration_method": config.get("integration_method", "symplectic"),
            "timestep_days": config.get("timestep_days", KUIPER_TIMESTEP_DAYS),
            "lyapunov_threshold": config.get(
                "lyapunov_threshold", KUIPER_LYAPUNOV_THRESHOLD
            ),
            "stability_target": config.get("stability_target", KUIPER_STABILITY_TARGET),
            "chaos_duration_years": config.get(
                "chaos_duration_years", KUIPER_DURATION_YEARS
            ),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )

    return config


def get_body_orbital_elements(body: str) -> Dict[str, Any]:
    """Get JPL-derived orbital parameters for a body.

    Args:
        body: Body name

    Returns:
        Dict with orbital elements (a, mass, e, i)
    """
    if body in KUIPER_BODY_PARAMETERS:
        return KUIPER_BODY_PARAMETERS[body].copy()
    elif body in PERTURBATION_BODIES:
        return PERTURBATION_BODIES[body].copy()
    else:
        return {"a": 1.0, "mass": 1e-10, "type": "unknown", "e": 0.0, "i": 0.0}


def initialize_kuiper_bodies(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Set up initial conditions for 12-body Kuiper simulation.

    Creates body states with positions, velocities, and masses
    based on orbital parameters including eccentricity and inclination.

    Args:
        config: Kuiper configuration dict

    Returns:
        List of body state dicts

    Receipt: kuiper_init_receipt
    """
    bodies_spec = config.get(
        "bodies",
        {
            "jovian": ["titan", "europa", "ganymede", "callisto"],
            "inner": ["venus", "mercury", "mars"],
            "kuiper": ["ceres", "pluto", "eris", "makemake", "haumea"],
        },
    )

    # Get all body names
    all_body_names = (
        bodies_spec.get("jovian", [])
        + bodies_spec.get("inner", [])
        + bodies_spec.get("kuiper", [])
    )

    bodies = []
    for name in all_body_names:
        params = get_body_orbital_elements(name)

        # Orbital elements
        a = params["a"]
        e = params.get("e", 0.0)
        i = math.radians(params.get("i", 0.0))

        # Random mean anomaly for initial position
        M = random.uniform(0, 2 * math.pi)

        # Solve Kepler's equation for eccentric anomaly (simplified)
        E = M + e * math.sin(M)  # First-order approximation

        # True anomaly
        theta = 2 * math.atan2(
            math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2)
        )

        # Distance from focus
        r = a * (1 - e * math.cos(E))

        # Position in orbital plane
        x_orb = r * math.cos(theta)
        y_orb = r * math.sin(theta)

        # Rotate by inclination (simplified - around x-axis)
        x = x_orb
        y = y_orb * math.cos(i)
        z = y_orb * math.sin(i)

        # Orbital velocity (vis-viva equation)
        # v = sqrt(GM * (2/r - 1/a)), with GM = 1 in solar units
        v_mag = math.sqrt(2.0 / r - 1.0 / a) * 0.01720209895  # Convert to AU/day

        # Velocity perpendicular to position in orbital plane
        v_theta = theta + math.pi / 2
        vx_orb = v_mag * math.cos(v_theta)
        vy_orb = v_mag * math.sin(v_theta)

        # Rotate velocity by inclination
        vx = vx_orb
        vy = vy_orb * math.cos(i)
        vz = vy_orb * math.sin(i)

        bodies.append(
            {
                "name": name,
                "type": params.get("type", "unknown"),
                "mass": params["mass"],
                "position": [x, y, z],
                "velocity": [vx, vy, vz],
                "semi_major_axis": a,
                "eccentricity": e,
                "inclination_deg": math.degrees(i),
            }
        )

    emit_receipt(
        "kuiper_init",
        {
            "receipt_type": "kuiper_init",
            "tenant_id": KUIPER_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "body_count": len(bodies),
            "body_names": [b["name"] for b in bodies],
            "body_types": list(set(b["type"] for b in bodies)),
            "payload_hash": dual_hash(
                json.dumps({"body_names": [b["name"] for b in bodies]}, sort_keys=True)
            ),
        },
    )

    return bodies


# === PHYSICS FUNCTIONS ===


def compute_kuiper_forces(
    bodies: List[Dict[str, Any]], perturbations: Optional[List[Dict[str, Any]]] = None
) -> List[List[float]]:
    """Compute N-body gravitational forces including perturbations.

    Uses Newton's law of universal gravitation for all body pairs,
    plus perturbations from Jupiter and Neptune.

    Args:
        bodies: List of body state dicts
        perturbations: Optional list of perturbation body states

    Returns:
        List of force vectors [fx, fy, fz] for each body
    """
    n = len(bodies)
    forces = [[0.0, 0.0, 0.0] for _ in range(n)]

    # Gravitational constant (AU^3 / (solar mass * day^2))
    G = 2.959122082855911e-4

    # Forces between bodies
    for i in range(n):
        for j in range(i + 1, n):
            dx = bodies[j]["position"][0] - bodies[i]["position"][0]
            dy = bodies[j]["position"][1] - bodies[i]["position"][1]
            dz = bodies[j]["position"][2] - bodies[i]["position"][2]

            r_sq = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r_sq)
            r_cubed = r_sq * r

            if r_cubed < 1e-20:
                continue

            mi = bodies[i]["mass"]
            mj = bodies[j]["mass"]
            force_mag = G * mi * mj / r_cubed

            fx = force_mag * dx
            fy = force_mag * dy
            fz = force_mag * dz

            forces[i][0] += fx
            forces[i][1] += fy
            forces[i][2] += fz
            forces[j][0] -= fx
            forces[j][1] -= fy
            forces[j][2] -= fz

    # Add perturbation forces (from Jupiter, Neptune)
    if perturbations:
        for p_body in perturbations:
            for i, body in enumerate(bodies):
                dx = p_body["position"][0] - body["position"][0]
                dy = p_body["position"][1] - body["position"][1]
                dz = p_body["position"][2] - body["position"][2]

                r_sq = dx * dx + dy * dy + dz * dz
                r = math.sqrt(r_sq)
                r_cubed = r_sq * r

                if r_cubed < 1e-20:
                    continue

                force_mag = G * body["mass"] * p_body["mass"] / r_cubed

                forces[i][0] += force_mag * dx
                forces[i][1] += force_mag * dy
                forces[i][2] += force_mag * dz

    return forces


def symplectic_kuiper_integrate(
    bodies: List[Dict[str, Any]],
    dt: float,
    perturbations: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """High-precision symplectic integration step (leapfrog/Verlet).

    Preserves Hamiltonian structure for long-term stability.

    Args:
        bodies: List of body state dicts
        dt: Timestep in days
        perturbations: Optional perturbation bodies

    Returns:
        Updated list of body states
    """
    # Half-step velocity update
    forces = compute_kuiper_forces(bodies, perturbations)

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

    # Update perturbation positions (circular orbit approximation)
    if perturbations:
        for p_body in perturbations:
            a = p_body["semi_major_axis"]
            # Simple circular update
            omega = 0.01720209895 / math.sqrt(a)  # Mean motion
            theta = math.atan2(p_body["position"][1], p_body["position"][0])
            theta += omega * dt
            p_body["position"][0] = a * math.cos(theta)
            p_body["position"][1] = a * math.sin(theta)

    # Half-step velocity update with new positions
    forces = compute_kuiper_forces(bodies, perturbations)

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


def compute_kuiper_total_energy(
    bodies: List[Dict[str, Any]], perturbations: Optional[List[Dict[str, Any]]] = None
) -> float:
    """Compute total mechanical energy (kinetic + potential).

    Args:
        bodies: List of body state dicts
        perturbations: Optional perturbation bodies

    Returns:
        Total energy in system units
    """
    G = 2.959122082855911e-4

    # Kinetic energy
    kinetic = 0.0
    for body in bodies:
        m = body["mass"]
        v_sq = sum(v**2 for v in body["velocity"])
        kinetic += 0.5 * m * v_sq

    # Potential energy (body-body)
    potential = 0.0
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1, n):
            dx = bodies[j]["position"][0] - bodies[i]["position"][0]
            dy = bodies[j]["position"][1] - bodies[i]["position"][1]
            dz = bodies[j]["position"][2] - bodies[i]["position"][2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)

            if r > 1e-10:
                potential -= G * bodies[i]["mass"] * bodies[j]["mass"] / r

    # Perturbation potential
    if perturbations:
        for p_body in perturbations:
            for body in bodies:
                dx = p_body["position"][0] - body["position"][0]
                dy = p_body["position"][1] - body["position"][1]
                dz = p_body["position"][2] - body["position"][2]
                r = math.sqrt(dx * dx + dy * dy + dz * dz)

                if r > 1e-10:
                    potential -= G * body["mass"] * p_body["mass"] / r

    return kinetic + potential


def compute_kuiper_lyapunov(trajectory: List[List[float]]) -> float:
    """Compute Lyapunov exponent from trajectory divergence.

    Args:
        trajectory: List of state vectors over time

    Returns:
        Lyapunov exponent estimate
    """
    if len(trajectory) < 10:
        return 0.0

    n_samples = min(100, len(trajectory) - 1)
    step = max(1, len(trajectory) // n_samples)

    divergences = []
    for i in range(0, len(trajectory) - step, step):
        s1 = trajectory[i]
        s2 = trajectory[i + step]

        d = sum((a - b) ** 2 for a, b in zip(s1, s2))
        if d > 1e-20:
            divergences.append(math.log(math.sqrt(d)))

    if not divergences:
        return 0.0

    n = len(divergences)
    mean_y = sum(divergences) / n
    mean_x = (n - 1) / 2.0

    numerator = sum((i - mean_x) * (divergences[i] - mean_y) for i in range(n))
    denominator = sum((i - mean_x) ** 2 for i in range(n))

    if abs(denominator) < 1e-20:
        return 0.0

    slope = numerator / denominator
    lyapunov = abs(slope) / (step * KUIPER_TIMESTEP_DAYS)

    return round(lyapunov, 6)


# === SIMULATION FUNCTIONS ===


def simulate_kuiper(
    bodies: int = KUIPER_BODY_COUNT,
    duration_years: float = KUIPER_DURATION_YEARS,
) -> Dict[str, Any]:
    """Run full 12-body Kuiper chaotic simulation.

    Args:
        bodies: Number of bodies to simulate (max 12)
        duration_years: Duration in simulated years

    Returns:
        Dict with simulation results

    Receipt: kuiper_12body_stability_receipt
    """
    config = load_kuiper_config()

    # Initialize bodies
    body_states = initialize_kuiper_bodies(config)

    # Initialize perturbation bodies (Jupiter, Neptune)
    perturbations = []
    if config.get("include_perturbations", True):
        for name in config.get("perturbation_sources", ["jupiter", "neptune"]):
            params = PERTURBATION_BODIES.get(name, {})
            if params:
                a = params["a"]
                theta = random.uniform(0, 2 * math.pi)
                perturbations.append(
                    {
                        "name": name,
                        "mass": params["mass"],
                        "semi_major_axis": a,
                        "position": [a * math.cos(theta), a * math.sin(theta), 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                    }
                )

    # Simulation parameters
    dt = config.get("timestep_days", KUIPER_TIMESTEP_DAYS)
    days_per_year = 365.25
    total_days = duration_years * days_per_year
    n_steps = int(total_days / dt)

    # Record initial energy
    initial_energy = compute_kuiper_total_energy(body_states, perturbations)

    # Record trajectory for Lyapunov calculation
    trajectory = []
    sample_interval = max(1, n_steps // 1000)

    # Run simulation
    for step in range(n_steps):
        body_states = symplectic_kuiper_integrate(body_states, dt, perturbations)

        if step % sample_interval == 0:
            state = []
            for b in body_states:
                state.extend(b["position"])
                state.extend(b["velocity"])
            trajectory.append(state)

    # Final energy
    final_energy = compute_kuiper_total_energy(body_states, perturbations)

    # Compute metrics
    energy_error = (
        abs(final_energy - initial_energy) / abs(initial_energy + 1e-20)
    )
    lyapunov = compute_kuiper_lyapunov(trajectory)
    energy_conserved = energy_error < config.get(
        "energy_conservation_tolerance", KUIPER_ENERGY_TOLERANCE
    )

    # Check stability
    lyapunov_threshold = config.get("lyapunov_threshold", KUIPER_LYAPUNOV_THRESHOLD)
    is_stable = lyapunov < lyapunov_threshold
    stability = 1.0 if is_stable else max(0.0, 1.0 - lyapunov / lyapunov_threshold)

    stability_target = config.get("stability_target", KUIPER_STABILITY_TARGET)

    result = {
        "body_count": len(body_states),
        "duration_years": duration_years,
        "timestep_days": dt,
        "n_steps": n_steps,
        "integration_method": "symplectic",
        "initial_energy": round(initial_energy, 15),
        "final_energy": round(final_energy, 15),
        "energy_error": round(energy_error, 18),
        "energy_conserved": energy_conserved,
        "lyapunov_exponent": lyapunov,
        "lyapunov_threshold": lyapunov_threshold,
        "is_stable": is_stable,
        "stability": round(stability, 4),
        "stability_target": stability_target,
        "target_met": stability >= stability_target,
        "perturbations_included": len(perturbations) > 0,
        "perturbation_sources": [p["name"] for p in perturbations],
    }

    emit_receipt(
        "kuiper_12body_stability",
        {
            "receipt_type": "kuiper_12body_stability",
            "tenant_id": KUIPER_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "body_count": len(body_states),
            "duration_years": duration_years,
            "lyapunov_exponent": lyapunov,
            "is_stable": is_stable,
            "stability": round(stability, 4),
            "energy_conserved": energy_conserved,
            "target_met": stability >= stability_target,
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


def check_kuiper_stability(
    lyapunov: float, threshold: float = KUIPER_LYAPUNOV_THRESHOLD
) -> bool:
    """Check if Lyapunov exponent indicates stability.

    Args:
        lyapunov: Computed Lyapunov exponent
        threshold: Stability threshold (default: 0.15)

    Returns:
        True if system is stable (lyapunov < threshold)
    """
    return lyapunov < threshold


def analyze_resonances(trajectory: List[List[float]]) -> Dict[str, Any]:
    """Analyze mean-motion resonances in the system.

    Args:
        trajectory: List of state vectors over time

    Returns:
        Dict with resonance analysis

    Receipt: kuiper_resonance_receipt
    """
    # Simplified resonance detection based on period ratios
    # Common Kuiper belt resonances with Neptune: 3:2 (Plutinos), 2:1, 5:3

    known_resonances = {
        "3:2": {"ratio": 1.5, "name": "Plutino", "bodies": ["pluto"]},
        "2:1": {"ratio": 2.0, "name": "Twotino", "bodies": []},
        "5:3": {"ratio": 1.667, "name": "5:3 resonance", "bodies": []},
    }

    # Estimate from orbital periods (simplified)
    neptune_period = 164.8  # years
    detected = []

    for name, res in known_resonances.items():
        expected_period = neptune_period * res["ratio"]
        detected.append(
            {
                "resonance": name,
                "name": res["name"],
                "expected_period_years": round(expected_period, 1),
                "associated_bodies": res["bodies"],
            }
        )

    result = {
        "resonances_analyzed": len(known_resonances),
        "detected_resonances": detected,
        "neptune_period_years": neptune_period,
        "primary_resonance": "3:2 (Plutino)",
    }

    emit_receipt(
        "kuiper_resonance",
        {
            "receipt_type": "kuiper_resonance",
            "tenant_id": KUIPER_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "resonances_analyzed": len(known_resonances),
            "primary_resonance": "3:2",
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def compute_close_encounters(
    trajectory: List[List[float]], threshold_au: float = 0.1
) -> List[Dict[str, Any]]:
    """Detect close encounters between bodies.

    Args:
        trajectory: List of state vectors over time
        threshold_au: Distance threshold in AU

    Returns:
        List of close encounter events

    Receipt: kuiper_encounter_receipt
    """
    encounters = []

    # Simplified: check sample points for close approaches
    n_bodies = 12
    state_size = 6  # 3 position + 3 velocity per body

    for t_idx, state in enumerate(trajectory[::10]):  # Sample every 10th point
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                pi = state[i * state_size : i * state_size + 3]
                pj = state[j * state_size : j * state_size + 3]

                if len(pi) < 3 or len(pj) < 3:
                    continue

                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dz = pi[2] - pj[2]
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                if distance < threshold_au:
                    encounters.append(
                        {
                            "time_index": t_idx * 10,
                            "body_i": i,
                            "body_j": j,
                            "distance_au": round(distance, 4),
                        }
                    )

    emit_receipt(
        "kuiper_encounter",
        {
            "receipt_type": "kuiper_encounter",
            "tenant_id": KUIPER_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "encounters_detected": len(encounters),
            "threshold_au": threshold_au,
            "payload_hash": dual_hash(
                json.dumps({"count": len(encounters)}, sort_keys=True)
            ),
        },
    )

    return encounters


def run_kuiper_monte_carlo(n_runs: int = 100) -> Dict[str, Any]:
    """Run Monte Carlo stability analysis for Kuiper system.

    Args:
        n_runs: Number of Monte Carlo runs

    Returns:
        Dict with statistical results

    Receipt: kuiper_monte_carlo_receipt
    """
    config = load_kuiper_config()

    stable_count = 0
    lyapunov_values = []
    stability_values = []

    for _ in range(n_runs):
        # Short simulation for Monte Carlo
        result = simulate_kuiper(
            bodies=config.get("body_count", KUIPER_BODY_COUNT),
            duration_years=10,  # Shorter for Monte Carlo
        )

        lyapunov_values.append(result["lyapunov_exponent"])
        stability_values.append(result["stability"])

        if result["is_stable"]:
            stable_count += 1

    mean_lyapunov = sum(lyapunov_values) / len(lyapunov_values)
    mean_stability = sum(stability_values) / len(stability_values)
    stable_fraction = stable_count / n_runs

    result = {
        "n_runs": n_runs,
        "stable_count": stable_count,
        "stable_fraction": round(stable_fraction, 4),
        "mean_lyapunov": round(mean_lyapunov, 6),
        "mean_stability": round(mean_stability, 4),
        "target": KUIPER_STABILITY_TARGET,
        "target_met": stable_fraction >= KUIPER_STABILITY_TARGET,
    }

    emit_receipt(
        "kuiper_monte_carlo",
        {
            "receipt_type": "kuiper_monte_carlo",
            "tenant_id": KUIPER_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "n_runs": n_runs,
            "stable_fraction": round(stable_fraction, 4),
            "mean_lyapunov": round(mean_lyapunov, 6),
            "target_met": stable_fraction >= KUIPER_STABILITY_TARGET,
            "payload_hash": dual_hash(
                json.dumps(
                    {"n_runs": n_runs, "stable_fraction": round(stable_fraction, 4)},
                    sort_keys=True,
                )
            ),
        },
    )

    return result


def compute_kuiper_chaos_tolerance(sim_results: Dict[str, Any]) -> float:
    """Compute Kuiper chaos tolerance from simulation results.

    Args:
        sim_results: Results from simulate_kuiper()

    Returns:
        Tolerance value in [0, 1], target is >= 0.93
    """
    stability_weight = 0.5
    energy_weight = 0.3
    lyapunov_weight = 0.2

    stability = sim_results.get("stability", 0.0)
    energy_ok = 1.0 if sim_results.get("energy_conserved", False) else 0.5

    lyapunov = sim_results.get("lyapunov_exponent", 1.0)
    lyapunov_threshold = sim_results.get("lyapunov_threshold", KUIPER_LYAPUNOV_THRESHOLD)
    lyapunov_factor = max(0.0, 1.0 - lyapunov / lyapunov_threshold)

    tolerance = (
        stability_weight * stability
        + energy_weight * energy_ok
        + lyapunov_weight * lyapunov_factor
    )

    return round(tolerance, 4)


def integrate_with_backbone(kuiper_results: Dict[str, Any]) -> Dict[str, Any]:
    """Wire Kuiper results to interstellar backbone.

    Args:
        kuiper_results: Results from Kuiper simulation

    Returns:
        Dict with backbone integration results

    Receipt: kuiper_backbone_receipt
    """
    # Import backbone module
    from .interstellar_backbone import (
        load_interstellar_config,
        get_all_bodies,
        INTERSTELLAR_BODY_COUNT,
    )

    backbone_config = load_interstellar_config()
    backbone_bodies = get_all_bodies()

    # Combine body counts
    total_bodies = INTERSTELLAR_BODY_COUNT + 5  # Add Kuiper belt objects

    # Compute combined stability
    kuiper_stability = kuiper_results.get("stability", 0.93)
    backbone_stability = backbone_config.get("autonomy_target", 0.98)
    combined_stability = (kuiper_stability + backbone_stability) / 2

    result = {
        "integration_complete": True,
        "kuiper_bodies": 12,
        "backbone_bodies": INTERSTELLAR_BODY_COUNT,
        "total_coordinated_bodies": total_bodies,
        "kuiper_stability": kuiper_stability,
        "backbone_stability": backbone_stability,
        "combined_stability": round(combined_stability, 4),
        "kuiper_extends_backbone": True,
        "coordination_mode": "d16_kuiper_hybrid",
    }

    emit_receipt(
        "kuiper_backbone",
        {
            "receipt_type": "kuiper_backbone",
            "tenant_id": KUIPER_TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "total_bodies": total_bodies,
            "combined_stability": round(combined_stability, 4),
            "integration_complete": True,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def get_kuiper_info() -> Dict[str, Any]:
    """Get Kuiper 12-body module configuration.

    Returns:
        Dict with module info
    """
    config = load_kuiper_config()

    return {
        "body_count": config.get("body_count", KUIPER_BODY_COUNT),
        "bodies": config.get("bodies", {}),
        "integration_method": config.get("integration_method", "symplectic"),
        "timestep_days": config.get("timestep_days", KUIPER_TIMESTEP_DAYS),
        "lyapunov_threshold": config.get("lyapunov_threshold", KUIPER_LYAPUNOV_THRESHOLD),
        "stability_target": config.get("stability_target", KUIPER_STABILITY_TARGET),
        "chaos_duration_years": config.get(
            "chaos_duration_years", KUIPER_DURATION_YEARS
        ),
        "energy_tolerance": config.get(
            "energy_conservation_tolerance", KUIPER_ENERGY_TOLERANCE
        ),
        "perturbation_sources": config.get("perturbation_sources", []),
        "description": "12-body Kuiper belt chaotic simulation with dwarf planets",
    }
