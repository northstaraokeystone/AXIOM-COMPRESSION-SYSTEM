"""starship_fleet.py - 1000+ Starship Launches Per Year Model

THE FLEET INSIGHT:
    A Starship doesn't deliver cargo. It delivers NEGENTROPY.
    500t payload = X joules of ordered work = Y bits of decision capacity.
    Entropy delivered = payload_kg × (ordered_energy_per_kg - ambient_entropy)

Source: SpaceProof v3.0 Multi-Tier Autonomy Network Evolution
Grok: "500t payload", "1000 flights/year target"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import math
import numpy as np

from ..core import emit_receipt, dual_hash

# === CONSTANTS ===

TENANT_ID = "spaceproof-fleet"

# Starship specifications (SpaceX official targets)
STARSHIP_PAYLOAD_KG = 500000  # 500t payload (Grok: "500t payload")
STARSHIP_FLIGHTS_PER_YEAR = 1000  # Grok: "1000 flights/year target"
ORDERED_ENERGY_PER_KG = 1e7  # J/kg (chemical energy in payload)

# Mars transfer window constraints
MARS_SYNODIC_PERIOD_MONTHS = 26  # Mars transfer windows
LAUNCHES_PER_WINDOW = 100  # Max launches per optimal window
WINDOW_DURATION_DAYS = 30  # Launch window duration

# Entropy constants
AMBIENT_ENTROPY_MARS = 1e6  # J/kg (Mars surface entropy)
AMBIENT_ENTROPY_ORBIT = 5e5  # J/kg (Mars orbit entropy)
AMBIENT_ENTROPY_DEEP_SPACE = 1e5  # J/kg (Deep space entropy)


class FleetStatus(Enum):
    """Fleet operational status."""

    NOMINAL = "nominal"
    STRESSED = "stressed"
    CRITICAL = "critical"


class Destination(Enum):
    """Starship delivery destinations."""

    MARS_SURFACE = "mars_surface"
    MARS_ORBIT = "mars_orbit"
    LUNAR_SURFACE = "lunar_surface"
    LEO = "leo"
    DEEP_SPACE = "deep_space"


@dataclass
class FleetConfig:
    """Configuration for fleet simulation.

    Attributes:
        n_starships: Number of Starships in fleet
        payload_kg: Payload per Starship in kg
        destinations: List of valid destinations
        launch_windows_per_year: Number of launch windows
        reliability: Launch success probability
    """

    n_starships: int = STARSHIP_FLIGHTS_PER_YEAR
    payload_kg: float = STARSHIP_PAYLOAD_KG
    destinations: List[str] = field(default_factory=lambda: ["mars_surface", "mars_orbit"])
    launch_windows_per_year: int = 12
    reliability: float = 0.98


@dataclass
class StarshipLaunch:
    """Record of a single Starship launch.

    Attributes:
        launch_id: Unique identifier
        timestamp: Launch time (ISO8601)
        payload_kg: Payload mass
        destination: Target destination
        entropy_delivered: Negentropy delivered (joules)
        success: Whether launch succeeded
    """

    launch_id: str
    timestamp: str
    payload_kg: float
    destination: str
    entropy_delivered: float
    success: bool


@dataclass
class FleetState:
    """State of Starship fleet at a point in time.

    Attributes:
        ts: Timestamp (ISO8601)
        year: Simulation year
        launches_this_year: Total launches in current year
        total_payload_delivered_kg: Cumulative payload delivered
        total_entropy_delivered: Cumulative entropy delivered
        active_starships: Number of operational Starships
        status: Fleet operational status
        launches: List of launch records
    """

    ts: str
    year: int
    launches_this_year: int
    total_payload_delivered_kg: float
    total_entropy_delivered: float
    active_starships: int
    status: str
    launches: List[StarshipLaunch] = field(default_factory=list)


# === AMBIENT ENTROPY LOOKUP ===

AMBIENT_ENTROPY: Dict[str, float] = {
    "mars_surface": AMBIENT_ENTROPY_MARS,
    "mars_orbit": AMBIENT_ENTROPY_ORBIT,
    "lunar_surface": AMBIENT_ENTROPY_ORBIT,
    "leo": AMBIENT_ENTROPY_ORBIT * 0.5,
    "deep_space": AMBIENT_ENTROPY_DEEP_SPACE,
}


def calculate_entropy_delivered(payload_kg: float, destination: str) -> float:
    """Calculate negentropy (ordered work) delivered by Starship.

    Entropy delivered = payload_kg × (ordered_energy_per_kg - ambient_entropy)

    Args:
        payload_kg: Payload mass in kg
        destination: Delivery destination

    Returns:
        Negentropy delivered in joules (always >= 0)
    """
    ambient = AMBIENT_ENTROPY.get(destination, AMBIENT_ENTROPY_MARS)
    entropy = payload_kg * (ORDERED_ENERGY_PER_KG - ambient)

    # Cannot deliver negative entropy (thermodynamic law)
    return max(0.0, entropy)


def generate_launch_windows(year: int, n_launches: int, seed: int = 42) -> List[Dict]:
    """Generate optimal launch windows for n Starships.

    Respects Mars transfer windows (26-month synodic period).

    Args:
        year: Simulation year
        n_launches: Number of launches to schedule
        seed: Random seed for reproducibility

    Returns:
        List of launch window dicts with timestamp and window_id
    """
    rng = np.random.default_rng(seed + year)
    windows = []

    # Distribute launches across available windows
    launches_per_window = n_launches // 12 + 1

    for month in range(12):
        if len(windows) >= n_launches:
            break

        # Not all months have optimal windows
        window_quality = 0.5 + 0.5 * math.sin(2 * math.pi * (month - 3) / MARS_SYNODIC_PERIOD_MONTHS)

        n_this_window = min(
            int(launches_per_window * window_quality * rng.uniform(0.8, 1.2)),
            LAUNCHES_PER_WINDOW,
        )

        for day in range(min(n_this_window, WINDOW_DURATION_DAYS)):
            if len(windows) >= n_launches:
                break

            windows.append(
                {
                    "window_id": f"W{year}-{month:02d}",
                    "timestamp": f"{year}-{month + 1:02d}-{day + 1:02d}T{rng.integers(0, 24):02d}:00:00Z",
                    "window_quality": window_quality,
                }
            )

    # Fill remaining launches if needed (fallback for high-demand scenarios)
    while len(windows) < n_launches:
        month = len(windows) % 12
        day = (len(windows) // 12) % 30 + 1
        window_quality = 0.5 + 0.5 * math.sin(2 * math.pi * (month - 3) / MARS_SYNODIC_PERIOD_MONTHS)
        windows.append(
            {
                "window_id": f"W{year}-{month:02d}",
                "timestamp": f"{year}-{month + 1:02d}-{day:02d}T{rng.integers(0, 24):02d}:00:00Z",
                "window_quality": window_quality,
            }
        )

    return windows[:n_launches]


def simulate_fleet(config: FleetConfig, duration_years: int, seed: int = 42) -> List[FleetState]:
    """Run full fleet simulation.

    Args:
        config: FleetConfig with fleet parameters
        duration_years: Number of years to simulate
        seed: Random seed

    Returns:
        List of FleetState, one per year
    """
    rng = np.random.default_rng(seed)
    states = []

    cumulative_payload = 0.0
    cumulative_entropy = 0.0

    for year in range(duration_years):
        year_launches = []
        year_payload = 0.0
        year_entropy = 0.0

        # Generate launch windows for this year
        windows = generate_launch_windows(year, config.n_starships, seed + year)

        for i, window in enumerate(windows):
            # Determine success based on reliability
            success = rng.random() < config.reliability

            # Select destination
            dest = rng.choice(config.destinations)

            # Calculate entropy delivered (0 if failed)
            if success:
                entropy = calculate_entropy_delivered(config.payload_kg, dest)
                payload = config.payload_kg
            else:
                entropy = 0.0
                payload = 0.0

            launch = StarshipLaunch(
                launch_id=f"SS-{year}-{i:04d}",
                timestamp=window["timestamp"],
                payload_kg=payload,
                destination=dest,
                entropy_delivered=entropy,
                success=success,
            )
            year_launches.append(launch)

            year_payload += payload
            year_entropy += entropy

            # Emit launch receipt
            emit_receipt(
                "starship_launch_receipt",
                {
                    "tenant_id": TENANT_ID,
                    "launch_id": launch.launch_id,
                    "timestamp": launch.timestamp,
                    "payload_kg": launch.payload_kg,
                    "destination": launch.destination,
                    "entropy_delivered": launch.entropy_delivered,
                    "success": launch.success,
                    "window_id": window["window_id"],
                },
            )

        cumulative_payload += year_payload
        cumulative_entropy += year_entropy

        # Determine fleet status
        success_rate = sum(1 for l in year_launches if l.success) / len(year_launches)
        if success_rate >= 0.95:
            status = FleetStatus.NOMINAL.value
        elif success_rate >= 0.80:
            status = FleetStatus.STRESSED.value
        else:
            status = FleetStatus.CRITICAL.value

        state = FleetState(
            ts=f"{year}-12-31T23:59:59Z",
            year=year,
            launches_this_year=len(year_launches),
            total_payload_delivered_kg=cumulative_payload,
            total_entropy_delivered=cumulative_entropy,
            active_starships=int(config.n_starships * success_rate),
            status=status,
            launches=year_launches,
        )
        states.append(state)

        # Emit yearly fleet state receipt
        emit_receipt(
            "fleet_state_receipt",
            {
                "tenant_id": TENANT_ID,
                "year": year,
                "launches_this_year": state.launches_this_year,
                "total_payload_delivered_kg": state.total_payload_delivered_kg,
                "total_entropy_delivered": state.total_entropy_delivered,
                "active_starships": state.active_starships,
                "status": state.status,
                "success_rate": success_rate,
            },
        )

    return states


def fleet_to_colony_entropy(fleet_state: FleetState, colony_ids: List[str]) -> Dict[str, float]:
    """Map Starship deliveries to colony entropy injection.

    Args:
        fleet_state: Current fleet state
        colony_ids: List of colony identifiers

    Returns:
        Dict mapping colony_id to entropy received
    """
    if not colony_ids:
        return {}

    # Filter to Mars surface deliveries
    mars_launches = [l for l in fleet_state.launches if l.destination == "mars_surface" and l.success]

    total_entropy = sum(l.entropy_delivered for l in mars_launches)

    # Distribute entropy across colonies (equal distribution for now)
    per_colony = total_entropy / len(colony_ids) if colony_ids else 0.0

    distribution = {cid: per_colony for cid in colony_ids}

    emit_receipt(
        "fleet_to_colony_receipt",
        {
            "tenant_id": TENANT_ID,
            "year": fleet_state.year,
            "total_entropy": total_entropy,
            "n_colonies": len(colony_ids),
            "per_colony_entropy": per_colony,
            "data_hash": dual_hash(str(distribution)),
        },
    )

    return distribution


def calculate_fleet_bandwidth(fleet_state: FleetState, bits_per_joule: float = 1e-6) -> float:
    """Calculate effective information bandwidth from fleet entropy.

    Converting energy to information capacity via Landauer's principle.

    Args:
        fleet_state: Current fleet state
        bits_per_joule: Information capacity per joule

    Returns:
        Effective bandwidth in bits/second (annualized)
    """
    # Total entropy delivered this year
    year_entropy = sum(l.entropy_delivered for l in fleet_state.launches if l.success)

    # Convert to bits and annualize (bits per second)
    seconds_per_year = 365.25 * 24 * 3600
    bandwidth_bps = (year_entropy * bits_per_joule) / seconds_per_year

    return bandwidth_bps
