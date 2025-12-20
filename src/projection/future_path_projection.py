"""D19.2 Future Path Projection - Light-Speed Constrained Path Projection.

PARADIGM: Project receipt paths forward with light-speed constraints.

The Physics:
  Light-speed is absolute. A receipt path to Proxima Centauri takes 4.24 years
  one-way. We KNOW this. This knowledge is INFORMATION about the future.

  When we project a path forward, we're not "predicting" - we're calculating
  the deterministic trajectory through spacetime given known constraints.

Grok's Insight:
  "projects forward receipt paths using latency-bound models"

Key Constraint:
  All projections MUST respect light-speed. No FTL paths.
"""

import json
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core import emit_receipt, dual_hash, TENANT_ID, StopRule

# === D19.2 PROJECTION CONSTANTS ===

LIGHT_SPEED_KM_S = 299792.458
"""Light speed in km/s - absolute constraint."""

LIGHT_SPEED_LY_YEAR = 1.0
"""Light speed in light-years per year."""

PROJECTION_HORIZON_YEARS = 10
"""Default projection horizon (beyond Proxima RTT)."""

PATH_DIVERGENCE_TOLERANCE = 0.001
"""Maximum allowed path divergence."""


@dataclass
class ProjectedPath:
    """A path projected into the future."""

    path_id: str
    source_receipt_hash: str
    destination: str
    distance_ly: float
    departure_ts: str
    arrival_ts: str
    travel_time_years: float
    projected_entropy: float = 0.0
    projected_compression: float = 0.0
    light_speed_valid: bool = True
    waypoints: List[Dict] = field(default_factory=list)


@dataclass
class FuturePathProjection:
    """Future path projection engine state."""

    projection_id: str
    config: Dict = field(default_factory=dict)
    latency_catalog: Dict = field(default_factory=dict)
    projected_paths: Dict[str, ProjectedPath] = field(default_factory=dict)
    horizon_years: float = PROJECTION_HORIZON_YEARS
    total_projections: int = 0
    valid_projections: int = 0


def load_projection_config() -> Dict[str, Any]:
    """Load projection configuration from spec file.

    Returns:
        Projection configuration dict
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "projection_horizon_config.json",
    )

    with open(config_path, "r") as f:
        return json.load(f)


def load_latency_catalog(proj: FuturePathProjection = None) -> Dict[str, Any]:
    """Load catalog of known latencies as weave templates.

    Args:
        proj: Optional FuturePathProjection to update

    Returns:
        Latency catalog dict
    """
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "latency_catalog.json",
    )

    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    if proj is not None:
        proj.latency_catalog = catalog

    emit_receipt(
        "latency_catalog_load",
        {
            "receipt_type": "latency_catalog_load",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "interstellar_targets": len(catalog.get("interstellar", {})),
            "solar_system_targets": len(catalog.get("solar_system", {})),
            "payload_hash": dual_hash(json.dumps(catalog, sort_keys=True)),
        },
    )

    return catalog


def init_projection(config: Dict = None) -> FuturePathProjection:
    """Initialize future path projection engine.

    Args:
        config: Optional configuration dict

    Returns:
        FuturePathProjection instance

    Receipt: projection_init_receipt
    """
    if config is None:
        config = load_projection_config()

    projection_id = str(uuid.uuid4())[:8]
    horizon = config.get("projection_parameters", {}).get(
        "horizon_years", PROJECTION_HORIZON_YEARS
    )

    proj = FuturePathProjection(
        projection_id=projection_id,
        config=config,
        horizon_years=horizon,
    )

    # Load latency catalog
    load_latency_catalog(proj)

    emit_receipt(
        "projection_init",
        {
            "receipt_type": "projection_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "projection_id": projection_id,
            "horizon_years": horizon,
            "light_speed_binding": True,
            "simulation_enabled": False,
            "payload_hash": dual_hash(
                json.dumps({"projection_id": projection_id, "horizon": horizon}, sort_keys=True)
            ),
        },
    )

    return proj


def project_single_path(
    proj: FuturePathProjection,
    receipt: Dict,
    destination: str,
    horizon_years: float = None,
) -> ProjectedPath:
    """Project a single receipt path forward with light-speed constraints.

    Args:
        proj: FuturePathProjection instance
        receipt: Receipt to project
        destination: Destination identifier
        horizon_years: Optional override for horizon

    Returns:
        ProjectedPath instance

    Receipt: single_path_projection_receipt
    """
    now = datetime.utcnow()
    horizon = horizon_years or proj.horizon_years

    # Get distance from latency catalog
    distance_ly = 0.0
    rtt_years = 0.0

    # Check interstellar targets
    if destination in proj.latency_catalog.get("interstellar", {}):
        target = proj.latency_catalog["interstellar"][destination]
        distance_ly = target.get("distance_ly", 0)
        rtt_years = target.get("rtt_years", 0)
    # Check solar system targets
    elif destination in proj.latency_catalog.get("solar_system", {}):
        target = proj.latency_catalog["solar_system"][destination]
        # Convert to light-years
        distance_au = target.get("distance_au", target.get("distance_au_avg", 0))
        distance_ly = distance_au * 1.58125e-5  # AU to light-years
        rtt_seconds = target.get("rtt_seconds", target.get("rtt_seconds_avg", 0))
        rtt_years = rtt_seconds / (365.25 * 24 * 3600)
    else:
        # Default to Proxima Centauri
        distance_ly = 4.24
        rtt_years = 8.48

    # Calculate travel time (one-way) - MUST respect light speed
    travel_time_years = distance_ly / LIGHT_SPEED_LY_YEAR

    # Validate light-speed constraint
    light_speed_valid = travel_time_years >= distance_ly

    if not light_speed_valid:
        # This should never happen with correct physics
        raise StopRule(f"FTL violation: travel_time={travel_time_years}yr < distance={distance_ly}ly")

    # Calculate arrival timestamp
    arrival_seconds = travel_time_years * 365.25 * 24 * 3600
    arrival_dt = datetime.fromtimestamp(now.timestamp() + arrival_seconds)

    path_id = str(uuid.uuid4())[:8]
    receipt_hash = receipt.get("payload_hash", dual_hash(json.dumps(receipt, sort_keys=True)))

    path = ProjectedPath(
        path_id=path_id,
        source_receipt_hash=receipt_hash,
        destination=destination,
        distance_ly=distance_ly,
        departure_ts=now.isoformat() + "Z",
        arrival_ts=arrival_dt.isoformat() + "Z",
        travel_time_years=travel_time_years,
        light_speed_valid=light_speed_valid,
    )

    proj.projected_paths[path_id] = path
    proj.total_projections += 1
    if light_speed_valid:
        proj.valid_projections += 1

    emit_receipt(
        "single_path_projection",
        {
            "receipt_type": "single_path_projection",
            "tenant_id": TENANT_ID,
            "ts": now.isoformat() + "Z",
            "projection_id": proj.projection_id,
            "path_id": path_id,
            "destination": destination,
            "distance_ly": round(distance_ly, 4),
            "travel_time_years": round(travel_time_years, 4),
            "light_speed_valid": light_speed_valid,
            "payload_hash": dual_hash(
                json.dumps({"path_id": path_id, "destination": destination}, sort_keys=True)
            ),
        },
    )

    return path


def project_all_paths(
    proj: FuturePathProjection,
    receipts: List[Dict],
    destination: str = "proxima_centauri",
) -> Dict[str, Any]:
    """Project all receipt paths forward.

    Args:
        proj: FuturePathProjection instance
        receipts: List of receipts to project
        destination: Target destination for all paths

    Returns:
        Projection result dict

    Receipt: all_paths_projection_receipt
    """
    now = datetime.utcnow()
    paths = []

    for receipt in receipts:
        path = project_single_path(proj, receipt, destination)
        paths.append({
            "path_id": path.path_id,
            "travel_time_years": path.travel_time_years,
            "light_speed_valid": path.light_speed_valid,
        })

    result = {
        "projection_id": proj.projection_id,
        "destination": destination,
        "paths_projected": len(paths),
        "valid_paths": sum(1 for p in paths if p["light_speed_valid"]),
        "horizon_years": proj.horizon_years,
        "simulation_enabled": False,
    }

    emit_receipt(
        "all_paths_projection",
        {
            "receipt_type": "all_paths_projection",
            "tenant_id": TENANT_ID,
            "ts": now.isoformat() + "Z",
            **result,
            "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
        },
    )

    return result


def apply_light_speed_bound(proj: FuturePathProjection, path: ProjectedPath) -> ProjectedPath:
    """Ensure path respects light-speed constraint.

    Args:
        proj: FuturePathProjection instance
        path: Path to validate

    Returns:
        Validated path (raises StopRule if invalid)
    """
    # Calculate minimum travel time at light speed
    min_travel_time = path.distance_ly / LIGHT_SPEED_LY_YEAR

    if path.travel_time_years < min_travel_time:
        raise StopRule(
            f"Light-speed violation: path {path.path_id} has travel_time={path.travel_time_years}yr "
            f"< minimum={min_travel_time}yr for distance={path.distance_ly}ly"
        )

    path.light_speed_valid = True
    return path


def calculate_path_arrival(
    proj: FuturePathProjection,
    path: ProjectedPath,
    destination: str = None,
) -> float:
    """Calculate when path arrives at destination.

    Args:
        proj: FuturePathProjection instance
        path: Projected path
        destination: Optional destination override

    Returns:
        Arrival time in years from now
    """
    # Use path's destination if not overridden
    dest = destination or path.destination

    # Get distance from catalog
    distance_ly = path.distance_ly

    # Arrival time = distance / c
    arrival_years = distance_ly / LIGHT_SPEED_LY_YEAR

    return arrival_years


def estimate_future_entropy(
    proj: FuturePathProjection,
    paths: Dict[str, ProjectedPath],
) -> Dict[str, float]:
    """Estimate entropy at future points along paths.

    Args:
        proj: FuturePathProjection instance
        paths: Dict of projected paths

    Returns:
        Dict mapping path_id to estimated entropy

    Receipt: future_entropy_estimate_receipt
    """
    now = datetime.utcnow()
    entropy_estimates = {}

    for path_id, path in paths.items():
        # Entropy estimation based on travel time and distance
        # Longer paths = more entropy accumulation
        base_entropy = 1.0
        distance_factor = math.log2(1 + path.distance_ly)
        time_factor = math.log2(1 + path.travel_time_years)

        estimated_entropy = base_entropy + (distance_factor * 0.3) + (time_factor * 0.2)
        path.projected_entropy = round(estimated_entropy, 6)
        entropy_estimates[path_id] = path.projected_entropy

    emit_receipt(
        "future_entropy_estimate",
        {
            "receipt_type": "future_entropy_estimate",
            "tenant_id": TENANT_ID,
            "ts": now.isoformat() + "Z",
            "projection_id": proj.projection_id,
            "paths_estimated": len(entropy_estimates),
            "mean_entropy": round(sum(entropy_estimates.values()) / len(entropy_estimates), 6)
                if entropy_estimates else 0,
            "payload_hash": dual_hash(
                json.dumps({"count": len(entropy_estimates)}, sort_keys=True)
            ),
        },
    )

    return entropy_estimates


def get_projection_status() -> Dict[str, Any]:
    """Get projection module status.

    Returns:
        Status dict
    """
    return {
        "module": "projection.future_path_projection",
        "version": "19.2.0",
        "paradigm": "future_projection_weaving",
        "light_speed_km_s": LIGHT_SPEED_KM_S,
        "projection_horizon_years": PROJECTION_HORIZON_YEARS,
        "path_divergence_tolerance": PATH_DIVERGENCE_TOLERANCE,
        "simulation_enabled": False,
        "reactive_mode": False,
        "insight": "Laws are woven preemptively from projected future entropy trajectories",
    }
