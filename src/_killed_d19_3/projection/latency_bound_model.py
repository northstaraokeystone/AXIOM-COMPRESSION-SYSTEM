"""D19.2 Latency Bound Model - Light-Speed Constrained N-Body Model.

PARADIGM: Light-speed constrained n-body model for path projection.

The Physics:
  c is the absolute speed limit. All paths MUST respect this constraint.
  Gravitational bending affects path geometry but not maximum speed.

Key Constraint:
  model.validate_light_speed() MUST pass for all paths.
"""

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

from src.core import emit_receipt, dual_hash, TENANT_ID, StopRule

# === MODEL CONSTANTS ===

LIGHT_SPEED_KM_S = 299792.458
"""Light speed in km/s."""

LIGHT_SPEED_M_S = 299792458
"""Light speed in m/s."""

GRAVITATIONAL_CONSTANT = 6.67430e-11
"""Gravitational constant G in m^3 kg^-1 s^-2."""


@dataclass
class CelestialBody:
    """A gravitational body in the model."""

    name: str
    mass_kg: float
    position_km: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_km_s: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class GeodesicPath:
    """A light-speed geodesic path."""

    path_id: str
    start_km: Tuple[float, float, float]
    end_km: Tuple[float, float, float]
    distance_km: float
    travel_time_seconds: float
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    gravitational_deflection_rad: float = 0.0
    light_speed_valid: bool = True


@dataclass
class LatencyBoundModel:
    """Light-speed constrained n-body model."""

    model_id: str
    bodies: Dict[str, CelestialBody] = field(default_factory=dict)
    paths: Dict[str, GeodesicPath] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)


def init_model(config: Dict = None) -> LatencyBoundModel:
    """Initialize light-speed bound n-body model.

    Args:
        config: Optional configuration dict

    Returns:
        LatencyBoundModel instance

    Receipt: latency_model_init_receipt
    """
    config = config or {}
    model_id = str(uuid.uuid4())[:8]

    model = LatencyBoundModel(model_id=model_id, config=config)

    # Add default solar system bodies
    default_bodies = [
        CelestialBody(name="sun", mass_kg=1.989e30, position_km=(0, 0, 0)),
        CelestialBody(name="earth", mass_kg=5.972e24, position_km=(1.496e8, 0, 0)),
        CelestialBody(name="mars", mass_kg=6.417e23, position_km=(2.279e8, 0, 0)),
        CelestialBody(name="jupiter", mass_kg=1.898e27, position_km=(7.785e8, 0, 0)),
    ]

    for body in default_bodies:
        model.bodies[body.name] = body

    emit_receipt(
        "latency_model_init",
        {
            "receipt_type": "latency_model_init",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "model_id": model_id,
            "bodies_count": len(model.bodies),
            "light_speed_km_s": LIGHT_SPEED_KM_S,
            "payload_hash": dual_hash(
                json.dumps(
                    {"model_id": model_id, "bodies": len(model.bodies)}, sort_keys=True
                )
            ),
        },
    )

    return model


def add_body(
    model: LatencyBoundModel,
    name: str,
    mass_kg: float,
    position_km: Tuple[float, float, float],
) -> CelestialBody:
    """Add a gravitational body to the model.

    Args:
        model: LatencyBoundModel instance
        name: Body name
        mass_kg: Body mass in kg
        position_km: Position in km (x, y, z)

    Returns:
        Added CelestialBody
    """
    body = CelestialBody(name=name, mass_kg=mass_kg, position_km=position_km)
    model.bodies[name] = body
    return body


def calculate_distance(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        start: Start position (x, y, z) in km
        end: End position (x, y, z) in km

    Returns:
        Distance in km
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_geodesic(
    model: LatencyBoundModel,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
) -> GeodesicPath:
    """Calculate light-speed geodesic path between two points.

    Args:
        model: LatencyBoundModel instance
        start: Start position in km
        end: End position in km

    Returns:
        GeodesicPath instance

    Receipt: geodesic_calculation_receipt
    """
    path_id = str(uuid.uuid4())[:8]

    # Calculate straight-line distance
    distance_km = calculate_distance(start, end)

    # Travel time at light speed
    travel_time_seconds = distance_km / LIGHT_SPEED_KM_S

    # Create path
    path = GeodesicPath(
        path_id=path_id,
        start_km=start,
        end_km=end,
        distance_km=distance_km,
        travel_time_seconds=travel_time_seconds,
        waypoints=[start, end],
        light_speed_valid=True,
    )

    # Apply gravitational bending
    path = apply_gravitational_bending(model, path)

    model.paths[path_id] = path

    emit_receipt(
        "geodesic_calculation",
        {
            "receipt_type": "geodesic_calculation",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "model_id": model.model_id,
            "path_id": path_id,
            "distance_km": round(distance_km, 2),
            "travel_time_seconds": round(travel_time_seconds, 4),
            "gravitational_deflection_rad": round(
                path.gravitational_deflection_rad, 10
            ),
            "light_speed_valid": path.light_speed_valid,
            "payload_hash": dual_hash(
                json.dumps(
                    {"path_id": path_id, "distance": distance_km}, sort_keys=True
                )
            ),
        },
    )

    return path


def apply_gravitational_bending(
    model: LatencyBoundModel,
    path: GeodesicPath,
) -> GeodesicPath:
    """Apply gravitational bending to path near massive objects.

    Uses simplified gravitational lensing formula.

    Args:
        model: LatencyBoundModel instance
        path: Path to bend

    Returns:
        Modified path with gravitational deflection
    """
    total_deflection = 0.0

    for body in model.bodies.values():
        # Calculate closest approach to body
        # Simplified: use perpendicular distance from line to body

        # Vector from start to end
        dx = path.end_km[0] - path.start_km[0]
        dy = path.end_km[1] - path.start_km[1]
        dz = path.end_km[2] - path.start_km[2]

        # Vector from start to body
        bx = body.position_km[0] - path.start_km[0]
        by = body.position_km[1] - path.start_km[1]
        bz = body.position_km[2] - path.start_km[2]

        # Project body onto path direction
        path_length_sq = dx * dx + dy * dy + dz * dz
        if path_length_sq > 0:
            t = max(0, min(1, (bx * dx + by * dy + bz * dz) / path_length_sq))
        else:
            t = 0

        # Closest point on path to body
        closest_x = path.start_km[0] + t * dx
        closest_y = path.start_km[1] + t * dy
        closest_z = path.start_km[2] + t * dz

        # Distance from body to closest point
        impact_param_km = calculate_distance(
            (closest_x, closest_y, closest_z),
            body.position_km,
        )

        # Convert to meters
        impact_param_m = impact_param_km * 1000

        if impact_param_m > 0:
            # Gravitational deflection: delta = 4GM / (c^2 * b)
            deflection = (4 * GRAVITATIONAL_CONSTANT * body.mass_kg) / (
                LIGHT_SPEED_M_S * LIGHT_SPEED_M_S * impact_param_m
            )
            total_deflection += abs(deflection)

    path.gravitational_deflection_rad = total_deflection
    return path


def validate_light_speed(model: LatencyBoundModel, path: GeodesicPath) -> bool:
    """Verify path respects light-speed constraint everywhere.

    Args:
        model: LatencyBoundModel instance
        path: Path to validate

    Returns:
        True if valid, raises StopRule if invalid
    """
    # Calculate implied speed
    if path.travel_time_seconds <= 0:
        raise StopRule(
            f"Invalid travel time: {path.travel_time_seconds}s for path {path.path_id}"
        )

    implied_speed = path.distance_km / path.travel_time_seconds

    if implied_speed > LIGHT_SPEED_KM_S * 1.0001:  # Allow tiny numerical error
        path.light_speed_valid = False
        raise StopRule(
            f"Light-speed violation: path {path.path_id} implies speed {implied_speed} km/s > c ({LIGHT_SPEED_KM_S} km/s)"
        )

    path.light_speed_valid = True
    return True


def get_arrival_time(model: LatencyBoundModel, path: GeodesicPath) -> float:
    """Get arrival time for path.

    Args:
        model: LatencyBoundModel instance
        path: Path to get arrival time for

    Returns:
        Arrival time in seconds
    """
    return path.travel_time_seconds


def get_model_status() -> Dict[str, Any]:
    """Get latency bound model status.

    Returns:
        Status dict
    """
    return {
        "module": "projection.latency_bound_model",
        "version": "19.2.0",
        "light_speed_km_s": LIGHT_SPEED_KM_S,
        "gravitational_constant": GRAVITATIONAL_CONSTANT,
        "gravitational_bending": True,
        "ftl_allowed": False,
        "validation_mode": "strict",
    }
