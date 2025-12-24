"""telemetry.py - Fleet Telemetry Domain Generator

D20 Production Evolution: NEW module for vehicle/satellite compression testing.

THE TELEMETRY INSIGHT:
    Telemetry is high-bandwidth, highly compressible.
    The patterns reveal both physics and anomalies.
    Compression = discovery of underlying structure.

Source: AXIOM D20 Production Evolution

Stakeholder Value:
- Elon/xAI: Tesla FSD telemetry compression
- Starlink: Constellation telemetry
- SpaceX: Launch telemetry
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from ..core import emit_receipt, dual_hash

# === CONSTANTS ===

TENANT_ID = "axiom-domain-telemetry"

# Telemetry rates (samples per second)
TESLA_FSD_TELEMETRY_HZ = 100  # 100 Hz sensor fusion
STARLINK_TELEMETRY_HZ = 10  # 10 Hz satellite status
ROCKET_TELEMETRY_HZ = 1000  # 1 kHz launch telemetry


@dataclass
class TelemetryConfig:
    """Configuration for telemetry generation."""

    fleet_type: str = "tesla"
    duration_sec: float = 10.0
    sample_rate_hz: float = 100.0
    channels: int = 10
    noise_level: float = 0.01


def generate(fleet_type: str, params: Optional[Dict] = None) -> Dict:
    """Generate telemetry stream for specified fleet type.

    Args:
        fleet_type: Type of fleet ("tesla", "starlink", "rocket")
        params: Optional parameters dict

    Returns:
        Dict with domain="telemetry", fleet_type, data arrays
    """
    if params is None:
        params = {}

    if fleet_type == "tesla":
        return tesla_stream(params)
    elif fleet_type == "starlink":
        return starlink_stream(params)
    elif fleet_type == "rocket":
        return rocket_stream(params)
    else:
        # Default to generic telemetry
        return _generic_stream(fleet_type, params)


def tesla_stream(params: Dict) -> Dict:
    """Generate Tesla FSD telemetry stream.

    Simulates:
    - Camera frame embeddings (compressed)
    - Radar/ultrasonic sensor data
    - Vehicle state (speed, steering, etc.)
    - Environmental context

    Args:
        params: Configuration parameters

    Returns:
        Telemetry dict with data arrays
    """
    duration = params.get("duration_sec", 10.0)
    sample_rate = params.get("sample_rate_hz", TESLA_FSD_TELEMETRY_HZ)
    n_samples = int(duration * sample_rate)

    # Generate time series
    t = np.linspace(0, duration, n_samples)

    # Vehicle dynamics (smooth with noise)
    speed_mph = 35 + 10 * np.sin(2 * np.pi * t / 60) + np.random.randn(n_samples) * 0.5
    steering_deg = 5 * np.sin(2 * np.pi * t / 30) + np.random.randn(n_samples) * 0.2
    acceleration = np.gradient(speed_mph) + np.random.randn(n_samples) * 0.1

    # Sensor data (highly correlated, compressible)
    radar_range = (
        50 + 20 * np.sin(2 * np.pi * t / 45) + np.random.randn(n_samples) * 1.0
    )
    ultrasonic = np.clip(radar_range * 0.1 + np.random.randn(n_samples) * 0.05, 0, 10)

    # Camera embedding (compressed representation)
    camera_embedding = np.random.randn(n_samples, 128) * 0.1
    # Add temporal correlation
    for i in range(1, n_samples):
        camera_embedding[i] = 0.9 * camera_embedding[i - 1] + 0.1 * camera_embedding[i]

    # Flatten for hashing
    data_bytes = np.concatenate(
        [t, speed_mph, steering_deg, acceleration, radar_range, ultrasonic]
    ).tobytes()
    data_hash = dual_hash(data_bytes)
    params_hash = dual_hash(str(params))

    result = {
        "domain": "telemetry",
        "fleet_type": "tesla",
        "params_hash": params_hash,
        "data_hash": data_hash,
        "duration_sec": duration,
        "sample_rate_hz": sample_rate,
        "n_samples": n_samples,
        "channels": {
            "t": t.tolist(),
            "speed_mph": speed_mph.tolist(),
            "steering_deg": steering_deg.tolist(),
            "acceleration": acceleration.tolist(),
            "radar_range": radar_range.tolist(),
            "ultrasonic": ultrasonic.tolist(),
        },
        "compression_hint": "high_temporal_correlation",
    }

    emit_receipt(
        "domain_receipt",
        {
            "tenant_id": TENANT_ID,
            "domain": "telemetry",
            "fleet_type": "tesla",
            "params_hash": params_hash,
            "data_hash": data_hash,
            "n_samples": n_samples,
        },
    )

    return result


def starlink_stream(params: Dict) -> Dict:
    """Generate Starlink constellation telemetry stream.

    Simulates:
    - Satellite position/velocity
    - Link quality metrics
    - Thermal status
    - Power levels

    Args:
        params: Configuration parameters

    Returns:
        Telemetry dict with data arrays
    """
    duration = params.get("duration_sec", 60.0)
    sample_rate = params.get("sample_rate_hz", STARLINK_TELEMETRY_HZ)
    n_satellites = params.get("n_satellites", 10)
    n_samples = int(duration * sample_rate)

    t = np.linspace(0, duration, n_samples)

    # Per-satellite telemetry
    satellite_data = []
    for sat_id in range(n_satellites):
        # Orbital parameters (periodic with slight phase offsets)
        phase = 2 * np.pi * sat_id / n_satellites
        lat = 45 * np.sin(2 * np.pi * t / 90 + phase)
        lon = 180 * np.sin(2 * np.pi * t / 93 + phase)
        alt = 550 + 5 * np.sin(2 * np.pi * t / 30 + phase)

        # Link metrics
        snr_db = 15 + 3 * np.sin(2 * np.pi * t / 60) + np.random.randn(n_samples) * 0.5
        throughput_mbps = 100 * (1 + 0.2 * np.sin(2 * np.pi * t / 45))

        # Thermal/power
        temp_c = -20 + 40 * np.sin(2 * np.pi * t / 45)  # Eclipse cycles
        power_w = 2000 + 500 * np.sin(2 * np.pi * t / 45)

        satellite_data.append(
            {
                "sat_id": sat_id,
                "lat": lat.tolist(),
                "lon": lon.tolist(),
                "alt_km": alt.tolist(),
                "snr_db": snr_db.tolist(),
                "throughput_mbps": throughput_mbps.tolist(),
                "temp_c": temp_c.tolist(),
                "power_w": power_w.tolist(),
            }
        )

    # Hash representative data
    data_bytes = np.array([sat_id for sat_id in range(n_satellites)]).tobytes()
    data_hash = dual_hash(data_bytes + str(n_samples).encode())
    params_hash = dual_hash(str(params))

    result = {
        "domain": "telemetry",
        "fleet_type": "starlink",
        "params_hash": params_hash,
        "data_hash": data_hash,
        "duration_sec": duration,
        "sample_rate_hz": sample_rate,
        "n_satellites": n_satellites,
        "n_samples": n_samples,
        "satellites": satellite_data,
        "compression_hint": "constellation_correlation",
    }

    emit_receipt(
        "domain_receipt",
        {
            "tenant_id": TENANT_ID,
            "domain": "telemetry",
            "fleet_type": "starlink",
            "params_hash": params_hash,
            "data_hash": data_hash,
            "n_satellites": n_satellites,
            "n_samples": n_samples,
        },
    )

    return result


def rocket_stream(params: Dict) -> Dict:
    """Generate SpaceX launch telemetry stream.

    Simulates:
    - Trajectory data (altitude, velocity, acceleration)
    - Engine telemetry (thrust, chamber pressure, temperature)
    - Attitude (pitch, yaw, roll)
    - Stage events

    Args:
        params: Configuration parameters

    Returns:
        Telemetry dict with data arrays
    """
    duration = params.get("duration_sec", 600.0)  # 10 minute launch
    sample_rate = params.get("sample_rate_hz", ROCKET_TELEMETRY_HZ)
    n_samples = int(duration * sample_rate)

    t = np.linspace(0, duration, n_samples)

    # Trajectory (simplified Falcon 9 profile)
    # Acceleration profile: max-Q around T+80s
    # g = 9.81 m/s^2 (for reference)
    thrust_accel = 15 * np.exp(-(((t - 200) / 300) ** 2)) + 10  # m/s^2

    # Velocity and altitude integration
    velocity = np.cumsum(thrust_accel / sample_rate)  # m/s
    altitude = np.cumsum(velocity / sample_rate)  # m

    # Engine telemetry (9 engines, then SECO1, stage sep, etc.)
    stage_sep_time = 150.0  # seconds
    is_stage1 = t < stage_sep_time

    thrust_kn = np.where(
        is_stage1,
        7500 + 500 * np.random.randn(n_samples),
        934 + 50 * np.random.randn(n_samples),
    )
    chamber_pressure_bar = thrust_kn / 100 + np.random.randn(n_samples) * 0.5
    nozzle_temp_c = 1800 + 200 * np.random.randn(n_samples)

    # Attitude (small oscillations during ascent)
    pitch_deg = 90 - 90 * np.clip(t / 300, 0, 1) + np.random.randn(n_samples) * 0.1
    yaw_deg = np.random.randn(n_samples) * 0.1
    roll_deg = np.random.randn(n_samples) * 0.1

    data_bytes = np.concatenate([t, altitude, velocity]).tobytes()
    data_hash = dual_hash(data_bytes)
    params_hash = dual_hash(str(params))

    result = {
        "domain": "telemetry",
        "fleet_type": "rocket",
        "params_hash": params_hash,
        "data_hash": data_hash,
        "duration_sec": duration,
        "sample_rate_hz": sample_rate,
        "n_samples": n_samples,
        "channels": {
            "t": t.tolist(),
            "altitude_m": altitude.tolist(),
            "velocity_mps": velocity.tolist(),
            "thrust_kn": thrust_kn.tolist(),
            "chamber_pressure_bar": chamber_pressure_bar.tolist(),
            "nozzle_temp_c": nozzle_temp_c.tolist(),
            "pitch_deg": pitch_deg.tolist(),
            "yaw_deg": yaw_deg.tolist(),
            "roll_deg": roll_deg.tolist(),
        },
        "events": [
            {"t": 0, "event": "liftoff"},
            {"t": 80, "event": "max_q"},
            {"t": stage_sep_time, "event": "stage_separation"},
            {"t": 480, "event": "SECO1"},
        ],
        "compression_hint": "high_bandwidth_correlated",
    }

    emit_receipt(
        "domain_receipt",
        {
            "tenant_id": TENANT_ID,
            "domain": "telemetry",
            "fleet_type": "rocket",
            "params_hash": params_hash,
            "data_hash": data_hash,
            "n_samples": n_samples,
        },
    )

    return result


def _generic_stream(fleet_type: str, params: Dict) -> Dict:
    """Generate generic telemetry stream.

    Args:
        fleet_type: Name of fleet type
        params: Configuration parameters

    Returns:
        Telemetry dict with generic data
    """
    duration = params.get("duration_sec", 10.0)
    sample_rate = params.get("sample_rate_hz", 100.0)
    channels = params.get("channels", 5)
    n_samples = int(duration * sample_rate)

    t = np.linspace(0, duration, n_samples)
    data = {
        f"channel_{i}": (np.random.randn(n_samples) * 0.1).tolist()
        for i in range(channels)
    }
    data["t"] = t.tolist()

    data_hash = dual_hash(str(data))
    params_hash = dual_hash(str(params))

    result = {
        "domain": "telemetry",
        "fleet_type": fleet_type,
        "params_hash": params_hash,
        "data_hash": data_hash,
        "duration_sec": duration,
        "sample_rate_hz": sample_rate,
        "n_samples": n_samples,
        "channels": data,
    }

    emit_receipt(
        "domain_receipt",
        {
            "tenant_id": TENANT_ID,
            "domain": "telemetry",
            "fleet_type": fleet_type,
            "params_hash": params_hash,
            "data_hash": data_hash,
        },
    )

    return result
