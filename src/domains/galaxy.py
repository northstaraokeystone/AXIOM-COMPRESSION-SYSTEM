"""galaxy.py - Galaxy Rotation Curve Domain Generator

D20 Production Evolution: Renamed from cosmos.py for stakeholder clarity.

THE GALAXY INSIGHT:
    Rotation curves reveal hidden physics.
    Newtonian at small r, dark matter at large r.
    The crossover encodes information about mass distribution.

Source: AXIOM D20 Production Evolution
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional
import numpy as np

from ..core import emit_receipt, dual_hash

# === CONSTANTS ===

TENANT_ID = "axiom-domain-galaxy"

# Physical constants
G = 4.302e-6  # Gravitational constant in kpc * (km/s)^2 / M_sun
M_SUN = 1.989e30  # Solar mass in kg
KPC_TO_M = 3.086e19  # kpc to meters

# Typical galaxy parameters
MILKY_WAY_MASS = 1.5e12  # M_sun
MILKY_WAY_DISK_SCALE = 3.0  # kpc


class PhysicsRegime(Enum):
    """Physics regime for rotation curve."""

    NEWTONIAN = "newtonian"
    DARK_MATTER = "dark_matter"
    MOND = "mond"
    NFW = "nfw"
    PBH_FOG = "pbh_fog"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class GalaxyParams:
    """Parameters for synthetic galaxy generation."""

    mass: float = 1e11  # Total mass in M_sun
    disk_scale: float = 3.0  # Disk scale length in kpc
    halo_scale: float = 10.0  # Halo scale length in kpc
    halo_mass: float = 1e12  # Dark matter halo mass in M_sun
    luminosity: float = 1e10  # Luminosity in L_sun
    inclination: float = 60.0  # Inclination angle in degrees


# === ROTATION CURVE PHYSICS ===


def newton_curve(r: np.ndarray, M: float) -> np.ndarray:
    """Compute Newtonian rotation velocity.

    V = sqrt(GM/r)

    Args:
        r: Radius in kpc
        M: Enclosed mass in M_sun

    Returns:
        Velocity in km/s
    """
    return np.sqrt(G * M / r)


def mond_curve(r: np.ndarray, M: float, a0: float = 1.2e-10) -> np.ndarray:
    """Compute MOND rotation velocity (deep MOND regime).

    In the deep MOND regime (a << a0): v^4 = G * M * a0

    Args:
        r: Radius in kpc
        M: Baryonic mass in M_sun
        a0: MOND acceleration scale in m/s^2 (default 1.2e-10)

    Returns:
        Velocity in km/s
    """
    # Convert a0 to natural units (kpc, km/s, M_sun)
    a0_natural = a0 * KPC_TO_M / 1e6  # km/s^2

    # Newtonian acceleration
    a_N = G * M / r**2

    # MOND interpolation function (simple form)
    x = a_N / a0_natural
    mu = x / np.sqrt(1 + x**2)

    # MOND velocity
    a_mond = a_N / mu
    return np.sqrt(a_mond * r)


def nfw_curve(r: np.ndarray, params: Dict) -> np.ndarray:
    """Compute velocity contribution from NFW dark matter halo.

    NFW profile: rho(r) = rho_s / (r/r_s * (1 + r/r_s)^2)

    Args:
        r: Radius in kpc
        params: Dict with M_vir, c, R_vir

    Returns:
        Velocity in km/s
    """
    M_vir = params.get("M_vir", 1e12)
    c = params.get("c", 10.0)
    R_vir = params.get("R_vir", 100.0)

    r_s = R_vir / c
    x = r / r_s

    # NFW enclosed mass factor
    f_c = np.log(1 + c) - c / (1 + c)
    M_enc = M_vir * (np.log(1 + x) - x / (1 + x)) / f_c

    return np.sqrt(G * M_enc / r)


def pbh_fog_curve(r: np.ndarray, params: Dict) -> np.ndarray:
    """Compute velocity for PBH fog model (novel dark matter alternative).

    Primordial black holes distributed as fog throughout the galaxy.

    Args:
        r: Radius in kpc
        params: Dict with pbh_density, pbh_mass

    Returns:
        Velocity in km/s
    """
    pbh_density = params.get("pbh_density", 1e6)  # M_sun/kpc^3
    core_radius = params.get("core_radius", 2.0)  # kpc

    # PBH fog mass enclosed (assuming isothermal-like distribution)
    M_enc = pbh_density * (4 / 3) * np.pi * r**3 * np.exp(-r / core_radius)

    return np.sqrt(G * M_enc / r)


def generate(
    regime: str,
    params: Optional[Dict] = None,
    n_points: int = 20,
    r_min: float = 0.5,
    r_max: float = 20.0,
    noise_level: float = 0.05,
) -> Dict:
    """Generate galaxy rotation curve data.

    Args:
        regime: Physics regime ("newtonian", "mond", "dark_matter", "nfw", "pbh_fog")
        params: Optional parameters dict
        n_points: Number of data points
        r_min: Minimum radius in kpc
        r_max: Maximum radius in kpc
        noise_level: Fractional noise level

    Returns:
        Dict with domain="galaxy", regime, data_hash, r, v, v_unc
    """
    if params is None:
        params = {}

    galaxy_params = GalaxyParams(
        mass=params.get("mass", 1e11),
        disk_scale=params.get("disk_scale", 3.0),
        halo_mass=params.get("halo_mass", 1e12),
    )

    # Generate radius array
    r = np.linspace(r_min, r_max, n_points)

    # Compute rotation curve based on regime
    if regime == "newtonian":
        v_true = newton_curve(r, galaxy_params.mass)
    elif regime == "mond":
        v_true = mond_curve(r, galaxy_params.mass)
    elif regime == "nfw" or regime == "dark_matter":
        v_true = nfw_curve(
            r,
            {
                "M_vir": galaxy_params.halo_mass,
                "c": params.get("c", 10.0),
                "R_vir": params.get("R_vir", 100.0),
            },
        )
    elif regime == "pbh_fog":
        v_true = pbh_fog_curve(r, params)
    else:  # mixed
        # Combine disk and halo
        v_disk = newton_curve(r, galaxy_params.mass * 0.3)
        v_halo = nfw_curve(
            r,
            {
                "M_vir": galaxy_params.halo_mass,
                "c": params.get("c", 10.0),
                "R_vir": params.get("R_vir", 100.0),
            },
        )
        v_true = np.sqrt(v_disk**2 + v_halo**2)

    # Add noise
    v_noise = v_true * (1 + noise_level * np.random.randn(n_points))
    v_unc = v_true * noise_level * np.ones(n_points)

    # Compute data hash
    data_bytes = np.concatenate([r, v_noise]).tobytes()
    data_hash = dual_hash(data_bytes)

    result = {
        "domain": "galaxy",
        "regime": regime,
        "params_hash": dual_hash(str(params)),
        "data_hash": data_hash,
        "r": r.tolist(),
        "v": v_noise.tolist(),
        "v_unc": v_unc.tolist(),
        "n_points": n_points,
    }

    # Emit domain receipt
    emit_receipt(
        "domain_receipt",
        {
            "tenant_id": TENANT_ID,
            "domain": "galaxy",
            "regime": regime,
            "params_hash": result["params_hash"],
            "data_hash": data_hash,
            "n_points": n_points,
        },
    )

    return result


def classify_regime(
    galaxy: Dict, v_pred: Optional[np.ndarray] = None
) -> Tuple[str, float]:
    """Classify the physics regime of a galaxy.

    Uses rotation curve shape to infer underlying physics.

    Args:
        galaxy: Galaxy dict with r, v arrays
        v_pred: Optional predicted velocities

    Returns:
        Tuple of (regime_name, confidence)
    """
    r = np.array(galaxy["r"])
    v = np.array(galaxy["v"])

    # Compute v^2 * r (should be constant for Newtonian)
    v2r = v**2 * r
    v2r_std = np.std(v2r) / np.mean(v2r)

    # Compute v gradient at large r (flat = dark matter dominated)
    if len(v) >= 5:
        v_gradient = (v[-1] - v[-5]) / (r[-1] - r[-5])
        flat_curve = abs(v_gradient) < 5  # km/s per kpc
    else:
        flat_curve = False

    # Classify
    if v2r_std < 0.1:
        return "newtonian", 1 - v2r_std
    elif flat_curve and v2r_std > 0.2:
        return "dark_matter", 0.8
    else:
        return "mixed", 0.6
