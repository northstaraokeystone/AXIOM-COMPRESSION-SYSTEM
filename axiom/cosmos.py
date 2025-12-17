"""cosmos.py - Galaxy Rotation Curve Generation and Loading

THE COSMOS INSIGHT:
    Rotation curves reveal hidden physics.
    Newtonian at small r, dark matter at large r.
    The crossover encodes information about mass distribution.

Source: AXIOM Validation Lock v1
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

# Import from src
try:
    from src.core import emit_receipt
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.core import emit_receipt


# === CONSTANTS ===

TENANT_ID = "axiom-cosmos"

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

def newtonian_velocity(r: np.ndarray, M: float) -> np.ndarray:
    """Compute Newtonian rotation velocity.

    v(r) = sqrt(G * M / r)

    Args:
        r: Radius in kpc
        M: Enclosed mass in M_sun

    Returns:
        Velocity in km/s
    """
    return np.sqrt(G * M / r)


def exponential_disk_velocity(r: np.ndarray, M_disk: float, R_d: float) -> np.ndarray:
    """Compute velocity contribution from exponential disk.

    Uses Freeman (1970) approximation for thin disk.

    Args:
        r: Radius in kpc
        M_disk: Total disk mass in M_sun
        R_d: Disk scale length in kpc

    Returns:
        Velocity in km/s
    """
    y = r / (2 * R_d)
    # Bessel function approximation for I0*K0 - I1*K1
    disk_factor = 3.36 * y**2 * (1 - 0.5 * y + 0.125 * y**2) * np.exp(-y)
    return np.sqrt(G * M_disk / R_d * disk_factor)


def nfw_halo_velocity(r: np.ndarray, M_vir: float, c: float, R_vir: float) -> np.ndarray:
    """Compute velocity contribution from NFW dark matter halo.

    NFW profile: rho(r) = rho_s / (r/r_s * (1 + r/r_s)^2)

    Args:
        r: Radius in kpc
        M_vir: Virial mass in M_sun
        c: Concentration parameter
        R_vir: Virial radius in kpc

    Returns:
        Velocity in km/s
    """
    r_s = R_vir / c
    x = r / r_s

    # NFW enclosed mass factor
    f_c = np.log(1 + c) - c / (1 + c)
    M_enc = M_vir * (np.log(1 + x) - x / (1 + x)) / f_c

    return np.sqrt(G * M_enc / r)


def mond_velocity(r: np.ndarray, M: float, a0: float = 1.2e-10) -> np.ndarray:
    """Compute MOND rotation velocity.

    MOND: a = a_N * mu(a_N/a0) where mu(x) = x/sqrt(1+x^2) (simple interpolation)

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


def generate_galaxy(
    galaxy_id: str = None,
    regime: PhysicsRegime = PhysicsRegime.MIXED,
    n_points: int = 20,
    r_min: float = 0.5,
    r_max: float = 20.0,
    params: GalaxyParams = None,
    noise_level: float = 0.05
) -> Dict:
    """Generate synthetic galaxy rotation curve.

    Args:
        galaxy_id: Galaxy identifier
        regime: Physics regime to use
        n_points: Number of data points
        r_min: Minimum radius in kpc
        r_max: Maximum radius in kpc
        params: Galaxy parameters
        noise_level: Fractional noise level

    Returns:
        Galaxy dict in standard format:
        {
            "id": str,
            "regime": str,
            "r": array,
            "v": array,
            "v_unc": array,
            "params": {...}
        }
    """
    if params is None:
        params = GalaxyParams()

    if galaxy_id is None:
        galaxy_id = f"SYN_{np.random.randint(10000):05d}"

    # Generate radius array
    r = np.linspace(r_min, r_max, n_points)

    # Compute rotation curve based on regime
    if regime == PhysicsRegime.NEWTONIAN:
        v_true = newtonian_velocity(r, params.mass)
    elif regime == PhysicsRegime.DARK_MATTER:
        v_disk = exponential_disk_velocity(r, params.mass, params.disk_scale)
        v_halo = nfw_halo_velocity(r, params.halo_mass, 10.0, 100.0)
        v_true = np.sqrt(v_disk**2 + v_halo**2)
    elif regime == PhysicsRegime.MOND:
        v_true = mond_velocity(r, params.mass)
    else:  # MIXED
        # Combine disk, bulge, and halo
        v_disk = exponential_disk_velocity(r, params.mass * 0.7, params.disk_scale)
        v_halo = nfw_halo_velocity(r, params.halo_mass, 10.0, 100.0)
        v_true = np.sqrt(v_disk**2 + v_halo**2)

    # Add noise
    v_noise = v_true * (1 + noise_level * np.random.randn(n_points))
    v_unc = v_true * noise_level * np.ones(n_points)

    galaxy = {
        "id": galaxy_id,
        "regime": regime.value,
        "r": r,
        "v": v_noise,
        "v_unc": v_unc,
        "params": {
            "source": "synthetic",
            "luminosity": params.luminosity,
            "disk_scale": params.disk_scale,
            "mass": params.mass,
            "halo_mass": params.halo_mass,
        }
    }

    # Emit receipt
    emit_receipt("synthetic_galaxy", {
        "tenant_id": TENANT_ID,
        "galaxy_id": galaxy_id,
        "regime": regime.value,
        "n_points": n_points,
        "r_range": [r_min, r_max],
        "noise_level": noise_level,
    })

    return galaxy


def generate_synthetic_dataset(
    n_galaxies: int = 30,
    regimes: List[PhysicsRegime] = None,
    seed: int = None
) -> List[Dict]:
    """Generate synthetic galaxy dataset with known physics.

    Args:
        n_galaxies: Number of galaxies to generate
        regimes: List of regimes to sample from
        seed: Random seed for reproducibility

    Returns:
        List of galaxy dicts
    """
    if seed is not None:
        np.random.seed(seed)

    if regimes is None:
        regimes = [PhysicsRegime.NEWTONIAN, PhysicsRegime.DARK_MATTER, PhysicsRegime.MOND, PhysicsRegime.MIXED]

    galaxies = []
    for i in range(n_galaxies):
        regime = regimes[i % len(regimes)]

        # Vary parameters
        params = GalaxyParams(
            mass=10 ** (10 + np.random.rand()),
            disk_scale=2 + 4 * np.random.rand(),
            halo_mass=10 ** (11 + np.random.rand()),
            luminosity=10 ** (9 + 2 * np.random.rand()),
        )

        galaxy = generate_galaxy(
            galaxy_id=f"SYN_{i:05d}",
            regime=regime,
            params=params,
            noise_level=0.03 + 0.04 * np.random.rand()
        )
        galaxies.append(galaxy)

    # Emit dataset receipt
    emit_receipt("synthetic_dataset", {
        "tenant_id": TENANT_ID,
        "n_galaxies": n_galaxies,
        "regimes": [r.value for r in regimes],
        "seed": seed,
    })

    return galaxies


def load_real_data(n_galaxies: int = 30) -> List[Dict]:
    """Load real galaxy data from SPARC.

    This is the bridge to real_data module.

    Args:
        n_galaxies: Number of galaxies to load

    Returns:
        List of galaxy dicts
    """
    from real_data.sparc import load_sparc
    return load_sparc(n_galaxies=n_galaxies)


def classify_regime(galaxy: Dict, v_pred: np.ndarray = None) -> Tuple[PhysicsRegime, float]:
    """Classify the physics regime of a galaxy.

    Uses rotation curve shape to infer underlying physics.

    Args:
        galaxy: Galaxy dict
        v_pred: Optional predicted velocities

    Returns:
        Tuple of (PhysicsRegime, confidence)
    """
    r = np.array(galaxy["r"])
    v = np.array(galaxy["v"])

    # Compute v^2 * r (should be constant for Newtonian)
    v2r = v ** 2 * r
    v2r_std = np.std(v2r) / np.mean(v2r)

    # Compute v gradient at large r (flat = dark matter dominated)
    if len(v) >= 5:
        v_gradient = (v[-1] - v[-5]) / (r[-1] - r[-5])
        flat_curve = abs(v_gradient) < 5  # km/s per kpc
    else:
        flat_curve = False

    # Classify
    if v2r_std < 0.1:
        return PhysicsRegime.NEWTONIAN, 1 - v2r_std
    elif flat_curve and v2r_std > 0.2:
        return PhysicsRegime.DARK_MATTER, 0.8
    else:
        return PhysicsRegime.MIXED, 0.6
