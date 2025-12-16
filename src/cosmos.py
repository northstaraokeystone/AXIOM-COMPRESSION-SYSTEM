"""cosmos.py - Synthetic Universe Generator

THE GROUND TRUTH ORACLE:
  Physics curves ARE the oracle.
  Each regime is a theorem made data.
  Noise isn't error—it's uncertainty for the witness.

4 physics regimes: Newtonian, MOND, NFW dark matter halo, PBH fog.
Pure functions for curves, receipt only on batch generation.

Source: CLAUDEME.md (§0, §8)
"""

import uuid
import numpy as np
from typing import Dict, List, Optional

from .core import dual_hash, emit_receipt, StopRule

# === PHYSICS CONSTANTS (Module Top) ===

G = 4.302e-6
"""Gravitational constant in galaxy units: kpc (km/s)² / M_sun"""

MOND_A0 = 1.2e-10
"""MOND acceleration scale in m/s²"""

TENANT_ID = "axiom-witness"
"""CLAUDEME tenant isolation"""

DEFAULT_N_POINTS = 100
"""Points per rotation curve"""

DEFAULT_NOISE = 0.03
"""3% noise fraction"""

REGIMES = ["newtonian", "mond", "nfw", "pbh_fog"]
"""Valid regime names"""


# === DEFAULT GALAXY PARAMETERS ===

DEFAULT_PARAMS = {
    "newtonian": {
        "M": 1e11,  # M_sun (total mass)
        "r_range": (0.5, 20.0),  # kpc
    },
    "mond": {
        "M": 1e10,  # M_sun (baryonic mass, smaller for deep MOND)
        "a0": MOND_A0,  # m/s²
        "r_range": (1.0, 30.0),  # kpc
    },
    "nfw": {
        "M_disk": 5e10,  # M_sun
        "V_200": 150.0,  # km/s (virial velocity)
        "c": 10,  # concentration parameter
        "r_s": 15.0,  # kpc (scale radius)
        "r_range": (0.5, 50.0),  # kpc
    },
    "pbh_fog": {
        "M_bar": 5e10,  # M_sun (baryonic)
        "f_pbh": 0.15,  # 15% PBH fraction of DM
        "M_pbh": 30.0,  # M_sun (individual PBH mass, LIGO range)
        "r_core": 3.0,  # kpc (fog core radius, distinct from NFW)
        "r_range": (0.5, 50.0),  # kpc
    },
}


# === STOPRULES ===

def stoprule_invalid_regime(regime: str) -> None:
    """Trigger stoprule for invalid regime.

    Emits anomaly receipt and raises StopRule.
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "regime",
        "baseline": 0.0,
        "delta": -1.0,
        "classification": "violation",
        "action": "halt",
        "invalid_regime": regime,
        "valid_regimes": REGIMES
    })
    raise StopRule(f"Invalid regime: {regime}")


def stoprule_negative_radius(r: np.ndarray) -> None:
    """Trigger stoprule for negative or zero radius.

    Emits anomaly receipt and raises StopRule.
    """
    emit_receipt("anomaly", {
        "tenant_id": TENANT_ID,
        "metric": "radius",
        "baseline": 0.0,
        "delta": float(np.min(r)),
        "classification": "violation",
        "action": "halt"
    })
    raise StopRule("Radius must be positive")


# === CURVE FUNCTIONS (Pure Math, No Receipts) ===

def newton_curve(r: np.ndarray, M: float) -> np.ndarray:
    """Keplerian rotation curve. The control case.

    Physics: V(r) = √(G·M/r)

    Args:
        r: Radius array in kpc, shape (n_points,)
        M: Total enclosed mass in solar masses

    Returns:
        V: Velocity array in km/s, shape (n_points,)

    Note: Pure function, no side effects. Does NOT emit receipt.
    """
    if np.any(r <= 0):
        stoprule_negative_radius(r)
    return np.sqrt(G * M / r)


def mond_curve(r: np.ndarray, M: float, a0: float = MOND_A0) -> np.ndarray:
    """Deep MOND regime rotation curve.

    Physics: In deep MOND (a << a₀):
        V⁴ = G·M·a₀ (asymptotic flat rotation)
        More precisely: V⁴ = V_N⁴ · μ(a/a₀) where μ is interpolation function

    For deep MOND, the asymptotic velocity is:
        V_∞ = (G·M·a₀)^(1/4)

    We use the simple interpolation:
        V⁴ = V_N⁴ + V_∞⁴

    Args:
        r: Radius array in kpc
        M: Baryonic mass in solar masses
        a0: MOND acceleration scale (need to convert units)

    Returns:
        V: Velocity array in km/s

    Key behavior:
        Inner region: rises more steeply than Newtonian
        Outer region: flattens (the MOND signature)
        KAN should discover r^0.25 scaling in outer regions

    Note: Does NOT emit receipt (pure math).
    """
    if np.any(r <= 0):
        stoprule_negative_radius(r)

    # Convert a0 from m/s² to galaxy units: kpc/s² → (km/s)²/kpc
    # 1 kpc = 3.086e19 m, so a0 in (km/s)²/kpc = a0 * 3.086e16
    a0_galaxy = a0 * 3.086e16  # (km/s)²/kpc

    # Newtonian velocity
    V_N = np.sqrt(G * M / r)

    # Asymptotic flat velocity in deep MOND
    V_inf = (G * M * a0_galaxy) ** 0.25

    # Simple interpolation: V⁴ = V_N⁴ + V_∞⁴
    return (V_N**4 + V_inf**4) ** 0.25


def nfw_curve(r: np.ndarray, M_disk: float, V_200: float, c: float, r_s: float) -> np.ndarray:
    """NFW dark matter halo + exponential disk.

    Physics:
        V²_total(r) = V²_disk(r) + V²_halo(r)
        V_halo = V_200 · √[f(c·x) / (x · f(c))]
        where f(y) = ln(1+y) - y/(1+y)
        and x = r/r_s

    Args:
        r: Radius array in kpc
        M_disk: Disk mass in solar masses
        V_200: Virial velocity in km/s
        c: Concentration parameter (dimensionless, typically 5-20)
        r_s: Scale radius in kpc

    Returns:
        V: Total velocity array in km/s

    Key behavior:
        Complex multi-term structure
        KAN should show higher complexity score
        Distinct shape from PBH fog

    Note: Does NOT emit receipt (pure math).
    """
    if np.any(r <= 0):
        stoprule_negative_radius(r)

    def f(y):
        """NFW profile function: f(y) = ln(1+y) - y/(1+y)"""
        return np.log(1 + y) - y / (1 + y)

    # NFW halo component
    x = r / r_s
    f_c = f(c)
    f_cx = f(c * x)

    # Avoid division by zero at very small x
    x_safe = np.maximum(x, 1e-10)
    V_halo_sq = V_200**2 * f_cx / (x_safe * f_c)
    V_halo_sq = np.maximum(V_halo_sq, 0)  # Ensure non-negative

    # Exponential disk component (Freeman disk with scale length ~ r_s/3)
    r_d = r_s / 3.0  # Disk scale length
    y = r / (2 * r_d)
    # Simplified exponential disk: V²_disk ≈ G·M_disk·r / (r + r_d)³ × f(r/r_d)
    V_disk_sq = G * M_disk * r**2 / (r + r_d)**3

    # Total velocity
    V_total = np.sqrt(V_disk_sq + V_halo_sq)

    return V_total


def pbh_fog_curve(r: np.ndarray, M_bar: float, f_pbh: float, M_pbh: float, r_core: float) -> np.ndarray:
    """Primordial Black Hole fog distribution. THE NOVEL CASE.

    Physics:
        V²_total(r) = V²_baryonic(r) + V²_fog(r)
        PBH fog density: ρ_fog(r) = ρ_0 · (1 + (r/r_core)²)^(-3/2)
        This is DIFFERENT from NFW: softer core, steeper outer falloff
        M_fog(r) = ∫₀ʳ ρ_fog(r') · 4πr'² dr'
        V_fog = √(G · M_fog(r) / r)

    Args:
        r: Radius array in kpc
        M_bar: Baryonic mass in solar masses
        f_pbh: PBH fraction of total DM (0-1)
        M_pbh: Individual PBH mass in solar masses (for normalization)
        r_core: Core radius in kpc (fog concentration)

    Returns:
        V: Total velocity array in km/s

    Key insight:
        If KAN compresses PBH fog BETTER than NFW on same total mass profile,
        we've witnessed something new. The distinct density profile should
        yield different spline patterns.

    Note: Does NOT emit receipt (pure math).
    """
    if np.any(r <= 0):
        stoprule_negative_radius(r)

    # Baryonic component (exponential disk approximation)
    r_d = 3.0  # kpc, typical disk scale length
    V_bar_sq = G * M_bar * r**2 / (r + r_d)**3

    # PBH fog component
    # Total DM mass inferred from f_pbh: M_DM = M_bar * f_pbh / (1 - f_pbh)
    M_fog_total = M_bar * f_pbh / (1 - f_pbh + 1e-10)

    # Enclosed fog mass for cored profile: ρ(r) ∝ (1 + (r/r_core)²)^(-3/2)
    # Integrated: M_fog(r) = M_fog_total * r³ / (r² + r_core²)^(3/2) * normalization
    # For this profile, analytic integral gives:
    # M(<r) = M_total * [r/√(r² + r_core²) - r_core·arcsinh(r/r_core)/r_core]
    # Simplified approximation that captures the core behavior:
    u = r / r_core
    M_fog_enclosed = M_fog_total * u**3 / (1 + u**2)**1.5

    V_fog_sq = G * M_fog_enclosed / r

    # Total velocity
    V_total = np.sqrt(V_bar_sq + V_fog_sq)

    return V_total


# === GALAXY GENERATION ===

def generate_galaxy(
    regime: str,
    n_points: int = DEFAULT_N_POINTS,
    noise: float = DEFAULT_NOISE,
    seed: int = 42,
    params: Optional[Dict] = None
) -> Dict:
    """Generate one complete galaxy rotation curve with metadata.

    Args:
        regime: One of REGIMES ("newtonian"|"mond"|"nfw"|"pbh_fog")
        n_points: Number of radial points (default DEFAULT_N_POINTS)
        noise: Noise fraction σ/V (default DEFAULT_NOISE)
        seed: Random seed for reproducibility
        params: Override default parameters (optional dict)

    Returns:
        dict with keys:
            id: str (galaxy identifier)
            regime: str (ground truth label)
            r: torch.Tensor of shape (n_points, 1)
            v: torch.Tensor of shape (n_points, 1) — observed (noisy)
            v_true: torch.Tensor of shape (n_points, 1) — ground truth
            v_unc: torch.Tensor of shape (n_points,) — uncertainty
            params: dict of parameters used
            seed: int

    Note: Does NOT emit receipt (single galaxy is component, batch is action).
    """
    # Validate regime
    if regime not in REGIMES:
        stoprule_invalid_regime(regime)

    # Set numpy random seed
    np.random.seed(seed)

    # Get parameters (merge defaults with overrides)
    base_params = DEFAULT_PARAMS[regime].copy()
    if params:
        base_params.update(params)

    # Generate r array (logarithmically spaced for better resolution)
    r_range = base_params.get("r_range", (0.5, 20.0))
    r = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), n_points)

    # Call appropriate curve function
    if regime == "newtonian":
        v_true = newton_curve(r, M=base_params["M"])
    elif regime == "mond":
        v_true = mond_curve(r, M=base_params["M"], a0=base_params.get("a0", MOND_A0))
    elif regime == "nfw":
        v_true = nfw_curve(
            r,
            M_disk=base_params["M_disk"],
            V_200=base_params["V_200"],
            c=base_params["c"],
            r_s=base_params["r_s"]
        )
    elif regime == "pbh_fog":
        v_true = pbh_fog_curve(
            r,
            M_bar=base_params["M_bar"],
            f_pbh=base_params["f_pbh"],
            M_pbh=base_params["M_pbh"],
            r_core=base_params["r_core"]
        )

    # Add Gaussian noise: V_obs = V_true + N(0, noise·V_true)
    v_unc = noise * v_true
    v_obs = v_true + np.random.randn(n_points) * v_unc

    # Generate unique galaxy_id
    galaxy_id = f"synth_{regime}_{seed:04d}"

    # Return numpy arrays with shape (n_points, 1) for r, v, v_true
    # NOTE: Using numpy arrays instead of torch tensors.
    # witness.py converts to numpy anyway (np.asarray), so this is compatible.
    return {
        "id": galaxy_id,
        "regime": regime,
        "r": r.reshape(-1, 1).astype(np.float32),
        "v": v_obs.reshape(-1, 1).astype(np.float32),
        "v_true": v_true.reshape(-1, 1).astype(np.float32),
        "v_unc": v_unc.astype(np.float32),
        "params": base_params,
        "seed": seed
    }


def batch_generate(
    n_per_regime: int = 25,
    noise: float = DEFAULT_NOISE,
    seed: int = 42
) -> List[Dict]:
    """Generate batch of galaxies across all 4 regimes.

    Args:
        n_per_regime: Galaxies per regime (total = 4 × n_per_regime)
        noise: Noise fraction for all galaxies
        seed: Base seed (galaxy i gets seed + i)

    Returns:
        List of galaxy dicts

    MUST emit receipt (cosmos_receipt).
    """
    results = []
    offset = 0

    for regime in REGIMES:
        for i in range(n_per_regime):
            galaxy = generate_galaxy(regime, noise=noise, seed=seed + offset)
            results.append(galaxy)
            offset += 1

    # Generate batch_id
    batch_id = str(uuid.uuid4())

    # Compute payload hash from all galaxy IDs and params
    payload_data = "|".join([g["id"] for g in results])

    # Emit cosmos_receipt
    emit_receipt("cosmos", {
        "tenant_id": TENANT_ID,
        "batch_id": batch_id,
        "n_galaxies": len(results),
        "regimes": {regime: n_per_regime for regime in REGIMES},
        "noise_fraction": noise,
        "seed": seed,
        "payload_hash": dual_hash(payload_data)
    })

    return results


# === PATHOLOGICAL CASES (GÖDEL Scenario) ===

def generate_pathological(pathology: str, n_points: int = DEFAULT_N_POINTS, seed: int = 42) -> Dict:
    """Generate edge-case galaxies for robustness testing.

    Pathology types:
        "constant": V = constant for all r → Zero information
        "noise": V = pure Gaussian noise → No law, complexity maximal
        "discontinuous": Step function in V → Non-physical
        "ambiguous": MOND-like Newtonian (borderline) → Regime uncertain

    Args:
        pathology: One of "constant", "noise", "discontinuous", "ambiguous"
        n_points: Number of radial points
        seed: Random seed

    Returns:
        Same dict structure as generate_galaxy with regime set to pathology type

    Note: Does NOT emit receipt (test utility).
    """
    np.random.seed(seed)

    # Generate r array
    r = np.logspace(np.log10(0.5), np.log10(30.0), n_points)

    if pathology == "constant":
        # Constant velocity everywhere
        v_true = np.ones(n_points) * 150.0  # 150 km/s constant
        v_obs = v_true.copy()
        v_unc = np.zeros(n_points)

    elif pathology == "noise":
        # Pure Gaussian noise, no underlying law
        v_true = np.abs(np.random.randn(n_points) * 100 + 150)  # Centered around 150 km/s
        v_obs = v_true.copy()
        v_unc = np.abs(v_true) * 0.5  # Large uncertainty

    elif pathology == "discontinuous":
        # Step function (non-physical)
        v_true = np.where(r < 10.0, 100.0, 200.0)  # Jump at r=10 kpc
        v_obs = v_true + np.random.randn(n_points) * 5
        v_unc = np.ones(n_points) * 5

    elif pathology == "ambiguous":
        # Borderline MOND/Newtonian - tuned to be indistinguishable
        M = 5e10  # Intermediate mass
        V_N = np.sqrt(G * M / r)
        # Add just a hint of flattening
        V_flat = 0.3 * (G * M * MOND_A0 * 3.086e16) ** 0.25
        v_true = np.sqrt(V_N**2 + V_flat**2 * 0.1)
        v_obs = v_true + np.random.randn(n_points) * DEFAULT_NOISE * v_true
        v_unc = DEFAULT_NOISE * v_true

    else:
        # Unknown pathology - generate noise as fallback
        v_true = np.abs(np.random.randn(n_points) * 100 + 150)
        v_obs = v_true.copy()
        v_unc = np.abs(v_true) * 0.5

    # Generate galaxy_id
    galaxy_id = f"pathological_{pathology}_{seed:04d}"

    # Return numpy arrays (same format as generate_galaxy)
    return {
        "id": galaxy_id,
        "regime": pathology,
        "r": r.reshape(-1, 1).astype(np.float32),
        "v": v_obs.reshape(-1, 1).astype(np.float32),
        "v_true": v_true.reshape(-1, 1).astype(np.float32),
        "v_unc": v_unc.astype(np.float32),
        "params": {"pathology": pathology},
        "seed": seed
    }
