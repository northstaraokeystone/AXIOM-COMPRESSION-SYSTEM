"""Synthetic galaxy rotation curve generator for AXIOM witness protocol.

4 physics regimes: Newtonian, MOND, NFW dark matter halo, PBH fog.
"""
import numpy as np

# Physics constants in galaxy units
G = 4.302e-6  # kpc·(km/s)²/M_☉
M_TYPICAL = 1e11  # M_☉
A0_MOND = 2.44e3  # (km/s)⁴/(kpc·(km/s)²) MOND scale giving V_flat~180km/s for M_TYPICAL
C_NFW = 15.0  # NFW concentration parameter
R_S = 5.0  # kpc (NFW scale radius)
V_200 = 150.0  # km/s (NFW virial velocity)


def newton_curve(r: np.ndarray, M: float = M_TYPICAL) -> np.ndarray:
    """Keplerian rotation curve: V(r) = sqrt(GM/r)."""
    return np.sqrt(G * M / r)


def mond_curve(r: np.ndarray, M: float = M_TYPICAL, a0: float = A0_MOND) -> np.ndarray:
    """Deep MOND interpolation: V⁴ = V_N⁴ + V_∞⁴ where V_∞ = (GMa₀)^(1/4)."""
    V_N = np.sqrt(G * M / r)  # Newtonian velocity
    V_inf = (G * M * a0) ** 0.25  # Asymptotic flat velocity in deep MOND
    return (V_N**4 + V_inf**4) ** 0.25


def nfw_curve(r: np.ndarray, v_200: float = V_200, c: float = C_NFW, r_200: float = None) -> np.ndarray:
    """NFW dark matter halo: V²(x) = V²_200 · (1/x) · g(cx)/g(c)."""
    if r_200 is None:
        r_200 = c * R_S
    x = r / r_200
    def g(y): return np.log(1.0 + y) - y / (1.0 + y)
    g_c = g(c)
    g_c = np.maximum(g_c, 1e-10)  # Avoid division by zero
    return v_200 * np.sqrt((1.0 / x) * g(c * x) / g_c)


def pbh_fog_curve(r: np.ndarray, M_bar: float = 5e10, f_pbh: float = 0.3, r_core: float = 2.0) -> np.ndarray:
    """PBH fog cored profile: V²_total = V²_baryonic + V²_fog."""
    V_bar_sq = G * M_bar / r
    M_fog_total = f_pbh * M_bar / (1.0 - f_pbh)
    M_fog_enclosed = M_fog_total * (1.0 - np.exp(-r / r_core)) * (r / (r + r_core))
    V_fog_sq = G * M_fog_enclosed / r
    return np.sqrt(V_bar_sq + V_fog_sq)


def generate_galaxy(regime: str, n_points: int = 100, noise_level: float = 0.05, seed: int = None) -> tuple:
    """Generate single galaxy rotation curve with noise."""
    regimes = {"newtonian": newton_curve, "mond": mond_curve, "nfw": nfw_curve, "pbh": pbh_fog_curve}
    regime_lower = regime.lower()
    if regime_lower not in regimes:
        raise ValueError(f"Unknown regime: {regime}. Must be one of {list(regimes.keys())}")
    r = np.logspace(np.log10(0.5), np.log10(30.0), n_points)
    V = regimes[regime_lower](r)
    if seed is not None:
        np.random.seed(seed)
    V_noisy = V * (1.0 + noise_level * np.random.randn(n_points))
    V_noisy = np.maximum(V_noisy, 1.0)
    return r, V_noisy


def batch_generate(n_per_regime: int = 25, noise_level: float = 0.05, seed: int = 42) -> list:
    """Generate batch of galaxies across all 4 regimes."""
    galaxies = []
    for regime in ["newtonian", "mond", "nfw", "pbh"]:
        for i in range(n_per_regime):
            galaxy_seed = seed + len(galaxies)
            r, v = generate_galaxy(regime, noise_level=noise_level, seed=galaxy_seed)
            galaxies.append({"id": f"synth_{regime}_{i:03d}", "regime": regime, "r": r, "v": v})
    return galaxies
