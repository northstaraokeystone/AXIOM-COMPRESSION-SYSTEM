"""witness.py - KAN-based Physics Discovery

THE WITNESS INSIGHT:
    The KAN witnesses the data and discovers the law.
    Kolmogorov-Arnold: f(x) = sum phi_i(sum psi_ij(x_j))
    Universe as computation: the law is the compression.

Source: AXIOM Validation Lock v1
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from pathlib import Path

# Import from src
try:
    from src.core import dual_hash, emit_receipt
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.core import dual_hash, emit_receipt


# === CONSTANTS ===

TENANT_ID = "axiom-witness"
DEFAULT_N_BASIS = 10
DEFAULT_DEGREE = 3
DEFAULT_EPOCHS = 100


# === KAN IMPLEMENTATION ===


@dataclass
class KANConfig:
    """Configuration for Kolmogorov-Arnold Network."""

    n_basis: int = DEFAULT_N_BASIS
    degree: int = DEFAULT_DEGREE
    learning_rate: float = 0.01
    regularization: float = 0.001


@dataclass
class KANState:
    """State of trained KAN."""

    weights: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_N_BASIS))
    knots: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_N_BASIS))
    loss_history: List[float] = field(default_factory=list)
    fitted: bool = False
    data_hash: str = ""


class KAN:
    """Kolmogorov-Arnold Network for univariate function approximation.

    The Kolmogorov-Arnold representation theorem states that any
    multivariate continuous function can be represented as:

        f(x1, ..., xn) = sum_{q=0}^{2n} Phi_q(sum_{p=1}^n psi_{q,p}(x_p))

    For rotation curves (univariate), this simplifies to:
        v(r) = sum_i phi_i(psi_i(r))

    We use B-spline basis functions for the inner (psi) and outer (phi).
    """

    def __init__(self, config: KANConfig = None):
        """Initialize KAN.

        Args:
            config: KAN configuration
        """
        self.config = config or KANConfig()
        self.state = KANState()

    def _create_basis(self, x: np.ndarray) -> np.ndarray:
        """Create B-spline basis functions.

        Args:
            x: Input values

        Returns:
            Basis matrix of shape (len(x), n_basis)
        """
        x_min, x_max = x.min(), x.max()
        n_basis = self.config.n_basis

        # Create knot vector
        knots = np.linspace(x_min, x_max, n_basis - self.config.degree + 1)
        self.state.knots = knots

        # Gaussian-like basis functions (stable approximation)
        basis = np.zeros((len(x), n_basis))
        width = (x_max - x_min) / (n_basis - 1) if n_basis > 1 else 1.0

        for i in range(n_basis):
            center = (
                x_min + (x_max - x_min) * i / (n_basis - 1) if n_basis > 1 else x_min
            )
            basis[:, i] = np.exp(-0.5 * ((x - center) / width) ** 2)

        return basis

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = DEFAULT_EPOCHS) -> Dict:
        """Fit KAN to data.

        Args:
            x: Input values
            y: Target values
            epochs: Number of training epochs

        Returns:
            Dict with training results
        """
        # Store data hash for provenance
        self.state.data_hash = dual_hash(str(x.tolist()) + str(y.tolist()))

        # Create basis
        basis = self._create_basis(x)

        # Initialize weights via least squares
        self.state.weights = np.linalg.lstsq(basis, y, rcond=None)[0]

        # Iterative refinement with regularization
        lr = self.config.learning_rate
        reg = self.config.regularization
        self.state.loss_history = []

        for epoch in range(epochs):
            pred = basis @ self.state.weights
            residual = y - pred
            mse = np.mean(residual**2)
            reg_loss = reg * np.sum(self.state.weights**2)
            self.state.loss_history.append(mse + reg_loss)

            # Gradient update with L2 regularization
            grad = -2 * basis.T @ residual / len(x) + 2 * reg * self.state.weights
            self.state.weights -= lr * grad

        self.state.fitted = True

        # Compute final metrics
        final_pred = basis @ self.state.weights
        r_squared = 1 - np.sum((y - final_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        final_mse = self.state.loss_history[-1] if self.state.loss_history else 0

        return {
            "epochs": epochs,
            "final_mse": float(final_mse),
            "r_squared": float(r_squared),
            "n_parameters": len(self.state.weights),
            "compression": self._compute_compression(len(x)),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output for given input.

        Args:
            x: Input values

        Returns:
            Predicted values
        """
        if not self.state.fitted:
            raise ValueError("KAN not fitted yet")

        basis = self._create_basis(x)
        return basis @ self.state.weights

    def _compute_compression(self, n_data_points: int) -> float:
        """Compute compression ratio.

        Args:
            n_data_points: Number of original data points

        Returns:
            Compression ratio (1.0 = no compression, higher = better)
        """
        # Original data: 2 values (x, y) per point, 64 bits each
        data_bits = n_data_points * 2 * 64

        # Model: weights + knots + metadata
        model_bits = len(self.state.weights) * 64 + len(self.state.knots) * 64 + 128

        return 1.0 - (model_bits / data_bits) if data_bits > 0 else 0.0

    def get_symbolic(self) -> str:
        """Extract symbolic approximation.

        Returns:
            String representation of learned function
        """
        if not self.state.fitted:
            return "not_fitted"

        # Find dominant basis functions
        dominant_idx = np.argsort(np.abs(self.state.weights))[-3:]
        terms = []

        for i in dominant_idx:
            coef = self.state.weights[i]
            if abs(coef) > 0.1:
                terms.append(f"{coef:.2f}*B_{i}(r)")

        return " + ".join(terms) if terms else "constant"


def train(kan: KAN, r: np.ndarray, v: np.ndarray, epochs: int = DEFAULT_EPOCHS) -> Dict:
    """Train KAN on rotation curve data.

    Args:
        kan: KAN instance
        r: Radius values
        v: Velocity values
        epochs: Training epochs

    Returns:
        Dict with training results and receipt info
    """
    result = kan.fit(r, v, epochs=epochs)

    # Emit witness receipt
    emit_receipt(
        "witness",
        {
            "tenant_id": TENANT_ID,
            "data_hash": kan.state.data_hash,
            "n_points": len(r),
            "epochs": epochs,
            "final_mse": result["final_mse"],
            "r_squared": result["r_squared"],
            "compression": result["compression"],
            "n_parameters": result["n_parameters"],
            "symbolic": kan.get_symbolic(),
        },
    )

    return result


def crossover_detection(
    kan: KAN, r_values: np.ndarray, threshold: float = 0.1
) -> List[Dict]:
    """Detect where KAN switches physics regimes.

    The crossover point reveals where Newtonian physics transitions
    to dark matter dominated dynamics.

    Args:
        kan: Trained KAN
        r_values: Radius values to evaluate
        threshold: Second derivative threshold for detection

    Returns:
        List of {transition_point_r, regime_from, regime_to, confidence}
    """
    if not kan.state.fitted:
        return []

    # Sample KAN at fine resolution
    r_fine = np.linspace(r_values.min(), r_values.max(), 200)
    v_fine = kan.predict(r_fine)

    # Compute first and second derivatives
    dv = np.gradient(v_fine, r_fine)
    d2v = np.gradient(dv, r_fine)

    # Identify significant changes in curvature
    crossovers = []
    d2v_normalized = np.abs(d2v) / np.max(np.abs(v_fine))

    # Find peaks in second derivative
    for i in range(1, len(d2v_normalized) - 1):
        is_peak = (
            d2v_normalized[i] > threshold
            and d2v_normalized[i] > d2v_normalized[i - 1]
            and d2v_normalized[i] > d2v_normalized[i + 1]
        )
        if is_peak:
            # Determine regime transition
            if dv[i] > 0:
                regime_from = "rising"
                regime_to = "flat" if i < len(dv) // 2 else "declining"
            else:
                regime_from = "flat"
                regime_to = "declining"

            crossovers.append(
                {
                    "transition_point_r": float(r_fine[i]),
                    "regime_from": regime_from,
                    "regime_to": regime_to,
                    "curvature": float(d2v[i]),
                    "confidence": float(min(1.0, d2v_normalized[i] / threshold)),
                }
            )

    # Emit detection receipt
    if crossovers:
        emit_receipt(
            "crossover_detection",
            {
                "tenant_id": TENANT_ID,
                "n_crossovers": len(crossovers),
                "threshold": threshold,
                "crossovers": crossovers,
            },
        )

    return crossovers


def extract_symbolic(kan: KAN, simplify: bool = True) -> Dict:
    """Extract symbolic expression from trained KAN.

    Args:
        kan: Trained KAN
        simplify: Whether to simplify the expression

    Returns:
        Dict with symbolic info
    """
    if not kan.state.fitted:
        return {"expression": "not_fitted", "confidence": 0}

    symbolic = kan.get_symbolic()

    # Known physical forms (for reference/future matching)
    # "newtonian": "sqrt(const/r)", "flat": "const", "rising_flat": "v_max*(1-exp(-r/r_s))"

    matched_form = None
    match_confidence = 0.0

    # Simple heuristic matching based on coefficient patterns
    weights = kan.state.weights
    if len(weights) > 0:
        # Check if dominated by single term (flat)
        max_weight = np.max(np.abs(weights))
        weight_ratio = (
            np.sum(np.abs(weights)) / (max_weight * len(weights))
            if max_weight > 0
            else 0
        )

        if weight_ratio < 0.3:
            matched_form = "concentrated"
            match_confidence = 1 - weight_ratio
        else:
            matched_form = "distributed"
            match_confidence = weight_ratio

    return {
        "expression": symbolic,
        "matched_form": matched_form,
        "confidence": match_confidence,
        "n_active_terms": int(np.sum(np.abs(weights) > 0.1)),
    }


# === BATCH PROCESSING ===


def batch_train(
    galaxies: List[Dict], config: KANConfig = None, epochs: int = DEFAULT_EPOCHS
) -> List[Dict]:
    """Train KAN on multiple galaxies.

    Args:
        galaxies: List of galaxy dicts
        config: KAN configuration
        epochs: Training epochs

    Returns:
        List of result dicts
    """
    results = []
    for galaxy in galaxies:
        kan = KAN(config)
        r = np.array(galaxy["r"])
        v = np.array(galaxy["v"])

        result = train(kan, r, v, epochs=epochs)
        result["galaxy_id"] = galaxy.get("id", "unknown")
        result["crossovers"] = crossover_detection(kan, r)
        result["symbolic"] = extract_symbolic(kan)

        results.append(result)

    # Emit batch receipt
    compressions = [r["compression"] for r in results]
    r_squareds = [r["r_squared"] for r in results]

    emit_receipt(
        "batch_witness",
        {
            "tenant_id": TENANT_ID,
            "n_galaxies": len(galaxies),
            "mean_compression": float(np.mean(compressions)),
            "mean_r_squared": float(np.mean(r_squareds)),
            "min_r_squared": float(np.min(r_squareds)),
            "max_r_squared": float(np.max(r_squareds)),
        },
    )

    return results
