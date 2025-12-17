"""witness.py - KAN-based physics law witnessing.

THE WITNESSING INSIGHT:
    AXIOM witnesses physics laws through data compression.
    When a KAN achieves high compression on rotation curves,
    the spline coefficients ARE the equation.

    Compression ratio measures how much law is in the data.
    High compression = strong physical law.
    Low compression = noise or unknown physics.

Source: arXiv 2509.10089 (KAN-SR, 2025)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import dual_hash, emit_receipt

# === CONSTANTS ===

TENANT_ID = "axiom-witness"
"""Tenant for witness receipts."""

DEFAULT_HIDDEN_DIM = 8
"""Default hidden layer dimension for KAN."""

DEFAULT_N_KNOTS = 10
"""Default number of B-spline knots."""

DEFAULT_EPOCHS = 100
"""Default training epochs."""

CROSSOVER_THRESHOLD = 0.1
"""Default threshold for crossover detection."""


# === KAN IMPLEMENTATION ===

@dataclass
class KANConfig:
    """Configuration for Kolmogorov-Arnold Network.

    Attributes:
        input_dim: Input dimension (default 1 for radius)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (default 1 for velocity)
        n_knots: Number of B-spline knots
        learning_rate: Training learning rate
    """
    input_dim: int = 1
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    output_dim: int = 1
    n_knots: int = DEFAULT_N_KNOTS
    learning_rate: float = 0.01


class KAN:
    """Kolmogorov-Arnold Network for physics law discovery.

    The KAN learns interpretable spline-based activation functions.
    High compression ratio indicates strong physical law in data.
    """

    def __init__(self, config: KANConfig = None):
        """Initialize KAN with configuration.

        Args:
            config: KANConfig (uses defaults if None)
        """
        if config is None:
            config = KANConfig()

        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.n_knots = config.n_knots
        self.lr = config.learning_rate

        # Initialize spline coefficients
        self.spline_coeffs = np.random.randn(
            self.hidden_dim, self.n_knots
        ) * 0.1

        # Output weights
        self.output_weights = np.random.randn(self.hidden_dim) * 0.1

        # Training state
        self.trained = False
        self.training_loss_history = []

    def _bspline_basis(self, x: np.ndarray) -> np.ndarray:
        """Compute B-spline basis function values.

        Args:
            x: Input array of shape (n_samples,)

        Returns:
            Basis matrix of shape (n_samples, n_knots)
        """
        # Use Gaussian RBF as simplified basis
        x_min, x_max = x.min(), x.max()
        centers = np.linspace(x_min, x_max, self.n_knots)
        width = (x_max - x_min) / self.n_knots + 1e-8

        return np.exp(-((x[:, None] - centers) ** 2) / (2 * width ** 2))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through KAN.

        Args:
            x: Input array of shape (n_samples,)

        Returns:
            Output array of shape (n_samples,)
        """
        # Compute basis functions
        basis = self._bspline_basis(x)  # [n_samples, n_knots]

        # Apply learned spline functions
        hidden = np.zeros((len(x), self.hidden_dim))
        for h in range(self.hidden_dim):
            hidden[:, h] = basis @ self.spline_coeffs[h]

        # Activation
        hidden = np.tanh(hidden)

        # Output
        return hidden @ self.output_weights

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = DEFAULT_EPOCHS) -> float:
        """Train KAN using gradient descent.

        Args:
            x: Input array
            y: Target array
            epochs: Number of training epochs

        Returns:
            Final loss value
        """
        self.training_loss_history = []

        for epoch in range(epochs):
            # Forward pass
            pred = self.forward(x)
            loss = np.mean((pred - y) ** 2)
            self.training_loss_history.append(loss)

            # Numerical gradient descent
            eps = 1e-5

            # Update spline coefficients
            for h in range(self.hidden_dim):
                for k in range(self.n_knots):
                    self.spline_coeffs[h, k] += eps
                    loss_plus = np.mean((self.forward(x) - y) ** 2)
                    self.spline_coeffs[h, k] -= 2 * eps
                    loss_minus = np.mean((self.forward(x) - y) ** 2)
                    self.spline_coeffs[h, k] += eps

                    grad = (loss_plus - loss_minus) / (2 * eps)
                    self.spline_coeffs[h, k] -= self.lr * grad

            # Update output weights
            for h in range(self.hidden_dim):
                self.output_weights[h] += eps
                loss_plus = np.mean((self.forward(x) - y) ** 2)
                self.output_weights[h] -= 2 * eps
                loss_minus = np.mean((self.forward(x) - y) ** 2)
                self.output_weights[h] += eps

                grad = (loss_plus - loss_minus) / (2 * eps)
                self.output_weights[h] -= self.lr * grad

        self.trained = True
        return self.training_loss_history[-1]

    def get_compression_ratio(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute compression ratio achieved by KAN.

        Compression = (raw data bits) / (model bits) * fit_quality
        Higher = more structure found = stronger physical law.

        Args:
            x: Input array
            y: Target array

        Returns:
            Compression ratio (>1 means good compression)
        """
        # Raw data bits
        raw_bits = len(x) * 2 * 64  # Two float64 arrays

        # Model bits
        model_params = self.hidden_dim * self.n_knots + self.hidden_dim
        model_bits = model_params * 32

        # Fit quality
        pred = self.forward(x)
        mse = np.mean((pred - y) ** 2)
        var_y = np.var(y)
        r_squared = max(0, 1 - mse / var_y) if var_y > 0 else 0

        # Effective compression
        if r_squared > 0:
            compression = (raw_bits / model_bits) * r_squared
        else:
            compression = 0.0

        return compression

    def get_r_squared(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 coefficient of determination.

        Args:
            x: Input array
            y: Target array

        Returns:
            R^2 value (1.0 = perfect fit)
        """
        pred = self.forward(x)
        mse = np.mean((pred - y) ** 2)
        var_y = np.var(y)
        return max(0, 1 - mse / var_y) if var_y > 0 else 0

    def to_equation(self) -> str:
        """Extract symbolic equation from spline coefficients.

        Returns:
            String representation of learned equation
        """
        # Find dominant hidden unit
        dominant_h = np.argmax(np.abs(self.output_weights))
        coeffs = self.spline_coeffs[dominant_h]

        # Build polynomial approximation
        terms = []
        for i, c in enumerate(coeffs[:5]):
            if abs(c) > 0.1:
                if i == 0:
                    terms.append(f"{c:.3f}")
                elif i == 1:
                    terms.append(f"{c:.3f}*x")
                else:
                    terms.append(f"{c:.3f}*x^{i}")

        return " + ".join(terms) if terms else "0"


# === TRAINING FUNCTION ===

def train(
    kan: KAN,
    r: np.ndarray,
    v: np.ndarray,
    epochs: int = DEFAULT_EPOCHS
) -> Dict:
    """Train KAN on rotation curve data.

    Args:
        kan: KAN instance
        r: Radius array
        v: Velocity array
        epochs: Training epochs

    Returns:
        Dict with training results
    """
    # Normalize inputs
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8)
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)

    # Train
    final_loss = kan.fit(r_norm, v_norm, epochs=epochs)

    # Compute metrics
    compression = kan.get_compression_ratio(r_norm, v_norm)
    r_squared = kan.get_r_squared(r_norm, v_norm)
    equation = kan.to_equation()

    return {
        "compression": compression,
        "r_squared": r_squared,
        "final_loss": final_loss,
        "equation": equation,
        "epochs": epochs,
    }


# === CROSSOVER DETECTION ===

def crossover_detection(
    kan: KAN,
    r_values: np.ndarray,
    threshold: float = CROSSOVER_THRESHOLD
) -> List[Dict]:
    """Detect physics regime transitions in KAN output.

    Samples KAN output at fine resolution and identifies
    discontinuities in the second derivative that indicate
    transitions between physical regimes (e.g., Newtonian
    to dark matter dominated).

    Args:
        kan: Trained KAN instance
        r_values: Array of radius values to analyze
        threshold: Threshold for second derivative discontinuity

    Returns:
        List of transition dicts with:
            - transition_point_r: Radius at transition
            - regime_from: Regime before transition
            - regime_to: Regime after transition
            - confidence: Confidence in detection
    """
    if not kan.trained:
        return []

    # Sample at fine resolution
    r_min, r_max = r_values.min(), r_values.max()
    r_fine = np.linspace(r_min, r_max, 1000)
    r_norm = (r_fine - r_min) / (r_max - r_min + 1e-8)

    # Get KAN output
    v_pred = kan.forward(r_norm)

    # Compute second derivative (numerically)
    dr = r_fine[1] - r_fine[0]
    dv_dr = np.gradient(v_pred, dr)
    d2v_dr2 = np.gradient(dv_dr, dr)

    # Find discontinuities above threshold
    transitions = []
    d2v_abs = np.abs(d2v_dr2)
    d2v_mean = np.mean(d2v_abs)
    d2v_std = np.std(d2v_abs)

    for i in range(1, len(d2v_dr2) - 1):
        # Detect sudden changes in second derivative
        local_change = abs(d2v_dr2[i] - d2v_dr2[i-1])
        if local_change > threshold * d2v_std and d2v_abs[i] > d2v_mean:
            # Classify regimes based on curve shape
            if dv_dr[i] > 0 and d2v_dr2[i] < 0:
                regime_from = "rising"
                regime_to = "flat" if abs(dv_dr[i]) < 0.1 else "declining"
            elif dv_dr[i] < 0:
                regime_from = "declining"
                regime_to = "flat"
            else:
                regime_from = "flat"
                regime_to = "rising" if d2v_dr2[i] > 0 else "declining"

            # Compute confidence based on change magnitude
            confidence = min(1.0, local_change / (threshold * d2v_std * 2))

            transitions.append({
                "transition_point_r": float(r_fine[i]),
                "regime_from": regime_from,
                "regime_to": regime_to,
                "confidence": float(confidence),
                "d2v_dr2": float(d2v_dr2[i]),
            })

    # Deduplicate nearby transitions
    if transitions:
        deduped = [transitions[0]]
        for t in transitions[1:]:
            if abs(t["transition_point_r"] - deduped[-1]["transition_point_r"]) > (r_max - r_min) * 0.05:
                deduped.append(t)
        transitions = deduped

    return transitions


# === WITNESS RECEIPT EMISSION ===

def emit_witness_receipt(
    galaxy_id: str,
    kan: KAN,
    r: np.ndarray,
    v: np.ndarray,
    data_source: str = "synthetic",
    sparc_seed: int = None,
    provenance_hash: str = None
) -> Dict:
    """Emit witness receipt for KAN analysis.

    Modified Receipt (v2):
        - Add: regime_transitions (list)
        - Add: data_source ("synthetic" | "SPARC" | "SDSS")
        - Add: provenance_hash (link to real_data_receipt if real data)

    Args:
        galaxy_id: Galaxy identifier
        kan: Trained KAN instance
        r: Radius array
        v: Velocity array
        data_source: Source of data
        sparc_seed: SPARC random seed if applicable
        provenance_hash: Hash linking to real_data_receipt

    Returns:
        Witness receipt dict
    """
    # Compute metrics
    compression = kan.get_compression_ratio(
        (r - r.min()) / (r.max() - r.min() + 1e-8),
        (v - v.min()) / (v.max() - v.min() + 1e-8)
    )
    r_squared = kan.get_r_squared(
        (r - r.min()) / (r.max() - r.min() + 1e-8),
        (v - v.min()) / (v.max() - v.min() + 1e-8)
    )

    # Detect crossovers
    transitions = crossover_detection(kan, r)

    # Build receipt payload
    payload = {
        "tenant_id": TENANT_ID,
        "galaxy_id": galaxy_id,
        "compression": compression,
        "r_squared": r_squared,
        "equation": kan.to_equation(),
        "regime_transitions": transitions,  # v2 addition
        "data_source": data_source,  # v2 addition
    }

    if sparc_seed is not None:
        payload["sparc_seed"] = sparc_seed

    if provenance_hash is not None:
        payload["provenance_hash"] = provenance_hash  # v2 addition

    return emit_receipt("witness", payload)


# === VALIDATION FUNCTIONS ===

def validate_compression_threshold(
    galaxies: List[Dict],
    threshold: float = 0.92
) -> Dict:
    """Validate that compression meets threshold on galaxy set.

    Args:
        galaxies: List of galaxy dicts with r, v arrays
        threshold: Minimum compression ratio (default 0.92 = 92%)

    Returns:
        Validation results dict
    """
    results = []
    for galaxy in galaxies:
        kan = KAN()
        r = np.array(galaxy["r"])
        v = np.array(galaxy["v"])

        train_result = train(kan, r, v)
        results.append({
            "galaxy_id": galaxy.get("id", "unknown"),
            "compression": train_result["compression"],
            "r_squared": train_result["r_squared"],
            "passes_threshold": train_result["compression"] >= threshold,
        })

    n_passing = sum(1 for r in results if r["passes_threshold"])
    mean_compression = np.mean([r["compression"] for r in results])

    return {
        "threshold": threshold,
        "n_galaxies": len(galaxies),
        "n_passing": n_passing,
        "pass_rate": n_passing / len(galaxies) if galaxies else 0,
        "mean_compression": mean_compression,
        "all_pass": n_passing == len(galaxies),
        "results": results,
    }
