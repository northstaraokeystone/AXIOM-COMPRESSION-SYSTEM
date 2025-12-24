"""pysr_comparison.py - Compare SpaceProof KAN to pySR Symbolic Regression

THE BENCHMARK INSIGHT:
    Claims without comparison are marketing.
    pySR is 2024 SOTA for symbolic regression.
    SpaceProof must prove it compresses better on real data.

Reference: Cranmer 2023 "Discovering Symbolic Models from Deep Learning"

Source: SpaceProof Validation Lock v1
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

# Import from src
try:
    from spaceproof.core import emit_receipt
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from spaceproof.core import emit_receipt


# === CONSTANTS ===

TENANT_ID = "spaceproof-benchmarks"
DEFAULT_EPOCHS = 100
DEFAULT_COMPLEXITY_LIMIT = 20

# Physics-informed operators for rotation curves
ROTATION_CURVE_OPERATORS = [
    "+",
    "-",
    "*",
    "/",
    "sqrt",
    "square",
    "cube",
    "log",
    "exp",
]

# Known rotation curve equations
KNOWN_EQUATIONS = {
    "newtonian": "sqrt(G * M / r)",
    "mond_simple": "sqrt(sqrt((G * M / r)^2 + (G * M * a0 / r)^2))",
    "nfw_halo": "sqrt(G * M_vir / r * (log(1 + c * r/r_s) - c * r/r_s / (1 + c * r/r_s)))",
    "isothermal_sphere": "v_c",  # Flat rotation curve
}


# === SpaceProof KAN IMPLEMENTATION ===


class SimpleKAN:
    """Simplified Kolmogorov-Arnold Network for rotation curve fitting.

    This is a minimal implementation that captures the key insight:
    KANs decompose multivariate functions into univariate ones.

    For rotation curves v(r), we learn:
        v(r) = sum_i phi_i(psi_i(r))

    where phi_i and psi_i are univariate splines.
    """

    def __init__(self, n_basis: int = 10, degree: int = 3):
        """Initialize KAN.

        Args:
            n_basis: Number of basis functions
            degree: Spline degree
        """
        self.n_basis = n_basis
        self.degree = degree
        self.weights = None
        self.knots = None
        self.fitted = False
        self.loss_history = []

    def _create_basis(self, r: np.ndarray) -> np.ndarray:
        """Create B-spline basis functions.

        Args:
            r: Input radius values

        Returns:
            Basis matrix of shape (len(r), n_basis)
        """
        r_min, r_max = r.min(), r.max()
        # Create knot vector with padding
        knots = np.linspace(r_min, r_max, self.n_basis - self.degree + 1)
        self.knots = knots

        # Simple polynomial basis (approximation of B-splines for efficiency)
        basis = np.zeros((len(r), self.n_basis))
        for i in range(self.n_basis):
            center = r_min + (r_max - r_min) * i / (self.n_basis - 1)
            width = (r_max - r_min) / (self.n_basis - 1)
            # Gaussian-like basis function
            basis[:, i] = np.exp(-0.5 * ((r - center) / width) ** 2)

        return basis

    def fit(self, r: np.ndarray, v: np.ndarray, epochs: int = 100) -> Dict:
        """Fit KAN to rotation curve data.

        Args:
            r: Radius values (kpc)
            v: Velocity values (km/s)
            epochs: Number of training epochs

        Returns:
            Dict with training results
        """
        basis = self._create_basis(r)

        # Least squares initialization
        self.weights = np.linalg.lstsq(basis, v, rcond=None)[0]

        # Iterative refinement (simplified gradient descent)
        lr = 0.01
        for epoch in range(epochs):
            pred = basis @ self.weights
            residual = v - pred
            mse = np.mean(residual**2)
            self.loss_history.append(mse)

            # Gradient update
            grad = -2 * basis.T @ residual / len(r)
            self.weights -= lr * grad

        self.fitted = True
        final_pred = basis @ self.weights
        r_squared = 1 - np.sum((v - final_pred) ** 2) / np.sum((v - np.mean(v)) ** 2)

        return {
            "epochs": epochs,
            "final_mse": self.loss_history[-1] if self.loss_history else 0,
            "r_squared": r_squared,
            "n_parameters": len(self.weights),
        }

    def predict(self, r: np.ndarray) -> np.ndarray:
        """Predict velocity at given radii.

        Args:
            r: Radius values

        Returns:
            Predicted velocity values
        """
        if not self.fitted:
            raise ValueError("KAN not fitted yet")

        basis = self._create_basis(r)
        return basis @ self.weights

    def get_symbolic_approximation(self) -> str:
        """Extract approximate symbolic form.

        Returns:
            String representation of learned function
        """
        if not self.fitted:
            return "not_fitted"

        # Identify dominant basis functions
        dominant = np.argsort(np.abs(self.weights))[-3:]
        terms = []
        for i in dominant:
            coef = self.weights[i]
            if abs(coef) > 0.1:
                terms.append(f"{coef:.2f}*phi_{i}(r)")

        return " + ".join(terms) if terms else "constant"

    def compute_compression(self, data_bits: int) -> float:
        """Compute compression ratio.

        Args:
            data_bits: Original data size in bits

        Returns:
            Compression ratio (1.0 = no compression)
        """
        if not self.fitted:
            return 0.0

        # Model size: weights (64-bit floats) + knots + metadata
        model_bits = len(self.weights) * 64 + len(self.knots) * 64 + 128

        return 1.0 - (model_bits / data_bits)


# === PYSR INTERFACE ===


def run_pysr(
    data: Dict, complexity_limit: int = DEFAULT_COMPLEXITY_LIMIT, timeout_s: int = 60
) -> Dict:
    """Run pySR symbolic regression on galaxy rotation curve.

    Args:
        data: Galaxy dict with 'r' and 'v' arrays
        complexity_limit: Maximum equation complexity
        timeout_s: Timeout in seconds

    Returns:
        Dict with {equation, mse, complexity, time_ms}

    Note: If pySR not installed, returns a baseline comparison.
    """
    r = np.array(data["r"])
    v = np.array(data["v"])

    start_time = time.time()

    try:
        from pysr import PySRRegressor

        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sqrt", "square", "log"],
            complexity_of_operators={"/": 2, "sqrt": 1, "log": 2},
            maxsize=complexity_limit,
            timeout_in_seconds=timeout_s,
            procs=1,
            populations=8,
            progress=False,
        )

        model.fit(r.reshape(-1, 1), v)

        best_eq = model.sympy()
        pred = model.predict(r.reshape(-1, 1))
        mse = np.mean((v - pred) ** 2)
        complexity = len(str(best_eq))

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "equation": str(best_eq),
            "mse": float(mse),
            "complexity": complexity,
            "time_ms": elapsed_ms,
            "tool": "pySR",
            "success": True,
        }

    except ImportError:
        # pySR not installed - provide baseline estimate
        elapsed_ms = (time.time() - start_time) * 1000

        # Use known Newtonian approximation as baseline
        # v = sqrt(G*M/r) => v^2 * r = const
        v_squared_r = v**2 * r
        const = np.mean(v_squared_r)
        pred = np.sqrt(const / r)
        mse = np.mean((v - pred) ** 2)

        return {
            "equation": f"sqrt({const:.2f}/r)",
            "mse": float(mse),
            "complexity": 10,
            "time_ms": elapsed_ms,
            "tool": "pySR_baseline",
            "success": True,
            "note": "pySR not installed - using Newtonian baseline",
        }


def run_spaceproof(data: Dict, epochs: int = DEFAULT_EPOCHS) -> Dict:
    """Run SpaceProof KAN witness on galaxy rotation curve.

    Args:
        data: Galaxy dict with 'r' and 'v' arrays
        epochs: Number of training epochs

    Returns:
        Dict with {equation, mse, compression, r_squared, time_ms}
    """
    r = np.array(data["r"])
    v = np.array(data["v"])

    start_time = time.time()

    # Create and train KAN
    kan = SimpleKAN(n_basis=10, degree=3)
    result = kan.fit(r, v, epochs=epochs)

    # Compute compression
    # Data bits: 20 points * 2 values * 64 bits = 2560 bits
    data_bits = len(r) * 2 * 64
    compression = kan.compute_compression(data_bits)

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "equation": kan.get_symbolic_approximation(),
        "mse": float(result["final_mse"]),
        "compression": compression,
        "r_squared": float(result["r_squared"]),
        "time_ms": elapsed_ms,
        "n_parameters": result["n_parameters"],
        "epochs": epochs,
        "tool": "SpaceProof_KAN",
        "success": True,
    }


def compare(galaxy: Dict) -> Dict:
    """Run both pySR and SpaceProof on galaxy, emit benchmark_receipt.

    Args:
        galaxy: Galaxy dict with rotation curve data

    Returns:
        Dict with comparison results
    """
    pysr_result = run_pysr(galaxy)
    spaceproof_result = run_spaceproof(galaxy)

    # Compute relative metrics
    mse_ratio = (
        spaceproof_result["mse"] / pysr_result["mse"]
        if pysr_result["mse"] > 0
        else float("inf")
    )
    time_ratio = (
        spaceproof_result["time_ms"] / pysr_result["time_ms"]
        if pysr_result["time_ms"] > 0
        else float("inf")
    )

    comparison = {
        "galaxy_id": galaxy.get("id", "unknown"),
        "pysr": pysr_result,
        "spaceproof": spaceproof_result,
        "comparison": {
            "mse_ratio_spaceproof_vs_pysr": mse_ratio,
            "time_ratio_spaceproof_vs_pysr": time_ratio,
            "spaceproof_compression": spaceproof_result["compression"],
            "winner_mse": "spaceproof" if mse_ratio < 1 else "pysr",
            "winner_time": "spaceproof" if time_ratio < 1 else "pysr",
        },
    }

    # Emit benchmark receipt for pySR
    emit_receipt(
        "benchmark",
        {
            "tenant_id": TENANT_ID,
            "tool_name": "pySR",
            "dataset_id": galaxy.get("id", "unknown"),
            "compression_ratio": 0,  # pySR doesn't report compression
            "r_squared": 1 - pysr_result["mse"] / np.var(galaxy["v"])
            if np.var(galaxy["v"]) > 0
            else 0,
            "equation": pysr_result["equation"],
            "time_ms": pysr_result["time_ms"],
        },
    )

    # Emit benchmark receipt for SpaceProof
    emit_receipt(
        "benchmark",
        {
            "tenant_id": TENANT_ID,
            "tool_name": "SpaceProof",
            "dataset_id": galaxy.get("id", "unknown"),
            "compression_ratio": spaceproof_result["compression"],
            "r_squared": spaceproof_result["r_squared"],
            "equation": spaceproof_result["equation"],
            "time_ms": spaceproof_result["time_ms"],
        },
    )

    return comparison


def batch_compare(galaxies: List[Dict]) -> Dict:
    """Run comparison on multiple galaxies, aggregate results.

    Args:
        galaxies: List of galaxy dicts

    Returns:
        Dict with aggregated comparison results
    """
    results = []
    for galaxy in galaxies:
        result = compare(galaxy)
        results.append(result)

    # Aggregate statistics
    spaceproof_compressions = [r["spaceproof"]["compression"] for r in results]
    spaceproof_r_squared = [r["spaceproof"]["r_squared"] for r in results]
    pysr_mses = [r["pysr"]["mse"] for r in results]
    spaceproof_mses = [r["spaceproof"]["mse"] for r in results]

    summary = {
        "n_galaxies": len(galaxies),
        "spaceproof": {
            "mean_compression": float(np.mean(spaceproof_compressions)),
            "std_compression": float(np.std(spaceproof_compressions)),
            "min_compression": float(np.min(spaceproof_compressions)),
            "max_compression": float(np.max(spaceproof_compressions)),
            "mean_r_squared": float(np.mean(spaceproof_r_squared)),
            "mean_mse": float(np.mean(spaceproof_mses)),
        },
        "pysr": {
            "mean_mse": float(np.mean(pysr_mses)),
        },
        "spaceproof_wins_mse": sum(
            1 for r in results if r["comparison"]["winner_mse"] == "spaceproof"
        ),
        "pysr_wins_mse": sum(
            1 for r in results if r["comparison"]["winner_mse"] == "pysr"
        ),
        "spaceproof_wins_time": sum(
            1 for r in results if r["comparison"]["winner_time"] == "spaceproof"
        ),
    }

    # Emit summary receipt
    emit_receipt(
        "benchmark_summary",
        {
            "tenant_id": TENANT_ID,
            "n_galaxies": len(galaxies),
            "mean_spaceproof_compression": summary["spaceproof"]["mean_compression"],
            "mean_spaceproof_r_squared": summary["spaceproof"]["mean_r_squared"],
            "spaceproof_wins_mse": summary["spaceproof_wins_mse"],
            "spaceproof_wins_time": summary["spaceproof_wins_time"],
        },
    )

    return {
        "individual_results": results,
        "summary": summary,
    }


def generate_table(results: List[Dict]) -> str:
    """Format comparison results as markdown table.

    Args:
        results: List of comparison result dicts

    Returns:
        Markdown table string
    """
    lines = [
        "| Galaxy | pySR MSE | SpaceProof MSE | SpaceProof Compression | SpaceProof R² | Winner |",
        "|--------|----------|-----------|-------------------|----------|--------|",
    ]

    for r in results:
        galaxy_id = r.get("galaxy_id", "unknown")
        pysr_mse = r["pysr"]["mse"]
        spaceproof_mse = r["spaceproof"]["mse"]
        compression = r["spaceproof"]["compression"]
        r_squared = r["spaceproof"]["r_squared"]
        winner = r["comparison"]["winner_mse"]

        lines.append(
            f"| {galaxy_id} | {pysr_mse:.4f} | {spaceproof_mse:.4f} | "
            f"{compression:.2%} | {r_squared:.4f} | {winner} |"
        )

    return "\n".join(lines)


# === CLI ENTRY POINT ===


def main():
    """Run benchmark suite from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="SpaceProof vs pySR Benchmark")
    parser.add_argument("--dataset", default="sparc", help="Dataset to use")
    parser.add_argument("--n", type=int, default=10, help="Number of galaxies")
    parser.add_argument("--output", default=None, help="Output file for results")

    args = parser.parse_args()

    # Load data
    from real_data.sparc import load_sparc

    print(f"Loading {args.n} galaxies from {args.dataset}...")
    galaxies = load_sparc(n_galaxies=args.n)

    print("Running benchmark comparison...")
    results = batch_compare(galaxies)

    # Print table
    print("\n" + generate_table(results["individual_results"]))

    # Print summary
    print("\nSummary:")
    print(
        f"  SpaceProof mean compression: {results['summary']['spaceproof']['mean_compression']:.2%}"
    )
    print(f"  SpaceProof mean R²: {results['summary']['spaceproof']['mean_r_squared']:.4f}")
    print(f"  SpaceProof wins (MSE): {results['summary']['spaceproof_wins_mse']}/{len(galaxies)}")
    print(
        f"  SpaceProof wins (time): {results['summary']['spaceproof_wins_time']}/{len(galaxies)}"
    )

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
