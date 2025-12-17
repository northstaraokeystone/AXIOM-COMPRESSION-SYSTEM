"""symbolic_baselines.py - Baseline Symbolic Regression Comparisons

Compare against:
    - AI Feynman (Udrescu & Tegmark 2020)
    - Eureqa (Schmidt & Lipson 2009) - legacy, stub only

These are historical baselines. pySR is the primary comparison.

Source: AXIOM Validation Lock v1
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

# Import from src
try:
    from src.core import emit_receipt
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.core import emit_receipt


# === CONSTANTS ===

TENANT_ID = "axiom-benchmarks"

# AI Feynman physics-based priors
AI_FEYNMAN_OPERATORS = {
    "basic": ["+", "-", "*", "/"],
    "advanced": ["sqrt", "square", "sin", "cos", "exp", "log"],
    "physics": ["arcsin", "arccos", "tanh"],
}


# === AI FEYNMAN IMPLEMENTATION ===

def run_ai_feynman(
    data: Dict,
    timeout_s: int = 60,
    use_physics_prior: bool = True
) -> Dict:
    """Run AI Feynman-style symbolic regression.

    AI Feynman uses dimensional analysis and physical priors
    to constrain the search space.

    For rotation curves, we use the prior:
        [v] = km/s, [r] = kpc
        [G*M] = km²/s² * kpc

    This suggests: v ~ sqrt(const / r) or v ~ const

    Args:
        data: Galaxy dict with 'r' and 'v' arrays
        timeout_s: Timeout in seconds
        use_physics_prior: Whether to use dimensional analysis

    Returns:
        Dict with results
    """
    r = np.array(data["r"])
    v = np.array(data["v"])

    start_time = time.time()

    # AI Feynman approach: Try known physical forms
    candidates = []

    # 1. Newtonian: v = sqrt(GM/r)
    # Fit: v^2 * r = const
    const1 = np.mean(v ** 2 * r)
    pred1 = np.sqrt(const1 / r)
    mse1 = np.mean((v - pred1) ** 2)
    candidates.append({
        "equation": f"sqrt({const1:.2f}/r)",
        "mse": mse1,
        "complexity": 8,
        "physics": "newtonian",
    })

    # 2. Flat rotation curve: v = const
    const2 = np.mean(v)
    pred2 = np.full_like(v, const2)
    mse2 = np.mean((v - pred2) ** 2)
    candidates.append({
        "equation": f"{const2:.2f}",
        "mse": mse2,
        "complexity": 1,
        "physics": "isothermal_halo",
    })

    # 3. Rising + flat: v = v_max * (1 - exp(-r/r_s))
    # Fit v_max and r_s
    v_max = np.max(v)
    r_half = r[np.argmin(np.abs(v - 0.5 * v_max))] if len(r) > 1 else r[0]
    r_s = r_half / np.log(2) if r_half > 0 else 1.0
    pred3 = v_max * (1 - np.exp(-r / r_s))
    mse3 = np.mean((v - pred3) ** 2)
    candidates.append({
        "equation": f"{v_max:.2f}*(1-exp(-r/{r_s:.2f}))",
        "mse": mse3,
        "complexity": 15,
        "physics": "exponential_disk",
    })

    # 4. MOND-like: v = sqrt(sqrt(a^2 + b/r))
    # Simplified MOND interpolation
    a = np.mean(v[-5:]) if len(v) >= 5 else np.max(v)  # Asymptotic velocity
    b = np.mean(v[:5] ** 4 * r[:5]) if len(v) >= 5 else a ** 4 * r[0]
    pred4 = np.power(a ** 4 + b / r, 0.25)
    mse4 = np.mean((v - pred4) ** 2)
    candidates.append({
        "equation": f"({a:.2f}^4 + {b:.2f}/r)^0.25",
        "mse": mse4,
        "complexity": 12,
        "physics": "mond_like",
    })

    # Select best candidate
    best = min(candidates, key=lambda x: x["mse"])

    elapsed_ms = (time.time() - start_time) * 1000

    result = {
        "equation": best["equation"],
        "mse": float(best["mse"]),
        "complexity": best["complexity"],
        "physics_form": best["physics"],
        "time_ms": elapsed_ms,
        "tool": "AI_Feynman",
        "candidates_evaluated": len(candidates),
        "success": True,
    }

    # Emit benchmark receipt
    emit_receipt("benchmark", {
        "tenant_id": TENANT_ID,
        "tool_name": "AI_Feynman",
        "dataset_id": data.get("id", "unknown"),
        "compression_ratio": 0,
        "r_squared": 1 - best["mse"] / np.var(v) if np.var(v) > 0 else 0,
        "equation": best["equation"],
        "time_ms": elapsed_ms,
    })

    return result


def run_eureqa_stub(data: Dict) -> Dict:
    """Stub for Eureqa comparison (legacy system, not available).

    Eureqa was the original symbolic regression system (2009)
    but is no longer publicly available.

    Args:
        data: Galaxy dict

    Returns:
        Dict with stub results
    """
    return {
        "equation": "unavailable",
        "mse": float("inf"),
        "complexity": 0,
        "time_ms": 0,
        "tool": "Eureqa",
        "success": False,
        "note": "Eureqa is legacy software and no longer available",
    }


def compare_all_baselines(galaxy: Dict) -> Dict:
    """Compare all baseline methods on a galaxy.

    Args:
        galaxy: Galaxy dict with rotation curve data

    Returns:
        Dict with all baseline results
    """
    from .pysr_comparison import run_pysr, run_axiom

    results = {
        "galaxy_id": galaxy.get("id", "unknown"),
        "pysr": run_pysr(galaxy),
        "ai_feynman": run_ai_feynman(galaxy),
        "eureqa": run_eureqa_stub(galaxy),
        "axiom": run_axiom(galaxy),
    }

    # Rank by MSE
    methods = ["pysr", "ai_feynman", "axiom"]
    mses = [(m, results[m]["mse"]) for m in methods if results[m]["success"]]
    mses.sort(key=lambda x: x[1])

    results["ranking_by_mse"] = [m[0] for m in mses]
    results["best_method"] = mses[0][0] if mses else None

    # Compute R² for ranking
    v = np.array(galaxy["v"])
    var_v = np.var(v)

    if var_v > 0:
        for method in methods:
            if results[method]["success"]:
                results[method]["r_squared"] = 1 - results[method]["mse"] / var_v

    return results


def batch_compare_baselines(galaxies: List[Dict]) -> Dict:
    """Run baseline comparison on multiple galaxies.

    Args:
        galaxies: List of galaxy dicts

    Returns:
        Dict with aggregated results
    """
    all_results = []
    method_wins = {"pysr": 0, "ai_feynman": 0, "axiom": 0}

    for galaxy in galaxies:
        result = compare_all_baselines(galaxy)
        all_results.append(result)

        if result["best_method"]:
            method_wins[result["best_method"]] += 1

    # Aggregate statistics
    summary = {
        "n_galaxies": len(galaxies),
        "method_wins": method_wins,
        "best_overall": max(method_wins, key=method_wins.get),
    }

    for method in ["pysr", "ai_feynman", "axiom"]:
        mses = [r[method]["mse"] for r in all_results if r[method]["success"]]
        if mses:
            summary[f"{method}_mean_mse"] = float(np.mean(mses))
            summary[f"{method}_std_mse"] = float(np.std(mses))

    # Emit summary receipt
    emit_receipt("baseline_comparison_summary", {
        "tenant_id": TENANT_ID,
        "n_galaxies": len(galaxies),
        "method_wins": method_wins,
        "best_overall": summary["best_overall"],
    })

    return {
        "individual_results": all_results,
        "summary": summary,
    }


def generate_baseline_table(results: List[Dict]) -> str:
    """Generate markdown table comparing all baselines.

    Args:
        results: List of comparison results

    Returns:
        Markdown table string
    """
    lines = [
        "| Galaxy | pySR MSE | AI Feynman MSE | AXIOM MSE | AXIOM R² | Best |",
        "|--------|----------|----------------|-----------|----------|------|",
    ]

    for r in results:
        galaxy_id = r.get("galaxy_id", "unknown")
        pysr_mse = r["pysr"]["mse"]
        ai_mse = r["ai_feynman"]["mse"]
        axiom_mse = r["axiom"]["mse"]
        axiom_r2 = r["axiom"].get("r_squared", 0)
        best = r.get("best_method", "unknown")

        lines.append(
            f"| {galaxy_id} | {pysr_mse:.4f} | {ai_mse:.4f} | "
            f"{axiom_mse:.4f} | {axiom_r2:.4f} | {best} |"
        )

    return "\n".join(lines)
