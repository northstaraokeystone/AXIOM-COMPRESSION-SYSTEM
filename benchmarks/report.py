"""report.py - Benchmark Report Generation

Generate comprehensive comparison reports for AXIOM validation.

Source: AXIOM Validation Lock v1
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np

# Import from src
try:
    from src.core import dual_hash, emit_receipt
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.core import dual_hash, emit_receipt


# === CONSTANTS ===

TENANT_ID = "axiom-benchmarks"

# Success criteria from spec
SUCCESS_CRITERIA = {
    "compression_min": 0.92,     # >= 92% compression on SPARC
    "r_squared_min": 0.98,       # >= 0.98 R² on rotation curves
    "bits_per_kg_tolerance": 0.15,  # ±15% of 60k kg baseline
    "n_scenarios": 10,           # 10 simulation scenarios
    "coverage_min": 0.95,        # 95% CI coverage
}


def format_comparison_table(results: Dict) -> str:
    """Format benchmark results as a markdown comparison table.

    Args:
        results: Dict with benchmark results

    Returns:
        Markdown table string
    """
    individual = results.get("individual_results", [])
    summary = results.get("summary", {})

    lines = [
        "# AXIOM Benchmark Results",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Individual Galaxy Results",
        "",
        "| Galaxy | AXIOM Compression | AXIOM R² | AXIOM MSE | pySR MSE | Winner |",
        "|--------|-------------------|----------|-----------|----------|--------|",
    ]

    for r in individual:
        galaxy_id = r.get("galaxy_id", "unknown")
        compression = r.get("axiom", {}).get("compression", 0)
        r_squared = r.get("axiom", {}).get("r_squared", 0)
        axiom_mse = r.get("axiom", {}).get("mse", 0)
        pysr_mse = r.get("pysr", {}).get("mse", 0)
        winner = r.get("comparison", {}).get("winner_mse", "unknown")

        # Highlight if meeting criteria
        compression_str = f"**{compression:.2%}**" if compression >= SUCCESS_CRITERIA["compression_min"] else f"{compression:.2%}"
        r_squared_str = f"**{r_squared:.4f}**" if r_squared >= SUCCESS_CRITERIA["r_squared_min"] else f"{r_squared:.4f}"

        lines.append(
            f"| {galaxy_id} | {compression_str} | {r_squared_str} | "
            f"{axiom_mse:.4f} | {pysr_mse:.4f} | {winner} |"
        )

    # Add summary section
    lines.extend([
        "",
        "## Summary Statistics",
        "",
        f"- **Galaxies tested**: {summary.get('n_galaxies', 0)}",
        f"- **Mean AXIOM compression**: {summary.get('axiom', {}).get('mean_compression', 0):.2%}",
        f"- **Mean AXIOM R²**: {summary.get('axiom', {}).get('mean_r_squared', 0):.4f}",
        f"- **AXIOM wins (MSE)**: {summary.get('axiom_wins_mse', 0)}/{summary.get('n_galaxies', 0)}",
        f"- **AXIOM wins (time)**: {summary.get('axiom_wins_time', 0)}/{summary.get('n_galaxies', 0)}",
        "",
        "## Success Criteria Check",
        "",
    ])

    # Check success criteria
    mean_compression = summary.get("axiom", {}).get("mean_compression", 0)
    mean_r_squared = summary.get("axiom", {}).get("mean_r_squared", 0)

    compression_pass = mean_compression >= SUCCESS_CRITERIA["compression_min"]
    r_squared_pass = mean_r_squared >= SUCCESS_CRITERIA["r_squared_min"]

    lines.extend([
        f"| Criterion | Target | Actual | Status |",
        f"|-----------|--------|--------|--------|",
        f"| Compression | ≥{SUCCESS_CRITERIA['compression_min']:.0%} | {mean_compression:.2%} | {'✓ PASS' if compression_pass else '✗ FAIL'} |",
        f"| R² | ≥{SUCCESS_CRITERIA['r_squared_min']} | {mean_r_squared:.4f} | {'✓ PASS' if r_squared_pass else '✗ FAIL'} |",
    ])

    return "\n".join(lines)


def generate_benchmark_report(
    benchmark_results: Dict,
    calibration_results: Dict = None,
    scenario_results: Dict = None
) -> str:
    """Generate full benchmark report.

    Args:
        benchmark_results: Results from batch_compare()
        calibration_results: Results from bits/kg calibration
        scenario_results: Results from scenario runs

    Returns:
        Full markdown report string
    """
    lines = [
        "# AXIOM Validation Lock Report",
        "",
        f"**Generated**: {datetime.utcnow().isoformat()}Z",
        "**Version**: 1.1 (Validation Lock)",
        "",
        "---",
        "",
    ]

    # Benchmark section
    lines.append(format_comparison_table(benchmark_results))

    # Calibration section
    if calibration_results:
        lines.extend([
            "",
            "---",
            "",
            "## bits/kg Calibration",
            "",
        ])

        bits_per_kg = calibration_results.get("mean_bits_per_kg", 0)
        ci = calibration_results.get("confidence_interval", [0, 0])
        baseline_kg = 60000
        tolerance = SUCCESS_CRITERIA["bits_per_kg_tolerance"]

        # Check if within tolerance
        within_tolerance = (
            bits_per_kg > 0 and
            abs(baseline_kg - bits_per_kg) / baseline_kg <= tolerance
        )

        lines.extend([
            f"- **Mean bits/kg**: {bits_per_kg:,.0f}",
            f"- **95% CI**: [{ci[0]:,.0f}, {ci[1]:,.0f}]",
            f"- **Baseline**: {baseline_kg:,} kg",
            f"- **Tolerance**: ±{tolerance:.0%}",
            f"- **Status**: {'✓ PASS' if within_tolerance else '✗ FAIL'}",
            f"- **Source**: {calibration_results.get('calibration_source', 'unknown')}",
        ])

    # Scenario section
    if scenario_results:
        lines.extend([
            "",
            "---",
            "",
            "## Scenario Results",
            "",
            "| Scenario | Status | Notes |",
            "|----------|--------|-------|",
        ])

        for scenario, result in scenario_results.items():
            status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
            notes = result.get("notes", "")
            lines.append(f"| {scenario} | {status} | {notes} |")

        n_scenarios = len(scenario_results)
        n_passed = sum(1 for r in scenario_results.values() if r.get("passed", False))
        lines.extend([
            "",
            f"**Scenarios**: {n_passed}/{n_scenarios} passed",
            f"**Target**: {SUCCESS_CRITERIA['n_scenarios']} scenarios",
        ])

    # Overall status
    lines.extend([
        "",
        "---",
        "",
        "## Overall Validation Status",
        "",
    ])

    # Compute overall pass/fail
    checks = []

    # Compression check
    mean_compression = benchmark_results.get("summary", {}).get("axiom", {}).get("mean_compression", 0)
    checks.append(("Compression ≥92%", mean_compression >= SUCCESS_CRITERIA["compression_min"]))

    # R² check
    mean_r_squared = benchmark_results.get("summary", {}).get("axiom", {}).get("mean_r_squared", 0)
    checks.append(("R² ≥0.98", mean_r_squared >= SUCCESS_CRITERIA["r_squared_min"]))

    # bits/kg check (if available)
    if calibration_results:
        bits_per_kg = calibration_results.get("mean_bits_per_kg", 0)
        baseline_kg = 60000
        within_tolerance = abs(baseline_kg - bits_per_kg) / baseline_kg <= 0.15
        checks.append(("bits/kg ±15%", within_tolerance))

    # Scenarios check (if available)
    if scenario_results:
        n_passed = sum(1 for r in scenario_results.values() if r.get("passed", False))
        checks.append((f"10 scenarios", n_passed >= 10))

    all_pass = all(c[1] for c in checks)

    lines.append("| Check | Status |")
    lines.append("|-------|--------|")
    for check_name, passed in checks:
        lines.append(f"| {check_name} | {'✓' if passed else '✗'} |")

    lines.extend([
        "",
        f"**OVERALL**: {'✓ VALIDATION PASSED' if all_pass else '✗ VALIDATION FAILED'}",
    ])

    return "\n".join(lines)


def emit_benchmark_summary(
    benchmark_results: Dict,
    calibration_results: Dict = None,
    scenario_results: Dict = None
) -> Dict:
    """Emit summary receipt for benchmark run.

    Args:
        benchmark_results: Results from batch_compare()
        calibration_results: Optional calibration results
        scenario_results: Optional scenario results

    Returns:
        Summary receipt dict
    """
    summary = benchmark_results.get("summary", {})

    # Compute validation status
    mean_compression = summary.get("axiom", {}).get("mean_compression", 0)
    mean_r_squared = summary.get("axiom", {}).get("mean_r_squared", 0)

    compression_pass = mean_compression >= SUCCESS_CRITERIA["compression_min"]
    r_squared_pass = mean_r_squared >= SUCCESS_CRITERIA["r_squared_min"]

    validation_status = {
        "compression_pass": compression_pass,
        "r_squared_pass": r_squared_pass,
    }

    if calibration_results:
        bits_per_kg = calibration_results.get("mean_bits_per_kg", 0)
        within_tolerance = abs(60000 - bits_per_kg) / 60000 <= 0.15 if bits_per_kg > 0 else False
        validation_status["bits_per_kg_pass"] = within_tolerance

    if scenario_results:
        n_passed = sum(1 for r in scenario_results.values() if r.get("passed", False))
        validation_status["scenarios_pass"] = n_passed >= 10

    all_pass = all(validation_status.values())

    return emit_receipt("validation_lock", {
        "tenant_id": TENANT_ID,
        "n_galaxies": summary.get("n_galaxies", 0),
        "mean_compression": mean_compression,
        "mean_r_squared": mean_r_squared,
        "axiom_wins_mse": summary.get("axiom_wins_mse", 0),
        "validation_status": validation_status,
        "all_pass": all_pass,
        "gate": "t48h",
        "slo": {
            "compression": f">= {SUCCESS_CRITERIA['compression_min']}",
            "r_squared": f">= {SUCCESS_CRITERIA['r_squared_min']}",
        },
    })


def save_report(
    report: str,
    output_path: str = "benchmark_report.md"
) -> str:
    """Save benchmark report to file.

    Args:
        report: Report markdown string
        output_path: Output file path

    Returns:
        Path to saved file
    """
    with open(output_path, 'w') as f:
        f.write(report)

    return output_path
