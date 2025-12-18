"""plot_curve.py - The Sovereignty Curve

Purpose: ONE graph. THE finding.

Output: outputs/sovereignty_curve.png

Source: Critical Review Dec 16, 2025 - "One curve, one finding"
"""

import os
from typing import List, Tuple, Optional

from .core import emit_receipt
from .sovereignty import SovereigntyConfig, compute_sovereignty


def generate_curve_data(
    crew_range: Tuple[int, int], bandwidth_mbps: float = 4.0, delay_s: float = 480.0
) -> List[Tuple[int, float]]:
    """Generate (crew, advantage) pairs for sovereignty curve.

    Args:
        crew_range: (min_crew, max_crew) to evaluate
        bandwidth_mbps: Communication bandwidth
        delay_s: One-way light delay

    Returns:
        List of (crew, advantage) tuples

    The curve shows sovereignty advantage vs crew size.
    Where advantage crosses zero is the threshold.
    """
    min_crew, max_crew = crew_range
    data = []

    for crew in range(min_crew, max_crew + 1):
        config = SovereigntyConfig(
            crew=crew, compute_flops=0.0, bandwidth_mbps=bandwidth_mbps, delay_s=delay_s
        )
        result = compute_sovereignty(config)
        data.append((crew, result.advantage))

    return data


def find_knee(curve_data: List[Tuple[int, float]]) -> int:
    """Find crew value where advantage crosses zero.

    Args:
        curve_data: List of (crew, advantage) from generate_curve_data()

    Returns:
        Crew value at knee (where advantage first becomes positive)

    Algorithm:
        Linear search for first positive advantage.
        If all negative, return max crew. If all positive, return min crew.
    """
    if not curve_data:
        return 0

    # Find first positive advantage
    for crew, advantage in curve_data:
        if advantage > 0:
            return crew

    # All negative - return last crew + 1 (no sovereignty achieved)
    return curve_data[-1][0] + 1


def plot_sovereignty_curve(
    curve_data: List[Tuple[int, float]],
    knee: int,
    output_path: str,
    title: str = "SOVEREIGNTY CURVE: Mars Colony",
    uncertainty: Optional[float] = None,
) -> None:
    """Generate matplotlib plot with annotations.

    Args:
        curve_data: List of (crew, advantage) from generate_curve_data()
        knee: Crew value at threshold (from find_knee())
        output_path: Path to save PNG
        title: Plot title
        uncertainty: Optional +/- crew uncertainty for annotation

    Output:
        Saves PNG to output_path

    The Plot:
        X-axis: Crew count
        Y-axis: Sovereignty advantage (bits/sec)
        Horizontal line at y=0 (sovereignty threshold)
        Vertical line at x=knee (threshold crew)
        Shaded regions for SOVEREIGN vs DEPENDENT
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
    except ImportError:
        # Fallback: create a text file describing the plot
        _plot_text_fallback(curve_data, knee, output_path, uncertainty)
        return

    # Extract data
    crews = [d[0] for d in curve_data]
    advantages = [d[1] for d in curve_data]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the curve
    ax.plot(crews, advantages, "b-", linewidth=2, label="Sovereignty Advantage")

    # Horizontal line at y=0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, label="Threshold")

    # Vertical line at knee
    ax.axvline(x=knee, color="red", linestyle="--", linewidth=1.5)

    # Shade regions
    ax.fill_between(
        crews,
        advantages,
        0,
        where=[a > 0 for a in advantages],
        alpha=0.3,
        color="green",
        label="SOVEREIGN",
    )
    ax.fill_between(
        crews,
        advantages,
        0,
        where=[a <= 0 for a in advantages],
        alpha=0.3,
        color="red",
        label="DEPENDENT",
    )

    # Annotate threshold
    uncertainty_str = f" +/- {uncertainty:.0f}" if uncertainty else ""
    ax.annotate(
        f"THRESHOLD: {knee}{uncertainty_str} crew",
        xy=(knee, 0),
        xytext=(knee + 10, max(advantages) * 0.3),
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    # Labels and title
    ax.set_xlabel("Crew Count", fontsize=12)
    ax.set_ylabel("Sovereignty Advantage (bits/sec)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    ax.legend(loc="upper left")

    # Grid
    ax.grid(True, alpha=0.3)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_text_fallback(
    curve_data: List[Tuple[int, float]],
    knee: int,
    output_path: str,
    uncertainty: Optional[float] = None,
) -> None:
    """Create text representation when matplotlib unavailable."""
    # Change extension to .txt
    txt_path = output_path.replace(".png", ".txt")

    uncertainty_str = f" +/- {uncertainty:.0f}" if uncertainty else ""

    lines = [
        "SOVEREIGNTY CURVE: Mars Colony",
        "=" * 40,
        "",
        "Crew    Advantage    Status",
        "-" * 40,
    ]

    for crew, advantage in curve_data:
        status = "SOVEREIGN" if advantage > 0 else "DEPENDENT"
        marker = " <-- THRESHOLD" if crew == knee else ""
        lines.append(f"{crew:4d}    {advantage:+10.2f}    {status}{marker}")

    lines.extend(["", "-" * 40, f"THRESHOLD: {knee}{uncertainty_str} crew", "=" * 40])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def format_finding(
    knee: int, bandwidth: float, delay: float, uncertainty: float
) -> str:
    """Format THE finding as human-readable text.

    Args:
        knee: Threshold crew count
        bandwidth: Bandwidth used (Mbps)
        delay: Delay used (seconds)
        uncertainty: Standard deviation of threshold

    Returns:
        The finding string

    Format:
        "At X +/- Y crew with A-B Mbps Starlink bandwidth and C-D minute
         light delay, a Mars colony achieves computational sovereignty--
         the point where local decision capacity exceeds Earth-dependent capacity."
    """
    delay_min = delay / 60  # Convert to minutes

    return (
        f"At {knee} +/- {uncertainty:.0f} crew with {bandwidth:.0f} Mbps bandwidth "
        f"and {delay_min:.0f}-minute light delay,\n"
        f"a Mars colony achieves computational sovereignty--\n"
        f"the point where local decision capacity exceeds Earth-dependent capacity."
    )


def emit_curve_receipt(knee: int, uncertainty: float, output_path: str) -> dict:
    """Emit receipt for sovereignty curve generation.

    MUST emit receipt per CLAUDEME.
    """
    return emit_receipt(
        "sovereignty_curve",
        {
            "tenant_id": "axiom-core",
            "threshold_crew": knee,
            "uncertainty": uncertainty,
            "output_path": output_path,
        },
    )


# === SENSITIVITY HEATMAP (v1.1 - Grok feedback Dec 16, 2025) ===


def generate_threshold_surface(
    bandwidth_range: Tuple[float, float] = (2.0, 10.0),
    delay_range: Tuple[float, float] = (180.0, 1320.0),
    steps: int = 20,
) -> Tuple[List[float], List[float], List[List[int]]]:
    """Generate 2D threshold surface for heatmap.

    Args:
        bandwidth_range: (min, max) bandwidth in Mbps
        delay_range: (min, max) delay in seconds
        steps: Grid resolution

    Returns:
        Tuple of (bandwidths, delays, thresholds_2d)
        thresholds_2d[i][j] = threshold at bandwidth[i], delay[j]

    Source: Grok Dec 16, 2025 - latency dominance visualization
    """
    from .sovereignty import find_threshold

    bw_min, bw_max = bandwidth_range
    delay_min, delay_max = delay_range

    bandwidths = [bw_min + (bw_max - bw_min) * i / (steps - 1) for i in range(steps)]
    delays = [
        delay_min + (delay_max - delay_min) * i / (steps - 1) for i in range(steps)
    ]

    thresholds = []
    for bw in bandwidths:
        row = []
        for delay in delays:
            t = find_threshold(bandwidth_mbps=bw, delay_s=delay)
            row.append(t)
        thresholds.append(row)

    return bandwidths, delays, thresholds


def plot_sensitivity_heatmap(
    output_path: str, title: str = "SOVEREIGNTY THRESHOLD SURFACE: Bandwidth vs Delay"
) -> None:
    """Generate 2D heatmap of threshold surface.

    Shows how threshold varies with both bandwidth and delay simultaneously.
    Reveals the latency-dominance Grok identified.

    Args:
        output_path: Path to save PNG
        title: Plot title

    Source: Grok Dec 16, 2025 - "It's primarily latency-limited"
    """
    bandwidths, delays, thresholds = generate_threshold_surface()

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
    except ImportError:
        _heatmap_text_fallback(bandwidths, delays, thresholds, output_path)
        return

    # Convert to numpy arrays
    bw_arr = np.array(bandwidths)
    delay_arr = np.array(delays) / 60  # Convert to minutes for display
    threshold_arr = np.array(thresholds)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(
        threshold_arr,
        aspect="auto",
        origin="lower",
        extent=[delay_arr[0], delay_arr[-1], bw_arr[0], bw_arr[-1]],
        cmap="RdYlGn_r",  # Red = high threshold (harder), Green = low (easier)
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sovereignty Threshold (crew)", fontsize=12)

    # Labels
    ax.set_xlabel("Light Delay (minutes)", fontsize=12)
    ax.set_ylabel("Bandwidth (Mbps)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add contour lines
    X, Y = np.meshgrid(delay_arr, bw_arr)
    contours = ax.contour(
        X, Y, threshold_arr, colors="black", alpha=0.5, linewidths=0.5
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%d")

    # Mark key scenarios
    # Opposition: 3 min, varies bandwidth
    ax.axvline(
        x=3, color="blue", linestyle="--", linewidth=1.5, label="Opposition (3 min)"
    )
    # Conjunction: 22 min, varies bandwidth
    ax.axvline(
        x=22, color="red", linestyle="--", linewidth=1.5, label="Conjunction (22 min)"
    )

    # Add annotation for latency dominance
    ax.annotate(
        "LATENCY\nDOMINATES",
        xy=(18, 8),
        fontsize=10,
        fontweight="bold",
        color="white",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
    )

    # Legend
    ax.legend(loc="upper left")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _heatmap_text_fallback(
    bandwidths: List[float],
    delays: List[float],
    thresholds: List[List[int]],
    output_path: str,
) -> None:
    """Create text representation when matplotlib unavailable."""
    txt_path = output_path.replace(".png", ".txt")

    lines = [
        "SOVEREIGNTY THRESHOLD SURFACE",
        "=" * 60,
        "",
        "Bandwidth (Mbps) vs Delay (minutes) -> Threshold (crew)",
        "",
        "Key insight: Latency dominates - vertical bands show",
        "threshold changes more with delay than bandwidth.",
        "",
        "-" * 60,
    ]

    # Header row with delays
    header = "BW\\Delay  "
    for delay in delays[::4]:  # Every 4th for readability
        header += f"{delay / 60:5.1f}m "
    lines.append(header)
    lines.append("-" * 60)

    # Data rows
    for i, bw in enumerate(bandwidths):
        if i % 4 == 0:  # Every 4th row for readability
            row = f"{bw:6.1f}   "
            for j, delay in enumerate(delays):
                if j % 4 == 0:
                    row += f"{thresholds[i][j]:6d} "
            lines.append(row)

    lines.extend(
        [
            "-" * 60,
            "",
            "FINDING: Threshold increases more along delay axis (horizontal)",
            "than along bandwidth axis (vertical), confirming latency dominance.",
            "=" * 60,
        ]
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def plot_model_comparison(
    output_path: str,
    bandwidth_mbps: float = 4.0,
    title: str = "MODEL COMPARISON: Linear vs Exponential Decay",
) -> None:
    """Plot linear vs exponential decay models side by side.

    Shows how the two models differ across delay values.

    Args:
        output_path: Path to save PNG
        bandwidth_mbps: Bandwidth for comparison
        title: Plot title

    Source: Grok paradigm shift - "bw * exp(-delay/tau)"
    """
    from .sovereignty import find_threshold, find_threshold_exponential
    from .entropy_shannon import external_rate, external_rate_exponential

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
    except ImportError:
        return  # Skip if matplotlib unavailable

    # Generate data
    delays = np.linspace(180, 1320, 50)  # 3 to 22 minutes

    thresholds_lin = [
        find_threshold(bandwidth_mbps=bandwidth_mbps, delay_s=d) for d in delays
    ]
    thresholds_exp = [
        find_threshold_exponential(bandwidth_mbps=bandwidth_mbps, delay_s=d)
        for d in delays
    ]

    rates_lin = [external_rate(bandwidth_mbps, d) for d in delays]
    rates_exp = [external_rate_exponential(bandwidth_mbps, d) for d in delays]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Thresholds
    ax1.plot(delays / 60, thresholds_lin, "b-", linewidth=2, label="Linear Model")
    ax1.plot(delays / 60, thresholds_exp, "r--", linewidth=2, label="Exponential Decay")
    ax1.set_xlabel("Light Delay (minutes)", fontsize=12)
    ax1.set_ylabel("Sovereignty Threshold (crew)", fontsize=12)
    ax1.set_title("Threshold Comparison", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark key points
    ax1.axvline(x=3, color="green", linestyle=":", alpha=0.7, label="Opposition")
    ax1.axvline(x=22, color="orange", linestyle=":", alpha=0.7, label="Conjunction")

    # Plot 2: External rates
    ax2.plot(delays / 60, rates_lin, "b-", linewidth=2, label="Linear: bw/(2*delay)")
    ax2.plot(
        delays / 60,
        rates_exp,
        "r--",
        linewidth=2,
        label="Exponential: bw*exp(-delay/tau)",
    )
    ax2.set_xlabel("Light Delay (minutes)", fontsize=12)
    ax2.set_ylabel("External Decision Rate", fontsize=12)
    ax2.set_title("External Rate Comparison", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def emit_heatmap_receipt(output_path: str) -> dict:
    """Emit receipt for heatmap generation.

    MUST emit receipt per CLAUDEME.
    """
    return emit_receipt(
        "sensitivity_heatmap",
        {
            "tenant_id": "axiom-core",
            "output_path": output_path,
            "bandwidth_range_mbps": [2.0, 10.0],
            "delay_range_s": [180, 1320],
            "finding": "latency_dominates",
        },
    )


# === ROI COMPARISON VISUALIZATION (v1.2 - Grok feedback Dec 16, 2025) ===
# Source: "Investing in τ yields higher ROI"


def plot_roi_comparison(
    investment_range: List[float],
    bw_base: float,
    tau_base: float,
    delays: List[float],
    output_path: str,
) -> None:
    """Generate ROI comparison plot: τ vs bandwidth at multiple delays.

    Shows two curves demonstrating that τ-reduction ROI exceeds
    bandwidth ROI at high delays (Mars distances).

    Args:
        investment_range: List of investment amounts in $M
        bw_base: Base bandwidth in Mbps
        tau_base: Base τ (decision latency) in seconds
        delays: List of delays to plot (seconds)
        output_path: Path to save PNG

    Output:
        ROI comparison chart with breakeven annotation

    Source: Grok Dec 16, 2025 - "τ-reduction ROI = 22x bandwidth ROI at conjunction"
    """
    from .sovereignty import (
        roi_tau_investment,
        roi_bandwidth_investment,
        find_breakeven_delay,
    )

    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        _roi_text_fallback(investment_range, bw_base, tau_base, delays, output_path)
        return

    # Create figure with subplots for each delay
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Colors for consistent styling
    colors = {"tau": "#2ecc71", "bw": "#e74c3c"}

    for idx, delay_s in enumerate(delays[:4]):  # Max 4 subplots
        ax = axes[idx]
        delay_min = delay_s / 60

        # Calculate ROIs for each investment level
        roi_taus = [
            roi_tau_investment(inv, tau_base, bw_base, delay_s)
            for inv in investment_range
        ]
        roi_bws = [
            roi_bandwidth_investment(inv, bw_base, tau_base, delay_s)
            for inv in investment_range
        ]

        # Plot
        ax.plot(
            investment_range,
            roi_taus,
            color=colors["tau"],
            linewidth=2.5,
            marker="o",
            label="τ-reduction (Autonomy)",
        )
        ax.plot(
            investment_range,
            roi_bws,
            color=colors["bw"],
            linewidth=2.5,
            marker="s",
            label="Bandwidth Upgrade",
        )

        # Fill between to highlight winner
        ax.fill_between(
            investment_range,
            roi_taus,
            roi_bws,
            where=[t > b for t, b in zip(roi_taus, roi_bws)],
            alpha=0.2,
            color=colors["tau"],
        )

        # Labels
        ax.set_xlabel("Investment ($M)", fontsize=10)
        ax.set_ylabel("ROI (rate gain per $M)", fontsize=10)
        ax.set_title(f"Delay: {delay_min:.0f} min", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(investment_range), max(investment_range))

        # Add winner annotation
        winner = "AUTONOMY" if roi_taus[-1] > roi_bws[-1] else "BANDWIDTH"
        ratio = max(roi_taus[-1], roi_bws[-1]) / max(
            0.001, min(roi_taus[-1], roi_bws[-1])
        )
        ax.annotate(
            f"{winner}\n({ratio:.1f}x better)",
            xy=(0.7, 0.85),
            xycoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            color=colors["tau"] if winner == "AUTONOMY" else colors["bw"],
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Add breakeven delay info to suptitle
    breakeven = find_breakeven_delay(bw_base, tau_base)
    fig.suptitle(
        f"ROI COMPARISON: τ-Reduction vs Bandwidth Upgrade\n"
        f"(Breakeven at {breakeven / 60:.1f} min - Mars is ALWAYS above this)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _roi_text_fallback(
    investment_range: List[float],
    bw_base: float,
    tau_base: float,
    delays: List[float],
    output_path: str,
) -> None:
    """Text fallback for ROI comparison when matplotlib unavailable."""
    from .sovereignty import roi_tau_investment, roi_bandwidth_investment

    txt_path = output_path.replace(".png", ".txt")
    lines = [
        "ROI COMPARISON: τ-Reduction vs Bandwidth Upgrade",
        "=" * 60,
        "",
    ]

    for delay_s in delays:
        delay_min = delay_s / 60
        lines.append(f"Delay: {delay_min:.0f} min")
        lines.append("-" * 40)
        lines.append("Investment    τ-ROI      BW-ROI     Winner")

        for inv in investment_range:
            roi_t = roi_tau_investment(inv, tau_base, bw_base, delay_s)
            roi_b = roi_bandwidth_investment(inv, bw_base, tau_base, delay_s)
            winner = "τ" if roi_t > roi_b else "BW"
            lines.append(f"${inv:>6.0f}M    {roi_t:>8.2f}  {roi_b:>8.2f}     {winner}")
        lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def plot_crew_vs_tau(
    tau_range: Tuple[float, float], bw_mbps: float, delay_s: float, output_path: str
) -> None:
    """Generate crew threshold vs τ plot.

    Shows how threshold drops from ~500 → ~180 → ~55 as τ decreases
    (better autonomy).

    Args:
        tau_range: (tau_min, tau_max) in seconds
        bw_mbps: Bandwidth in Mbps
        delay_s: One-way light delay in seconds
        output_path: Path to save PNG

    Source: Grok Dec 16, 2025 - "potentially dropping threshold to 20-30 crew"
    """
    from .sovereignty import threshold_from_tau, TAU_BASE_CURRENT_S
    from .entropy_shannon import tau_cost

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
    except ImportError:
        _crew_text_fallback(tau_range, bw_mbps, delay_s, output_path)
        return

    tau_min, tau_max = tau_range
    tau_values = np.linspace(tau_min, tau_max, 50)

    # Calculate thresholds and costs
    thresholds = [threshold_from_tau(tau, bw_mbps, delay_s) for tau in tau_values]
    costs = [tau_cost(tau, TAU_BASE_CURRENT_S) for tau in tau_values]

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot threshold (left axis)
    line1 = ax1.plot(
        tau_values, thresholds, "b-", linewidth=2.5, label="Crew Threshold"
    )
    ax1.set_xlabel("τ (Decision Latency, seconds)", fontsize=12)
    ax1.set_ylabel("Crew Threshold", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot cost (right axis)
    line2 = ax2.plot(
        tau_values, costs, "r--", linewidth=2, label="Investment Cost ($M)"
    )
    ax2.set_ylabel("Investment Cost ($M)", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Add key points
    key_taus = [300, 100, 30]
    for tau in key_taus:
        if tau_min <= tau <= tau_max:
            t = threshold_from_tau(tau, bw_mbps, delay_s)
            c = tau_cost(tau, TAU_BASE_CURRENT_S)
            ax1.axvline(x=tau, color="gray", linestyle=":", alpha=0.5)
            ax1.annotate(
                f"τ={tau}s\n{t} crew\n${c:.0f}M",
                xy=(tau, t),
                xytext=(tau + 20, t + 50),
                fontsize=9,
                ha="left",
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
            )

    ax1.set_title(
        f"CREW THRESHOLD vs DECISION LATENCY τ\n"
        f"(Delay: {delay_s / 60:.0f} min, Bandwidth: {bw_mbps} Mbps)",
        fontsize=14,
        fontweight="bold",
    )

    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Lower τ (better) on the right

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _crew_text_fallback(
    tau_range: Tuple[float, float], bw_mbps: float, delay_s: float, output_path: str
) -> None:
    """Text fallback for crew vs tau plot."""
    from .sovereignty import threshold_from_tau
    from .entropy_shannon import tau_cost, TAU_BASE_CURRENT_S

    txt_path = output_path.replace(".png", ".txt")
    tau_min, tau_max = tau_range

    lines = [
        "CREW THRESHOLD vs DECISION LATENCY τ",
        f"Delay: {delay_s / 60:.0f} min, Bandwidth: {bw_mbps} Mbps",
        "=" * 50,
        "",
        "τ (s)    Threshold    Cost ($M)",
        "-" * 50,
    ]

    for tau in [tau_max, 200, 150, 100, 75, 50, tau_min]:
        if tau_min <= tau <= tau_max:
            t = threshold_from_tau(tau, bw_mbps, delay_s)
            c = tau_cost(tau, TAU_BASE_CURRENT_S)
            lines.append(f"{tau:>6.0f}    {t:>8d}    {c:>10.0f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def plot_investment_decision(
    budget_m: float, bw_base: float, tau_base: float, delay_s: float, output_path: str
) -> None:
    """Generate investment decision quadrant chart.

    Shows when to invest in τ vs bandwidth given budget and delay.

    Args:
        budget_m: Available budget in millions USD
        bw_base: Current bandwidth in Mbps
        tau_base: Current τ (decision latency) in seconds
        delay_s: One-way light delay in seconds
        output_path: Path to save PNG

    Output:
        Decision quadrant showing optimal investment strategy
    """
    from .sovereignty import (
        compare_investment_roi,
        find_breakeven_delay,
        crew_reduction_from_autonomy,
    )

    pass  # Imports removed - unused

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
    except ImportError:
        return  # Skip if matplotlib unavailable

    # Calculate key metrics
    breakeven = find_breakeven_delay(bw_base, tau_base)
    comparison = compare_investment_roi(budget_m, bw_base, tau_base, delay_s)
    crew_impact = crew_reduction_from_autonomy(budget_m, bw_base, delay_s, tau_base)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create delay range for analysis
    delays = np.linspace(60, 1800, 100)  # 1 min to 30 min

    # Calculate ROI ratio at each delay
    roi_ratios = []
    for d in delays:
        comp = compare_investment_roi(budget_m, bw_base, tau_base, d)
        if comp["roi_bw"] > 0:
            ratio = comp["roi_tau"] / comp["roi_bw"]
        else:
            ratio = 100  # Very high if bandwidth ROI is 0
        roi_ratios.append(ratio)

    # Plot
    ax.fill_between(
        delays / 60,
        1,
        roi_ratios,
        where=[r > 1 for r in roi_ratios],
        alpha=0.3,
        color="green",
        label="Invest in AUTONOMY (τ)",
    )
    ax.fill_between(
        delays / 60,
        roi_ratios,
        1,
        where=[r <= 1 for r in roi_ratios],
        alpha=0.3,
        color="red",
        label="Invest in BANDWIDTH",
    )
    ax.plot(delays / 60, roi_ratios, "b-", linewidth=2)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1.5, label="Breakeven")
    ax.axvline(x=breakeven / 60, color="orange", linestyle=":", linewidth=2)

    # Add Mars range annotation
    ax.axvspan(3, 22, alpha=0.1, color="purple", label="Mars Range (3-22 min)")

    # Annotate breakeven
    ax.annotate(
        f"BREAKEVEN\n{breakeven / 60:.1f} min",
        xy=(breakeven / 60, 1),
        xytext=(breakeven / 60 + 2, 3),
        fontsize=11,
        fontweight="bold",
        color="orange",
        arrowprops=dict(arrowstyle="->", color="orange"),
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add current scenario annotation
    ax.annotate(
        f"At {delay_s / 60:.0f} min delay:\n"
        f"τ ROI = {comparison['roi_tau']:.1f}\n"
        f"BW ROI = {comparison['roi_bw']:.1f}\n"
        f"→ {comparison['winner'].upper()} wins\n"
        f"Crew: {crew_impact['before']}→{crew_impact['after']}",
        xy=(delay_s / 60, comparison["ratio"]),
        xytext=(0.02, 0.98),
        textcoords="axes fraction",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_xlabel("Light Delay (minutes)", fontsize=12)
    ax.set_ylabel("τ-ROI / Bandwidth-ROI Ratio", fontsize=12)
    ax.set_title(
        f"INVESTMENT DECISION: Budget ${budget_m:.0f}M\n"
        f'"Don\'t buy more bandwidth for Mars. Buy better decisions."',
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.set_xlim(0, 30)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def annotate_breakeven(ax, breakeven_delay: float) -> None:
    """Helper function to annotate breakeven point on a plot.

    Args:
        ax: Matplotlib axis object
        breakeven_delay: Breakeven delay in seconds
    """
    ax.axvline(x=breakeven_delay / 60, color="orange", linestyle="--", linewidth=2)
    ax.annotate(
        f"BREAKEVEN: {breakeven_delay / 60:.1f} min",
        xy=(breakeven_delay / 60, ax.get_ylim()[1] * 0.9),
        fontsize=10,
        fontweight="bold",
        color="orange",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def emit_roi_visualization_receipt(output_path: str) -> dict:
    """Emit receipt for ROI visualization.

    MUST emit receipt per CLAUDEME.
    """
    return emit_receipt(
        "roi_visualization",
        {
            "tenant_id": "axiom-core",
            "output_path": output_path,
            "finding": "autonomy_roi_dominates_at_mars",
        },
    )
