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
    crew_range: Tuple[int, int],
    bandwidth_mbps: float = 4.0,
    delay_s: float = 480.0
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
            crew=crew,
            compute_flops=0.0,
            bandwidth_mbps=bandwidth_mbps,
            delay_s=delay_s
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
    uncertainty: Optional[float] = None
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
        matplotlib.use('Agg')  # Non-interactive backend
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
    ax.plot(crews, advantages, 'b-', linewidth=2, label='Sovereignty Advantage')

    # Horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='Threshold')

    # Vertical line at knee
    ax.axvline(x=knee, color='red', linestyle='--', linewidth=1.5)

    # Shade regions
    ax.fill_between(crews, advantages, 0,
                    where=[a > 0 for a in advantages],
                    alpha=0.3, color='green', label='SOVEREIGN')
    ax.fill_between(crews, advantages, 0,
                    where=[a <= 0 for a in advantages],
                    alpha=0.3, color='red', label='DEPENDENT')

    # Annotate threshold
    uncertainty_str = f" +/- {uncertainty:.0f}" if uncertainty else ""
    ax.annotate(
        f'THRESHOLD: {knee}{uncertainty_str} crew',
        xy=(knee, 0),
        xytext=(knee + 10, max(advantages) * 0.3),
        fontsize=12,
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='red')
    )

    # Labels and title
    ax.set_xlabel('Crew Count', fontsize=12)
    ax.set_ylabel('Sovereignty Advantage (bits/sec)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='upper left')

    # Grid
    ax.grid(True, alpha=0.3)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_text_fallback(
    curve_data: List[Tuple[int, float]],
    knee: int,
    output_path: str,
    uncertainty: Optional[float] = None
) -> None:
    """Create text representation when matplotlib unavailable."""
    # Change extension to .txt
    txt_path = output_path.replace('.png', '.txt')

    uncertainty_str = f" +/- {uncertainty:.0f}" if uncertainty else ""

    lines = [
        "SOVEREIGNTY CURVE: Mars Colony",
        "=" * 40,
        "",
        "Crew    Advantage    Status",
        "-" * 40
    ]

    for crew, advantage in curve_data:
        status = "SOVEREIGN" if advantage > 0 else "DEPENDENT"
        marker = " <-- THRESHOLD" if crew == knee else ""
        lines.append(f"{crew:4d}    {advantage:+10.2f}    {status}{marker}")

    lines.extend([
        "",
        "-" * 40,
        f"THRESHOLD: {knee}{uncertainty_str} crew",
        "=" * 40
    ])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))


def format_finding(
    knee: int,
    bandwidth: float,
    delay: float,
    uncertainty: float
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
    return emit_receipt("sovereignty_curve", {
        "tenant_id": "axiom-core",
        "threshold_crew": knee,
        "uncertainty": uncertainty,
        "output_path": output_path
    })
