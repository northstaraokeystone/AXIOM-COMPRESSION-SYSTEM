"""Mars path CLI commands.

Commands:
- cmd_mars_status: Show Mars path status
- cmd_mars_simulate: Run dome simulation
- cmd_mars_isru: Show ISRU metrics
- cmd_mars_sovereignty: Check sovereignty threshold

Source: SpaceProof scalable paths architecture - Mars autonomous habitat
"""

import json
from typing import Dict, Any, Optional

from .core import (
    stub_status,
    simulate_dome,
    compute_isru_closure,
    compute_sovereignty,
    get_mars_info,
)


def cmd_mars_status(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Show Mars path status.

    Args:
        args: Optional arguments (unused)

    Returns:
        Status dict
    """
    status = stub_status()

    print("=" * 60)
    print("MARS PATH STATUS")
    print("=" * 60)
    print(f"Ready: {status['ready']}")
    print(f"Stage: {status['stage']}")
    print(f"Version: {status['version']}")
    print(f"\nEvolution path: {' -> '.join(status['evolution_path'])}")
    print("\nCurrent capabilities:")
    for cap in status.get("current_capabilities", []):
        print(f"  - {cap}")
    print("\nPending capabilities:")
    for cap in status.get("pending_capabilities", []):
        print(f"  - {cap}")

    return status


def cmd_mars_simulate(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run dome simulation.

    Args:
        args: Optional arguments:
            - crew: Number of crew (default: 50)
            - duration: Duration in days (default: 365)

    Returns:
        Simulation results
    """
    if args is None:
        args = {}

    crew = args.get("crew", 50)
    duration = args.get("duration", 365)

    result = simulate_dome(crew=crew, duration_days=duration)

    print("=" * 60)
    print("MARS DOME SIMULATION (STUB)")
    print("=" * 60)
    print(f"Crew: {crew}")
    print(f"Duration: {duration} days")
    print("\nResources Required:")
    for resource, amount in result["resources_required"].items():
        print(f"  {resource}: {amount:,.1f}")
    print("\nISRU Projected:")
    for resource, amount in result["isru_projected"].items():
        print(f"  {resource}: {amount:,.1f}")
    print(f"\nISRU Closure: {result['isru_closure_projected']:.1%}")
    print("\n[STUB MODE - Full simulation pending]")

    return result


def cmd_mars_isru(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Show ISRU metrics.

    Args:
        args: Optional arguments with resource data

    Returns:
        ISRU calculation results
    """
    if args is None:
        args = {}

    # Default resource values (stub)
    resources = args.get(
        "resources",
        {"water": (75, 100), "o2": (90, 100), "power": (95, 100), "food": (30, 100)},
    )

    closure = compute_isru_closure(resources)

    print("=" * 60)
    print("MARS ISRU METRICS")
    print("=" * 60)
    print(f"Closure Ratio: {closure:.1%}")
    print("Target: 85%")
    print(f"Gap: {0.85 - closure:.1%}")
    print(f"Target Met: {'YES' if closure >= 0.85 else 'NO'}")
    print("\nResource Breakdown:")
    for resource, values in resources.items():
        if isinstance(values, (list, tuple)):
            local, total = values
        else:
            local, total = values.get("local", 0), values.get("required", 0)
        pct = local / total * 100 if total > 0 else 0
        print(f"  {resource}: {local}/{total} ({pct:.0f}%)")

    return {"closure": closure, "resources": resources}


def cmd_mars_sovereignty(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check sovereignty threshold.

    Args:
        args: Optional arguments:
            - crew: Number of crew (default: 50)
            - bandwidth: Bandwidth in Mbps (default: 100)
            - latency: Latency in seconds (default: 1200)

    Returns:
        Sovereignty check results
    """
    if args is None:
        args = {}

    crew = args.get("crew", 50)
    bandwidth = args.get("bandwidth", 100.0)
    latency = args.get("latency", 1200.0)

    is_sovereign = compute_sovereignty(
        crew=crew, bandwidth_mbps=bandwidth, latency_s=latency
    )

    print("=" * 60)
    print("MARS SOVEREIGNTY CHECK")
    print("=" * 60)
    print(f"Crew: {crew}")
    print(f"Bandwidth: {bandwidth} Mbps")
    print(f"Latency: {latency}s (one-way)")
    print(f"\nInternal Rate: {crew * 1000} bps")
    effective_ext = (bandwidth * 1_000_000) / (latency * 2)
    print(f"External Rate: {effective_ext:.0f} bps")
    print(f"\nSOVEREIGN: {'YES' if is_sovereign else 'NO'}")
    print(f"Advantage Ratio: {crew * 1000 / effective_ext:.2f}x")

    return {"is_sovereign": is_sovereign, "crew": crew}


def cmd_mars_info(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Show Mars path configuration.

    Args:
        args: Optional arguments (unused)

    Returns:
        Path info dict
    """
    info = get_mars_info()

    print("=" * 60)
    print("MARS PATH INFO")
    print("=" * 60)
    print(f"Version: {info['version']}")
    print(f"Status: {info['status']}")
    print(f"Description: {info['description']}")
    print("\nConfig:")
    print(json.dumps(info["config"], indent=2))
    print(f"\nDependencies: {info['dependencies']}")
    print(f"Receipts: {info['receipts']}")

    return info
