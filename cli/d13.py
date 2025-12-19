"""D13 recursion CLI commands.

Commands for D13 fractal recursion operations targeting alpha >= 3.70.
"""

from argparse import Namespace
from typing import Any, Dict


def cmd_d13_info(args: Namespace) -> Dict[str, Any]:
    """Show D13 configuration and status.

    Args:
        args: CLI arguments

    Returns:
        Dict with D13 info
    """
    from src.fractal_layers import get_d13_info

    info = get_d13_info()

    print("\n=== D13 RECURSION INFO ===")
    print(f"Version: {info.get('version', '1.0.0')}")
    print("\nD13 Config:")
    d13_config = info.get("d13_config", {})
    print(f"  Recursion depth: {d13_config.get('recursion_depth', 13)}")
    print(f"  Alpha floor: {d13_config.get('alpha_floor', 3.68)}")
    print(f"  Alpha target: {d13_config.get('alpha_target', 3.70)}")
    print(f"  Alpha ceiling: {d13_config.get('alpha_ceiling', 3.72)}")
    print(f"  Uplift: {d13_config.get('uplift', 0.32)}")

    print("\nSolar Hub Config:")
    hub_config = info.get("solar_hub_config", {})
    print(f"  Planets: {hub_config.get('planets', [])}")
    print(f"  Autonomy target: {hub_config.get('autonomy_target', 0.95)}")

    print("\nLES Config:")
    les_config = info.get("les_config", {})
    print(f"  Model: {les_config.get('model', 'large_eddy_simulation')}")
    print(f"  Subgrid model: {les_config.get('subgrid_model', 'smagorinsky')}")

    print("\nZK Config:")
    zk_config = info.get("zk_config", {})
    print(f"  Proof system: {zk_config.get('proof_system', 'groth16')}")
    print(f"  Resilience target: {zk_config.get('resilience_target', 1.0)}")

    return info


def cmd_d13_push(args: Namespace) -> Dict[str, Any]:
    """Run D13 recursion push for alpha >= 3.70.

    Args:
        args: CLI arguments (may include --simulate)

    Returns:
        Dict with push results
    """
    from src.fractal_layers import d13_push, D13_TREE_MIN

    simulate = getattr(args, "simulate", False)
    tree_size = getattr(args, "tree_size", D13_TREE_MIN)
    base_alpha = getattr(args, "base_alpha", 3.38)

    result = d13_push(tree_size=tree_size, base_alpha=base_alpha, simulate=simulate)

    print("\n=== D13 PUSH RESULTS ===")
    print(f"Mode: {result['mode']}")
    print(f"Tree size: {result['tree_size']:,}")
    print(f"Base alpha: {result['base_alpha']}")
    print(f"Depth: {result['depth']}")
    print(f"\nEffective alpha: {result['eff_alpha']}")
    print(f"Instability: {result['instability']}")
    print("\nTargets:")
    print(f"  Floor met (>= 3.68): {result['floor_met']}")
    print(f"  Target met (>= 3.70): {result['target_met']}")
    print(f"  Ceiling met (>= 3.72): {result['ceiling_met']}")
    print(f"\nSLO passed: {result['slo_passed']}")
    print(f"Gate: {result['gate']}")

    return result


def cmd_d13_solar_hybrid(args: Namespace) -> Dict[str, Any]:
    """Run integrated D13 + Solar hub hybrid.

    Args:
        args: CLI arguments

    Returns:
        Dict with hybrid results
    """
    from src.solar_orbital_hub import d13_solar_hybrid

    simulate = getattr(args, "simulate", False)
    tree_size = getattr(args, "tree_size", 10**12)
    base_alpha = getattr(args, "base_alpha", 3.38)

    result = d13_solar_hybrid(
        tree_size=tree_size, base_alpha=base_alpha, simulate=simulate
    )

    print("\n=== D13 + SOLAR HUB HYBRID ===")
    print(f"Mode: {result['mode']}")

    print("\nD13 Result:")
    d13 = result.get("d13_result", {})
    print(f"  Eff alpha: {d13.get('eff_alpha', 0)}")
    print(f"  Floor met: {d13.get('floor_met', False)}")
    print(f"  Target met: {d13.get('target_met', False)}")

    print("\nSolar Hub Result:")
    hub = result.get("hub_result", {})
    print(f"  Autonomy: {hub.get('autonomy', 0)}")
    print(f"  Sync cycles: {hub.get('sync_cycles', 0)}")
    print(f"  Hub operational: {hub.get('hub_operational', False)}")

    print(f"\nCombined autonomy: {result.get('combined_autonomy', 0)}")
    print(f"Integration status: {result.get('integration_status', 'unknown')}")
    print(f"Gate: {result.get('gate', 't24h')}")

    return result
