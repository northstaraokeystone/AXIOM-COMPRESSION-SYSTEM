"""D14 fractal recursion CLI commands.

Commands for D14 depth-14 fractal recursion operations.
"""

from argparse import Namespace
from typing import Any, Dict


def cmd_d14_info(args: Namespace) -> Dict[str, Any]:
    """Show D14 configuration.

    Args:
        args: CLI arguments

    Returns:
        Dict with D14 info
    """
    from src.fractal_layers import get_d14_info

    info = get_d14_info()

    print("\n=== D14 FRACTAL RECURSION CONFIGURATION ===")
    print(f"Version: {info.get('version', '1.0.0')}")

    d14_config = info.get("d14_config", {})
    print("\nD14 Recursion:")
    print(f"  Recursion depth: {d14_config.get('recursion_depth', 14)}")
    print(f"  Alpha floor: {d14_config.get('alpha_floor', 3.73)}")
    print(f"  Alpha target: {d14_config.get('alpha_target', 3.75)}")
    print(f"  Alpha ceiling: {d14_config.get('alpha_ceiling', 3.77)}")
    print(f"  Uplift: {d14_config.get('uplift', 0.34)}")
    print(f"  Adaptive termination: {d14_config.get('adaptive_termination', True)}")

    interstellar = info.get("interstellar_config", {})
    print("\nInterstellar Backbone:")
    print(f"  Body count: {interstellar.get('body_count', 7)}")
    print(f"  Autonomy target: {interstellar.get('autonomy_target', 0.98)}")

    plonk = info.get("plonk_config", {})
    print("\nPLONK ZK:")
    print(f"  Proof system: {plonk.get('proof_system', 'plonk')}")
    print(f"  Universal setup: {plonk.get('universal_setup', True)}")

    print(f"\nDescription: {info.get('description', '')}")

    return info


def cmd_d14_push(args: Namespace) -> Dict[str, Any]:
    """Run D14 recursion push for alpha >= 3.75.

    Args:
        args: CLI arguments

    Returns:
        Dict with D14 push results
    """
    from src.fractal_layers import d14_push, D14_TREE_MIN

    tree_size = getattr(args, "tree_size", D14_TREE_MIN)
    base_alpha = getattr(args, "base_alpha", 3.41)
    simulate = getattr(args, "simulate", False)
    adaptive = getattr(args, "adaptive", True)

    result = d14_push(
        tree_size=tree_size,
        base_alpha=base_alpha,
        simulate=simulate,
        adaptive=adaptive,
    )

    print("\n=== D14 RECURSION PUSH ===")
    print(f"Mode: {result.get('mode', 'execute')}")
    print(f"Tree size: {result.get('tree_size', 0):,}")
    print(f"Base alpha: {result.get('base_alpha', 0)}")
    print(f"Depth: {result.get('depth', 14)}")
    print(f"Adaptive: {result.get('adaptive', True)}")

    print("\nResults:")
    print(f"  Effective alpha: {result.get('eff_alpha', 0)}")
    print(f"  Instability: {result.get('instability', 0)}")

    print("\nTargets:")
    print(f"  Floor met (>= 3.73): {result.get('floor_met', False)}")
    print(f"  Target met (>= 3.75): {result.get('target_met', False)}")
    print(f"  Ceiling met (>= 3.77): {result.get('ceiling_met', False)}")

    print(f"\nSLO Passed: {result.get('slo_passed', False)}")
    print(f"Gate: {result.get('gate', 't24h')}")

    return result


def cmd_d14_interstellar_hybrid(args: Namespace) -> Dict[str, Any]:
    """Run integrated D14 + interstellar backbone.

    Args:
        args: CLI arguments

    Returns:
        Dict with hybrid results
    """
    from src.interstellar_backbone import d14_interstellar_hybrid

    tree_size = getattr(args, "tree_size", 10**12)
    base_alpha = getattr(args, "base_alpha", 3.41)
    simulate = getattr(args, "simulate", False)

    result = d14_interstellar_hybrid(
        tree_size=tree_size,
        base_alpha=base_alpha,
        simulate=simulate,
    )

    print("\n=== D14 + INTERSTELLAR HYBRID ===")
    print(f"Mode: {result.get('mode', 'execute')}")

    d14 = result.get("d14_result", {})
    print("\nD14 Fractal:")
    print(f"  Effective alpha: {d14.get('eff_alpha', 0)}")
    print(f"  Floor met: {d14.get('floor_met', False)}")
    print(f"  Target met: {d14.get('target_met', False)}")

    backbone = result.get("backbone_result", {})
    print("\nInterstellar Backbone:")
    print(f"  Sync cycles: {backbone.get('sync_cycles', 0)}")
    print(f"  Autonomy: {backbone.get('autonomy', 0):.2%}")
    print(f"  Target met: {backbone.get('target_met', False)}")

    print(f"\nCombined alpha: {result.get('combined_alpha', 0)}")
    print(f"Combined autonomy: {result.get('combined_autonomy', 0):.2%}")
    print(f"Integration status: {result.get('integration_status', 'unknown')}")
    print(f"Gate: {result.get('gate', 't24h')}")

    return result
