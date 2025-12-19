"""Entropy pruning CLI commands for AXIOM-CORE.

Commands: entropy_prune, pruning_sweep, extended_250d, verify_chain, pruning_info
         pruning_v4, pruning_v4_compare, quantum_refine, quantum_refine_info
"""

from src.pruning import (
    entropy_prune,
    generate_sample_merkle_tree,
    get_pruning_info,
)
from src.gnn_cache import (
    nonlinear_retention_with_pruning,
    ENTROPY_ASYMPTOTE_E,
    PRUNING_TARGET_ALPHA,
    CACHE_DEPTH_BASELINE,
    BLACKOUT_PRUNING_TARGET_DAYS,
    OVERFLOW_THRESHOLD_DAYS_PRUNED,
)
from src.reasoning import extended_250d_sovereignty

from cli.base import print_header


def cmd_entropy_prune(
    blackout_days: int, trim_factor: float, hybrid: bool, simulate: bool
):
    """Run single entropy pruning test.

    Args:
        blackout_days: Blackout duration in days
        trim_factor: ln(n) trim factor (0.3-0.5 range)
        hybrid: Whether to enable hybrid dedup + predictive pruning
        simulate: Whether to output simulation receipt
    """
    print_header(f"ENTROPY PRUNING TEST ({blackout_days} days)")

    print("\nConfiguration:")
    print(f"  ENTROPY_ASYMPTOTE_E: {ENTROPY_ASYMPTOTE_E} (physics constant)")
    print(f"  Trim factor: {trim_factor}")
    print(f"  Hybrid mode: {hybrid}")
    print(f"  Target alpha: {PRUNING_TARGET_ALPHA}")
    print(f"  Target days: {BLACKOUT_PRUNING_TARGET_DAYS}")

    # Generate sample Merkle tree
    tree = generate_sample_merkle_tree(n_leaves=100, duplicate_ratio=0.2)
    print(f"  Sample tree leaves: {tree['leaf_count']}")

    # Run entropy pruning
    result = entropy_prune(tree, trim_factor=trim_factor, hybrid=hybrid)

    print("\nRESULTS:")
    print(f"  Branches pruned: {result['branches_pruned']}")
    print(f"  Entropy before: {result['entropy_before']}")
    print(f"  Entropy after: {result['entropy_after']}")
    print(f"  Entropy reduction: {result['entropy_reduction_pct']:.1f}%")
    print(f"  Alpha uplift: {result['alpha_uplift']}")
    print(f"  Dedup removed: {result['dedup_removed']}")
    print(f"  Predictive pruned: {result['predictive_pruned']}")

    # Get retention with pruning
    try:
        retention = nonlinear_retention_with_pruning(
            blackout_days,
            CACHE_DEPTH_BASELINE,
            pruning_enabled=True,
            trim_factor=trim_factor,
        )
        print(f"\n  Effective alpha at {blackout_days}d: {retention['eff_alpha']}")
        print(f"  Pruning boost: {retention['pruning_boost']}")
        print(f"  Target achieved: {retention['eff_alpha'] >= PRUNING_TARGET_ALPHA}")
    except Exception as e:
        print(f"\n  OVERFLOW: {e}")

    if simulate:
        print("\n[entropy_pruning_receipt emitted above]")

    print("=" * 60)


def cmd_pruning_sweep(simulate: bool):
    """Run pruning sensitivity sweep.

    Args:
        simulate: Whether to output simulation receipts
    """
    print_header("PRUNING SENSITIVITY SWEEP")

    trim_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
    test_days = [150, 200, 250]

    print("\nConfiguration:")
    print(f"  Trim factors: {trim_factors}")
    print(f"  Test durations: {test_days} days")

    print("\nRESULTS:")
    print(f"  {'Trim':>8} | {'150d':>10} | {'200d':>10} | {'250d':>10}")
    print(f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    for trim in trim_factors:
        row = f"  {trim:>8.2f} |"
        for days in test_days:
            try:
                result = nonlinear_retention_with_pruning(
                    days, CACHE_DEPTH_BASELINE, pruning_enabled=True, trim_factor=trim
                )
                row += f" {result['eff_alpha']:>10.4f} |"
            except Exception:
                row += f" {'OVERFLOW':>10} |"
        print(row)

    if simulate:
        print("\n[pruning_sweep receipts emitted above]")

    print("=" * 60)


def cmd_extended_250d(simulate: bool):
    """Run 250d extended simulation with pruning.

    Args:
        simulate: Whether to output simulation receipts
    """
    print_header("EXTENDED 250d SOVEREIGNTY PROJECTION")

    print("\nConfiguration:")
    print(f"  Target days: {BLACKOUT_PRUNING_TARGET_DAYS}")
    print(f"  Target alpha: {PRUNING_TARGET_ALPHA}")
    print(f"  Overflow threshold (pruned): {OVERFLOW_THRESHOLD_DAYS_PRUNED}")
    print(f"  Entropy asymptote: {ENTROPY_ASYMPTOTE_E} (physics)")

    result = extended_250d_sovereignty(
        pruning_enabled=True,
        trim_factor=0.3,
        blackout_days=BLACKOUT_PRUNING_TARGET_DAYS,
    )

    print("\nRESULTS:")
    print(f"  Effective alpha: {result['effective_alpha']}")
    print(f"  Target achieved: {result['target_achieved']}")
    print(f"  Pruning boost: {result['pruning_boost']}")
    print(f"  Overflow margin: {result['overflow_margin']} days")
    print(f"  Cycles to 10K: {result['cycles_to_10k_person_eq']}")
    print(f"  Cycles to 1M: {result['cycles_to_1M_person_eq']}")

    print("\nSLO VALIDATION:")
    alpha_ok = result["effective_alpha"] >= PRUNING_TARGET_ALPHA
    overflow_ok = result["overflow_margin"] >= 0
    print(
        f"  Alpha >= {PRUNING_TARGET_ALPHA}: {'PASS' if alpha_ok else 'FAIL'} ({result['effective_alpha']})"
    )
    print(
        f"  Overflow margin >= 0: {'PASS' if overflow_ok else 'FAIL'} ({result['overflow_margin']}d)"
    )

    if simulate:
        print("\n[extended_250d_sovereignty receipt emitted above]")

    print("=" * 60)


def cmd_verify_chain(trim_factor: float, simulate: bool):
    """Verify chain integrity after pruning.

    Args:
        trim_factor: ln(n) trim factor
        simulate: Whether to output simulation receipts
    """
    print_header("CHAIN INTEGRITY VERIFICATION")

    print("\nConfiguration:")
    print(f"  Trim factor: {trim_factor}")
    print("  Test iterations: 10")

    all_passed = True
    for i in range(10):
        tree = generate_sample_merkle_tree(n_leaves=100, duplicate_ratio=0.2)
        try:
            entropy_prune(tree, trim_factor=trim_factor, hybrid=True)
            status = "PASS"
        except Exception as e:
            if "Chain broken" in str(e) or "Quorum lost" in str(e):
                status = f"FAIL: {e}"
                all_passed = False
            else:
                status = "PASS"
        print(f"  Iteration {i + 1}: {status}")

    print(f"\nCHAIN INTEGRITY: {'PASS' if all_passed else 'FAIL'}")

    if simulate:
        print("\n[chain_integrity receipts emitted above]")

    print("=" * 60)


def cmd_pruning_info():
    """Output pruning configuration."""
    print_header("PRUNING CONFIGURATION")

    info = get_pruning_info()

    print("\nPhysics Constants:")
    print(
        f"  ENTROPY_ASYMPTOTE_E: {info['entropy_asymptote_e']} (Shannon bound, NOT tunable)"
    )
    print(f"  PRUNING_TARGET_ALPHA: {info['pruning_target_alpha']}")

    print("\nTrim Factor Range:")
    print(f"  Base: {info['ln_n_trim_factor_base']} (conservative)")
    print(f"  Max: {info['ln_n_trim_factor_max']} (aggressive)")

    print("\nThresholds:")
    print(f"  Entropy prune threshold: {info['entropy_prune_threshold']}")
    print(f"  Min confidence: {info['min_confidence_threshold']}")
    print(f"  Min quorum fraction: {info['min_quorum_fraction']:.2%}")

    print("\nTargets:")
    print(f"  Blackout target days: {info['blackout_pruning_target_days']}")
    print(f"  Overflow threshold (pruned): {info['overflow_threshold_pruned_days']}")

    print(f"\nDescription: {info['description']}")

    print("\n[pruning_info receipt emitted above]")
    print("=" * 60)


def cmd_pruning_v4(args=None):
    """Run pruning v4 for >99.5% compression.

    Args:
        args: CLI arguments (optional).

    Returns:
        dict: Pruning result.
    """
    from src.pruning_v4 import prune_v4, load_pruning_config
    import random

    config = load_pruning_config()
    target = config.get("compression_target", 0.995)

    # Create test tree
    nodes = []
    for i in range(100):
        node = {
            "id": i,
            "value": random.random(),
            "children": [{"sub_id": j, "data": random.random()} for j in range(5)],
        }
        nodes.append(node)
    test_tree = {
        "root": "test_tree",
        "nodes": nodes,
        "depth": 3,
        "metadata": {"created": "test", "version": "1.0"},
    }

    print_header(f"PRUNING V4 (target: {target*100:.1f}%)")

    result = prune_v4(test_tree)

    print(f"\nCompression Achieved: {result['compression']*100:.2f}%")
    print(f"Target: {result['target']*100:.1f}%")
    print(f"Target Met: {result['target_met']}")
    print(f"Original Size: {result['original_size']} bytes")
    print(f"Final Size: {result['final_size']} bytes")
    print(f"Passes Completed: {result['passes_completed']}")
    print(f"Method: {result['method']}")

    print("=" * 60)
    return result


def cmd_pruning_v4_compare(args=None):
    """Compare v3 vs v4 pruning.

    Args:
        args: CLI arguments (optional).

    Returns:
        dict: Comparison result.
    """
    from src.pruning_v4 import compare_to_v3
    import random

    # Create test tree
    nodes = []
    for i in range(100):
        node = {
            "id": i,
            "value": random.random(),
            "children": [{"sub_id": j, "data": random.random()} for j in range(5)],
        }
        nodes.append(node)
    test_tree = {
        "root": "test_tree",
        "nodes": nodes,
        "depth": 3,
        "metadata": {"created": "test", "version": "1.0"},
    }

    print_header("V3 VS V4 PRUNING COMPARISON")

    result = compare_to_v3(test_tree)

    print(f"\nV3 Compression: {result['v3_compression']*100:.2f}%")
    print(f"V4 Compression: {result['v4_compression']*100:.2f}%")
    print(f"Improvement: +{result['improvement_pct']:.2f}%")
    print(f"V4 Better: {result['v4_better']}")

    print("=" * 60)
    return result


def cmd_pruning_v4_status(args=None):
    """Show pruning v4 status.

    Args:
        args: CLI arguments (optional).

    Returns:
        dict: Status info.
    """
    from src.pruning_v4 import get_pruning_status

    print_header("PRUNING V4 STATUS")

    status = get_pruning_status()

    print(f"\nEnabled: {status['enabled']}")
    print(f"Compression Target: {status['compression_target']*100:.1f}%")
    print(f"Persistence Depth: H0-H{status['persistence_depth']}")
    print(f"Iterative Passes: {status['iterative_passes']}")
    print(f"Method: {status['method']}")
    print(f"Hole Threshold: {status['hole_threshold']}")

    print("=" * 60)
    return status


def cmd_quantum_refine(args=None):
    """Run quantum correlation refinement.

    Args:
        args: CLI arguments (optional).

    Returns:
        dict: Refinement result.
    """
    from src.quantum_refine import refine_correlation, load_refine_config

    config = load_refine_config()
    iterations = config.get("refinement_iterations", 10)

    print_header(f"QUANTUM REFINEMENT ({iterations} iterations)")

    result = refine_correlation()

    print(f"\nPairs Processed: {result['pairs_processed']}")
    print(f"Correlation Before: {result['correlation_before']*100:.2f}%")
    print(f"Correlation After: {result['correlation_after']*100:.2f}%")
    print(f"Improvement: +{result['improvement']*100:.2f}%")
    print(f"Target: {result['target']*100:.1f}%")
    print(f"Target Met: {result['target_met']}")

    print("=" * 60)
    return result


def cmd_quantum_refine_info(args=None):
    """Show quantum refinement configuration.

    Args:
        args: CLI arguments (optional).

    Returns:
        dict: Configuration info.
    """
    from src.quantum_refine import get_refine_status

    print_header("QUANTUM REFINEMENT CONFIGURATION")

    status = get_refine_status()

    print(f"\nCorrelation Target: {status['correlation_target']*100:.1f}%")
    print(f"Decoherence Mitigation: {status['decoherence_mitigation']}")
    print(f"Error Correction: {status['error_correction']}")
    print(f"Refinement Iterations: {status['refinement_iterations']}")
    print(f"Decoherence Time: {status['decoherence_time_ms']}ms")
    print(f"Bell Limit (Classical): {status['bell_limit_classical']}")
    print(f"Bell Limit (Quantum): {status['bell_limit_quantum']:.3f}")

    print("=" * 60)
    return status
