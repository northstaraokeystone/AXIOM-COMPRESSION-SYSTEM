"""cli/args.py - Argument definitions for AXIOM CLI.

Extracted from cli.py to keep main entry point under 600 lines.
"""

import argparse
from src.partition import NODE_BASELINE
from src.timeline import C_BASE_DEFAULT, P_FACTOR_DEFAULT


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all CLI options.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="AXIOM-CORE CLI - The Sovereignty Calculator"
    )
    parser.add_argument(
        "command", nargs="?", help="Command: baseline, bootstrap, curve, full"
    )

    # Timeline args
    _add_timeline_args(parser)

    # Partition flags
    _add_partition_args(parser)

    # Reroute/blackout flags
    _add_blackout_args(parser)

    # Extended sweep/retention flags
    _add_extended_args(parser)

    # GNN/cache flags
    _add_gnn_args(parser)

    # Pruning flags
    _add_pruning_args(parser)

    # Ablation flags
    _add_ablation_args(parser)

    # RL flags
    _add_rl_args(parser)

    # Adaptive depth flags
    _add_depth_args(parser)

    # 500-run sweep flags
    _add_sweep_args(parser)

    # Pipeline flags
    _add_pipeline_args(parser)

    # Scale flags
    _add_scale_args(parser)

    # Fractal ceiling breach flags
    _add_fractal_args(parser)

    # Full sweep and hybrid flags
    _add_hybrid_args(parser)

    # Benchmark flags
    _add_benchmark_args(parser)

    # Path exploration flags
    _add_path_args(parser)

    # ISRU/D5 flags
    _add_isru_args(parser)

    # Titan/D6 flags
    _add_titan_args(parser)

    # Adversarial audit flags
    _add_audit_args(parser)

    # Europa/D7 flags
    _add_europa_args(parser)

    # NREL flags
    _add_nrel_args(parser)

    # D8 flags
    _add_d8_args(parser)

    # D9 flags
    _add_d9_args(parser)

    # Drone flags
    _add_drone_args(parser)

    # Randomized paths flags
    _add_randomized_args(parser)

    # D10/Jovian flags
    _add_d10_args(parser)

    # Callisto flags
    _add_callisto_args(parser)

    # Quantum-resistant flags
    _add_quantum_resist_args(parser)

    # Dust dynamics flags
    _add_dust_args(parser)

    # D13 flags
    _add_d13_args(parser)

    # Solar hub flags
    _add_solar_args(parser)

    # LES flags
    _add_les_args(parser)

    # ZK flags
    _add_zk_args(parser)

    # D14 flags
    _add_d14_args(parser)

    # Interstellar backbone flags
    _add_interstellar_args(parser)

    # Atacama real-time flags
    _add_atacama_args(parser)

    # PLONK flags
    _add_plonk_args(parser)

    # D17 Heliosphere flags
    _add_d17_args(parser)

    # Heliosphere-Oort flags
    _add_heliosphere_args(parser)

    # D18 Interstellar relay flags
    _add_d18_args(parser)

    # Interstellar relay flags
    _add_relay_args(parser)

    # Quantum alternative flags
    _add_quantum_alt_args(parser)

    # Elon-sphere flags
    _add_elonsphere_args(parser)

    # Federation flags
    _add_federation_args(parser)

    return parser


def _add_timeline_args(parser: argparse.ArgumentParser) -> None:
    """Add timeline simulation arguments."""
    parser.add_argument(
        "--c_base",
        type=float,
        default=C_BASE_DEFAULT,
        help="Initial person-eq capacity (default: 50)",
    )
    parser.add_argument(
        "--p_factor",
        type=float,
        default=P_FACTOR_DEFAULT,
        help="Propulsion growth factor (default: 1.8)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0,
        help="Latency in seconds (0=Earth, 1200=Mars max)",
    )
    parser.add_argument(
        "--simulate_timeline",
        action="store_true",
        help="Run sovereignty timeline simulation",
    )


def _add_partition_args(parser: argparse.ArgumentParser) -> None:
    """Add partition simulation arguments."""
    parser.add_argument(
        "--partition",
        type=float,
        default=None,
        help="Run single partition simulation at specified loss percentage (0-1)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=NODE_BASELINE,
        help="Specify node count for simulation (default: 5)",
    )
    parser.add_argument(
        "--stress_quorum",
        action="store_true",
        help="Run full stress sweep (1000 iterations, 0-40%% loss)",
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Output simulation receipt to stdout"
    )


def _add_blackout_args(parser: argparse.ArgumentParser) -> None:
    """Add blackout/reroute arguments."""
    parser.add_argument(
        "--reroute", action="store_true", help="Enable adaptive rerouting"
    )
    parser.add_argument(
        "--reroute_enabled", action="store_true", help="Alias for --reroute"
    )
    parser.add_argument(
        "--blackout", type=int, default=None, help="Blackout duration in days"
    )
    parser.add_argument(
        "--blackout_sweep", action="store_true", help="Run full blackout sweep"
    )
    parser.add_argument(
        "--algo_info", action="store_true", help="Output reroute algorithm spec"
    )


def _add_extended_args(parser: argparse.ArgumentParser) -> None:
    """Add extended sweep arguments."""
    parser.add_argument(
        "--extended_sweep",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Extended blackout sweep range",
    )
    parser.add_argument(
        "--retention_curve", action="store_true", help="Output retention curve JSON"
    )
    parser.add_argument(
        "--gnn_stub", action="store_true", help="Echo GNN sensitivity stub"
    )


def _add_gnn_args(parser: argparse.ArgumentParser) -> None:
    """Add GNN/cache arguments."""
    parser.add_argument(
        "--gnn_nonlinear", action="store_true", help="Use GNN nonlinear model"
    )
    parser.add_argument(
        "--cache_depth", type=int, default=int(1e8), help="Cache depth (default: 1e8)"
    )
    parser.add_argument(
        "--cache_sweep", action="store_true", help="Run cache depth sweep"
    )
    parser.add_argument(
        "--extreme_sweep",
        type=int,
        default=None,
        metavar="DAYS",
        help="Run extreme blackout sweep to days",
    )
    parser.add_argument(
        "--overflow_test", action="store_true", help="Test cache overflow"
    )
    parser.add_argument(
        "--innovation_stubs", action="store_true", help="Echo innovation stubs"
    )


def _add_pruning_args(parser: argparse.ArgumentParser) -> None:
    """Add pruning arguments."""
    parser.add_argument(
        "--entropy_prune", action="store_true", help="Enable entropy pruning"
    )
    parser.add_argument(
        "--trim_factor", type=float, default=0.3, help="Trim factor (default: 0.3)"
    )
    parser.add_argument(
        "--hybrid_prune", action="store_true", help="Enable hybrid pruning"
    )
    parser.add_argument(
        "--pruning_sweep", action="store_true", help="Run pruning sweep"
    )
    parser.add_argument(
        "--extended_250d", action="store_true", help="Run 250d simulation"
    )
    parser.add_argument(
        "--verify_chain", action="store_true", help="Verify chain integrity"
    )
    parser.add_argument(
        "--pruning_info", action="store_true", help="Output pruning config"
    )


def _add_ablation_args(parser: argparse.ArgumentParser) -> None:
    """Add ablation arguments."""
    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        metavar="MODE",
        help="Ablation mode (full/no_cache/no_prune/baseline)",
    )
    parser.add_argument(
        "--ablation_sweep", action="store_true", help="Run ablation sweep"
    )
    parser.add_argument(
        "--ceiling_track",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Ceiling gap analysis for alpha",
    )
    parser.add_argument(
        "--formula_check", action="store_true", help="Validate alpha formula"
    )
    parser.add_argument(
        "--isolate_layers", action="store_true", help="Isolate layer contributions"
    )


def _add_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RL tuning arguments."""
    parser.add_argument("--rl_tune", action="store_true", help="Enable RL auto-tuning")
    parser.add_argument(
        "--rl_episodes", type=int, default=100, help="RL episodes (default: 100)"
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Enable adaptive depth/config"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable all dynamic features"
    )
    parser.add_argument("--tune_sweep", action="store_true", help="Run retention sweep")
    parser.add_argument(
        "--show_rl_history", action="store_true", help="Output RL history"
    )
    parser.add_argument(
        "--validate_no_static", action="store_true", help="Verify no static configs"
    )
    parser.add_argument(
        "--rl_info", action="store_true", help="Output RL tuning config"
    )
    parser.add_argument(
        "--adaptive_info", action="store_true", help="Output adaptive config"
    )
    parser.add_argument(
        "--rl_status", action="store_true", help="Output RL integration status"
    )


def _add_depth_args(parser: argparse.ArgumentParser) -> None:
    """Add depth scaling arguments."""
    parser.add_argument(
        "--adaptive_depth_run", action="store_true", help="Run with adaptive depth"
    )
    parser.add_argument(
        "--rl_sweep", type=int, default=500, help="Informed RL runs (default: 500)"
    )
    parser.add_argument(
        "--depth_scaling_test", action="store_true", help="Run depth scaling test"
    )
    parser.add_argument(
        "--compute_depth", action="store_true", help="Show computed depth"
    )
    parser.add_argument(
        "--tree_size", type=int, default=int(1e6), help="Tree size (default: 1e6)"
    )
    parser.add_argument(
        "--depth_scaling_info", action="store_true", help="Output depth scaling config"
    )
    parser.add_argument(
        "--efficient_sweep_info", action="store_true", help="Output sweep config"
    )


def _add_sweep_args(parser: argparse.ArgumentParser) -> None:
    """Add 500-run sweep arguments."""
    parser.add_argument(
        "--rl_500_sweep", action="store_true", help="Run 500-run RL sweep"
    )
    parser.add_argument(
        "--lr_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Override LR bounds",
    )
    parser.add_argument(
        "--retention_target", type=float, default=1.05, help="Retention target"
    )
    parser.add_argument(
        "--quantum_estimate", action="store_true", help="Show quantum estimate"
    )
    parser.add_argument(
        "--rl_500_sweep_info", action="store_true", help="Output 500-run config"
    )


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add pipeline arguments."""
    parser.add_argument(
        "--lr_pilot",
        type=int,
        default=None,
        const=50,
        nargs="?",
        metavar="RUNS",
        help="Run LR pilot (default: 50 runs)",
    )
    parser.add_argument(
        "--quantum_sim",
        type=int,
        default=None,
        const=10,
        nargs="?",
        metavar="RUNS",
        help="Run quantum sim (default: 10 runs)",
    )
    parser.add_argument(
        "--post_tune_execute", action="store_true", help="Execute tuned sweep"
    )
    parser.add_argument(
        "--full_pipeline", action="store_true", help="Run complete pipeline"
    )
    parser.add_argument(
        "--show_bounds", action="store_true", help="Show LR bounds comparison"
    )
    parser.add_argument("--pilot_info", action="store_true", help="Output pilot config")
    parser.add_argument(
        "--quantum_rl_hybrid_info", action="store_true", help="Output quantum-RL config"
    )
    parser.add_argument(
        "--pipeline_info", action="store_true", help="Output pipeline config"
    )
    parser.add_argument(
        "--pilot_runs", type=int, default=50, help="Pilot runs (default: 50)"
    )
    parser.add_argument(
        "--quantum_runs", type=int, default=10, help="Quantum runs (default: 10)"
    )
    parser.add_argument(
        "--sweep_runs", type=int, default=500, help="Sweep runs (default: 500)"
    )


def _add_scale_args(parser: argparse.ArgumentParser) -> None:
    """Add scale validation arguments."""
    parser.add_argument(
        "--multi_scale_sweep",
        type=str,
        default=None,
        metavar="TYPE",
        help='Run multi-scale sweep ("all" or "1e9")',
    )
    parser.add_argument(
        "--scalability_gate_test",
        action="store_true",
        help="Test scalability gate status",
    )
    parser.add_argument(
        "--scale_info", action="store_true", help="Show scale validation configuration"
    )
    parser.add_argument(
        "--fractal_info", action="store_true", help="Show fractal layer configuration"
    )


def _add_fractal_args(parser: argparse.ArgumentParser) -> None:
    """Add fractal ceiling breach arguments."""
    parser.add_argument(
        "--alpha_boost",
        type=str,
        default=None,
        metavar="MODE",
        help="Alpha boost mode (off, quantum, fractal, hybrid)",
    )
    parser.add_argument(
        "--fractal_push", action="store_true", help="Run fractal ceiling breach"
    )
    parser.add_argument(
        "--fractal_info_hybrid",
        action="store_true",
        help="Show fractal hybrid configuration",
    )
    parser.add_argument(
        "--base_alpha",
        type=float,
        default=2.99,
        help="Base alpha for fractal/hybrid (default: 2.99)",
    )


def _add_hybrid_args(parser: argparse.ArgumentParser) -> None:
    """Add hybrid sweep arguments."""
    parser.add_argument(
        "--full_500_sweep",
        action="store_true",
        help="Run full 500-sweep with quantum-fractal hybrid",
    )
    parser.add_argument(
        "--hybrid_boost_info",
        action="store_true",
        help="Show hybrid boost configuration",
    )


def _add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add benchmark arguments."""
    parser.add_argument(
        "--hybrid_10e12", action="store_true", help="Run 10^12 hybrid benchmark"
    )
    parser.add_argument(
        "--release_gate", action="store_true", help="Check release gate 3.1 status"
    )
    parser.add_argument(
        "--fractal_recursion",
        action="store_true",
        help="Run fractal recursion ceiling breach",
    )
    parser.add_argument(
        "--fractal_recursion_sweep",
        action="store_true",
        help="Sweep through all recursion depths",
    )
    parser.add_argument(
        "--benchmark_info", action="store_true", help="Show benchmark configuration"
    )
    parser.add_argument(
        "--recursion_depth",
        type=int,
        default=3,
        help="Fractal recursion depth (1-5, default: 3)",
    )


def _add_path_args(parser: argparse.ArgumentParser) -> None:
    """Add path exploration arguments."""
    parser.add_argument(
        "--d4_push", action="store_true", help="Run D4 recursion for alpha>=3.2"
    )
    parser.add_argument("--d4_info", action="store_true", help="Show D4 configuration")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Select exploration path (mars/multiplanet/agi)",
    )
    parser.add_argument(
        "--path_status", action="store_true", help="Show all path statuses"
    )
    parser.add_argument(
        "--path_list", action="store_true", help="List registered paths"
    )
    parser.add_argument(
        "--path_cmd", type=str, default=None, help="Run command on selected path"
    )
    parser.add_argument(
        "--mars_status",
        action="store_true",
        help="Shortcut: --path mars --path_cmd status",
    )
    parser.add_argument(
        "--multiplanet_status",
        action="store_true",
        help="Shortcut: --path multiplanet --path_cmd status",
    )
    parser.add_argument(
        "--agi_status",
        action="store_true",
        help="Shortcut: --path agi --path_cmd status",
    )
    parser.add_argument(
        "--registry_info", action="store_true", help="Show path registry info"
    )


def _add_isru_args(parser: argparse.ArgumentParser) -> None:
    """Add ISRU/D5 arguments."""
    parser.add_argument(
        "--d5_push", action="store_true", help="Run D5 recursion for alpha>=3.25"
    )
    parser.add_argument(
        "--d5_info", action="store_true", help="Show D5 + ISRU configuration"
    )
    parser.add_argument(
        "--d5_isru_hybrid", action="store_true", help="Run integrated D5+ISRU hybrid"
    )
    parser.add_argument(
        "--moxie_info", action="store_true", help="Show MOXIE calibration data"
    )
    parser.add_argument(
        "--isru_simulate", action="store_true", help="Run ISRU O2 production simulation"
    )
    parser.add_argument(
        "--isru_closure", action="store_true", help="Show ISRU closure metrics"
    )
    parser.add_argument(
        "--isru_info", action="store_true", help="Show ISRU module info"
    )
    parser.add_argument(
        "--crew", type=int, default=4, help="Crew size for ISRU simulation (default: 4)"
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="Simulation hours for ISRU (default: 24)"
    )
    parser.add_argument(
        "--moxie_units",
        type=int,
        default=10,
        help="Number of MOXIE units (default: 10)",
    )


def _add_titan_args(parser: argparse.ArgumentParser) -> None:
    """Add Titan/D6 arguments."""
    parser.add_argument(
        "--d6_push", action="store_true", help="Run D6 recursion for alpha>=3.33"
    )
    parser.add_argument(
        "--d6_info", action="store_true", help="Show D6 + Titan configuration"
    )
    parser.add_argument(
        "--d6_titan_hybrid", action="store_true", help="Run integrated D6+Titan hybrid"
    )
    parser.add_argument(
        "--titan_info", action="store_true", help="Show Titan methane hybrid info"
    )
    parser.add_argument(
        "--titan_config", action="store_true", help="Show Titan configuration from spec"
    )
    parser.add_argument(
        "--titan_simulate",
        action="store_true",
        help="Run Titan methane harvest simulation",
    )
    parser.add_argument(
        "--titan_duration",
        type=int,
        default=30,
        help="Titan simulation duration in days (default: 30)",
    )
    parser.add_argument(
        "--titan_extraction_rate",
        type=float,
        default=10.0,
        help="Titan extraction rate kg/hr (default: 10.0)",
    )
    parser.add_argument(
        "--perovskite_info",
        action="store_true",
        help="Show perovskite efficiency configuration",
    )
    parser.add_argument(
        "--perovskite_project",
        action="store_true",
        help="Project perovskite efficiency timeline",
    )
    parser.add_argument(
        "--perovskite_years",
        type=int,
        default=10,
        help="Perovskite projection years (default: 10)",
    )
    parser.add_argument(
        "--perovskite_growth",
        type=float,
        default=0.10,
        help="Perovskite annual growth rate (default: 0.10)",
    )


def _add_audit_args(parser: argparse.ArgumentParser) -> None:
    """Add adversarial audit arguments."""
    parser.add_argument(
        "--audit_info", action="store_true", help="Show adversarial audit configuration"
    )
    parser.add_argument(
        "--audit_config", action="store_true", help="Show adversarial config from spec"
    )
    parser.add_argument(
        "--audit_run", action="store_true", help="Run adversarial audit"
    )
    parser.add_argument(
        "--audit_noise",
        type=float,
        default=0.05,
        help="Adversarial noise level (default: 0.05)",
    )
    parser.add_argument(
        "--audit_iterations",
        type=int,
        default=100,
        help="Adversarial test iterations (default: 100)",
    )
    parser.add_argument(
        "--audit_stress", action="store_true", help="Run adversarial stress test"
    )
    parser.add_argument(
        "--audit_injection", action="store_true", help="Run injection attack audit"
    )
    parser.add_argument(
        "--audit_poisoning", action="store_true", help="Run poisoning attack audit"
    )
    parser.add_argument(
        "--audit_expanded", action="store_true", help="Run all expanded audits"
    )


def _add_europa_args(parser: argparse.ArgumentParser) -> None:
    """Add Europa/D7 arguments."""
    parser.add_argument(
        "--d7_push", action="store_true", help="Run D7 recursion for alpha>=3.40"
    )
    parser.add_argument(
        "--d7_info", action="store_true", help="Show D7 + Europa configuration"
    )
    parser.add_argument(
        "--d7_europa_hybrid",
        action="store_true",
        help="Run integrated D7+Europa hybrid",
    )
    parser.add_argument(
        "--europa_info", action="store_true", help="Show Europa ice drilling info"
    )
    parser.add_argument(
        "--europa_config",
        action="store_true",
        help="Show Europa configuration from spec",
    )
    parser.add_argument(
        "--europa_simulate",
        action="store_true",
        help="Run Europa ice drilling simulation",
    )
    parser.add_argument(
        "--europa_depth",
        type=int,
        default=1000,
        help="Europa drilling depth in meters (default: 1000)",
    )
    parser.add_argument(
        "--europa_duration",
        type=int,
        default=30,
        help="Europa simulation duration in days (default: 30)",
    )
    parser.add_argument(
        "--europa_drill_rate",
        type=float,
        default=2.0,
        help="Europa drill rate m/hr (default: 2.0)",
    )
    parser.add_argument(
        "--europa_resupply",
        type=float,
        default=180.0,
        help="Europa resupply interval in days (default: 180)",
    )


def _add_nrel_args(parser: argparse.ArgumentParser) -> None:
    """Add NREL validation arguments."""
    parser.add_argument(
        "--nrel_info", action="store_true", help="Show NREL validation configuration"
    )
    parser.add_argument(
        "--nrel_config", action="store_true", help="Show NREL configuration from spec"
    )
    parser.add_argument(
        "--nrel_validate", action="store_true", help="Run NREL efficiency validation"
    )
    parser.add_argument(
        "--nrel_efficiency",
        type=float,
        default=0.256,
        help="NREL efficiency to validate (default: 0.256)",
    )
    parser.add_argument(
        "--nrel_project", action="store_true", help="Project NREL degradation over time"
    )
    parser.add_argument(
        "--nrel_years", type=int, default=25, help="NREL projection years (default: 25)"
    )
    parser.add_argument(
        "--nrel_initial",
        type=float,
        default=0.0,
        help="NREL initial efficiency for projection (default: lab value)",
    )
    parser.add_argument(
        "--nrel_compare", action="store_true", help="Compare NREL to MOXIE efficiency"
    )
    parser.add_argument(
        "--moxie_efficiency",
        type=float,
        default=0.0,
        help="MOXIE efficiency for comparison (default: baseline)",
    )


def _add_d8_args(parser: argparse.ArgumentParser) -> None:
    """Add D8 arguments."""
    parser.add_argument(
        "--d8_push", action="store_true", help="Run D8 recursion for alpha>=3.45"
    )
    parser.add_argument(
        "--d8_info", action="store_true", help="Show D8 + unified RL configuration"
    )
    parser.add_argument(
        "--d8_multi_sync", action="store_true", help="Run integrated D8+sync"
    )
    parser.add_argument(
        "--sync_info", action="store_true", help="Show sync configuration"
    )
    parser.add_argument("--sync_run", action="store_true", help="Run sync cycle")
    parser.add_argument(
        "--sync_efficiency", action="store_true", help="Show efficiency metrics"
    )
    parser.add_argument(
        "--atacama_info", action="store_true", help="Show Atacama configuration"
    )
    parser.add_argument(
        "--atacama_validate", action="store_true", help="Run Atacama validation"
    )
    parser.add_argument(
        "--encrypt_info", action="store_true", help="Show encryption configuration"
    )
    parser.add_argument(
        "--encrypt_keygen", action="store_true", help="Generate fractal key"
    )
    parser.add_argument(
        "--encrypt_audit", action="store_true", help="Run full encryption audit"
    )
    parser.add_argument(
        "--encrypt_side_channel", action="store_true", help="Test side-channel defense"
    )
    parser.add_argument(
        "--encrypt_inversion", action="store_true", help="Test model inversion defense"
    )
    parser.add_argument(
        "--encrypt_key_depth",
        type=int,
        default=6,
        help="Fractal key depth (default: 6)",
    )
    parser.add_argument(
        "--encrypt_iterations",
        type=int,
        default=100,
        help="Encryption test iterations (default: 100)",
    )


def _add_d9_args(parser: argparse.ArgumentParser) -> None:
    """Add D9/Ganymede arguments."""
    parser.add_argument(
        "--d9_push", action="store_true", help="Run D9 recursion for alpha>=3.50"
    )
    parser.add_argument("--d9_info", action="store_true", help="Show D9 configuration")
    parser.add_argument(
        "--d9_ganymede_hybrid", action="store_true", help="Run integrated D9+Ganymede"
    )
    parser.add_argument(
        "--ganymede_info", action="store_true", help="Show Ganymede configuration"
    )
    parser.add_argument(
        "--ganymede_config", action="store_true", help="Show Ganymede config from spec"
    )
    parser.add_argument(
        "--ganymede_navigate", action="store_true", help="Run navigation simulation"
    )
    parser.add_argument(
        "--ganymede_nav_mode",
        type=str,
        default="field_following",
        help="Navigation mode (field_following, magnetopause_crossing, polar_transit)",
    )
    parser.add_argument(
        "--ganymede_duration",
        type=int,
        default=24,
        help="Navigation duration in hours (default: 24)",
    )
    parser.add_argument(
        "--ganymede_autonomy", action="store_true", help="Show autonomy metrics"
    )


def _add_drone_args(parser: argparse.ArgumentParser) -> None:
    """Add drone array arguments."""
    parser.add_argument(
        "--drone_info", action="store_true", help="Show drone array configuration"
    )
    parser.add_argument(
        "--drone_config", action="store_true", help="Show drone config from spec"
    )
    parser.add_argument(
        "--drone_coverage", action="store_true", help="Run drone coverage simulation"
    )
    parser.add_argument(
        "--drone_sample", action="store_true", help="Run drone dust sampling"
    )
    parser.add_argument(
        "--drone_validate", action="store_true", help="Run full drone validation"
    )
    parser.add_argument(
        "--drone_count", type=int, default=10, help="Number of drones (default: 10)"
    )
    parser.add_argument(
        "--drone_area",
        type=float,
        default=1000.0,
        help="Area to cover in km2 (default: 1000)",
    )
    parser.add_argument(
        "--drone_duration",
        type=int,
        default=60,
        help="Sampling duration in seconds (default: 60)",
    )


def _add_randomized_args(parser: argparse.ArgumentParser) -> None:
    """Add randomized paths arguments."""
    parser.add_argument(
        "--randomized_info",
        action="store_true",
        help="Show randomized paths configuration",
    )
    parser.add_argument(
        "--randomized_config",
        action="store_true",
        help="Show randomized config from spec",
    )
    parser.add_argument(
        "--randomized_generate", action="store_true", help="Generate execution tree"
    )
    parser.add_argument(
        "--randomized_audit", action="store_true", help="Run full randomized audit"
    )
    parser.add_argument(
        "--randomized_timing", action="store_true", help="Test timing resilience"
    )
    parser.add_argument(
        "--randomized_power", action="store_true", help="Test power resilience"
    )
    parser.add_argument(
        "--randomized_cache", action="store_true", help="Test cache resilience"
    )
    parser.add_argument(
        "--randomized_depth",
        type=int,
        default=8,
        help="Execution tree depth (default: 8)",
    )
    parser.add_argument(
        "--randomized_iterations",
        type=int,
        default=100,
        help="Randomized test iterations (default: 100)",
    )
    parser.add_argument(
        "--threat_level",
        type=str,
        default="medium",
        help="Threat level for path depth recommendation (low, medium, high, critical)",
    )


def _add_d10_args(parser: argparse.ArgumentParser) -> None:
    """Add D10/Jovian arguments."""
    parser.add_argument(
        "--d10_push", action="store_true", help="Run D10 recursion for alpha>=3.55"
    )
    parser.add_argument(
        "--d10_info", action="store_true", help="Show D10 configuration"
    )
    parser.add_argument(
        "--d10_jovian_hub", action="store_true", help="Run integrated D10+Jovian hub"
    )
    parser.add_argument(
        "--jovian_info", action="store_true", help="Show Jovian hub configuration"
    )
    parser.add_argument(
        "--jovian_sync", action="store_true", help="Run Jovian sync cycle"
    )
    parser.add_argument(
        "--jovian_autonomy", action="store_true", help="Show Jovian system autonomy"
    )
    parser.add_argument(
        "--jovian_coordinate", action="store_true", help="Run full Jovian coordination"
    )


def _add_callisto_args(parser: argparse.ArgumentParser) -> None:
    """Add Callisto arguments."""
    parser.add_argument(
        "--callisto_info", action="store_true", help="Show Callisto configuration"
    )
    parser.add_argument(
        "--callisto_config", action="store_true", help="Show Callisto config from spec"
    )
    parser.add_argument(
        "--callisto_ice", action="store_true", help="Show Callisto ice availability"
    )
    parser.add_argument(
        "--callisto_extract", action="store_true", help="Run Callisto extraction sim"
    )
    parser.add_argument(
        "--callisto_radiation",
        action="store_true",
        help="Show Callisto radiation advantage",
    )
    parser.add_argument(
        "--callisto_autonomy", action="store_true", help="Show Callisto autonomy"
    )
    parser.add_argument(
        "--callisto_hub", action="store_true", help="Evaluate Callisto hub suitability"
    )
    parser.add_argument(
        "--callisto_rate",
        type=float,
        default=100.0,
        help="Callisto extraction rate kg/hr (default: 100)",
    )
    parser.add_argument(
        "--callisto_duration",
        type=int,
        default=30,
        help="Callisto extraction duration days (default: 30)",
    )


def _add_quantum_resist_args(parser: argparse.ArgumentParser) -> None:
    """Add quantum-resistant arguments."""
    parser.add_argument(
        "--quantum_resist_info",
        action="store_true",
        help="Show quantum-resistant config",
    )
    parser.add_argument(
        "--quantum_resist_config",
        action="store_true",
        help="Show quantum config from spec",
    )
    parser.add_argument(
        "--quantum_keygen", action="store_true", help="Generate quantum-resistant key"
    )
    parser.add_argument(
        "--quantum_key_size",
        type=int,
        default=256,
        help="Quantum key size in bits (default: 256)",
    )
    parser.add_argument(
        "--quantum_audit", action="store_true", help="Run full quantum-resistant audit"
    )
    parser.add_argument(
        "--quantum_spectre", action="store_true", help="Test Spectre defense"
    )
    parser.add_argument(
        "--quantum_cache", action="store_true", help="Test quantum cache timing defense"
    )
    parser.add_argument(
        "--spectre_v1", action="store_true", help="Test Spectre v1 defense"
    )
    parser.add_argument(
        "--spectre_v2", action="store_true", help="Test Spectre v2 defense"
    )
    parser.add_argument(
        "--spectre_v4", action="store_true", help="Test Spectre v4 defense"
    )
    parser.add_argument(
        "--quantum_iterations",
        type=int,
        default=100,
        help="Quantum test iterations (default: 100)",
    )


def _add_dust_args(parser: argparse.ArgumentParser) -> None:
    """Add dust dynamics arguments."""
    parser.add_argument(
        "--dust_info", action="store_true", help="Show dust dynamics configuration"
    )
    parser.add_argument(
        "--dust_config", action="store_true", help="Show dust config from spec"
    )
    parser.add_argument(
        "--dust_dynamics", action="store_true", help="Run dust dynamics validation"
    )
    parser.add_argument(
        "--dust_settling", action="store_true", help="Simulate dust settling"
    )
    parser.add_argument(
        "--dust_particle", action="store_true", help="Analyze particle distribution"
    )
    parser.add_argument(
        "--dust_solar_impact", action="store_true", help="Compute solar panel impact"
    )
    parser.add_argument(
        "--dust_mars", action="store_true", help="Project Mars conditions from Atacama"
    )
    parser.add_argument(
        "--dust_depth_mm",
        type=float,
        default=1.0,
        help="Dust depth in mm for solar impact (default: 1.0)",
    )
    parser.add_argument(
        "--dust_duration",
        type=int,
        default=30,
        help="Dust settling duration days (default: 30)",
    )


def _add_d13_args(parser: argparse.ArgumentParser) -> None:
    """Add D13 recursion arguments."""
    parser.add_argument(
        "--d13_push", action="store_true", help="Run D13 recursion for alpha>=3.70"
    )
    parser.add_argument(
        "--d13_info", action="store_true", help="Show D13 configuration"
    )
    parser.add_argument(
        "--d13_solar_hybrid", action="store_true", help="Run integrated D13+Solar hub"
    )


def _add_solar_args(parser: argparse.ArgumentParser) -> None:
    """Add Solar orbital hub arguments."""
    parser.add_argument(
        "--solar_info", action="store_true", help="Show Solar hub configuration"
    )
    parser.add_argument(
        "--solar_positions", action="store_true", help="Show orbital positions"
    )
    parser.add_argument(
        "--solar_windows", action="store_true", help="Show communication windows"
    )
    parser.add_argument(
        "--solar_transfer", action="store_true", help="Simulate resource transfer"
    )
    parser.add_argument(
        "--solar_sync", action="store_true", help="Run coordination sync"
    )
    parser.add_argument(
        "--solar_autonomy", action="store_true", help="Show Solar hub autonomy"
    )
    parser.add_argument(
        "--solar_simulate", action="store_true", help="Run full Solar hub simulation"
    )
    parser.add_argument(
        "--solar_timestamp",
        type=float,
        default=0.0,
        help="Days since epoch for orbital positions (default: 0)",
    )
    parser.add_argument(
        "--solar_duration",
        type=int,
        default=365,
        help="Duration in days for Solar sim (default: 365)",
    )
    parser.add_argument(
        "--from_planet",
        type=str,
        default="mars",
        help="Source planet for transfer (default: mars)",
    )
    parser.add_argument(
        "--to_planet",
        type=str,
        default="venus",
        help="Destination planet for transfer (default: venus)",
    )
    parser.add_argument(
        "--resource",
        type=str,
        default="water_ice",
        help="Resource to transfer (default: water_ice)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=1000.0,
        help="Amount to transfer in kg (default: 1000)",
    )


def _add_les_args(parser: argparse.ArgumentParser) -> None:
    """Add LES (Large Eddy Simulation) arguments."""
    parser.add_argument(
        "--les_info", action="store_true", help="Show LES configuration"
    )
    parser.add_argument(
        "--les_simulate", action="store_true", help="Run LES simulation"
    )
    parser.add_argument(
        "--les_dust_devil", action="store_true", help="Simulate Mars dust devil"
    )
    parser.add_argument(
        "--les_compare", action="store_true", help="Compare LES vs RANS"
    )
    parser.add_argument(
        "--les_validate", action="store_true", help="Run full LES validation"
    )
    parser.add_argument(
        "--les_reynolds",
        type=float,
        default=50000,
        help="Reynolds number for LES (default: 50000)",
    )
    parser.add_argument(
        "--les_duration",
        type=float,
        default=100.0,
        help="LES simulation duration in seconds (default: 100)",
    )
    parser.add_argument(
        "--dust_devil_diameter",
        type=float,
        default=50.0,
        help="Dust devil diameter in meters (default: 50)",
    )
    parser.add_argument(
        "--dust_devil_height",
        type=float,
        default=500.0,
        help="Dust devil height in meters (default: 500)",
    )
    parser.add_argument(
        "--dust_devil_intensity",
        type=float,
        default=0.7,
        help="Dust devil intensity 0-1 (default: 0.7)",
    )


def _add_zk_args(parser: argparse.ArgumentParser) -> None:
    """Add ZK (Zero-Knowledge) proof arguments."""
    parser.add_argument("--zk_info", action="store_true", help="Show ZK configuration")
    parser.add_argument("--zk_setup", action="store_true", help="Run ZK trusted setup")
    parser.add_argument("--zk_prove", action="store_true", help="Generate ZK proof")
    parser.add_argument("--zk_verify", action="store_true", help="Verify ZK proof")
    parser.add_argument(
        "--zk_attestation", action="store_true", help="Create ZK attestation"
    )
    parser.add_argument("--zk_audit", action="store_true", help="Run full ZK audit")
    parser.add_argument(
        "--zk_benchmark", action="store_true", help="Benchmark ZK proof system"
    )
    parser.add_argument(
        "--zk_circuit_size",
        type=int,
        default=2**20,
        help="ZK circuit size in constraints (default: 2^20)",
    )
    parser.add_argument(
        "--zk_iterations",
        type=int,
        default=10,
        help="ZK benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--zk_attestation_count",
        type=int,
        default=5,
        help="ZK audit attestation count (default: 5)",
    )
    parser.add_argument(
        "--enclave_id",
        type=str,
        default="test_enclave",
        help="Enclave ID for attestation (default: test_enclave)",
    )
    parser.add_argument(
        "--code_hash",
        type=str,
        default="test_code_hash",
        help="Code hash for attestation (default: test_code_hash)",
    )
    parser.add_argument(
        "--config_hash",
        type=str,
        default="test_config_hash",
        help="Config hash for attestation (default: test_config_hash)",
    )


def _add_d14_args(parser: argparse.ArgumentParser) -> None:
    """Add D14 fractal recursion arguments."""
    parser.add_argument(
        "--d14_push", action="store_true", help="Run D14 recursion for alpha>=3.75"
    )
    parser.add_argument(
        "--d14_info", action="store_true", help="Show D14 configuration"
    )
    parser.add_argument(
        "--d14_interstellar_hybrid",
        action="store_true",
        help="Run integrated D14+interstellar hybrid",
    )


def _add_interstellar_args(parser: argparse.ArgumentParser) -> None:
    """Add interstellar backbone arguments."""
    parser.add_argument(
        "--interstellar_info",
        action="store_true",
        help="Show interstellar backbone configuration",
    )
    parser.add_argument(
        "--interstellar_bodies",
        action="store_true",
        help="List all 7 bodies in the backbone",
    )
    parser.add_argument(
        "--interstellar_positions",
        action="store_true",
        help="Show current body positions",
    )
    parser.add_argument(
        "--interstellar_windows",
        action="store_true",
        help="Show communication windows between bodies",
    )
    parser.add_argument(
        "--interstellar_sync",
        action="store_true",
        help="Run backbone coordination sync",
    )
    parser.add_argument(
        "--interstellar_autonomy",
        action="store_true",
        help="Show backbone autonomy metrics",
    )
    parser.add_argument(
        "--interstellar_failover",
        action="store_true",
        help="Test emergency failover for a body",
    )
    parser.add_argument(
        "--interstellar_body",
        type=str,
        default="europa",
        help="Body for failover test (default: europa)",
    )
    parser.add_argument(
        "--interstellar_timestamp",
        type=float,
        default=0.0,
        help="Timestamp for positions/windows (default: 0)",
    )
    parser.add_argument(
        "--interstellar_duration",
        type=int,
        default=60,
        help="Duration for sync in days (default: 60)",
    )


def _add_atacama_args(parser: argparse.ArgumentParser) -> None:
    """Add Atacama real-time LES arguments."""
    parser.add_argument(
        "--atacama_realtime",
        action="store_true",
        help="Run Atacama real-time LES mode",
    )
    parser.add_argument(
        "--atacama_dust_devil",
        action="store_true",
        help="Track dust devil in real-time",
    )
    parser.add_argument(
        "--atacama_correlation",
        action="store_true",
        help="Compute real-time correlation",
    )
    parser.add_argument(
        "--atacama_full_validation",
        action="store_true",
        help="Run full Atacama validation",
    )
    parser.add_argument(
        "--atacama_sampling_hz",
        type=int,
        default=100,
        help="Drone sampling rate in Hz (default: 100)",
    )
    parser.add_argument(
        "--atacama_realtime_duration",
        type=float,
        default=60.0,
        help="Real-time simulation duration in seconds (default: 60)",
    )


def _add_plonk_args(parser: argparse.ArgumentParser) -> None:
    """Add PLONK ZK proof arguments."""
    parser.add_argument(
        "--plonk_info", action="store_true", help="Show PLONK configuration"
    )
    parser.add_argument(
        "--plonk_setup", action="store_true", help="Run universal trusted setup"
    )
    parser.add_argument(
        "--plonk_prove", action="store_true", help="Generate PLONK proof"
    )
    parser.add_argument(
        "--plonk_verify", action="store_true", help="Verify PLONK proof"
    )
    parser.add_argument(
        "--plonk_recursive",
        action="store_true",
        help="Generate recursive proof (proof of proofs)",
    )
    parser.add_argument(
        "--plonk_attestation",
        action="store_true",
        help="Create PLONK attestation",
    )
    parser.add_argument(
        "--plonk_audit", action="store_true", help="Run full PLONK audit"
    )
    parser.add_argument(
        "--plonk_benchmark", action="store_true", help="Benchmark PLONK proof system"
    )
    parser.add_argument(
        "--plonk_compare", action="store_true", help="Compare PLONK vs Groth16"
    )
    parser.add_argument(
        "--plonk_participants",
        type=int,
        default=10,
        help="Number of setup participants (default: 10)",
    )
    parser.add_argument(
        "--plonk_recursive_count",
        type=int,
        default=3,
        help="Number of proofs to aggregate (default: 3)",
    )
    parser.add_argument(
        "--plonk_iterations",
        type=int,
        default=10,
        help="Benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--plonk_attestation_count",
        type=int,
        default=5,
        help="PLONK audit attestation count (default: 5)",
    )

    # D15 flags
    _add_d15_args(parser)

    # Chaos simulation flags
    _add_chaos_args(parser)

    # Halo2 flags
    _add_halo2_args(parser)

    # Atacama 200Hz flags
    _add_atacama_200hz_args(parser)


def _add_d15_args(parser: argparse.ArgumentParser) -> None:
    """Add D15 quantum-entangled fractal recursion arguments."""
    parser.add_argument(
        "--d15_push", action="store_true", help="Run D15 recursion for alpha>=3.81"
    )
    parser.add_argument(
        "--d15_info", action="store_true", help="Show D15 configuration"
    )
    parser.add_argument(
        "--d15_chaos_hybrid",
        action="store_true",
        help="Run integrated D15+chaos+backbone hybrid",
    )
    parser.add_argument(
        "--d15_entanglement",
        action="store_true",
        help="Test quantum entanglement correlation",
    )


def _add_chaos_args(parser: argparse.ArgumentParser) -> None:
    """Add chaotic n-body simulation arguments."""
    parser.add_argument(
        "--chaos_info", action="store_true", help="Show chaos simulation configuration"
    )
    parser.add_argument(
        "--chaos_simulate", action="store_true", help="Run chaotic n-body simulation"
    )
    parser.add_argument(
        "--chaos_stability", action="store_true", help="Check simulation stability"
    )
    parser.add_argument(
        "--chaos_monte_carlo",
        action="store_true",
        help="Run Monte Carlo stability analysis",
    )
    parser.add_argument(
        "--chaos_backbone_tolerance",
        action="store_true",
        help="Compute backbone chaos tolerance",
    )
    parser.add_argument(
        "--chaos_lyapunov", action="store_true", help="Compute Lyapunov exponent"
    )
    parser.add_argument(
        "--chaos_iterations",
        type=int,
        default=1000,
        help="Chaos simulation iterations (default: 1000)",
    )
    parser.add_argument(
        "--chaos_dt",
        type=float,
        default=0.001,
        help="Chaos simulation time step (default: 0.001)",
    )
    parser.add_argument(
        "--chaos_monte_carlo_runs",
        type=int,
        default=100,
        help="Monte Carlo runs (default: 100)",
    )


def _add_halo2_args(parser: argparse.ArgumentParser) -> None:
    """Add Halo2 recursive ZK proof arguments."""
    parser.add_argument(
        "--halo2_info", action="store_true", help="Show Halo2 configuration"
    )
    parser.add_argument(
        "--halo2_prove", action="store_true", help="Generate Halo2 proof"
    )
    parser.add_argument(
        "--halo2_verify", action="store_true", help="Verify Halo2 proof"
    )
    parser.add_argument(
        "--halo2_recursive",
        action="store_true",
        help="Generate recursive proof (infinite depth capable)",
    )
    parser.add_argument(
        "--halo2_attestation",
        action="store_true",
        help="Create Halo2 attestation",
    )
    parser.add_argument(
        "--halo2_audit", action="store_true", help="Run full Halo2 audit"
    )
    parser.add_argument(
        "--halo2_benchmark", action="store_true", help="Benchmark Halo2 proof system"
    )
    parser.add_argument(
        "--halo2_compare", action="store_true", help="Compare Halo2 vs PLONK vs Groth16"
    )
    parser.add_argument(
        "--halo2_infinite_chain",
        action="store_true",
        help="Generate infinite attestation chain",
    )
    parser.add_argument(
        "--halo2_recursive_depth",
        type=int,
        default=5,
        help="Recursive proof depth (default: 5)",
    )
    parser.add_argument(
        "--halo2_iterations",
        type=int,
        default=10,
        help="Halo2 benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--halo2_attestation_count",
        type=int,
        default=5,
        help="Halo2 audit attestation count (default: 5)",
    )
    parser.add_argument(
        "--halo2_chain_depth",
        type=int,
        default=10,
        help="Infinite chain depth (default: 10)",
    )


def _add_atacama_200hz_args(parser: argparse.ArgumentParser) -> None:
    """Add Atacama 200Hz drone sampling arguments."""
    parser.add_argument(
        "--atacama_200hz",
        action="store_true",
        help="Run Atacama 200Hz mode",
    )
    parser.add_argument(
        "--atacama_200hz_info",
        action="store_true",
        help="Show Atacama 200Hz configuration",
    )
    parser.add_argument(
        "--atacama_200hz_predict",
        action="store_true",
        help="Run dust devil prediction at 200Hz",
    )
    parser.add_argument(
        "--atacama_200hz_duration",
        type=float,
        default=60.0,
        help="200Hz simulation duration in seconds (default: 60)",
    )

    # D16 flags
    _add_d16_args(parser)

    # Kuiper 12-body flags
    _add_kuiper_args(parser)

    # Bulletproofs flags
    _add_bulletproofs_args(parser)

    # ML ensemble flags
    _add_ml_ensemble_args(parser)


def _add_d16_args(parser: argparse.ArgumentParser) -> None:
    """Add D16 topological fractal recursion arguments."""
    parser.add_argument(
        "--d16_push", action="store_true", help="Run D16 recursion for alpha>=3.91"
    )
    parser.add_argument(
        "--d16_info", action="store_true", help="Show D16 configuration"
    )
    parser.add_argument(
        "--d16_topological",
        action="store_true",
        help="Show D16 topological metrics",
    )
    parser.add_argument(
        "--d16_homology",
        action="store_true",
        help="Compute persistent homology",
    )
    parser.add_argument(
        "--d16_kuiper_hybrid",
        action="store_true",
        help="Run integrated D16+Kuiper hybrid",
    )


def _add_kuiper_args(parser: argparse.ArgumentParser) -> None:
    """Add Kuiper 12-body chaos arguments."""
    parser.add_argument(
        "--kuiper_info", action="store_true", help="Show Kuiper configuration"
    )
    parser.add_argument(
        "--kuiper_bodies", action="store_true", help="List all 12 bodies"
    )
    parser.add_argument(
        "--kuiper_simulate", action="store_true", help="Run 12-body simulation"
    )
    parser.add_argument(
        "--kuiper_stability", action="store_true", help="Check Kuiper stability"
    )
    parser.add_argument(
        "--kuiper_resonances", action="store_true", help="Analyze resonances"
    )
    parser.add_argument(
        "--kuiper_encounters", action="store_true", help="Find close encounters"
    )
    parser.add_argument(
        "--kuiper_monte_carlo",
        action="store_true",
        help="Run Monte Carlo stability analysis",
    )
    parser.add_argument(
        "--kuiper_duration",
        type=float,
        default=10.0,
        help="Kuiper simulation duration in years (default: 10)",
    )


def _add_bulletproofs_args(parser: argparse.ArgumentParser) -> None:
    """Add Bulletproofs high-load stress testing arguments."""
    parser.add_argument(
        "--bulletproofs_info", action="store_true", help="Show Bulletproofs config"
    )
    parser.add_argument(
        "--bulletproofs_prove", action="store_true", help="Generate Bulletproof"
    )
    parser.add_argument(
        "--bulletproofs_verify", action="store_true", help="Verify Bulletproof"
    )
    parser.add_argument(
        "--bulletproofs_aggregate",
        action="store_true",
        help="Aggregate Bulletproofs",
    )
    parser.add_argument(
        "--bulletproofs_stress",
        action="store_true",
        help="Run high-load stress test",
    )
    parser.add_argument(
        "--bulletproofs_chain",
        action="store_true",
        help="Generate proof chain",
    )
    parser.add_argument(
        "--bulletproofs_audit",
        action="store_true",
        help="Run full Bulletproofs audit",
    )
    parser.add_argument(
        "--bulletproofs_benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    parser.add_argument(
        "--bulletproofs_compare",
        action="store_true",
        help="Compare Bulletproofs vs Halo2",
    )
    parser.add_argument(
        "--bulletproofs_depth",
        type=int,
        default=100,
        help="Bulletproofs stress test depth (default: 100)",
    )


def _add_ml_ensemble_args(parser: argparse.ArgumentParser) -> None:
    """Add ML ensemble forecasting arguments."""
    parser.add_argument(
        "--ml_ensemble_info",
        action="store_true",
        help="Show ML ensemble configuration",
    )
    parser.add_argument(
        "--ml_ensemble_train",
        action="store_true",
        help="Train ML ensemble",
    )
    parser.add_argument(
        "--ml_ensemble_predict",
        action="store_true",
        help="Make 60s dust forecast",
    )
    parser.add_argument(
        "--ml_ensemble_accuracy",
        action="store_true",
        help="Check forecast accuracy",
    )
    parser.add_argument(
        "--ml_ensemble_horizon",
        type=int,
        default=60,
        help="Prediction horizon in seconds (default: 60)",
    )


def _add_d17_args(parser: argparse.ArgumentParser) -> None:
    """Add D17 depth-first Heliosphere arguments."""
    parser.add_argument(
        "--d17_push", action="store_true", help="Run D17 recursion for alpha>=3.92"
    )
    parser.add_argument(
        "--d17_info", action="store_true", help="Show D17 configuration"
    )
    parser.add_argument(
        "--d17_depthfirst",
        action="store_true",
        help="Show D17 depth-first metrics",
    )
    parser.add_argument(
        "--d17_plateau_check",
        action="store_true",
        help="Check for asymptotic plateau",
    )
    parser.add_argument(
        "--d17_heliosphere_hybrid",
        action="store_true",
        help="Run integrated D17+Heliosphere hybrid",
    )
    parser.add_argument(
        "--bulletproofs_infinite_chain",
        action="store_true",
        help="Run 10k infinite chain test",
    )
    parser.add_argument(
        "--bulletproofs_10k_stress",
        action="store_true",
        help="Run 10k stress test",
    )
    parser.add_argument(
        "--ml_ensemble_90s",
        action="store_true",
        help="Run 90s ML ensemble forecast",
    )
    parser.add_argument(
        "--ml_ensemble_90s_info",
        action="store_true",
        help="Show 90s ML ensemble configuration",
    )


def _add_heliosphere_args(parser: argparse.ArgumentParser) -> None:
    """Add Heliosphere-Oort cloud arguments."""
    parser.add_argument(
        "--heliosphere_info",
        action="store_true",
        help="Show Heliosphere configuration",
    )
    parser.add_argument(
        "--heliosphere_zones",
        action="store_true",
        help="Show Heliosphere zone boundaries",
    )
    parser.add_argument(
        "--heliosphere_status",
        action="store_true",
        help="Show current Heliosphere status",
    )
    parser.add_argument(
        "--oort_info", action="store_true", help="Show Oort cloud configuration"
    )
    parser.add_argument(
        "--oort_simulate",
        action="store_true",
        help="Run Oort cloud coordination simulation",
    )
    parser.add_argument(
        "--oort_latency",
        action="store_true",
        help="Show Oort latency metrics",
    )
    parser.add_argument(
        "--oort_autonomy",
        action="store_true",
        help="Show Oort autonomy level",
    )
    parser.add_argument(
        "--oort_compression",
        action="store_true",
        help="Show compression metrics",
    )
    parser.add_argument(
        "--oort_stability",
        action="store_true",
        help="Check Oort coordination stability",
    )
    parser.add_argument(
        "--oort_au",
        type=float,
        default=50000,
        help="Oort simulation distance in AU (default: 50000)",
    )


def _add_d18_args(parser: argparse.ArgumentParser) -> None:
    """Add D18 interstellar relay arguments."""
    parser.add_argument(
        "--d18_push", action="store_true", help="Run D18 recursion for alpha>=3.91"
    )
    parser.add_argument(
        "--d18_info", action="store_true", help="Show D18 configuration"
    )
    parser.add_argument(
        "--d18_pruning",
        action="store_true",
        help="Show D18 pruning v3 metrics",
    )
    parser.add_argument(
        "--d18_compression",
        action="store_true",
        help="Show D18 compression metrics",
    )
    parser.add_argument(
        "--d18_interstellar_hybrid",
        action="store_true",
        help="Run integrated D18+relay+quantum hybrid",
    )


def _add_relay_args(parser: argparse.ArgumentParser) -> None:
    """Add interstellar relay arguments."""
    parser.add_argument(
        "--relay_info",
        action="store_true",
        help="Show interstellar relay configuration",
    )
    parser.add_argument(
        "--relay_simulate",
        action="store_true",
        help="Run Proxima relay simulation",
    )
    parser.add_argument(
        "--relay_latency",
        action="store_true",
        help="Show relay latency metrics",
    )
    parser.add_argument(
        "--relay_nodes",
        action="store_true",
        help="List relay chain nodes",
    )
    parser.add_argument(
        "--relay_stress",
        action="store_true",
        help="Run relay stress test",
    )
    parser.add_argument(
        "--relay_duration",
        type=int,
        default=365,
        help="Relay simulation duration in days (default: 365)",
    )
    parser.add_argument(
        "--relay_distance",
        type=float,
        default=4.24,
        help="Relay distance in light-years (default: 4.24)",
    )
    parser.add_argument(
        "--relay_iterations",
        type=int,
        default=100,
        help="Relay stress test iterations (default: 100)",
    )


def _add_quantum_alt_args(parser: argparse.ArgumentParser) -> None:
    """Add quantum alternative arguments."""
    parser.add_argument(
        "--quantum_alt_info",
        action="store_true",
        help="Show quantum alternative configuration",
    )
    parser.add_argument(
        "--quantum_alt_simulate",
        action="store_true",
        help="Run quantum non-local simulation",
    )
    parser.add_argument(
        "--quantum_alt_correlation",
        action="store_true",
        help="Show quantum correlation metrics",
    )
    parser.add_argument(
        "--quantum_alt_bell",
        action="store_true",
        help="Run Bell inequality check",
    )
    parser.add_argument(
        "--quantum_pairs",
        type=int,
        default=1000,
        help="Number of entanglement pairs (default: 1000)",
    )


def _add_elonsphere_args(parser: argparse.ArgumentParser) -> None:
    """Add Elon-sphere arguments."""
    # Starlink
    parser.add_argument(
        "--starlink_info",
        action="store_true",
        help="Show Starlink relay status",
    )
    parser.add_argument(
        "--starlink_simulate",
        action="store_true",
        help="Run Starlink simulation",
    )
    parser.add_argument(
        "--starlink_nodes",
        type=int,
        default=10,
        help="Number of Starlink nodes (default: 10)",
    )

    # Grok
    parser.add_argument(
        "--grok_info",
        action="store_true",
        help="Show Grok inference status",
    )
    parser.add_argument(
        "--grok_tune",
        action="store_true",
        help="Run Grok tuning",
    )
    parser.add_argument(
        "--grok_agents",
        type=int,
        default=8,
        help="Number of Grok agents (default: 8)",
    )

    # xAI
    parser.add_argument(
        "--xai_info",
        action="store_true",
        help="Show xAI compute status",
    )
    parser.add_argument(
        "--xai_simulate",
        action="store_true",
        help="Run xAI quantum simulation",
    )
    parser.add_argument(
        "--xai_pairs",
        type=int,
        default=1000,
        help="xAI simulation pairs (default: 1000)",
    )
    parser.add_argument(
        "--xai_iterations",
        type=int,
        default=100,
        help="xAI simulation iterations (default: 100)",
    )

    # Dojo
    parser.add_argument(
        "--dojo_info",
        action="store_true",
        help="Show Dojo offload status",
    )
    parser.add_argument(
        "--dojo_offload",
        action="store_true",
        help="Run Dojo offload",
    )
    parser.add_argument(
        "--dojo_depth",
        type=int,
        default=18,
        help="Dojo training depth (default: 18)",
    )
    parser.add_argument(
        "--dojo_batch_size",
        type=int,
        default=1000000,
        help="Dojo batch size (default: 1000000)",
    )


def _add_federation_args(parser: argparse.ArgumentParser) -> None:
    """Add federation arguments."""
    parser.add_argument(
        "--federation_info",
        action="store_true",
        help="Show multi-star federation status",
    )
    parser.add_argument(
        "--federation_consensus",
        action="store_true",
        help="Run federation consensus",
    )
    parser.add_argument(
        "--federation_lag",
        type=float,
        default=4.24,
        help="Federation communication lag in years (default: 4.24)",
    )
