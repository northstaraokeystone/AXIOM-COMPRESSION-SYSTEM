"""cli/dispatch.py - Command dispatch logic for AXIOM CLI.

Extracted from cli.py to keep main entry point under 600 lines.
"""

from src.rl_tune import RL_LR_MIN, RL_LR_MAX

# Import all command handlers
from cli import (
    # Core
    cmd_baseline,
    cmd_bootstrap,
    cmd_curve,
    cmd_full,
    # Partition
    cmd_partition,
    cmd_stress_quorum,
    # Blackout
    cmd_reroute,
    cmd_algo_info,
    cmd_blackout,
    cmd_blackout_sweep,
    cmd_simulate_timeline,
    cmd_extended_sweep,
    cmd_retention_curve,
    cmd_gnn_stub,
    cmd_gnn_nonlinear,
    cmd_cache_sweep,
    cmd_extreme_sweep,
    cmd_overflow_test,
    cmd_innovation_stubs,
    # Pruning
    cmd_entropy_prune,
    cmd_pruning_sweep,
    cmd_extended_250d,
    cmd_verify_chain,
    cmd_pruning_info,
    # Ablation
    cmd_ablate,
    cmd_ablation_sweep,
    cmd_ceiling_track,
    cmd_formula_check,
    cmd_isolate_layers,
    # Depth
    cmd_adaptive_depth_run,
    cmd_depth_scaling_test,
    cmd_compute_depth_single,
    cmd_depth_scaling_info,
    cmd_efficient_sweep_info,
    # RL
    cmd_rl_info,
    cmd_adaptive_info,
    cmd_rl_status,
    cmd_validate_no_static,
    cmd_rl_tune,
    cmd_dynamic_mode,
    cmd_tune_sweep,
    cmd_rl_500_sweep,
    cmd_rl_500_sweep_info,
    # Quantum
    cmd_quantum_estimate,
    cmd_quantum_sim,
    cmd_quantum_rl_hybrid_info,
    # Pipeline
    cmd_lr_pilot,
    cmd_post_tune_execute,
    cmd_full_pipeline,
    cmd_pilot_info,
    cmd_pipeline_info,
    # Scale
    cmd_multi_scale_sweep,
    cmd_scalability_gate_test,
    cmd_scale_info,
    cmd_fractal_info,
    # Fractal ceiling breach
    cmd_fractal_push,
    cmd_alpha_boost,
    cmd_fractal_info_hybrid,
    # Full sweep
    cmd_full_500_sweep,
    # Info
    cmd_hybrid_boost_info,
    # Benchmark (10^12 scale)
    cmd_hybrid_10e12,
    cmd_release_gate,
    cmd_fractal_recursion,
    cmd_fractal_recursion_sweep,
    cmd_benchmark_info,
    # Paths
    cmd_path_status,
    cmd_path_list,
    cmd_path_run,
    cmd_mars_status,
    cmd_multiplanet_status,
    cmd_agi_status,
    cmd_d4_push,
    cmd_d4_info,
    cmd_registry_info,
    # ISRU hybrid (D5 + MOXIE)
    cmd_moxie_info,
    cmd_isru_simulate,
    cmd_isru_closure,
    cmd_d5_isru_hybrid,
    cmd_d5_push_isru,
    cmd_d5_info_isru,
    cmd_isru_info,
    # D6 + Titan hybrid
    cmd_d6_info,
    cmd_d6_push,
    cmd_d6_titan_hybrid,
    cmd_titan_info,
    cmd_titan_config,
    cmd_titan_simulate,
    cmd_perovskite_info,
    cmd_perovskite_project,
    # Adversarial audit
    cmd_audit_info,
    cmd_audit_config,
    cmd_audit_run,
    cmd_audit_stress,
    cmd_d7_info,
    cmd_d7_push,
    cmd_d7_europa_hybrid,
    cmd_europa_info,
    cmd_europa_config,
    cmd_europa_simulate,
    cmd_nrel_info,
    cmd_nrel_config,
    cmd_nrel_validate,
    cmd_nrel_project,
    cmd_nrel_compare,
    # D8 + Multi-planet sync
    cmd_sync_info,
    cmd_sync_run,
    cmd_sync_efficiency,
    cmd_d8_multi_sync,
    # D8 + Fractal encryption
    cmd_encrypt_info,
    cmd_encrypt_keygen,
    cmd_encrypt_audit,
    cmd_encrypt_side_channel,
    cmd_encrypt_inversion,
    # D8 + Atacama validation
    cmd_atacama_info,
    cmd_atacama_validate,
    # D8
    cmd_d8_info,
    cmd_d8_push,
    # D9 + Ganymede
    cmd_d9_info,
    cmd_d9_push,
    cmd_d9_ganymede_hybrid,
    cmd_ganymede_info,
    cmd_ganymede_config,
    cmd_ganymede_navigate,
    cmd_ganymede_autonomy,
    # Atacama drone arrays
    cmd_drone_info,
    cmd_drone_config,
    cmd_drone_coverage,
    cmd_drone_sample,
    cmd_drone_validate,
    # Randomized execution paths
    cmd_randomized_info,
    cmd_randomized_config,
    cmd_randomized_generate,
    cmd_randomized_audit,
    cmd_randomized_timing,
    cmd_randomized_power,
    cmd_randomized_cache,
    # D10 + Jovian hub
    cmd_d10_info,
    cmd_d10_push,
    cmd_d10_jovian_hub,
    cmd_jovian_info,
    cmd_jovian_sync,
    cmd_jovian_autonomy,
    cmd_jovian_coordinate,
    # Callisto
    cmd_callisto_info,
    cmd_callisto_config,
    cmd_callisto_ice,
    cmd_callisto_extract,
    cmd_callisto_radiation,
    cmd_callisto_autonomy,
    cmd_callisto_hub_suitability,
    # Quantum-resistant
    cmd_quantum_resist_info,
    cmd_quantum_resist_config,
    cmd_quantum_keygen,
    cmd_quantum_resist_audit,
    cmd_quantum_spectre,
    cmd_quantum_resist_cache,
    cmd_quantum_spectre_v1,
    cmd_quantum_spectre_v2,
    cmd_quantum_spectre_v4,
    # Dust dynamics
    cmd_dust_dynamics_info,
    cmd_dust_dynamics_config,
    cmd_dust_dynamics,
    cmd_dust_settling,
    cmd_dust_particle,
    cmd_dust_solar_impact,
    cmd_dust_mars_projection,
)

# D13 commands
from cli.d13 import (
    cmd_d13_info,
    cmd_d13_push,
    cmd_d13_solar_hybrid,
)

# Solar orbital hub commands
from cli.solar import (
    cmd_solar_info,
    cmd_solar_positions,
    cmd_solar_windows,
    cmd_solar_transfer,
    cmd_solar_sync,
    cmd_solar_autonomy,
    cmd_solar_simulate,
)

# LES dust dynamics commands
from cli.les import (
    cmd_les_info,
    cmd_les_simulate,
    cmd_les_dust_devil,
    cmd_les_compare,
    cmd_les_validate,
)

# ZK proof commands
from cli.zk import (
    cmd_zk_info,
    cmd_zk_setup,
    cmd_zk_prove,
    cmd_zk_verify,
    cmd_zk_attestation,
    cmd_zk_audit,
    cmd_zk_benchmark,
)

# D14 fractal recursion commands
from cli.d14 import (
    cmd_d14_info,
    cmd_d14_push,
    cmd_d14_interstellar_hybrid,
)

# Interstellar backbone commands
from cli.interstellar import (
    cmd_interstellar_info,
    cmd_interstellar_bodies,
    cmd_interstellar_positions,
    cmd_interstellar_windows,
    cmd_interstellar_sync,
    cmd_interstellar_autonomy,
    cmd_interstellar_failover,
)

# PLONK ZK proof commands
from cli.plonk import (
    cmd_plonk_info,
    cmd_plonk_setup,
    cmd_plonk_prove,
    cmd_plonk_verify,
    cmd_plonk_recursive,
    cmd_plonk_attestation,
    cmd_plonk_audit,
    cmd_plonk_benchmark,
    cmd_plonk_compare,
)

# D15 quantum-entangled fractal commands
from cli.d15 import (
    cmd_d15_info,
    cmd_d15_push,
    cmd_d15_chaos_hybrid,
    cmd_d15_entanglement,
)

# Chaotic n-body simulation commands
from cli.chaos import (
    cmd_chaos_info,
    cmd_chaos_simulate,
    cmd_chaos_stability,
    cmd_chaos_monte_carlo,
    cmd_chaos_backbone_tolerance,
    cmd_chaos_lyapunov,
)

# Halo2 recursive ZK proof commands
from cli.halo2 import (
    cmd_halo2_info,
    cmd_halo2_prove,
    cmd_halo2_verify,
    cmd_halo2_recursive,
    cmd_halo2_attestation,
    cmd_halo2_audit,
    cmd_halo2_benchmark,
    cmd_halo2_compare,
    cmd_halo2_infinite_chain,
)

# D16 topological fractal + Kuiper commands
from cli.d16 import (
    cmd_d16_info,
    cmd_d16_push,
    cmd_d16_topological,
    cmd_d16_homology,
    cmd_d16_kuiper_hybrid,
    cmd_kuiper_info,
    cmd_kuiper_simulate,
    cmd_kuiper_resonances,
    cmd_bulletproofs_info,
    cmd_bulletproofs_stress,
    cmd_bulletproofs_chain,
    cmd_ml_ensemble_info,
    cmd_ml_ensemble_predict,
    cmd_ml_ensemble_train,
)

# D17 depth-first fractal + Heliosphere/Oort commands
from cli.d17 import (
    cmd_d17_info,
    cmd_d17_push,
    cmd_d17_depthfirst,
    cmd_d17_plateau_check,
    cmd_d17_heliosphere_hybrid,
    cmd_heliosphere_info,
    cmd_heliosphere_zones,
    cmd_heliosphere_status,
    cmd_oort_info,
    cmd_oort_simulate,
    cmd_oort_latency,
    cmd_oort_autonomy,
    cmd_oort_compression,
    cmd_oort_stability,
    cmd_bulletproofs_infinite_chain,
    cmd_bulletproofs_10k_stress,
    cmd_ml_ensemble_90s,
    cmd_ml_ensemble_90s_info,
)

# D18 interstellar relay commands
from cli.d18 import (
    cmd_d18_info,
    cmd_d18_push,
    cmd_d18_pruning,
    cmd_d18_compression,
    cmd_d18_interstellar_hybrid,
)

# D19 swarm intelligence commands
from cli.d19 import (
    cmd_d19_info,
    cmd_d19_run,
    cmd_d19_gate_1,
    cmd_d19_gate_2,
    cmd_d19_gate_1_2,
    cmd_d19_gate_3,
    cmd_d19_gate_4,
    cmd_d19_gate_5,
    cmd_d19_tweet,
)

# Interstellar relay commands
from cli.relay import (
    cmd_relay_info,
    cmd_relay_simulate,
    cmd_relay_latency,
    cmd_relay_nodes,
    cmd_relay_stress,
)

# Quantum alternative commands
from cli.quantum import (
    cmd_quantum_alt_info,
    cmd_quantum_alt_simulate,
    cmd_quantum_alt_correlation,
    cmd_quantum_alt_bell,
)

# Elon-sphere commands
from cli.elonsphere import (
    cmd_starlink_info,
    cmd_starlink_simulate,
    cmd_grok_info,
    cmd_grok_tune,
    cmd_xai_info,
    cmd_xai_simulate,
    cmd_dojo_info,
    cmd_dojo_offload,
    cmd_federation_info,
    cmd_federation_consensus,
)

# Live relay HIL commands
from cli.live_relay import (
    cmd_live_relay_info,
    cmd_live_relay_connect,
    cmd_live_relay_test,
    cmd_live_relay_mars,
    cmd_live_relay_stress,
    cmd_live_relay_status,
)

# Lag consensus commands
from cli.consensus import (
    cmd_consensus_info,
    cmd_consensus_init,
    cmd_consensus_simulate,
    cmd_consensus_election,
    cmd_consensus_status,
)

# Pruning v4 + quantum refine commands
from cli.pruning import (
    cmd_pruning_v4,
    cmd_pruning_v4_compare,
    cmd_pruning_v4_status,
    cmd_quantum_refine,
    cmd_quantum_refine_info,
)

# Mars relay commands
from cli.mars_relay import (
    cmd_mars_relay_info,
    cmd_mars_relay_deploy,
    cmd_mars_relay_mesh,
    cmd_mars_relay_proof,
    cmd_mars_relay_latency,
    cmd_mars_relay_status,
    cmd_mars_relay_opposition,
    cmd_mars_relay_conjunction,
    cmd_mars_relay_stress,
)

# Multi-planet federation commands
from cli.federation_multiplanet import (
    cmd_federation_multiplanet_info,
    cmd_federation_multiplanet_init,
    cmd_federation_multiplanet_add,
    cmd_federation_multiplanet_sync,
    cmd_federation_multiplanet_consensus,
    cmd_federation_multiplanet_arbitrate,
    cmd_federation_multiplanet_status,
    cmd_federation_multiplanet_health,
    cmd_federation_multiplanet_partition,
    cmd_federation_multiplanet_recover,
)

# Gravity adaptive commands
from cli.gravity_adaptive import (
    cmd_gravity_info,
    cmd_gravity_adjust,
    cmd_gravity_consensus,
    cmd_gravity_packet,
    cmd_gravity_validate,
    cmd_gravity_status,
    cmd_gravity_all_planets,
)

# Quantum v2 commands
from cli.quantum_v2 import (
    cmd_quantum_v2_info,
    cmd_quantum_v2_refine,
    cmd_quantum_v2_iterative,
    cmd_quantum_v2_compare,
    cmd_quantum_v2_decoherence,
    cmd_quantum_v2_correction,
    cmd_quantum_v2_status,
    cmd_quantum_v2_validate,
)

# Swarm testnet commands
from cli.swarm import (
    cmd_swarm_info,
    cmd_swarm_init,
    cmd_swarm_deploy,
    cmd_swarm_mesh,
    cmd_swarm_consensus,
    cmd_swarm_stress,
    cmd_swarm_status,
    cmd_testnet_info,
    cmd_testnet_ethereum,
    cmd_testnet_solana,
    cmd_testnet_bridge,
    cmd_testnet_cross_chain,
    cmd_testnet_sync,
    cmd_testnet_status,
    cmd_testnet_stress,
)


def dispatch(args, docstring: str) -> None:
    """Dispatch command based on parsed arguments.

    Args:
        args: Parsed arguments from argparse
        docstring: CLI docstring to show for help
    """
    reroute_enabled = args.reroute or args.reroute_enabled

    # Info commands (no args needed)
    if args.rl_info:
        return cmd_rl_info()
    if args.adaptive_info:
        return cmd_adaptive_info()
    if args.rl_status:
        return cmd_rl_status()
    if args.validate_no_static:
        return cmd_validate_no_static()
    if args.depth_scaling_info:
        return cmd_depth_scaling_info()
    if args.efficient_sweep_info:
        return cmd_efficient_sweep_info()
    if args.rl_500_sweep_info:
        return cmd_rl_500_sweep_info()
    if args.pilot_info:
        return cmd_pilot_info()
    if args.quantum_rl_hybrid_info:
        return cmd_quantum_rl_hybrid_info()
    if args.pipeline_info:
        return cmd_pipeline_info()
    if args.scale_info:
        return cmd_scale_info()
    if args.fractal_info:
        return cmd_fractal_info()
    if args.algo_info:
        return cmd_algo_info()
    if args.gnn_stub:
        return cmd_gnn_stub()
    if args.innovation_stubs:
        return cmd_innovation_stubs()
    if args.pruning_info:
        return cmd_pruning_info()
    if args.formula_check:
        return cmd_formula_check()
    if args.retention_curve:
        return cmd_retention_curve()

    # Benchmark (10^12 scale) commands
    if args.benchmark_info:
        return cmd_benchmark_info()
    if args.hybrid_10e12:
        return cmd_hybrid_10e12(args.tree_size, args.base_alpha, args.simulate)
    if args.release_gate:
        return cmd_release_gate(args.simulate)
    if args.fractal_recursion_sweep:
        return cmd_fractal_recursion_sweep(
            args.tree_size, args.base_alpha, args.simulate
        )
    if args.fractal_recursion:
        return cmd_fractal_recursion(
            args.tree_size, args.base_alpha, args.recursion_depth, args.simulate
        )

    # Path exploration commands
    if args.d4_info:
        return cmd_d4_info()
    if args.registry_info:
        return cmd_registry_info()
    if args.d4_push:
        return cmd_d4_push(args.tree_size, args.base_alpha, args.simulate)
    if args.path_status:
        return cmd_path_status(args.path)
    if args.path_list:
        return cmd_path_list()
    if args.mars_status:
        return cmd_mars_status()
    if args.multiplanet_status:
        return cmd_multiplanet_status()
    if args.agi_status:
        return cmd_agi_status()
    if args.path is not None and args.path_cmd is not None:
        return cmd_path_run(args.path, args.path_cmd)

    # D5 + ISRU hybrid commands
    if args.moxie_info:
        return cmd_moxie_info()
    if args.isru_info:
        return cmd_isru_info()
    if args.d5_info:
        return cmd_d5_info_isru()
    if args.isru_closure:
        return cmd_isru_closure(args.simulate)
    if args.isru_simulate:
        return cmd_isru_simulate(args.hours, args.crew, args.moxie_units, args.simulate)
    if args.d5_push:
        return cmd_d5_push_isru(args.tree_size, args.base_alpha, args.simulate)
    if args.d5_isru_hybrid:
        return cmd_d5_isru_hybrid(
            args.tree_size,
            args.base_alpha,
            args.crew,
            args.hours,
            args.moxie_units,
            args.simulate,
        )

    # D6 + Titan hybrid commands
    if args.d6_info:
        return cmd_d6_info()
    if args.titan_info:
        return cmd_titan_info()
    if args.titan_config:
        return cmd_titan_config()
    if args.perovskite_info:
        return cmd_perovskite_info()
    if args.d6_push:
        return cmd_d6_push(args.tree_size, args.base_alpha, args.simulate)
    if args.titan_simulate:
        return cmd_titan_simulate(
            args.titan_duration, args.titan_extraction_rate, args.simulate
        )
    if args.d6_titan_hybrid:
        return cmd_d6_titan_hybrid(
            args.tree_size, args.base_alpha, args.titan_duration, args.simulate
        )
    if args.perovskite_project:
        return cmd_perovskite_project(args.perovskite_years, args.perovskite_growth)

    # Adversarial audit commands
    if args.audit_info:
        return cmd_audit_info()
    if args.audit_config:
        return cmd_audit_config()
    if args.audit_run:
        return cmd_audit_run(args.audit_noise, args.audit_iterations, args.simulate)
    if args.audit_stress:
        return cmd_audit_stress(None, 50, args.simulate)

    # D7 + Europa hybrid commands
    if args.d7_info:
        return cmd_d7_info()
    if args.europa_info:
        return cmd_europa_info()
    if args.europa_config:
        return cmd_europa_config()
    if args.d7_push:
        return cmd_d7_push(args.tree_size, args.base_alpha, args.simulate)
    if args.europa_simulate:
        return cmd_europa_simulate(
            args.europa_depth,
            args.europa_duration,
            args.europa_drill_rate,
            args.simulate,
        )
    if args.d7_europa_hybrid:
        return cmd_d7_europa_hybrid(
            args.tree_size,
            args.base_alpha,
            args.europa_depth,
            args.europa_duration,
            args.simulate,
        )

    # NREL perovskite validation commands
    if args.nrel_info:
        return cmd_nrel_info()
    if args.nrel_config:
        return cmd_nrel_config()
    if args.nrel_validate:
        return cmd_nrel_validate(args.nrel_efficiency, args.simulate)
    if args.nrel_project:
        return cmd_nrel_project(args.nrel_years, args.nrel_initial, args.simulate)
    if args.nrel_compare:
        return cmd_nrel_compare(
            args.nrel_efficiency, args.moxie_efficiency, args.simulate
        )

    # D8 + Multi-planet sync commands
    if args.d8_info:
        return cmd_d8_info()
    if args.sync_info:
        return cmd_sync_info()
    if args.encrypt_info:
        return cmd_encrypt_info()
    if args.atacama_info:
        return cmd_atacama_info()
    if args.d8_push:
        return cmd_d8_push(args.tree_size, args.base_alpha, args.simulate)
    if args.d8_multi_sync:
        return cmd_d8_multi_sync(args.tree_size, args.base_alpha, args.simulate)
    if args.sync_run:
        return cmd_sync_run(args.simulate)
    if args.sync_efficiency:
        return cmd_sync_efficiency()
    if args.atacama_validate:
        return cmd_atacama_validate(args.simulate)
    if args.encrypt_keygen:
        return cmd_encrypt_keygen(args.encrypt_key_depth)
    if args.encrypt_audit:
        return cmd_encrypt_audit(args.simulate)
    if args.encrypt_side_channel:
        return cmd_encrypt_side_channel(args.encrypt_iterations)
    if args.encrypt_inversion:
        return cmd_encrypt_inversion(args.encrypt_iterations)

    # D9 + Ganymede + randomized paths commands
    if args.d9_info:
        return cmd_d9_info()
    if args.ganymede_info:
        return cmd_ganymede_info()
    if args.ganymede_config:
        return cmd_ganymede_config()
    if args.drone_info:
        return cmd_drone_info()
    if args.drone_config:
        return cmd_drone_config()
    if args.randomized_info:
        return cmd_randomized_info()
    if args.randomized_config:
        return cmd_randomized_config()
    if args.d9_push:
        return cmd_d9_push(args.tree_size, args.base_alpha, args.simulate)
    if args.d9_ganymede_hybrid:
        return cmd_d9_ganymede_hybrid(
            args.tree_size,
            args.base_alpha,
            args.ganymede_nav_mode,
            args.ganymede_duration,
            args.simulate,
        )
    if args.ganymede_navigate:
        return cmd_ganymede_navigate(
            args.ganymede_nav_mode, args.ganymede_duration, args.simulate
        )
    if args.ganymede_autonomy:
        return cmd_ganymede_autonomy(args.simulate)
    if args.drone_coverage:
        return cmd_drone_coverage(args.drone_count, args.drone_area, args.simulate)
    if args.drone_sample:
        return cmd_drone_sample(10, args.drone_duration, args.simulate)
    if args.drone_validate:
        return cmd_drone_validate(
            args.drone_count, args.drone_area, args.drone_duration, args.simulate
        )
    if args.randomized_generate:
        return cmd_randomized_generate(args.randomized_depth)
    if args.randomized_audit:
        return cmd_randomized_audit(args.randomized_iterations, args.simulate)
    if args.randomized_timing:
        return cmd_randomized_timing(args.randomized_iterations, args.simulate)
    if args.randomized_power:
        return cmd_randomized_power(args.randomized_iterations, args.simulate)
    if args.randomized_cache:
        return cmd_randomized_cache(args.randomized_iterations, args.simulate)

    # D10 + Jovian hub commands
    if args.d10_info:
        return cmd_d10_info()
    if args.d10_push:
        return cmd_d10_push(args.tree_size, args.base_alpha, args.simulate)
    if args.d10_jovian_hub:
        return cmd_d10_jovian_hub(args.tree_size, args.base_alpha, args.simulate)
    if args.jovian_info:
        return cmd_jovian_info()
    if args.jovian_sync:
        return cmd_jovian_sync(args.simulate)
    if args.jovian_autonomy:
        return cmd_jovian_autonomy(args.simulate)
    if args.jovian_coordinate:
        return cmd_jovian_coordinate(args.simulate)

    # Callisto commands
    if args.callisto_info:
        return cmd_callisto_info()
    if args.callisto_config:
        return cmd_callisto_config()
    if args.callisto_ice:
        return cmd_callisto_ice(args.simulate)
    if args.callisto_extract:
        return cmd_callisto_extract(
            args.callisto_duration, args.callisto_rate, args.simulate
        )
    if args.callisto_radiation:
        return cmd_callisto_radiation(args.simulate)
    if args.callisto_autonomy:
        return cmd_callisto_autonomy(args.simulate)
    if args.callisto_hub:
        return cmd_callisto_hub_suitability(args.simulate)

    # Quantum-resistant commands
    if args.quantum_resist_info:
        return cmd_quantum_resist_info()
    if args.quantum_resist_config:
        return cmd_quantum_resist_config()
    if args.quantum_keygen:
        return cmd_quantum_keygen(args.quantum_key_size)
    if args.quantum_audit:
        return cmd_quantum_resist_audit(args.quantum_iterations, args.simulate)
    if args.quantum_spectre:
        return cmd_quantum_spectre(args.quantum_iterations, args.simulate)
    if args.quantum_cache:
        return cmd_quantum_resist_cache(args.quantum_iterations, args.simulate)
    if args.spectre_v1:
        return cmd_quantum_spectre_v1(args.quantum_iterations, args.simulate)
    if args.spectre_v2:
        return cmd_quantum_spectre_v2(args.quantum_iterations, args.simulate)
    if args.spectre_v4:
        return cmd_quantum_spectre_v4(args.quantum_iterations, args.simulate)

    # Dust dynamics commands
    if args.dust_info:
        return cmd_dust_dynamics_info()
    if args.dust_config:
        return cmd_dust_dynamics_config()
    if args.dust_dynamics:
        return cmd_dust_dynamics(args.simulate)
    if args.dust_settling:
        return cmd_dust_settling(args.dust_duration, args.simulate)
    if args.dust_particle:
        return cmd_dust_particle(args.simulate)
    if args.dust_solar_impact:
        return cmd_dust_solar_impact(args.dust_depth_mm, args.simulate)
    if args.dust_mars:
        return cmd_dust_mars_projection(args.simulate)

    # D13 commands
    if args.d13_info:
        return cmd_d13_info(args)
    if args.d13_push:
        return cmd_d13_push(args)
    if args.d13_solar_hybrid:
        return cmd_d13_solar_hybrid(args)

    # Solar orbital hub commands
    if args.solar_info:
        return cmd_solar_info(args)
    if args.solar_positions:
        return cmd_solar_positions(args)
    if args.solar_windows:
        return cmd_solar_windows(args)
    if args.solar_transfer:
        return cmd_solar_transfer(args)
    if args.solar_sync:
        return cmd_solar_sync(args)
    if args.solar_autonomy:
        return cmd_solar_autonomy(args)
    if args.solar_simulate:
        return cmd_solar_simulate(args)

    # LES dust dynamics commands
    if args.les_info:
        return cmd_les_info(args)
    if args.les_simulate:
        return cmd_les_simulate(args)
    if args.les_dust_devil:
        return cmd_les_dust_devil(args)
    if args.les_compare:
        return cmd_les_compare(args)
    if args.les_validate:
        return cmd_les_validate(args)

    # ZK proof commands
    if args.zk_info:
        return cmd_zk_info(args)
    if args.zk_setup:
        return cmd_zk_setup(args)
    if args.zk_prove:
        return cmd_zk_prove(args)
    if args.zk_verify:
        return cmd_zk_verify(args)
    if args.zk_attestation:
        return cmd_zk_attestation(args)
    if args.zk_audit:
        return cmd_zk_audit(args)
    if args.zk_benchmark:
        return cmd_zk_benchmark(args)

    # D14 fractal recursion commands
    if args.d14_info:
        return cmd_d14_info(args)
    if args.d14_push:
        return cmd_d14_push(args)
    if args.d14_interstellar_hybrid:
        return cmd_d14_interstellar_hybrid(args)

    # Interstellar backbone commands
    if args.interstellar_info:
        return cmd_interstellar_info(args)
    if args.interstellar_bodies:
        return cmd_interstellar_bodies(args)
    if args.interstellar_positions:
        # Use interstellar_timestamp for positions
        args.timestamp = args.interstellar_timestamp
        return cmd_interstellar_positions(args)
    if args.interstellar_windows:
        # Use interstellar_timestamp for windows
        args.timestamp = args.interstellar_timestamp
        return cmd_interstellar_windows(args)
    if args.interstellar_sync:
        # Use interstellar_duration for sync
        args.duration_days = args.interstellar_duration
        return cmd_interstellar_sync(args)
    if args.interstellar_autonomy:
        return cmd_interstellar_autonomy(args)
    if args.interstellar_failover:
        # Use interstellar_body for failover
        args.body = args.interstellar_body
        return cmd_interstellar_failover(args)

    # Atacama real-time commands
    if args.atacama_realtime:
        from src.cfd_dust_dynamics import atacama_les_realtime
        import json

        result = atacama_les_realtime(
            duration_sec=args.atacama_realtime_duration,
            sampling_hz=args.atacama_sampling_hz,
        )
        print("\n=== ATACAMA REAL-TIME LES ===")
        print(f"Duration: {result.get('duration_sec', 0)} sec")
        print(f"Sampling: {result.get('sampling_hz', 0)} Hz")
        print(f"Samples collected: {result.get('samples_collected', 0)}")
        print(f"Correlation: {result.get('correlation', 0):.4f}")
        print(f"Target met: {result.get('target_met', False)}")
        return
    if args.atacama_dust_devil:
        from src.cfd_dust_dynamics import track_dust_devil
        import json

        result = track_dust_devil(duration_sec=args.atacama_realtime_duration)
        print("\n=== ATACAMA DUST DEVIL TRACKING ===")
        print(f"Duration: {result.get('duration_sec', 0)} sec")
        print(f"Samples: {result.get('samples_collected', 0)}")
        print(f"Max velocity: {result.get('max_velocity_ms', 0):.2f} m/s")
        print(f"Max diameter: {result.get('max_diameter_m', 0):.1f} m")
        print(f"Max height: {result.get('max_height_m', 0):.1f} m")
        return
    if args.atacama_correlation:
        from src.cfd_dust_dynamics import compute_realtime_correlation

        result = compute_realtime_correlation()
        print("\n=== ATACAMA CORRELATION ===")
        print(f"Correlation: {result.get('correlation', 0):.4f}")
        print(f"Target: {result.get('target', 0):.4f}")
        print(f"Target met: {result.get('target_met', False)}")
        return
    if args.atacama_full_validation:
        from src.cfd_dust_dynamics import run_atacama_validation
        import json

        result = run_atacama_validation()
        print("\n=== ATACAMA FULL VALIDATION ===")
        print(json.dumps(result, indent=2))
        return

    # PLONK ZK proof commands
    if args.plonk_info:
        return cmd_plonk_info(args)
    if args.plonk_setup:
        args.participants = args.plonk_participants
        return cmd_plonk_setup(args)
    if args.plonk_prove:
        return cmd_plonk_prove(args)
    if args.plonk_verify:
        return cmd_plonk_verify(args)
    if args.plonk_recursive:
        args.count = args.plonk_recursive_count
        return cmd_plonk_recursive(args)
    if args.plonk_attestation:
        return cmd_plonk_attestation(args)
    if args.plonk_audit:
        args.count = args.plonk_attestation_count
        return cmd_plonk_audit(args)
    if args.plonk_benchmark:
        args.iterations = args.plonk_iterations
        return cmd_plonk_benchmark(args)
    if args.plonk_compare:
        return cmd_plonk_compare(args)

    # D15 quantum-entangled fractal commands
    if args.d15_info:
        return cmd_d15_info(args)
    if args.d15_push:
        return cmd_d15_push(args)
    if args.d15_chaos_hybrid:
        return cmd_d15_chaos_hybrid(args)
    if args.d15_entanglement:
        return cmd_d15_entanglement(args)

    # Chaotic n-body simulation commands
    if args.chaos_info:
        return cmd_chaos_info(args)
    if args.chaos_simulate:
        return cmd_chaos_simulate(args)
    if args.chaos_stability:
        return cmd_chaos_stability(args)
    if args.chaos_monte_carlo:
        return cmd_chaos_monte_carlo(args)
    if args.chaos_backbone_tolerance:
        return cmd_chaos_backbone_tolerance(args)
    if args.chaos_lyapunov:
        return cmd_chaos_lyapunov(args)

    # Halo2 recursive ZK proof commands
    if args.halo2_info:
        return cmd_halo2_info(args)
    if args.halo2_prove:
        return cmd_halo2_prove(args)
    if args.halo2_verify:
        return cmd_halo2_verify(args)
    if args.halo2_recursive:
        return cmd_halo2_recursive(args)
    if args.halo2_attestation:
        return cmd_halo2_attestation(args)
    if args.halo2_audit:
        args.count = args.halo2_attestation_count
        return cmd_halo2_audit(args)
    if args.halo2_benchmark:
        args.iterations = args.halo2_iterations
        return cmd_halo2_benchmark(args)
    if args.halo2_compare:
        return cmd_halo2_compare(args)
    if args.halo2_infinite_chain:
        return cmd_halo2_infinite_chain(args)

    # D16 topological fractal + Kuiper commands
    if args.d16_info:
        return cmd_d16_info(args)
    if args.d16_push:
        return cmd_d16_push(args)
    if args.d16_topological:
        return cmd_d16_topological(args)
    if args.d16_homology:
        return cmd_d16_homology(args)
    if args.d16_kuiper_hybrid:
        return cmd_d16_kuiper_hybrid(args)
    if args.kuiper_info:
        return cmd_kuiper_info(args)
    if args.kuiper_simulate:
        return cmd_kuiper_simulate(args)
    if args.kuiper_resonances:
        return cmd_kuiper_resonances(args)
    if args.bulletproofs_info:
        return cmd_bulletproofs_info(args)
    if args.bulletproofs_stress:
        return cmd_bulletproofs_stress(args)
    if args.bulletproofs_chain:
        return cmd_bulletproofs_chain(args)
    if args.ml_ensemble_info:
        return cmd_ml_ensemble_info(args)
    if args.ml_ensemble_predict:
        return cmd_ml_ensemble_predict(args)
    if args.ml_ensemble_train:
        return cmd_ml_ensemble_train(args)

    # D17 depth-first fractal + Heliosphere/Oort commands
    if args.d17_info:
        return cmd_d17_info(args)
    if args.d17_push:
        return cmd_d17_push(args)
    if args.d17_depthfirst:
        return cmd_d17_depthfirst(args)
    if args.d17_plateau_check:
        return cmd_d17_plateau_check(args)
    if args.d17_heliosphere_hybrid:
        return cmd_d17_heliosphere_hybrid(args)
    if args.heliosphere_info:
        return cmd_heliosphere_info(args)
    if args.heliosphere_zones:
        return cmd_heliosphere_zones(args)
    if args.heliosphere_status:
        return cmd_heliosphere_status(args)
    if args.oort_info:
        return cmd_oort_info(args)
    if args.oort_simulate:
        return cmd_oort_simulate(args)
    if args.oort_latency:
        return cmd_oort_latency(args)
    if args.oort_autonomy:
        return cmd_oort_autonomy(args)
    if args.oort_compression:
        return cmd_oort_compression(args)
    if args.oort_stability:
        return cmd_oort_stability(args)
    if args.bulletproofs_infinite_chain:
        return cmd_bulletproofs_infinite_chain(args)
    if args.bulletproofs_10k_stress:
        return cmd_bulletproofs_10k_stress(args)
    if args.ml_ensemble_90s:
        return cmd_ml_ensemble_90s(args)
    if args.ml_ensemble_90s_info:
        return cmd_ml_ensemble_90s_info(args)

    # D18 interstellar relay commands
    if args.d18_info:
        return cmd_d18_info(args)
    if args.d18_push:
        return cmd_d18_push(args)
    if args.d18_pruning:
        return cmd_d18_pruning(args)
    if args.d18_compression:
        return cmd_d18_compression(args)
    if args.d18_interstellar_hybrid:
        return cmd_d18_interstellar_hybrid(args)

    # D19 swarm intelligence commands
    if args.d19_info:
        return cmd_d19_info(args)
    if args.d19_run:
        return cmd_d19_run(args)
    if args.d19_gate_1:
        return cmd_d19_gate_1(args)
    if args.d19_gate_2:
        return cmd_d19_gate_2(args)
    if args.d19_gate_1_2:
        return cmd_d19_gate_1_2(args)
    if args.d19_gate_3:
        return cmd_d19_gate_3(args)
    if args.d19_gate_4:
        return cmd_d19_gate_4(args)
    if args.d19_gate_5:
        return cmd_d19_gate_5(args)
    if args.d19_tweet:
        return cmd_d19_tweet(args)

    # Interstellar relay commands
    if args.relay_info:
        return cmd_relay_info(args)
    if args.relay_simulate:
        return cmd_relay_simulate(args)
    if args.relay_latency:
        return cmd_relay_latency(args)
    if args.relay_nodes:
        return cmd_relay_nodes(args)
    if args.relay_stress:
        return cmd_relay_stress(args)

    # Quantum alternative commands
    if args.quantum_alt_info:
        return cmd_quantum_alt_info(args)
    if args.quantum_alt_simulate:
        return cmd_quantum_alt_simulate(args)
    if args.quantum_alt_correlation:
        return cmd_quantum_alt_correlation(args)
    if args.quantum_alt_bell:
        return cmd_quantum_alt_bell(args)

    # Elon-sphere commands
    if args.starlink_info:
        return cmd_starlink_info(args)
    if args.starlink_simulate:
        return cmd_starlink_simulate(args)
    if args.grok_info:
        return cmd_grok_info(args)
    if args.grok_tune:
        return cmd_grok_tune(args)
    if args.xai_info:
        return cmd_xai_info(args)
    if args.xai_simulate:
        return cmd_xai_simulate(args)
    if args.dojo_info:
        return cmd_dojo_info(args)
    if args.dojo_offload:
        return cmd_dojo_offload(args)

    # Federation commands
    if args.federation_info:
        return cmd_federation_info(args)
    if args.federation_consensus:
        return cmd_federation_consensus(args)

    # Live relay HIL commands
    if args.live_relay_info:
        return cmd_live_relay_info(args)
    if args.live_relay_connect:
        return cmd_live_relay_connect(args)
    if args.live_relay_test:
        args.duration = args.hil_duration
        return cmd_live_relay_test(args)
    if args.live_relay_mars:
        args.duration = args.mars_proof_hours
        return cmd_live_relay_mars(args)
    if args.live_relay_stress:
        args.iterations = args.hil_iterations
        return cmd_live_relay_stress(args)
    if args.live_relay_status:
        return cmd_live_relay_status(args)

    # Lag consensus commands
    if args.consensus_info:
        return cmd_consensus_info(args)
    if args.consensus_init:
        args.nodes = args.consensus_nodes
        return cmd_consensus_init(args)
    if args.consensus_simulate:
        args.nodes = args.consensus_nodes
        args.latency = args.consensus_latency
        return cmd_consensus_simulate(args)
    if args.consensus_election:
        args.nodes = args.consensus_nodes
        return cmd_consensus_election(args)
    if args.consensus_status:
        return cmd_consensus_status(args)

    # Pruning v4 commands
    if args.pruning_v4:
        args.target = args.pruning_v4_target
        return cmd_pruning_v4(args)
    if args.pruning_v4_compare:
        return cmd_pruning_v4_compare(args)
    if args.pruning_v4_status:
        return cmd_pruning_v4_status(args)

    # Quantum refine commands
    if args.quantum_refine:
        args.iterations = args.quantum_refine_iterations
        return cmd_quantum_refine(args)
    if args.quantum_refine_info:
        return cmd_quantum_refine_info(args)

    # Mars relay commands
    if args.mars_relay_info:
        return cmd_mars_relay_info(args)
    if args.mars_relay_deploy:
        return cmd_mars_relay_deploy(args)
    if args.mars_relay_mesh:
        return cmd_mars_relay_mesh(args)
    if args.mars_relay_proof:
        return cmd_mars_relay_proof(args)
    if args.mars_relay_latency:
        return cmd_mars_relay_latency(args)
    if args.mars_relay_status:
        return cmd_mars_relay_status(args)
    if args.mars_relay_opposition:
        return cmd_mars_relay_opposition(args)
    if args.mars_relay_conjunction:
        return cmd_mars_relay_conjunction(args)
    if args.mars_relay_stress:
        return cmd_mars_relay_stress(args)

    # Multi-planet federation commands
    if args.federation_multiplanet_info:
        return cmd_federation_multiplanet_info(args)
    if args.federation_multiplanet_init:
        return cmd_federation_multiplanet_init(args)
    if args.federation_multiplanet_add:
        return cmd_federation_multiplanet_add(args)
    if args.federation_multiplanet_sync:
        return cmd_federation_multiplanet_sync(args)
    if args.federation_multiplanet_consensus:
        return cmd_federation_multiplanet_consensus(args)
    if args.federation_multiplanet_arbitrate:
        return cmd_federation_multiplanet_arbitrate(args)
    if args.federation_multiplanet_status:
        return cmd_federation_multiplanet_status(args)
    if args.federation_multiplanet_health:
        return cmd_federation_multiplanet_health(args)
    if args.federation_multiplanet_partition:
        return cmd_federation_multiplanet_partition(args)
    if args.federation_multiplanet_recover:
        return cmd_federation_multiplanet_recover(args)

    # Gravity adaptive commands
    if args.gravity_info:
        return cmd_gravity_info(args)
    if args.gravity_adjust:
        return cmd_gravity_adjust(args)
    if args.gravity_consensus:
        return cmd_gravity_consensus(args)
    if args.gravity_packet:
        return cmd_gravity_packet(args)
    if args.gravity_validate:
        return cmd_gravity_validate(args)
    if args.gravity_status:
        return cmd_gravity_status(args)
    if args.gravity_all_planets:
        return cmd_gravity_all_planets(args)

    # Quantum v2 commands
    if args.quantum_v2_info:
        return cmd_quantum_v2_info(args)
    if args.quantum_v2_refine:
        return cmd_quantum_v2_refine(args)
    if args.quantum_v2_iterative:
        return cmd_quantum_v2_iterative(args)
    if args.quantum_v2_compare:
        return cmd_quantum_v2_compare(args)
    if args.quantum_v2_decoherence:
        return cmd_quantum_v2_decoherence(args)
    if args.quantum_v2_correction:
        return cmd_quantum_v2_correction(args)
    if args.quantum_v2_status:
        return cmd_quantum_v2_status(args)
    if args.quantum_v2_validate:
        return cmd_quantum_v2_validate(args)

    # Swarm testnet commands
    if args.swarm_info:
        return cmd_swarm_info(args)
    if args.swarm_init:
        return cmd_swarm_init(args)
    if args.swarm_deploy:
        return cmd_swarm_deploy(args)
    if args.swarm_mesh:
        return cmd_swarm_mesh(args)
    if args.swarm_consensus:
        return cmd_swarm_consensus(args)
    if args.swarm_stress:
        return cmd_swarm_stress(args)
    if args.swarm_status:
        return cmd_swarm_status(args)

    # Parallel testnet commands
    if args.testnet_info:
        return cmd_testnet_info(args)
    if args.testnet_ethereum:
        return cmd_testnet_ethereum(args)
    if args.testnet_solana:
        return cmd_testnet_solana(args)
    if args.testnet_bridge:
        return cmd_testnet_bridge(args)
    if args.testnet_cross_chain:
        return cmd_testnet_cross_chain(args)
    if args.testnet_sync:
        return cmd_testnet_sync(args)
    if args.testnet_status:
        return cmd_testnet_status(args)
    if args.testnet_stress:
        return cmd_testnet_stress(args)

    # Atacama 200Hz commands
    if args.atacama_200hz:
        from src.cfd_dust_dynamics import atacama_200hz

        result = atacama_200hz(
            duration_sec=args.atacama_200hz_duration,
        )
        print("\n=== ATACAMA 200Hz MODE ===")
        print(f"Duration: {result.get('duration_sec', 0)} sec")
        print(f"Sampling: {result.get('sampling_hz', 200)} Hz")
        print(f"Samples collected: {result.get('samples_collected', 0)}")
        print(f"Correlation: {result.get('correlation', 0):.4f}")
        print(f"Target met: {result.get('target_met', False)}")
        return
    if args.atacama_200hz_info:
        from src.cfd_dust_dynamics import get_atacama_200hz_info

        result = get_atacama_200hz_info()
        print("\n=== ATACAMA 200Hz CONFIGURATION ===")
        print(f"Sampling rate: {result.get('sampling_hz', 200)} Hz")
        print(f"Correlation target: {result.get('correlation_target', 0.97)}")
        print("Upgrade from: 100Hz")
        return
    if args.atacama_200hz_predict:
        from src.cfd_dust_dynamics import predict_dust_devil

        result = predict_dust_devil(
            duration_sec=args.atacama_200hz_duration,
            sampling_hz=200,
        )
        print("\n=== ATACAMA 200Hz DUST DEVIL PREDICTION ===")
        print(f"Duration: {result.get('duration_sec', 0)} sec")
        print(f"Predictions made: {result.get('predictions_made', 0)}")
        print(f"Prediction accuracy: {result.get('accuracy', 0):.2%}")
        print(f"Lead time: {result.get('lead_time_sec', 0):.1f} sec")
        return

    # Expanded AGI audit commands
    if args.audit_expanded:
        from src.agi_audit_expanded import run_expanded_audit
        import json

        result = run_expanded_audit(attack_type="all", iterations=args.audit_iterations)
        print(json.dumps(result, indent=2))
        return
    if args.audit_injection:
        from src.agi_audit_expanded import run_expanded_audit
        import json

        result = run_expanded_audit(
            attack_type="injection", iterations=args.audit_iterations
        )
        print(json.dumps(result, indent=2))
        return
    if args.audit_poisoning:
        from src.agi_audit_expanded import run_expanded_audit
        import json

        result = run_expanded_audit(
            attack_type="poisoning", iterations=args.audit_iterations
        )
        print(json.dumps(result, indent=2))
        return

    # Scale commands
    if args.scalability_gate_test:
        return cmd_scalability_gate_test()
    if args.multi_scale_sweep is not None:
        return cmd_multi_scale_sweep(args.multi_scale_sweep, args.simulate)

    # Fractal ceiling breach commands
    if args.fractal_info_hybrid:
        return cmd_fractal_info_hybrid()
    if args.hybrid_boost_info:
        return cmd_hybrid_boost_info()
    if args.fractal_push:
        return cmd_fractal_push(args.tree_size, args.base_alpha, args.simulate)
    if args.alpha_boost is not None:
        return cmd_alpha_boost(
            args.alpha_boost, args.tree_size, args.base_alpha, args.simulate
        )
    if args.full_500_sweep:
        lr_range = tuple(args.lr_range) if args.lr_range else (RL_LR_MIN, RL_LR_MAX)
        return cmd_full_500_sweep(
            args.tree_size, lr_range, args.retention_target, args.simulate
        )

    # RL commands
    if args.rl_tune and args.blackout is not None:
        return cmd_rl_tune(
            args.blackout,
            args.rl_episodes,
            True,
            args.adaptive or args.dynamic,
            args.simulate,
        )
    if args.dynamic and args.blackout is not None:
        return cmd_dynamic_mode(args.blackout, args.rl_episodes, args.simulate)
    if args.tune_sweep:
        return cmd_tune_sweep(args.simulate)
    if args.quantum_estimate:
        return cmd_quantum_estimate(args.retention_target)
    if args.rl_500_sweep:
        lr_range = tuple(args.lr_range) if args.lr_range else (RL_LR_MIN, RL_LR_MAX)
        return cmd_rl_500_sweep(
            args.tree_size, lr_range, args.retention_target, args.simulate
        )

    # Pipeline commands
    if args.full_pipeline:
        return cmd_full_pipeline(
            args.pilot_runs,
            args.quantum_runs,
            args.sweep_runs,
            args.tree_size,
            args.simulate,
        )
    if args.post_tune_execute:
        return cmd_post_tune_execute(args.tree_size, args.simulate)
    if args.lr_pilot is not None:
        runs = args.lr_pilot if args.lr_pilot else 50
        return cmd_lr_pilot(runs, args.tree_size, args.simulate, args.show_bounds)
    if args.quantum_sim is not None:
        runs = args.quantum_sim if args.quantum_sim else 10
        return cmd_quantum_sim(runs, args.simulate)

    # Depth commands
    if args.compute_depth:
        return cmd_compute_depth_single(args.tree_size)
    if args.depth_scaling_test:
        return cmd_depth_scaling_test()
    if args.adaptive_depth_run:
        return cmd_adaptive_depth_run(args.rl_sweep, args.tree_size, args.simulate)

    # Ablation commands
    if args.ceiling_track is not None:
        return cmd_ceiling_track(args.ceiling_track)
    if args.isolate_layers:
        blackout_days = args.blackout if args.blackout is not None else 150
        return cmd_isolate_layers(blackout_days, args.simulate)
    if args.ablation_sweep:
        blackout_days = args.blackout if args.blackout is not None else 150
        return cmd_ablation_sweep(blackout_days, args.simulate)
    if args.ablate is not None:
        blackout_days = args.blackout if args.blackout is not None else 150
        return cmd_ablate(args.ablate, blackout_days, args.simulate)

    # Pruning commands
    if args.pruning_sweep:
        return cmd_pruning_sweep(args.simulate)
    if args.extended_250d:
        return cmd_extended_250d(args.simulate)
    if args.verify_chain:
        return cmd_verify_chain(args.trim_factor, args.simulate)
    if args.entropy_prune and args.blackout is not None:
        hybrid = args.hybrid_prune or True
        return cmd_entropy_prune(args.blackout, args.trim_factor, hybrid, args.simulate)

    # GNN/cache commands
    if args.cache_sweep:
        return cmd_cache_sweep(args.simulate)
    if args.overflow_test:
        return cmd_overflow_test(args.simulate)
    if args.extreme_sweep is not None:
        return cmd_extreme_sweep(args.extreme_sweep, args.cache_depth, args.simulate)
    if args.gnn_nonlinear and args.blackout is not None:
        return cmd_gnn_nonlinear(args.blackout, args.cache_depth, args.simulate)

    # Extended sweep
    if args.extended_sweep is not None:
        return cmd_extended_sweep(
            args.extended_sweep[0], args.extended_sweep[1], args.simulate
        )

    # Blackout commands
    if args.blackout_sweep:
        return cmd_blackout_sweep(reroute_enabled)
    if args.blackout is not None:
        return cmd_blackout(args.blackout, reroute_enabled, args.simulate)
    if reroute_enabled and args.partition is None and args.blackout is None:
        return cmd_reroute(args.simulate)

    # Partition commands
    if args.stress_quorum:
        return cmd_stress_quorum()
    if args.partition is not None:
        return cmd_partition(args.partition, args.nodes, args.simulate)

    # Timeline
    if args.simulate_timeline:
        return cmd_simulate_timeline(args.c_base, args.p_factor, args.tau)

    # Positional commands
    if args.command is None:
        print(docstring)
        return

    cmd = args.command.lower()
    if cmd == "baseline":
        cmd_baseline()
    elif cmd == "bootstrap":
        cmd_bootstrap()
    elif cmd == "curve":
        cmd_curve()
    elif cmd == "full":
        cmd_full()
    else:
        print(f"Unknown command: {cmd}")
        print(docstring)
