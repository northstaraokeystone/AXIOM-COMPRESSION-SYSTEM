"""ISRU hybrid CLI commands for AXIOM-CORE.

Commands for D5+ISRU hybrid operations:
- cmd_moxie_info: Show MOXIE calibration data
- cmd_isru_simulate: Run ISRU simulation
- cmd_isru_closure: Show closure metrics
- cmd_d5_isru_hybrid: Run integrated D5+ISRU
- cmd_d5_push: Run D5 recursion push
- cmd_d5_info: Show D5 configuration
"""

from typing import Dict, Any

from src.fractal_layers import (
    get_d5_info,
    d5_push,
    D5_ALPHA_TARGET,
    D5_ALPHA_FLOOR,
    D5_ALPHA_CEILING,
    D5_TREE_MIN,
    D5_UPLIFT,
)
from src.isru_hybrid import (
    moxie_calibration,
    simulate_o2_production,
    compute_isru_closure,
    d5_isru_hybrid,
    get_isru_info,
    MOXIE_O2_TOTAL_G,
    MOXIE_O2_PEAK_G_HR,
    MOXIE_O2_AVG_G_HR,
    ISRU_CLOSURE_TARGET,
)

from cli.base import print_header, print_receipt_note


def cmd_moxie_info() -> Dict[str, Any]:
    """Show MOXIE calibration data.

    Returns:
        MOXIE calibration dict
    """
    print_header("MOXIE CALIBRATION DATA")

    calibration = moxie_calibration()

    print("\nNASA Perseverance MOXIE (Dec 2025):")
    print(f"  O2 Total: {calibration['o2_total_g']}g")
    print(f"  O2 Peak Rate: {calibration['o2_peak_g_hr']} g/hr")
    print(f"  O2 Avg Rate: {calibration['o2_avg_g_hr']} g/hr")
    print(f"  Runs: {calibration.get('runs', 16)}")
    print(f"  Efficiency: {calibration.get('conversion_efficiency', 0.06) * 100:.1f}%")

    print("\nValidation:")
    print(f"  Expected total: {MOXIE_O2_TOTAL_G}g")
    print(f"  Validated: {'YES' if calibration['validated'] else 'NO'}")

    print_receipt_note("moxie_calibration")
    print("=" * 60)

    return calibration


def cmd_isru_simulate(
    hours: int = 24, crew: int = 4, moxie_units: int = 10, simulate: bool = False
) -> Dict[str, Any]:
    """Run ISRU O2 production simulation.

    Args:
        hours: Simulation duration in hours
        crew: Number of crew members
        moxie_units: Number of MOXIE units
        simulate: Output simulation receipt

    Returns:
        Simulation result dict
    """
    print_header(f"ISRU SIMULATION {'(SIMULATE)' if simulate else ''}")

    print("\nConfiguration:")
    print(f"  Duration: {hours} hours")
    print(f"  Crew: {crew}")
    print(f"  MOXIE units: {moxie_units}")

    result = simulate_o2_production(hours, crew, moxie_units)

    print("\nO2 Production:")
    print(
        f"  Production: {result['production_g']:.2f}g ({result['production_kg']:.4f}kg)"
    )
    print(f"  Peak capacity: {result['peak_production_kg']:.4f}kg")

    print("\nO2 Consumption:")
    print(f"  Consumption: {result['consumption_kg']:.4f}kg")

    print("\nBalance:")
    print(f"  Net balance: {result['balance_kg']:.4f}kg")
    print(f"  Self-sufficient: {'YES' if result['self_sufficient'] else 'NO'}")
    print(f"  Rate per crew: {result['rate_per_crew_kg_day']:.4f} kg/day")

    if simulate:
        print_receipt_note("isru_production")

    print("=" * 60)

    return result


def cmd_isru_closure(simulate: bool = False) -> Dict[str, Any]:
    """Show ISRU closure metrics.

    Args:
        simulate: Output simulation receipt

    Returns:
        Closure metrics dict
    """
    print_header("ISRU CLOSURE METRICS")

    # Run simulation with default parameters
    production_result = simulate_o2_production(24, 4, 10)

    production = {"o2": production_result["production_kg"]}
    consumption = {"o2": production_result["consumption_kg"]}

    closure = compute_isru_closure(production, consumption)

    print("\nClosure Ratio:")
    print(f"  Current: {closure:.4f}")
    print(f"  Target: {ISRU_CLOSURE_TARGET}")
    print(f"  Gap: {ISRU_CLOSURE_TARGET - closure:.4f}")
    print(f"  Target met: {'YES' if closure >= ISRU_CLOSURE_TARGET else 'NO'}")

    print("\nResource Balance:")
    print(f"  O2 production: {production['o2']:.4f}kg")
    print(f"  O2 consumption: {consumption['o2']:.4f}kg")

    if simulate:
        print_receipt_note("isru_closure")

    print("=" * 60)

    return {
        "closure": closure,
        "target": ISRU_CLOSURE_TARGET,
        "target_met": closure >= ISRU_CLOSURE_TARGET,
        "production": production,
        "consumption": consumption,
    }


def cmd_d5_isru_hybrid(
    tree_size: int = D5_TREE_MIN,
    base_alpha: float = 3.0,
    crew: int = 4,
    hours: int = 24,
    moxie_units: int = 10,
    simulate: bool = False,
) -> Dict[str, Any]:
    """Run integrated D5+ISRU hybrid.

    Args:
        tree_size: Tree size for D5 recursion
        base_alpha: Base alpha before uplift
        crew: Number of crew members
        hours: Simulation duration in hours
        moxie_units: Number of MOXIE units
        simulate: Output simulation receipt

    Returns:
        Hybrid result dict
    """
    print_header(f"D5+ISRU HYBRID {'(SIMULATE)' if simulate else ''}")

    print("\nConfiguration:")
    print(f"  Tree size: {tree_size:,}")
    print(f"  Base alpha: {base_alpha}")
    print(f"  Crew: {crew}")
    print(f"  Hours: {hours}")
    print(f"  MOXIE units: {moxie_units}")

    result = d5_isru_hybrid(tree_size, base_alpha, crew, hours, moxie_units)

    print("\nD5 RECURSION:")
    print(f"  Effective alpha: {result['d5_result']['eff_alpha']}")
    print(f"  Uplift: +{result['d5_result']['uplift']}")
    print(f"  Floor met (3.23): {'YES' if result['d5_result']['floor_met'] else 'NO'}")
    print(
        f"  Target met (3.25): {'YES' if result['d5_result']['target_met'] else 'NO'}"
    )

    print("\nISRU SIMULATION:")
    print(f"  O2 production: {result['isru_result']['production_kg']:.4f}kg")
    print(f"  O2 consumption: {result['isru_result']['consumption_kg']:.4f}kg")
    print(f"  O2 balance: {result['isru_result']['balance_kg']:.4f}kg")
    print(
        f"  Self-sufficient: {'YES' if result['isru_result']['self_sufficient'] else 'NO'}"
    )

    print("\nCLOSURE:")
    print(f"  Closure ratio: {result['closure']['ratio']}")
    print(f"  Target (0.85): {'MET' if result['closure']['target_met'] else 'NOT MET'}")

    print("\nCOMBINED SLO:")
    combined = result["combined_slo"]
    print(
        f"  Alpha target ({combined['alpha_target']}): {'PASS' if combined['alpha_met'] else 'FAIL'}"
    )
    print(
        f"  Closure target ({combined['closure_target']}): {'PASS' if combined['closure_met'] else 'FAIL'}"
    )
    print(f"  ALL TARGETS: {'PASS' if combined['all_targets_met'] else 'FAIL'}")

    if simulate:
        print_receipt_note("d5_isru_hybrid")

    print("=" * 60)

    return result


def cmd_d5_push_isru(
    tree_size: int = D5_TREE_MIN, base_alpha: float = 3.0, simulate: bool = False
) -> Dict[str, Any]:
    """Run D5 recursion push for alpha >= 3.25.

    Args:
        tree_size: Tree size (default: 10^12)
        base_alpha: Base alpha (default: 3.0)
        simulate: Simulation mode

    Returns:
        D5 push results
    """
    result = d5_push(tree_size, base_alpha, simulate)

    print_header(f"D5 PUSH {'(SIMULATE)' if simulate else ''}")

    print(f"\nTree size: {tree_size:,}")
    print(f"Base alpha: {base_alpha}")
    print(f"Depth: {result['depth']}")

    print(f"\nEffective alpha: {result['eff_alpha']}")
    print(f"Instability: {result['instability']}")

    print(f"\nFloor met ({D5_ALPHA_FLOOR}): {'YES' if result['floor_met'] else 'NO'}")
    print(f"Target met ({D5_ALPHA_TARGET}): {'YES' if result['target_met'] else 'NO'}")
    print(
        f"Ceiling met ({D5_ALPHA_CEILING}): {'YES' if result['ceiling_met'] else 'NO'}"
    )

    print(f"\nSLO passed: {'YES' if result['slo_passed'] else 'NO'}")
    print(f"Gate: {result['gate']}")

    if simulate:
        print_receipt_note("d5_push")

    print("=" * 60)

    return result


def cmd_d5_info_isru() -> Dict[str, Any]:
    """Show D5 + ISRU configuration.

    Returns:
        D5 info dict
    """
    info = get_d5_info()

    print_header("D5 + ISRU CONFIGURATION")

    print(f"\nVersion: {info['version']}")

    print("\nD5 Config:")
    d5_config = info["d5_config"]
    print(f"  Recursion depth: {d5_config.get('recursion_depth', 5)}")
    print(f"  Alpha floor: {d5_config.get('alpha_floor', D5_ALPHA_FLOOR)}")
    print(f"  Alpha target: {d5_config.get('alpha_target', D5_ALPHA_TARGET)}")
    print(f"  Alpha ceiling: {d5_config.get('alpha_ceiling', D5_ALPHA_CEILING)}")
    print(f"  Uplift: +{d5_config.get('uplift', D5_UPLIFT)}")

    print("\nUplift by depth:")
    for depth, uplift in info.get("uplift_by_depth", {}).items():
        print(f"  Depth {depth}: +{uplift}")

    print("\nMOXIE Calibration:")
    moxie = info.get("moxie_calibration", {})
    print(f"  Source: {moxie.get('source', 'NASA Perseverance MOXIE')}")
    print(f"  O2 total: {moxie.get('o2_total_g', MOXIE_O2_TOTAL_G)}g")
    print(f"  O2 peak: {moxie.get('o2_peak_g_hr', MOXIE_O2_PEAK_G_HR)} g/hr")
    print(f"  O2 avg: {moxie.get('o2_avg_g_hr', MOXIE_O2_AVG_G_HR)} g/hr")

    print("\nISRU Config:")
    isru = info.get("isru_config", {})
    print(f"  Closure target: {isru.get('closure_target', ISRU_CLOSURE_TARGET)}")
    print(f"  Resources: {isru.get('resources', ['o2', 'h2o', 'ch4'])}")

    print_receipt_note("d5_info")
    print("=" * 60)

    return info


def cmd_isru_info() -> Dict[str, Any]:
    """Show ISRU module information.

    Returns:
        ISRU info dict
    """
    info = get_isru_info()

    print_header("ISRU HYBRID MODULE INFO")

    print("\nMOXIE Calibration:")
    moxie = info["moxie_calibration"]
    print(f"  O2 total: {moxie['o2_total_g']}g")
    print(f"  O2 peak: {moxie['o2_peak_g_hr']} g/hr")
    print(f"  O2 avg: {moxie['o2_avg_g_hr']} g/hr")
    print(f"  Runs: {moxie['runs']}")
    print(f"  Efficiency: {moxie['efficiency'] * 100:.1f}%")

    print("\nISRU Config:")
    isru = info["isru_config"]
    print(f"  Closure target: {isru['closure_target']}")
    print(f"  Resources: {isru['resources']}")
    print(f"  Sabatier efficiency: {isru['sabatier_efficiency']}")
    print(f"  Electrolysis efficiency: {isru['electrolysis_efficiency']}")

    print("\nD5 Integration:")
    d5 = info["d5_integration"]
    print(f"  Alpha target: {d5['alpha_target']}")
    print(f"  Uplift: +{d5['uplift']}")
    print(f"  Tree min: {d5['tree_min']:,}")

    print("\nConsumption Rates:")
    rates = info["consumption_rates"]
    print(f"  O2: {rates['o2_kg_day']} kg/day per crew")
    print(f"  H2O: {rates['h2o_kg_day']} kg/day per crew")

    print_receipt_note("isru_info")
    print("=" * 60)

    return info
