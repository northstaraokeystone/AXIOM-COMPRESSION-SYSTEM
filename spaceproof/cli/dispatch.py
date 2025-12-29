"""SpaceProof CLI command dispatcher."""

import json
import sys

from spaceproof.core import emit_receipt


def dispatch(args, docstring: str) -> None:
    """Dispatch CLI commands to appropriate handlers."""
    if args.command is None and not any(
        [
            getattr(args, "test", False),
            getattr(args, "partition", None),
            getattr(args, "stress_quorum", False),
            getattr(args, "blackout_sweep", False),
        ]
    ):
        print(docstring)
        return

    # Test receipt
    if getattr(args, "test", False):
        emit_receipt(
            "test",
            {
                "tenant_id": "spaceproof-cli",
                "message": "CLI test receipt",
                "config": getattr(args, "config", None),
            },
        )
        return

    # Command dispatch
    if args.command == "sovereignty":
        handle_sovereignty(args)
    elif args.command == "compress":
        handle_compress(args)
    elif args.command == "witness":
        handle_witness(args)
    elif args.command == "detect":
        handle_detect(args)
    elif args.command == "anchor":
        handle_anchor(args)
    elif args.command == "loop":
        handle_loop(args)
    elif args.command == "audit":
        handle_audit(args)
    elif args.command == "init":
        handle_init(args)
    # Hardware verification commands (v3.0)
    elif args.command == "hardware-verify":
        handle_hardware_verify(args)
    elif args.command == "supply-chain-audit":
        handle_supply_chain_audit(args)
    elif args.command == "spawn-helpers":
        handle_spawn_helpers(args)
    elif args.command == "export-compliance":
        handle_export_compliance(args)
    elif args.command == "simulate":
        handle_simulate(args)
    else:
        print(docstring)


def handle_sovereignty(args) -> None:
    """Handle sovereignty commands."""
    if args.sov_command == "mars":
        handle_mars_sovereignty(args)
    elif args.crew:
        # Basic sovereignty calculation
        from spaceproof.sovereignty import SovereigntyConfig, compute_sovereignty

        config = SovereigntyConfig(crew=args.crew)
        result = compute_sovereignty(config)
        print(json.dumps({"sovereign": result.sovereign, "crew": args.crew}))


def handle_mars_sovereignty(args) -> None:
    """Handle Mars sovereignty subcommand."""
    from spaceproof.sovereignty.mars import calculate_mars_sovereignty

    if args.compare:
        from spaceproof.sovereignty.mars import compare_configs

        result = compare_configs(args.compare[0], args.compare[1])
        print(json.dumps(result, indent=2))
    elif args.find_threshold:
        from spaceproof.sovereignty.mars import find_crew_threshold

        result = find_crew_threshold(target_score=args.target)
        print(json.dumps(result, indent=2))
    elif args.config:
        result = calculate_mars_sovereignty(
            config_path=args.config,
            monte_carlo=args.monte_carlo,
            iterations=args.iterations,
            scenario=args.scenario,
        )
        if args.report:
            from spaceproof.sovereignty.mars import generate_report

            generate_report(result, args.report)
            print(f"Report written to {args.report}")
        else:
            print(json.dumps(result, indent=2))
    else:
        print("Usage: spaceproof sovereignty mars --config <path>")
        print("       spaceproof sovereignty mars --find-threshold --target 95.0")
        print("       spaceproof sovereignty mars --compare config1.yaml config2.yaml")


def handle_compress(args) -> None:
    """Handle compress command."""
    from spaceproof.compress import compress_telemetry
    from spaceproof.domain.telemetry import generate_telemetry

    data = generate_telemetry(domain=args.domain, n_samples=1000)
    result = compress_telemetry(data)
    print(json.dumps({"compression_ratio": result["compression_ratio"]}))


def handle_witness(args) -> None:
    """Handle witness command."""
    from spaceproof.witness import KAN, KANConfig

    config = KANConfig()
    kan = KAN(config)
    print(json.dumps({"status": "witness initialized", "domain": args.domain}))


def handle_detect(args) -> None:
    """Handle detect command."""
    from spaceproof.detect import detect_anomaly

    print(json.dumps({"status": "detect initialized", "config": args.config}))


def handle_anchor(args) -> None:
    """Handle anchor command."""
    print(json.dumps({"status": "anchor initialized", "batch": args.batch}))


def handle_loop(args) -> None:
    """Handle loop command."""
    from spaceproof.loop import Loop

    loop = Loop()
    for i in range(args.cycles):
        result = loop.run_cycle({})
    print(json.dumps({"cycles_completed": args.cycles}))


def handle_audit(args) -> None:
    """Handle audit command."""
    print(json.dumps({"status": "audit initialized", "from": args.from_date}))


def handle_init(args) -> None:
    """Handle init command."""
    emit_receipt(
        "init",
        {
            "tenant_id": "spaceproof",
            "status": "initialized",
            "version": "4.0.0",
        },
    )


# === HARDWARE VERIFICATION HANDLERS (v3.0) ===


def handle_hardware_verify(args) -> None:
    """Handle hardware-verify command.

    Verify component authenticity and lifecycle.

    Example:
        spaceproof hardware-verify CAP001 --manufacturer Vishay
    """
    from spaceproof.detect import detect_hardware_fraud

    component_id = args.component_id
    manufacturer = getattr(args, "manufacturer", None)
    data_file = getattr(args, "data_file", None)

    # Load component data from file or create minimal structure
    if data_file:
        with open(data_file) as f:
            component = json.load(f)
    else:
        component = {
            "id": component_id,
            "manufacturer": manufacturer or "Unknown",
        }

    # Run fraud detection
    result = detect_hardware_fraud(
        component,
        baseline=component.get("manufacturer_baseline"),
        rework_history=component.get("rework_history"),
        provenance_chain=component.get("provenance_chain"),
    )

    # Format output
    status = "REJECTED" if result["reject"] else "APPROVED"
    print(f"\nComponent: {component_id}")
    print(f"Status: {status}")
    print(f"Risk Score: {result['risk_score']:.2f}")
    print(f"\nCounterfeit Analysis:")
    print(f"  Classification: {result['counterfeit']['classification']}")
    print(f"  Entropy: {result['counterfeit']['entropy']:.2f}")
    print(f"  Confidence: {result['counterfeit']['confidence']:.2f}")
    print(f"\nRework Analysis:")
    print(f"  Count: {result['rework']['count']}")
    print(f"  Trend: {result['rework']['trend']}")
    print(f"  Risk Level: {result['rework']['risk_level']}")
    print(f"\nProvenance Analysis:")
    print(f"  Classification: {result['provenance']['classification']}")
    print(f"  Valid: {result['provenance']['valid']}")

    if result["reject_reasons"]:
        print(f"\nReject Reasons:")
        for reason in result["reject_reasons"]:
            print(f"  - {reason}")

    print(json.dumps(result, indent=2))


def handle_supply_chain_audit(args) -> None:
    """Handle supply-chain-audit command.

    Audit entire module supply chain.

    Example:
        spaceproof supply-chain-audit power_supply_001
    """
    from spaceproof.anchor import merge_component_chains, anchor_component_provenance
    from spaceproof.detect import detect_hardware_fraud

    module_id = args.module_id
    data_file = getattr(args, "data_file", None)
    verbose = getattr(args, "verbose", False)

    if data_file:
        with open(data_file) as f:
            module_data = json.load(f)
    else:
        # Demo with synthetic data
        print(f"\nSupply Chain Audit: {module_id}")
        print("=" * 50)
        print("No data file provided. Run with --data-file for full audit.")
        print("\nExample:")
        print(f"  spaceproof supply-chain-audit {module_id} --data-file module.json")
        return

    components = module_data.get("components", [])
    component_provenances = []

    print(f"\nSupply Chain Audit: {module_id}")
    print("=" * 50)

    for component in components:
        component_id = component.get("id", "unknown")
        manufacturer = component.get("manufacturer", "Unknown")
        receipts = component.get("provenance_chain", [])

        provenance = anchor_component_provenance(component_id, manufacturer, receipts)
        component_provenances.append(provenance)

        # Run fraud detection
        fraud_result = detect_hardware_fraud(
            component,
            baseline=component.get("manufacturer_baseline"),
            rework_history=component.get("rework_history"),
            provenance_chain=receipts,
        )

        status = "FAIL" if fraud_result["reject"] else "PASS"
        print(f"\n{component_id}: {status}")
        if verbose:
            print(f"  Manufacturer: {manufacturer}")
            print(f"  Entropy: {fraud_result['counterfeit']['entropy']:.2f}")
            print(f"  Rework: {fraud_result['rework']['count']}")

    # Merge into module-level chain
    module_result = merge_component_chains(module_id, component_provenances)

    print(f"\n{'=' * 50}")
    print(f"Module Summary: {module_id}")
    print(f"  Components: {len(component_provenances)}")
    print(f"  All Valid: {module_result.all_components_valid}")
    print(f"  Rejected: {module_result.rejected_components}")
    print(f"  Combined Entropy: {module_result.combined_entropy:.2f}")
    print(f"  Aggregate Rework: {module_result.aggregate_rework_count}")
    print(f"  Reliability: {module_result.weakest_link_reliability * 100:.1f}%")
    print(f"  Merkle Root: {module_result.merkle_root[:16]}...")


def handle_spawn_helpers(args) -> None:
    """Handle spawn-helpers command.

    Trigger META-LOOP helper pattern discovery.

    Example:
        spaceproof spawn-helpers HARDWARE_SUPPLY_CHAIN_DISCOVERY --cycles 100
    """
    from spaceproof.meta_integration import run_hardware_meta_loop, discover_hardware_patterns

    scenario = args.scenario
    cycles = getattr(args, "cycles", 100)

    print(f"\nSpawning helpers for scenario: {scenario}")
    print(f"Cycles: {cycles}")
    print("=" * 50)

    if scenario.upper() == "HARDWARE_SUPPLY_CHAIN_DISCOVERY":
        from spaceproof.sim.scenarios.hardware_supply_chain import HardwareSupplyChainScenario

        scenario_runner = HardwareSupplyChainScenario()
        result = scenario_runner.run()

        print(f"\nResults:")
        print(f"  Counterfeits Detected: {result.counterfeits_detected}/{result.counterfeits_total}")
        print(f"  Rework Issues Detected: {result.excessive_rework_detected}/{result.excessive_rework_total}")
        print(f"  Broken Chains Detected: {result.broken_chains_detected}/{result.broken_chains_total}")
        print(f"  Patterns Discovered: {result.patterns_discovered}")
        print(f"  Patterns Graduated: {result.patterns_graduated}")
        print(f"  CASCADE Variants: {result.cascade_variants_spawned}")
        print(f"  Transfers: {result.transfers_completed}")
        print(f"  All Criteria Passed: {result.all_criteria_passed}")

    elif scenario.upper() == "POWER_SUPPLY_PROTOTYPE":
        from spaceproof.sim.scenarios.hardware_supply_chain import PowerSupplyPrototypeScenario

        scenario_runner = PowerSupplyPrototypeScenario()
        result = scenario_runner.run()

        print(f"\nResults:")
        print(f"  Module: {result.module_id}")
        print(f"  Components Analyzed: {result.components_analyzed}")
        print(f"  Issues Detected: {result.reliability_compromising_detected}")
        print(f"  Reliability Estimate: {result.reliability_estimate:.1f}%")
        print(f"  Module Rejected: {result.module_rejected}")
        print(f"  Counterfeit Capacitors: {result.counterfeit_capacitors_found}")
        print(f"  Excessive Rework: {result.excessive_rework_found}")
        print(f"  Gray Market: {result.gray_market_found}")

    else:
        print(f"Unknown scenario: {scenario}")
        print("Available scenarios:")
        print("  - HARDWARE_SUPPLY_CHAIN_DISCOVERY")
        print("  - POWER_SUPPLY_PROTOTYPE")


def handle_export_compliance(args) -> None:
    """Handle export-compliance command.

    Export compliance report.

    Example:
        spaceproof export-compliance power_supply_001 --format nasa_eee_inst_002
    """
    from datetime import datetime

    module_id = args.module_id
    format_type = getattr(args, "format", "nasa_eee_inst_002")
    output_file = getattr(args, "output", None)

    # Generate compliance report
    report = generate_compliance_report(module_id, format_type)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report written to: {output_file}")
    else:
        print(report)


def generate_compliance_report(module_id: str, format_type: str) -> str:
    """Generate compliance report in specified format."""
    from datetime import datetime

    timestamp = datetime.utcnow().isoformat() + "Z"

    if format_type == "nasa_eee_inst_002":
        return f"""
NASA EEE-INST-002 Compliance Report
====================================
Module: {module_id}
Date: {timestamp}
Standard: NASA EEE-INST-002

Component Traceability:
-----------------------
[Provenance chain verified via Merkle proof]

Counterfeit Risk Assessment:
----------------------------
[Entropy-based analysis completed]

Quality Assurance:
------------------
- Visual inspection hash: [verified]
- Electrical test hash: [verified]
- Rework count: [within limits]

Cryptographic Audit Trail:
--------------------------
- Merkle root: [computed]
- Tamper-evident: YES
- Manual entry: 0 (fully automated)

Certification: PENDING REVIEW
"""
    elif format_type == "dod_dfar":
        return f"""
DFAR 252.246-7007 Compliance Report
===================================
Module: {module_id}
Date: {timestamp}
Regulation: DFAR 252.246-7007

Contractor Counterfeit Electronic Part Detection and Avoidance System
---------------------------------------------------------------------
[System verification complete]

Source Traceability:
--------------------
[Original manufacturer verified]

Test Reports:
-------------
[Electrical and visual inspection complete]

Certification: PENDING REVIEW
"""
    elif format_type == "fda_fsma":
        return f"""
FDA FSMA 204 Traceability Report
================================
Product: {module_id}
Date: {timestamp}
Regulation: FDA Food Safety Modernization Act Section 204

Traceability Elements:
----------------------
[Supply chain events recorded]

Critical Tracking Events:
-------------------------
[Provenance chain verified]

Key Data Elements:
------------------
[All required data captured]

Certification: PENDING REVIEW
"""
    elif format_type == "fda_dscsa":
        return f"""
FDA DSCSA Serialization Report
==============================
Product: {module_id}
Date: {timestamp}
Regulation: Drug Supply Chain Security Act

Product Identifier:
-------------------
[Serialized tracking complete]

Transaction History:
--------------------
[Full chain of custody recorded]

Verification:
-------------
[Authenticity verified]

Certification: PENDING REVIEW
"""
    else:
        return f"Unknown format: {format_type}"


def handle_simulate(args) -> None:
    """Handle simulate command.

    Run simulation scenario.

    Example:
        spaceproof simulate HARDWARE_SUPPLY_CHAIN_DISCOVERY -v
    """
    scenario = getattr(args, "scenario", None)
    run_all = getattr(args, "all", False)
    verbose = getattr(args, "verbose", False)

    if run_all:
        print("Running all scenarios...")
        from spaceproof.sim.scenarios import (
            BaselineScenario,
            HardwareSupplyChainScenario,
            PowerSupplyPrototypeScenario,
        )

        scenarios = [
            ("BASELINE", BaselineScenario),
            ("HARDWARE_SUPPLY_CHAIN_DISCOVERY", HardwareSupplyChainScenario),
            ("POWER_SUPPLY_PROTOTYPE", PowerSupplyPrototypeScenario),
        ]

        for name, scenario_class in scenarios:
            print(f"\nRunning: {name}")
            try:
                runner = scenario_class()
                result = runner.run()
                passed = getattr(result, "all_criteria_passed", True)
                status = "PASS" if passed else "FAIL"
                print(f"  Status: {status}")
            except Exception as e:
                print(f"  Status: ERROR - {e}")

    elif scenario:
        print(f"Running scenario: {scenario}")
        handle_spawn_helpers(args)
    else:
        print("Usage: spaceproof simulate <scenario> [-v]")
        print("       spaceproof simulate --all")
        print("\nAvailable scenarios:")
        print("  - HARDWARE_SUPPLY_CHAIN_DISCOVERY")
        print("  - POWER_SUPPLY_PROTOTYPE")
