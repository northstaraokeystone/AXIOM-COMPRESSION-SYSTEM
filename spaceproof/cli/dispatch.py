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
