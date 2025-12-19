"""CLI commands for gravity adaptive operations."""

import json


def cmd_gravity_info(args) -> None:
    """Show gravity configuration."""
    from src.gravity_adaptive import get_gravity_status

    status = get_gravity_status()
    print("\n=== GRAVITY ADAPTIVE CONFIGURATION ===")
    print(f"Adaptive enabled: {status['adaptive_enabled']}")
    print(f"Planets configured: {status['planets_configured']}")
    print(f"\n=== DEFAULT GRAVITY MAP ===")
    for planet, gravity in status['default_gravity_map'].items():
        print(f"  {planet}: {gravity}g")


def cmd_gravity_adjust(args) -> None:
    """Adjust parameters for planet gravity."""
    from src.gravity_adaptive import adjust_for_gravity, get_planet_gravity

    planet = getattr(args, "gravity_planet", "mars")
    gravity_g = get_planet_gravity(planet)

    result = adjust_for_gravity(gravity_g)
    print(f"\n=== GRAVITY ADJUSTMENT FOR {planet.upper()} ===")
    print(f"Gravity: {result['gravity_g']}g")
    print(f"Gravity ratio: {result['gravity_ratio']:.3f}")
    print(f"Timing factor: {result['timing_factor']:.3f}")
    print(f"Consensus multiplier: {result['consensus_multiplier']:.3f}")
    print(f"Packet multiplier: {result['packet_multiplier']:.3f}")
    print(f"Autonomy adjustment: {result['autonomy_adjustment']:.6f}")


def cmd_gravity_consensus(args) -> None:
    """Show gravity-adjusted consensus timing."""
    from src.gravity_adaptive import adjust_consensus_timing, get_planet_gravity

    planet = getattr(args, "gravity_planet", "mars")
    gravity_g = get_planet_gravity(planet)

    result = adjust_consensus_timing(gravity_g)
    print(f"\n=== CONSENSUS TIMING FOR {planet.upper()} ({gravity_g}g) ===")
    print(f"Base heartbeat: {result['base_heartbeat_ms']} ms")
    print(f"Adjusted heartbeat: {result['adjusted_heartbeat_ms']} ms")
    print(f"Base election timeout: {result['base_election_timeout_ms']} ms")
    print(f"Adjusted election timeout: {result['adjusted_election_timeout_ms']} ms")
    print(f"Timing factor: {result['timing_factor']:.3f}")


def cmd_gravity_packet(args) -> None:
    """Show gravity-adjusted packet timing."""
    from src.gravity_adaptive import adjust_packet_timing, get_planet_gravity

    planet = getattr(args, "gravity_planet", "mars")
    gravity_g = get_planet_gravity(planet)

    result = adjust_packet_timing(gravity_g)
    print(f"\n=== PACKET TIMING FOR {planet.upper()} ({gravity_g}g) ===")
    print(f"Base timeout: {result['base_timeout_ms']} ms")
    print(f"Adjusted timeout: {result['adjusted_timeout_ms']} ms")
    print(f"Base retry delay: {result['base_retry_delay_ms']} ms")
    print(f"Adjusted retry delay: {result['adjusted_retry_delay_ms']} ms")
    print(f"Adjustment factor: {result['adjustment_factor']:.3f}")


def cmd_gravity_validate(args) -> None:
    """Validate gravity adjustment for a planet."""
    from src.gravity_adaptive import validate_gravity_adjustment

    planet = getattr(args, "gravity_planet", "mars")

    result = validate_gravity_adjustment(planet)
    print(f"\n=== GRAVITY VALIDATION FOR {planet.upper()} ===")
    print(f"Valid: {result['valid']}")
    print(f"Gravity: {result['gravity_g']}g")
    print(f"Timing factor: {result['adjustment']['timing_factor']:.3f}")
    print(f"Consensus multiplier: {result['adjustment']['consensus_multiplier']:.3f}")
    print(f"Packet multiplier: {result['adjustment']['packet_multiplier']:.3f}")


def cmd_gravity_status(args) -> None:
    """Show gravity status."""
    from src.gravity_adaptive import get_gravity_status

    status = get_gravity_status()
    print("\n=== GRAVITY STATUS ===")
    print(json.dumps(status, indent=2))


def cmd_gravity_all_planets(args) -> None:
    """Show adjustments for all planets."""
    from src.gravity_adaptive import get_all_planet_adjustments

    adjustments = get_all_planet_adjustments()
    print("\n=== GRAVITY ADJUSTMENTS FOR ALL PLANETS ===")
    for planet, adj in adjustments.items():
        print(f"\n{planet.upper()} ({adj['gravity_g']}g):")
        print(f"  Timing factor: {adj['timing_factor']:.3f}")
        print(f"  Consensus multiplier: {adj['consensus_multiplier']:.3f}")
        print(f"  Packet multiplier: {adj['packet_multiplier']:.3f}")
        print(f"  Autonomy adjustment: {adj['autonomy_adjustment']:.6f}")
