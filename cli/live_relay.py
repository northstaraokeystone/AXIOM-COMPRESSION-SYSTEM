"""Live relay CLI commands.

Provides CLI commands for hardware-in-loop testing with Starlink analogs.
"""

from argparse import Namespace
from typing import Any, Dict


def cmd_live_relay_info(args: Namespace) -> Dict[str, Any]:
    """Show live relay configuration.

    Args:
        args: CLI arguments.

    Returns:
        dict: Configuration info.
    """
    from src.live_relay_hil import load_hil_config

    config = load_hil_config()

    print("\n=== LIVE RELAY HIL CONFIGURATION ===")
    print(f"Enabled: {config['enabled']}")
    print(f"Mode: {config['mode']}")
    print(f"Priority: {config['priority']}")
    print("\n--- Starlink Analog ---")
    analog = config["starlink_analog_config"]
    print(f"  Latency: {analog['latency_ms']}ms")
    print(f"  Bandwidth: {analog['bandwidth_gbps']} Gbps")
    print(f"  Packet Loss: {analog['packet_loss_rate']*100:.2f}%")
    print(f"  Timeout: {analog['timeout_ms']}ms")
    print("\n--- Mars HIL ---")
    mars = config["mars_hil_config"]
    print(f"  Enabled: {mars['enabled']}")
    print(f"  Latency Range: {mars['latency_min_minutes']}-{mars['latency_max_minutes']} min")
    print(f"  Proof Duration: {mars['proof_duration_hours']} hours")

    return config


def cmd_live_relay_connect(args: Namespace) -> Dict[str, Any]:
    """Connect to Starlink analog hardware.

    Args:
        args: CLI arguments.

    Returns:
        dict: Connection result.
    """
    from src.live_relay_hil import connect_starlink_analog

    print("\n=== CONNECTING TO STARLINK ANALOG ===")
    result = connect_starlink_analog()

    if result["connected"]:
        print("Connection: SUCCESS")
        print(f"Mode: {result['mode']}")
        print(f"Latency: {result['config']['latency_ms']}ms")
    else:
        print("Connection: FAILED")

    return result


def cmd_live_relay_test(args: Namespace) -> Dict[str, Any]:
    """Run HIL test.

    Args:
        args: CLI arguments.

    Returns:
        dict: Test result.
    """
    from src.live_relay_hil import run_hil_test

    duration = getattr(args, "duration", 60)

    print(f"\n=== RUNNING HIL TEST ({duration}s) ===")
    result = run_hil_test(duration_s=duration)

    print(f"Test Passed: {result['test_passed']}")
    print(f"Packets Sent: {result['packets_sent']}")
    print(f"Packets Received: {result['packets_received']}")
    print(f"Loss Rate: {result['loss_rate']*100:.2f}%")
    print(f"Avg Latency: {result['avg_latency_ms']:.2f}ms")
    print(f"Throughput: {result['throughput_pps']:.1f} pps")

    return result


def cmd_live_relay_mars(args: Namespace) -> Dict[str, Any]:
    """Run Mars HIL proof.

    Args:
        args: CLI arguments.

    Returns:
        dict: Proof result.
    """
    from src.live_relay_hil import mars_hil_proof

    duration = getattr(args, "duration", 1)

    print(f"\n=== RUNNING MARS HIL PROOF ({duration}h) ===")
    result = mars_hil_proof(duration_hours=duration)

    print(f"Proof Passed: {result['proof_passed']}")
    print(f"Messages Sent: {result['messages_sent']}")
    print(f"Messages Received: {result['messages_received']}")
    print(f"Success Rate: {result['success_rate']*100:.2f}%")
    print(f"Autonomy Target: {result['autonomy_target']*100:.1f}%")
    print(f"Opposition Latency: {result['opposition_latency_min']:.1f} min")
    print(f"Conjunction Latency: {result['conjunction_latency_min']:.1f} min")

    return result


def cmd_live_relay_stress(args: Namespace) -> Dict[str, Any]:
    """Run stress test.

    Args:
        args: CLI arguments.

    Returns:
        dict: Stress test result.
    """
    from src.live_relay_hil import stress_test_hil

    iterations = getattr(args, "iterations", 100)

    print(f"\n=== RUNNING STRESS TEST ({iterations} iterations) ===")
    result = stress_test_hil(iterations=iterations)

    print(f"Stress Passed: {result['stress_passed']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Success Count: {result['success_count']}")
    print(f"Success Rate: {result['success_rate']*100:.2f}%")
    print(f"Avg Latency: {result['avg_latency_ms']:.2f}ms")
    print(f"Throughput: {result['throughput_ops']:.1f} ops/s")

    return result


def cmd_live_relay_status(args: Namespace) -> Dict[str, Any]:
    """Show HIL status.

    Args:
        args: CLI arguments.

    Returns:
        dict: Status info.
    """
    from src.live_relay_hil import get_hil_status

    status = get_hil_status()

    print("\n=== HIL STATUS ===")
    print(f"HIL Enabled: {status['hil_enabled']}")
    print(f"Mode: {status['mode']}")
    print(f"Priority: {status['priority']}")
    print("\n--- Starlink Analog ---")
    analog = status["starlink_analog"]
    print(f"  Latency: {analog['latency_ms']}ms")
    print(f"  Bandwidth: {analog['bandwidth_gbps']} Gbps")
    print("\n--- Mars HIL ---")
    mars = status["mars_hil"]
    print(f"  Enabled: {mars['enabled']}")
    print(f"  Latency Range: {mars['latency_range_min']}")

    return status
