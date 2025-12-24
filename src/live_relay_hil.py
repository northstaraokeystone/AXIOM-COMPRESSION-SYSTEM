"""Hardware-in-loop interface for Starlink analog testing.

Provides hardware-in-loop testing capabilities for validating interstellar
relay protocols using Starlink analog hardware. Enables transition from
simulation to live hardware validation.

Receipt Types:
    - live_relay_config_receipt: Configuration loaded
    - hil_connection_receipt: Connection established
    - hil_latency_receipt: Latency measured
    - hil_packet_receipt: Packet sent/received
    - live_relay_hil_receipt: HIL test result
    - mars_hil_proof_receipt: Mars proof result
    - starlink_analog_receipt: Analog status

StopRules:
    - stoprule_hil_connection_failed: Connection failure
    - stoprule_hil_timeout: Timeout exceeded
    - stoprule_hil_packet_loss_exceeded: Loss above threshold
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from src.core import TENANT_ID, StopRule, dual_hash, emit_receipt

# Live relay constants
LIVE_RELAY_ENABLED = True
STARLINK_ANALOG_LATENCY_MS = 500
STARLINK_ANALOG_PACKET_LOSS = 0.001
STARLINK_ANALOG_BANDWIDTH_GBPS = 100
HIL_TIMEOUT_MS = 5000
HIL_RETRY_COUNT = 3
HIL_FAILURE_THRESHOLD = 0.01

# Mars constants
MARS_LATENCY_MIN = 3  # minutes one-way (opposition)
MARS_LATENCY_MAX = 22  # minutes one-way (conjunction)
MARS_HIL_ENABLED = True
MARS_PROOF_DURATION_HOURS = 24


class HILConnectionError(StopRule):
    """Hardware-in-loop connection failure."""

    pass


class HILTimeoutError(StopRule):
    """Hardware-in-loop timeout exceeded."""

    pass


class HILPacketLossError(StopRule):
    """Hardware-in-loop packet loss exceeded threshold."""

    pass


def load_hil_config() -> Dict[str, Any]:
    """Load HIL configuration from spec file.

    Returns:
        dict: HIL configuration including starlink analog settings.

    Receipt:
        live_relay_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "live_relay_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    config = {
        "enabled": spec.get("live_relay_config", {}).get("enabled", LIVE_RELAY_ENABLED),
        "mode": spec.get("live_relay_config", {}).get("mode", "hardware_in_loop"),
        "priority": spec.get("live_relay_config", {}).get(
            "priority", "live_verifiability_before_recursion"
        ),
        "starlink_analog_config": spec.get(
            "starlink_analog_config",
            {
                "latency_ms": STARLINK_ANALOG_LATENCY_MS,
                "packet_loss_rate": STARLINK_ANALOG_PACKET_LOSS,
                "bandwidth_gbps": STARLINK_ANALOG_BANDWIDTH_GBPS,
                "timeout_ms": HIL_TIMEOUT_MS,
                "retry_count": HIL_RETRY_COUNT,
                "failure_threshold": HIL_FAILURE_THRESHOLD,
            },
        ),
        "mars_hil_config": spec.get(
            "mars_hil_config",
            {
                "enabled": MARS_HIL_ENABLED,
                "latency_min_minutes": MARS_LATENCY_MIN,
                "latency_max_minutes": MARS_LATENCY_MAX,
                "proof_duration_hours": MARS_PROOF_DURATION_HOURS,
            },
        ),
    }

    emit_receipt(
        "live_relay_config_receipt",
        {
            "receipt_type": "live_relay_config_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "enabled": config["enabled"],
            "mode": config["mode"],
            "priority": config["priority"],
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )
    return config


def connect_starlink_analog() -> Dict[str, Any]:
    """Establish connection to Starlink analog hardware.

    Returns:
        dict: Connection result with status.

    Receipt:
        hil_connection_receipt
        starlink_analog_receipt

    StopRule:
        HILConnectionError if connection fails after retries.
    """
    from hardware.starlink_analog.interface import StarlinkAnalogInterface

    config = load_hil_config()
    analog_config = config["starlink_analog_config"]

    interface = StarlinkAnalogInterface(
        latency_ms=analog_config.get("latency_ms", STARLINK_ANALOG_LATENCY_MS),
        packet_loss_rate=analog_config.get(
            "packet_loss_rate", STARLINK_ANALOG_PACKET_LOSS
        ),
        bandwidth_gbps=analog_config.get(
            "bandwidth_gbps", STARLINK_ANALOG_BANDWIDTH_GBPS
        ),
        timeout_ms=analog_config.get("timeout_ms", HIL_TIMEOUT_MS),
        mock_mode=True,  # Using mock for now
    )

    retry_count = analog_config.get("retry_count", HIL_RETRY_COUNT)
    connected = False

    for attempt in range(retry_count):
        if interface.connect():
            connected = True
            break
        time.sleep(0.1 * (attempt + 1))  # Exponential backoff

    if not connected:
        emit_receipt(
            "hil_connection_receipt",
            {
                "receipt_type": "hil_connection_receipt",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "connected": False,
                "error": "connection_failed",
                "attempts": retry_count,
                "payload_hash": dual_hash(json.dumps({"connected": False})),
            },
        )
        raise HILConnectionError("Failed to connect to Starlink analog after retries")

    result = {
        "connected": True,
        "interface": interface,
        "config": analog_config,
        "mode": "mock",
    }

    emit_receipt(
        "hil_connection_receipt",
        {
            "receipt_type": "hil_connection_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "connected": True,
            "latency_ms": analog_config["latency_ms"],
            "bandwidth_gbps": analog_config["bandwidth_gbps"],
            "payload_hash": dual_hash(json.dumps({"connected": True})),
        },
    )

    emit_receipt(
        "starlink_analog_receipt",
        {
            "receipt_type": "starlink_analog_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "status": "connected",
            "mode": "mock",
            "payload_hash": dual_hash(json.dumps({"status": "connected"})),
        },
    )

    return result


def disconnect_starlink_analog(
    interface: Optional[Any] = None,
) -> Dict[str, Any]:
    """Clean disconnect from Starlink analog hardware.

    Args:
        interface: StarlinkAnalogInterface instance (optional).

    Returns:
        dict: Disconnect result.

    Receipt:
        hil_connection_receipt
    """
    if interface is not None:
        interface.disconnect()

    result = {"disconnected": True}

    emit_receipt(
        "hil_connection_receipt",
        {
            "receipt_type": "hil_connection_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "connected": False,
            "disconnected": True,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def send_packet(
    interface: Any, data: bytes, timeout_ms: int = HIL_TIMEOUT_MS
) -> Dict[str, Any]:
    """Send packet to Starlink analog with timeout.

    Args:
        interface: StarlinkAnalogInterface instance.
        data: Data bytes to send.
        timeout_ms: Timeout in milliseconds.

    Returns:
        dict: Send result.

    Receipt:
        hil_packet_receipt

    StopRule:
        HILTimeoutError if timeout exceeded.
    """
    start = time.time()

    success = interface.send(data)

    elapsed_ms = (time.time() - start) * 1000

    if elapsed_ms > timeout_ms:
        emit_receipt(
            "hil_packet_receipt",
            {
                "receipt_type": "hil_packet_receipt",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "direction": "send",
                "success": False,
                "error": "timeout",
                "elapsed_ms": elapsed_ms,
                "payload_hash": dual_hash(json.dumps({"success": False})),
            },
        )
        raise HILTimeoutError(f"Send timeout: {elapsed_ms}ms > {timeout_ms}ms")

    result = {
        "success": success,
        "direction": "send",
        "bytes": len(data),
        "elapsed_ms": elapsed_ms,
    }

    emit_receipt(
        "hil_packet_receipt",
        {
            "receipt_type": "hil_packet_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "direction": "send",
            "success": success,
            "bytes": len(data),
            "elapsed_ms": elapsed_ms,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def receive_packet(interface: Any, timeout_ms: int = HIL_TIMEOUT_MS) -> Dict[str, Any]:
    """Receive packet from Starlink analog with timeout.

    Args:
        interface: StarlinkAnalogInterface instance.
        timeout_ms: Timeout in milliseconds.

    Returns:
        dict: Receive result with data.

    Receipt:
        hil_packet_receipt

    StopRule:
        HILTimeoutError if timeout exceeded.
    """
    start = time.time()

    data = interface.receive(timeout_ms)

    elapsed_ms = (time.time() - start) * 1000

    if elapsed_ms > timeout_ms:
        emit_receipt(
            "hil_packet_receipt",
            {
                "receipt_type": "hil_packet_receipt",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "direction": "receive",
                "success": False,
                "error": "timeout",
                "elapsed_ms": elapsed_ms,
                "payload_hash": dual_hash(json.dumps({"success": False})),
            },
        )
        raise HILTimeoutError(f"Receive timeout: {elapsed_ms}ms > {timeout_ms}ms")

    result = {
        "success": data is not None,
        "direction": "receive",
        "bytes": len(data) if data else 0,
        "elapsed_ms": elapsed_ms,
        "data": data,
    }

    emit_receipt(
        "hil_packet_receipt",
        {
            "receipt_type": "hil_packet_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "direction": "receive",
            "success": data is not None,
            "bytes": len(data) if data else 0,
            "elapsed_ms": elapsed_ms,
            "payload_hash": dual_hash(json.dumps({"success": data is not None})),
        },
    )
    return result


def measure_latency(interface: Any, iterations: int = 10) -> Dict[str, Any]:
    """Measure latency over multiple iterations.

    Args:
        interface: StarlinkAnalogInterface instance.
        iterations: Number of measurement iterations.

    Returns:
        dict: Latency statistics.

    Receipt:
        hil_latency_receipt
    """
    latencies = []
    test_data = b"latency_probe_" + str(time.time()).encode()

    for _ in range(iterations):
        start = time.time()
        interface.send(test_data)
        interface.receive(HIL_TIMEOUT_MS)
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

    result = {
        "iterations": iterations,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "avg_ms": sum(latencies) / len(latencies),
        "latencies": latencies,
    }

    emit_receipt(
        "hil_latency_receipt",
        {
            "receipt_type": "hil_latency_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "iterations": iterations,
            "min_ms": result["min_ms"],
            "max_ms": result["max_ms"],
            "avg_ms": result["avg_ms"],
            "payload_hash": dual_hash(json.dumps(result, default=str)),
        },
    )
    return result


def simulate_packet_loss(interface: Any, rate: float) -> Dict[str, Any]:
    """Inject packet loss simulation.

    Args:
        interface: StarlinkAnalogInterface instance.
        rate: Loss rate to simulate (0-1).

    Returns:
        dict: Packet loss simulation result.

    Receipt:
        hil_packet_receipt
    """
    interface.packet_loss_rate = rate
    result = {"loss_rate_set": rate}

    emit_receipt(
        "hil_packet_receipt",
        {
            "receipt_type": "hil_packet_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "set_loss_rate",
            "rate": rate,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def run_hil_test(duration_s: int = 60) -> Dict[str, Any]:
    """Run full hardware-in-loop test.

    Args:
        duration_s: Test duration in seconds.

    Returns:
        dict: HIL test results.

    Receipt:
        live_relay_hil_receipt

    StopRule:
        HILPacketLossError if loss exceeds threshold.
    """
    config = load_hil_config()
    conn = connect_starlink_analog()
    interface = conn["interface"]

    start_time = time.time()
    packets_sent = 0
    packets_received = 0
    packets_lost = 0
    total_latency = 0.0

    while (time.time() - start_time) < duration_s:
        test_data = f"hil_test_{packets_sent}".encode()

        try:
            send_result = send_packet(interface, test_data)
            if send_result["success"]:
                packets_sent += 1

                recv_result = receive_packet(interface)
                if recv_result["success"]:
                    packets_received += 1
                    total_latency += recv_result["elapsed_ms"]
                else:
                    packets_lost += 1
            else:
                packets_lost += 1
        except HILTimeoutError:
            packets_lost += 1

        # Check loss threshold
        if packets_sent > 0:
            loss_rate = packets_lost / packets_sent
            if loss_rate > config["starlink_analog_config"]["failure_threshold"]:
                emit_receipt(
                    "live_relay_hil_receipt",
                    {
                        "receipt_type": "live_relay_hil_receipt",
                        "tenant_id": TENANT_ID,
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "test_passed": False,
                        "error": "packet_loss_exceeded",
                        "loss_rate": loss_rate,
                        "threshold": config["starlink_analog_config"][
                            "failure_threshold"
                        ],
                        "payload_hash": dual_hash(json.dumps({"test_passed": False})),
                    },
                )
                disconnect_starlink_analog(interface)
                raise HILPacketLossError(
                    f"Packet loss {loss_rate:.2%} exceeded threshold"
                )

        time.sleep(0.01)  # Small delay between packets

    disconnect_starlink_analog(interface)

    avg_latency = total_latency / max(1, packets_received)
    loss_rate = packets_lost / max(1, packets_sent)

    result = {
        "test_passed": True,
        "duration_s": duration_s,
        "packets_sent": packets_sent,
        "packets_received": packets_received,
        "packets_lost": packets_lost,
        "loss_rate": loss_rate,
        "avg_latency_ms": avg_latency,
        "throughput_pps": packets_sent / duration_s,
    }

    emit_receipt(
        "live_relay_hil_receipt",
        {
            "receipt_type": "live_relay_hil_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "test_passed": True,
            "duration_s": duration_s,
            "packets_sent": packets_sent,
            "packets_received": packets_received,
            "loss_rate": loss_rate,
            "avg_latency_ms": avg_latency,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def validate_relay_chain(nodes: int = 5) -> Dict[str, Any]:
    """Validate multi-node relay chain.

    Args:
        nodes: Number of relay nodes in chain.

    Returns:
        dict: Relay chain validation result.

    Receipt:
        live_relay_hil_receipt
    """
    config = load_hil_config()
    per_hop_latency = config["starlink_analog_config"]["latency_ms"]

    # Simulate relay chain
    total_latency = per_hop_latency * nodes

    # Run validation
    conn = connect_starlink_analog()
    interface = conn["interface"]

    # Simulate chain traversal
    chain_results = []
    for node_id in range(nodes):
        test_data = f"relay_node_{node_id}".encode()
        send_result = send_packet(interface, test_data)
        recv_result = receive_packet(interface)
        chain_results.append(
            {
                "node_id": node_id,
                "send_success": send_result["success"],
                "recv_success": recv_result["success"],
                "latency_ms": recv_result.get("elapsed_ms", 0),
            }
        )

    disconnect_starlink_analog(interface)

    all_success = all(r["send_success"] and r["recv_success"] for r in chain_results)
    actual_latency = sum(r["latency_ms"] for r in chain_results)

    result = {
        "validated": all_success,
        "nodes": nodes,
        "expected_latency_ms": total_latency,
        "actual_latency_ms": actual_latency,
        "chain_results": chain_results,
    }

    emit_receipt(
        "live_relay_hil_receipt",
        {
            "receipt_type": "live_relay_hil_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "test_type": "relay_chain_validation",
            "validated": all_success,
            "nodes": nodes,
            "expected_latency_ms": total_latency,
            "actual_latency_ms": actual_latency,
            "payload_hash": dual_hash(json.dumps(result, default=str)),
        },
    )
    return result


def mars_hil_proof(duration_hours: float = 1.0) -> Dict[str, Any]:
    """Run Mars latency proof-of-concept.

    Simulates Mars communication latencies (3-22 minutes) to validate
    autonomy and coordination protocols.

    Args:
        duration_hours: Proof duration in hours.

    Returns:
        dict: Mars HIL proof result.

    Receipt:
        mars_hil_proof_receipt
    """
    from hardware.starlink_analog.latency_sim import simulate_mars_latency

    config = load_hil_config()
    mars_config = config["mars_hil_config"]

    # Get latency bounds
    opposition_latency = simulate_mars_latency("opposition")
    conjunction_latency = simulate_mars_latency("conjunction")

    # Run proof simulation
    conn = connect_starlink_analog()
    interface = conn["interface"]

    start_time = time.time()
    duration_s = duration_hours * 3600
    messages_sent = 0
    messages_received = 0
    autonomy_periods = 0

    # Simulate varying Mars latencies
    phases = ["opposition", "average", "conjunction"]
    phase_idx = 0

    while (time.time() - start_time) < duration_s:
        # Cycle through phases
        current_phase = phases[phase_idx % len(phases)]
        current_latency_min = simulate_mars_latency(current_phase)

        # Simulate autonomy period (round trip time)
        autonomy_periods += 1

        # Send test message
        test_data = f"mars_msg_{messages_sent}_phase_{current_phase}".encode()
        send_result = send_packet(interface, test_data)
        if send_result["success"]:
            messages_sent += 1

        # Simulate latency (scaled down for testing)
        time.sleep(min(0.1, current_latency_min / 60))

        recv_result = receive_packet(interface)
        if recv_result["success"]:
            messages_received += 1

        phase_idx += 1
        time.sleep(0.01)

    disconnect_starlink_analog(interface)

    success_rate = messages_received / max(1, messages_sent)
    autonomy_achieved = success_rate >= mars_config["autonomy_target"]

    result = {
        "proof_passed": autonomy_achieved,
        "duration_hours": duration_hours,
        "messages_sent": messages_sent,
        "messages_received": messages_received,
        "success_rate": success_rate,
        "autonomy_target": mars_config["autonomy_target"],
        "autonomy_periods": autonomy_periods,
        "opposition_latency_min": opposition_latency,
        "conjunction_latency_min": conjunction_latency,
    }

    emit_receipt(
        "mars_hil_proof_receipt",
        {
            "receipt_type": "mars_hil_proof_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "proof_passed": autonomy_achieved,
            "duration_hours": duration_hours,
            "success_rate": success_rate,
            "autonomy_target": mars_config["autonomy_target"],
            "opposition_latency_min": opposition_latency,
            "conjunction_latency_min": conjunction_latency,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def stress_test_hil(iterations: int = 100) -> Dict[str, Any]:
    """Run stress test on HIL system.

    Args:
        iterations: Number of stress iterations.

    Returns:
        dict: Stress test results.

    Receipt:
        live_relay_hil_receipt
    """
    conn = connect_starlink_analog()
    interface = conn["interface"]

    results = []
    start_time = time.time()

    for i in range(iterations):
        test_data = f"stress_test_{i}_{'x' * 1000}".encode()  # 1KB+ payload
        iter_start = time.time()

        send_result = send_packet(interface, test_data)
        recv_result = receive_packet(interface)

        iter_time = (time.time() - iter_start) * 1000
        results.append(
            {
                "iteration": i,
                "send_success": send_result["success"],
                "recv_success": recv_result["success"],
                "latency_ms": iter_time,
            }
        )

    disconnect_starlink_analog(interface)

    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r["send_success"] and r["recv_success"])
    latencies = [r["latency_ms"] for r in results]

    result = {
        "stress_passed": success_count >= iterations * 0.99,  # 99% success
        "iterations": iterations,
        "success_count": success_count,
        "success_rate": success_count / iterations,
        "total_time_s": total_time,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "throughput_ops": iterations / total_time,
    }

    emit_receipt(
        "live_relay_hil_receipt",
        {
            "receipt_type": "live_relay_hil_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "test_type": "stress_test",
            "stress_passed": result["stress_passed"],
            "iterations": iterations,
            "success_rate": result["success_rate"],
            "avg_latency_ms": result["avg_latency_ms"],
            "throughput_ops": result["throughput_ops"],
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def get_hil_status() -> Dict[str, Any]:
    """Get current HIL system status.

    Returns:
        dict: Current HIL status.

    Receipt:
        live_relay_hil_receipt
    """
    config = load_hil_config()

    status = {
        "hil_enabled": config["enabled"],
        "mode": config["mode"],
        "priority": config["priority"],
        "starlink_analog": {
            "latency_ms": config["starlink_analog_config"]["latency_ms"],
            "bandwidth_gbps": config["starlink_analog_config"]["bandwidth_gbps"],
            "packet_loss_rate": config["starlink_analog_config"]["packet_loss_rate"],
        },
        "mars_hil": {
            "enabled": config["mars_hil_config"]["enabled"],
            "latency_range_min": f"{config['mars_hil_config']['latency_min_minutes']}-{config['mars_hil_config']['latency_max_minutes']}",
        },
    }

    emit_receipt(
        "live_relay_hil_receipt",
        {
            "receipt_type": "live_relay_hil_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "status_type": "system_status",
            "hil_enabled": status["hil_enabled"],
            "mode": status["mode"],
            "payload_hash": dual_hash(json.dumps(status)),
        },
    )
    return status


# StopRule functions for external use
def stoprule_hil_connection_failed(error: str) -> None:
    """Raise StopRule for connection failure."""
    raise HILConnectionError(f"HIL connection failed: {error}")


def stoprule_hil_timeout(latency: float) -> None:
    """Raise StopRule for timeout exceeded."""
    raise HILTimeoutError(f"HIL timeout exceeded: {latency}ms")


def stoprule_hil_packet_loss_exceeded(rate: float) -> None:
    """Raise StopRule for packet loss exceeded."""
    raise HILPacketLossError(f"HIL packet loss exceeded: {rate:.2%}")
