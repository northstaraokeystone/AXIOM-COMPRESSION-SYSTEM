"""Starlink analog hardware interface abstraction.

Provides a hardware interface abstraction for Starlink analog testing.
Supports both real hardware connections and mock interfaces for unit testing.

Receipt Types:
    - starlink_analog_connect_receipt: Connection established
    - starlink_analog_disconnect_receipt: Connection closed
    - starlink_analog_send_receipt: Data sent
    - starlink_analog_receive_receipt: Data received
    - starlink_analog_stats_receipt: Connection statistics
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from src.core import TENANT_ID, dual_hash, emit_receipt

# Default configuration
DEFAULT_LATENCY_MS = 500
DEFAULT_PACKET_LOSS = 0.001
DEFAULT_BANDWIDTH_GBPS = 100
DEFAULT_TIMEOUT_MS = 5000


@dataclass
class ConnectionStats:
    """Statistics for analog connection."""

    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    packets_lost: int = 0
    total_latency_ms: float = 0.0
    connection_uptime_s: float = 0.0
    error_count: int = 0


@dataclass
class StarlinkAnalogInterface:
    """Interface to Starlink analog hardware.

    Abstracts hardware communication for testing interstellar relay protocols.
    Can operate in mock mode for unit testing without actual hardware.
    """

    latency_ms: float = DEFAULT_LATENCY_MS
    packet_loss_rate: float = DEFAULT_PACKET_LOSS
    bandwidth_gbps: float = DEFAULT_BANDWIDTH_GBPS
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    mock_mode: bool = True
    _connected: bool = field(default=False, init=False)
    _stats: ConnectionStats = field(default_factory=ConnectionStats, init=False)
    _connect_time: Optional[float] = field(default=None, init=False)

    def connect(self) -> bool:
        """Establish connection to analog hardware.

        Returns:
            bool: True if connection successful, False otherwise.

        Receipt:
            starlink_analog_connect_receipt
        """
        if self._connected:
            return True

        # Simulate connection delay
        if self.mock_mode:
            time.sleep(0.01)  # 10ms simulated connection time
            self._connected = True
            self._connect_time = time.time()
            self._stats = ConnectionStats()

            emit_receipt(
                "starlink_analog_connect_receipt",
                {
                    "receipt_type": "starlink_analog_connect_receipt",
                    "tenant_id": TENANT_ID,
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "connected": True,
                    "mock_mode": self.mock_mode,
                    "latency_ms": self.latency_ms,
                    "bandwidth_gbps": self.bandwidth_gbps,
                    "payload_hash": dual_hash(
                        json.dumps({"connected": True, "mock_mode": self.mock_mode})
                    ),
                },
            )
            return True

        # Real hardware connection would go here
        # For now, we only support mock mode
        return False

    def disconnect(self) -> bool:
        """Clean disconnect from analog hardware.

        Returns:
            bool: True if disconnect successful, False otherwise.

        Receipt:
            starlink_analog_disconnect_receipt
        """
        if not self._connected:
            return True

        uptime = time.time() - self._connect_time if self._connect_time else 0.0
        self._stats.connection_uptime_s = uptime

        self._connected = False
        self._connect_time = None

        emit_receipt(
            "starlink_analog_disconnect_receipt",
            {
                "receipt_type": "starlink_analog_disconnect_receipt",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "disconnected": True,
                "uptime_s": uptime,
                "final_stats": {
                    "bytes_sent": self._stats.bytes_sent,
                    "bytes_received": self._stats.bytes_received,
                    "packets_sent": self._stats.packets_sent,
                    "packets_received": self._stats.packets_received,
                },
                "payload_hash": dual_hash(
                    json.dumps({"disconnected": True, "uptime_s": uptime})
                ),
            },
        )
        return True

    def send(self, data: bytes) -> bool:
        """Send data to analog hardware.

        Args:
            data: Bytes to send.

        Returns:
            bool: True if send successful, False if packet lost.

        Receipt:
            starlink_analog_send_receipt
        """
        if not self._connected:
            return False

        # Simulate latency
        time.sleep(self.latency_ms / 1000.0 / 100)  # Scaled down for testing

        # Simulate packet loss
        if random.random() < self.packet_loss_rate:
            self._stats.packets_lost += 1
            self._stats.packets_sent += 1
            emit_receipt(
                "starlink_analog_send_receipt",
                {
                    "receipt_type": "starlink_analog_send_receipt",
                    "tenant_id": TENANT_ID,
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "success": False,
                    "reason": "packet_lost",
                    "bytes": len(data),
                    "payload_hash": dual_hash(json.dumps({"success": False})),
                },
            )
            return False

        self._stats.bytes_sent += len(data)
        self._stats.packets_sent += 1
        self._stats.total_latency_ms += self.latency_ms

        emit_receipt(
            "starlink_analog_send_receipt",
            {
                "receipt_type": "starlink_analog_send_receipt",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "success": True,
                "bytes": len(data),
                "latency_ms": self.latency_ms,
                "payload_hash": dual_hash(json.dumps({"success": True, "bytes": len(data)})),
            },
        )
        return True

    def receive(self, timeout_ms: Optional[int] = None) -> Optional[bytes]:
        """Receive data from analog hardware.

        Args:
            timeout_ms: Timeout in milliseconds. Uses default if None.

        Returns:
            bytes: Received data, or None if timeout/error.

        Receipt:
            starlink_analog_receive_receipt
        """
        if not self._connected:
            return None

        _ = timeout_ms or self.timeout_ms  # Used for timeout tracking

        # In mock mode, simulate receiving echo data
        if self.mock_mode:
            time.sleep(self.latency_ms / 1000.0 / 100)  # Scaled down

            # Simulate packet loss on receive
            if random.random() < self.packet_loss_rate:
                self._stats.packets_lost += 1
                return None

            # Generate mock response
            mock_data = b"mock_response_" + str(time.time()).encode()
            self._stats.bytes_received += len(mock_data)
            self._stats.packets_received += 1
            self._stats.total_latency_ms += self.latency_ms

            emit_receipt(
                "starlink_analog_receive_receipt",
                {
                    "receipt_type": "starlink_analog_receive_receipt",
                    "tenant_id": TENANT_ID,
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "success": True,
                    "bytes": len(mock_data),
                    "latency_ms": self.latency_ms,
                    "payload_hash": dual_hash(
                        json.dumps({"success": True, "bytes": len(mock_data)})
                    ),
                },
            )
            return mock_data

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics.

        Returns:
            dict: Connection statistics.

        Receipt:
            starlink_analog_stats_receipt
        """
        if self._connect_time:
            self._stats.connection_uptime_s = time.time() - self._connect_time

        stats = {
            "connected": self._connected,
            "bytes_sent": self._stats.bytes_sent,
            "bytes_received": self._stats.bytes_received,
            "packets_sent": self._stats.packets_sent,
            "packets_received": self._stats.packets_received,
            "packets_lost": self._stats.packets_lost,
            "packet_loss_rate_actual": (
                self._stats.packets_lost / max(1, self._stats.packets_sent)
            ),
            "total_latency_ms": self._stats.total_latency_ms,
            "avg_latency_ms": (
                self._stats.total_latency_ms / max(1, self._stats.packets_sent)
            ),
            "connection_uptime_s": self._stats.connection_uptime_s,
            "error_count": self._stats.error_count,
        }

        emit_receipt(
            "starlink_analog_stats_receipt",
            {
                "receipt_type": "starlink_analog_stats_receipt",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "stats": stats,
                "payload_hash": dual_hash(json.dumps(stats, sort_keys=True)),
            },
        )
        return stats


def create_mock_interface(
    latency_ms: float = DEFAULT_LATENCY_MS,
    packet_loss_rate: float = DEFAULT_PACKET_LOSS,
    bandwidth_gbps: float = DEFAULT_BANDWIDTH_GBPS,
) -> StarlinkAnalogInterface:
    """Create a mock interface for testing.

    Args:
        latency_ms: Simulated latency in milliseconds.
        packet_loss_rate: Simulated packet loss rate (0-1).
        bandwidth_gbps: Simulated bandwidth in Gbps.

    Returns:
        StarlinkAnalogInterface: Configured mock interface.
    """
    return StarlinkAnalogInterface(
        latency_ms=latency_ms,
        packet_loss_rate=packet_loss_rate,
        bandwidth_gbps=bandwidth_gbps,
        mock_mode=True,
    )


def load_interface_config() -> Dict[str, Any]:
    """Load interface configuration from spec file.

    Returns:
        dict: Interface configuration.
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "live_relay_spec.json",
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)
    return spec.get("starlink_analog_config", {})
