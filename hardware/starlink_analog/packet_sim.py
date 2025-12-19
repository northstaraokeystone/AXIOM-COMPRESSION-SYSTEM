"""Packet loss and corruption simulation for Starlink analog testing.

Provides utilities for simulating packet loss, burst loss patterns,
and data corruption scenarios for hardware-in-loop testing.

Receipt Types:
    - packet_sim_loss_receipt: Packet loss event
    - packet_sim_burst_receipt: Burst loss pattern
    - packet_sim_corruption_receipt: Data corruption event
"""

import json
import random
from datetime import datetime
from typing import List, Tuple

from src.core import TENANT_ID, dual_hash, emit_receipt

# Default parameters
DEFAULT_LOSS_RATE = 0.001  # 0.1%
DEFAULT_BURST_LENGTH = 3
DEFAULT_CORRUPTION_RATE = 0.0001  # 0.01%


def simulate_packet_loss(rate: float = DEFAULT_LOSS_RATE) -> bool:
    """Simulate probabilistic packet loss.

    Args:
        rate: Packet loss rate (0-1).

    Returns:
        bool: True if packet is lost, False if delivered.

    Receipt:
        packet_sim_loss_receipt (only when loss occurs)
    """
    lost = random.random() < rate

    if lost:
        emit_receipt(
            "packet_sim_loss_receipt",
            {
                "receipt_type": "packet_sim_loss_receipt",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "lost": True,
                "rate": rate,
                "payload_hash": dual_hash(json.dumps({"lost": True, "rate": rate})),
            },
        )

    return lost


def simulate_burst_loss(burst_length: int = DEFAULT_BURST_LENGTH) -> List[bool]:
    """Simulate burst loss pattern.

    Generates a burst loss pattern where consecutive packets are likely
    to be lost together, simulating real network conditions.

    Args:
        burst_length: Expected length of burst.

    Returns:
        list: Boolean list indicating loss status for each packet in burst.

    Receipt:
        packet_sim_burst_receipt
    """
    # Gilbert-Elliott model simplified
    # Once in loss state, high probability of continued loss
    pattern = []
    in_burst = random.random() < 0.1  # 10% chance to enter burst state

    for _ in range(burst_length):
        if in_burst:
            lost = random.random() < 0.8  # 80% loss in burst
            pattern.append(lost)
            if not lost:
                in_burst = False  # Exit burst state
        else:
            lost = random.random() < 0.01  # 1% loss normally
            pattern.append(lost)
            if lost:
                in_burst = True  # Enter burst state

    emit_receipt(
        "packet_sim_burst_receipt",
        {
            "receipt_type": "packet_sim_burst_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "burst_length": burst_length,
            "pattern": pattern,
            "loss_count": sum(pattern),
            "loss_rate": sum(pattern) / len(pattern),
            "payload_hash": dual_hash(
                json.dumps({"burst_length": burst_length, "pattern": pattern})
            ),
        },
    )
    return pattern


def simulate_corruption(
    data: bytes, rate: float = DEFAULT_CORRUPTION_RATE
) -> Tuple[bytes, int]:
    """Simulate bit corruption in data.

    Args:
        data: Original data bytes.
        rate: Bit corruption rate (0-1).

    Returns:
        tuple: (Corrupted data, number of corrupted bits).

    Receipt:
        packet_sim_corruption_receipt
    """
    data_array = bytearray(data)
    corrupted_bits = 0

    for i in range(len(data_array)):
        for bit in range(8):
            if random.random() < rate:
                data_array[i] ^= 1 << bit  # Flip bit
                corrupted_bits += 1

    corrupted_data = bytes(data_array)

    if corrupted_bits > 0:
        emit_receipt(
            "packet_sim_corruption_receipt",
            {
                "receipt_type": "packet_sim_corruption_receipt",
                "tenant_id": TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "original_size": len(data),
                "corrupted_bits": corrupted_bits,
                "corruption_rate_actual": corrupted_bits / (len(data) * 8),
                "data_integrity": corrupted_bits == 0,
                "payload_hash": dual_hash(
                    json.dumps(
                        {"original_size": len(data), "corrupted_bits": corrupted_bits}
                    )
                ),
            },
        )

    return corrupted_data, corrupted_bits


def calculate_ber(errors: int, total_bits: int) -> float:
    """Calculate Bit Error Rate.

    Args:
        errors: Number of bit errors.
        total_bits: Total bits transmitted.

    Returns:
        float: Bit Error Rate.
    """
    return errors / max(1, total_bits)


def simulate_congestion_loss(queue_depth: int, max_queue: int) -> bool:
    """Simulate congestion-based packet loss.

    Args:
        queue_depth: Current queue depth.
        max_queue: Maximum queue size.

    Returns:
        bool: True if packet dropped due to congestion.
    """
    if queue_depth >= max_queue:
        return True  # Tail drop

    # Random Early Detection (RED) style
    threshold = max_queue * 0.7
    if queue_depth > threshold:
        drop_prob = (queue_depth - threshold) / (max_queue - threshold)
        return random.random() < drop_prob

    return False


def get_packet_sim_config() -> dict:
    """Get default packet simulation configuration.

    Returns:
        dict: Packet simulation parameters.
    """
    return {
        "default_loss_rate": DEFAULT_LOSS_RATE,
        "default_burst_length": DEFAULT_BURST_LENGTH,
        "default_corruption_rate": DEFAULT_CORRUPTION_RATE,
        "interstellar_loss_rate": 0.001,  # Higher for space links
        "mars_loss_rate": 0.0005,  # Mars communication
        "ber_threshold": 1e-9,  # Acceptable BER
    }
