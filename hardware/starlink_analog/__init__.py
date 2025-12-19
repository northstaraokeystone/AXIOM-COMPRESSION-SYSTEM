"""Starlink analog hardware interface package.

Provides hardware-in-loop testing abstractions for Starlink satellite relay
simulation. Enables validation of interstellar coordination protocols using
ground-based analog hardware.

Exports:
    - interface: Hardware interface abstraction
    - latency_sim: Latency simulation utilities
    - packet_sim: Packet loss/corruption simulation
"""

from hardware.starlink_analog import interface
from hardware.starlink_analog import latency_sim
from hardware.starlink_analog import packet_sim

__all__ = ["interface", "latency_sim", "packet_sim"]
