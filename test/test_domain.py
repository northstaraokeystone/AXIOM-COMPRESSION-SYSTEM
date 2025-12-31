"""Tests for spaceproof.domain module."""

from spaceproof.domain import galaxy, colony, telemetry


def test_galaxy_module_exists():
    """galaxy module is importable."""
    assert galaxy is not None


def test_colony_module_exists():
    """colony module is importable."""
    assert colony is not None


def test_telemetry_module_exists():
    """telemetry module is importable."""
    assert telemetry is not None


def test_galaxy_has_generate():
    """galaxy module has generate function."""
    assert hasattr(galaxy, "generate")


def test_colony_has_generate():
    """colony module has generate function."""
    assert hasattr(colony, "generate")


def test_telemetry_has_generate():
    """telemetry module has generate function."""
    assert hasattr(telemetry, "generate")
