"""Tests for Multi-planet path.

Tests:
- test_multiplanet_spec_loads: Spec loads with valid dual-hash
- test_multiplanet_sequence: assert sequence == ["asteroid", "mars", "europa", "titan"]
- test_multiplanet_latency_bounds: assert all bodies have latency bounds
- test_multiplanet_autonomy_increases: assert autonomy requirement increases with distance
- test_multiplanet_receipt_emitted: assert mp_status_receipt emitted
- test_multiplanet_cli_routes: assert CLI commands route correctly

Source: AXIOM scalable paths architecture - Multi-planet expansion
"""


def test_multiplanet_spec_loads():
    """Spec loads with valid structure."""
    from src.paths import load_path_spec

    spec = load_path_spec("multiplanet")

    assert spec is not None
    assert spec["path"] == "multiplanet"
    assert "version" in spec
    assert "config" in spec
    assert spec["status"] == "stub"


def test_multiplanet_stub_ready():
    """Stub status reports ready."""
    from src.paths.multiplanet import stub_status

    status = stub_status()

    assert status["ready"] is True
    assert status["stage"] == "stub"
    assert "evolution_path" in status


def test_multiplanet_sequence():
    """Expansion sequence is correct (ordered by increasing autonomy)."""
    from src.paths.multiplanet import get_sequence, EXPANSION_SEQUENCE

    sequence = get_sequence()

    assert sequence == ["asteroid", "mars", "europa", "ganymede", "titan"]
    assert sequence == list(EXPANSION_SEQUENCE)


def test_multiplanet_latency_bounds():
    """All bodies have latency bounds."""
    from src.paths import load_path_spec

    spec = load_path_spec("multiplanet")
    latency_bounds = spec["config"]["relay_latency_bounds"]

    for body in ["asteroid", "mars", "europa", "titan"]:
        assert body in latency_bounds
        bounds = latency_bounds[body]
        assert len(bounds) == 2
        assert bounds[0] < bounds[1]  # min < max


def test_multiplanet_autonomy_increases():
    """Autonomy requirement increases with distance."""
    from src.paths.multiplanet import AUTONOMY_REQUIREMENT, EXPANSION_SEQUENCE

    prev_autonomy = 0.0
    for body in EXPANSION_SEQUENCE:
        current_autonomy = AUTONOMY_REQUIREMENT[body]
        assert current_autonomy > prev_autonomy, f"Autonomy should increase: {body}"
        prev_autonomy = current_autonomy


def test_multiplanet_autonomy_values():
    """Autonomy values match spec."""
    from src.paths import load_path_spec

    spec = load_path_spec("multiplanet")
    autonomy = spec["config"]["autonomy_requirement"]

    assert autonomy["asteroid"] == 0.7
    assert autonomy["mars"] == 0.85
    assert autonomy["europa"] == 0.95
    assert autonomy["titan"] == 0.99


def test_multiplanet_bandwidth_budget():
    """Bandwidth budget is defined for all bodies."""
    from src.paths import load_path_spec

    spec = load_path_spec("multiplanet")
    bandwidth = spec["config"]["bandwidth_budget_mbps"]

    assert bandwidth["asteroid"] == 500
    assert bandwidth["mars"] == 100
    assert bandwidth["europa"] == 20
    assert bandwidth["titan"] == 5


def test_multiplanet_compression_target():
    """Telemetry compression target is defined."""
    from src.paths import load_path_spec

    spec = load_path_spec("multiplanet")
    config = spec["config"]

    assert config.get("telemetry_compression_target") == 0.95


def test_multiplanet_get_body_config():
    """Body config returns expected structure."""
    from src.paths.multiplanet import get_body_config

    config = get_body_config("mars")

    assert config["body"] == "mars"
    assert config["sequence_position"] == 2
    assert "latency_min_min" in config
    assert "latency_max_min" in config
    assert "autonomy_requirement" in config
    assert "bandwidth_budget_mbps" in config


def test_multiplanet_latency_budget():
    """Latency budget computation works."""
    from src.paths.multiplanet import compute_latency_budget

    budget = compute_latency_budget("mars")

    assert budget["body"] == "mars"
    assert budget["one_way_min_min"] == 3
    assert budget["one_way_max_min"] == 22
    assert budget["round_trip_min_min"] == 6
    assert budget["round_trip_max_min"] == 44


def test_multiplanet_autonomy_requirement():
    """Autonomy requirement function works."""
    from src.paths.multiplanet import compute_autonomy_requirement

    autonomy = compute_autonomy_requirement("titan")

    assert autonomy == 0.99


def test_multiplanet_simulate_body():
    """Body simulation returns expected structure."""
    from src.paths.multiplanet import simulate_body

    result = simulate_body("asteroid")

    assert result["stub_mode"] is True
    assert result["body"] == "asteroid"
    assert "autonomy_required" in result


def test_multiplanet_telemetry_compression():
    """Telemetry compression computation works."""
    from src.paths.multiplanet import compute_telemetry_compression

    result = compute_telemetry_compression("mars", data_rate_mbps=1000)

    assert result["body"] == "mars"
    assert result["raw_data_rate_mbps"] == 1000
    assert result["bandwidth_budget_mbps"] == 100
    assert result["compression_needed"] == 10.0  # 1000/100


def test_multiplanet_dependencies():
    """Dependencies include mars."""
    from src.paths import load_path_spec

    spec = load_path_spec("multiplanet")

    assert "mars" in spec["dependencies"]


def test_multiplanet_receipts_defined():
    """Receipt types are defined."""
    from src.paths import load_path_spec

    spec = load_path_spec("multiplanet")

    receipts = spec.get("receipts", [])
    assert "mp_status" in receipts
    assert "mp_sequence" in receipts
    assert "mp_body" in receipts
    assert "mp_telemetry" in receipts


def test_multiplanet_cli_status():
    """CLI status command works."""
    from src.paths.multiplanet.cli import cmd_multiplanet_status

    result = cmd_multiplanet_status()

    assert result["ready"] is True
    assert "version" in result


def test_multiplanet_cli_sequence():
    """CLI sequence command works."""
    from src.paths.multiplanet.cli import cmd_multiplanet_sequence

    result = cmd_multiplanet_sequence()

    assert "sequence" in result
    assert result["sequence"] == ["asteroid", "mars", "europa", "ganymede", "titan"]
