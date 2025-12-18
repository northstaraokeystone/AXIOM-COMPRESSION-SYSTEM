"""Tests for Mars habitat path.

Tests:
- test_mars_spec_loads: Spec loads with valid dual-hash
- test_mars_stub_ready: assert stub_status()['ready']
- test_mars_isru_target: assert isru_closure_target == 0.85
- test_mars_sovereignty_defined: assert sovereignty_threshold exists
- test_mars_receipt_emitted: assert mars_status_receipt emitted
- test_mars_cli_routes: assert CLI commands route correctly

Source: AXIOM scalable paths architecture - Mars autonomous habitat
"""


def test_mars_spec_loads():
    """Spec loads with valid structure."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")

    assert spec is not None
    assert spec["path"] == "mars"
    assert "version" in spec
    assert "config" in spec
    assert spec["status"] == "stub"


def test_mars_stub_ready():
    """Stub status reports ready."""
    from src.paths.mars import stub_status

    status = stub_status()

    assert status["ready"] is True
    assert status["stage"] == "stub"
    assert "evolution_path" in status


def test_mars_isru_target():
    """ISRU closure target is 0.85."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")
    config = spec.get("config", {})

    assert config.get("isru_closure_target") == 0.85


def test_mars_isru_uplift_target():
    """ISRU uplift target is 0.15."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")
    config = spec.get("config", {})

    assert config.get("isru_uplift_target") == 0.15


def test_mars_sovereignty_defined():
    """Sovereignty threshold is defined."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")
    config = spec.get("config", {})

    assert "sovereignty_threshold" in config
    assert config["sovereignty_threshold"] is True


def test_mars_crew_range():
    """Crew range is defined."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")
    config = spec.get("config", {})

    assert config.get("crew_range") == [4, 1000]


def test_mars_decision_rate():
    """Decision rate target is defined."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")
    config = spec.get("config", {})

    assert config.get("decision_rate_target_bps") == 1000


def test_mars_dome_resources():
    """Dome resources are defined."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")
    config = spec.get("config", {})

    resources = config.get("dome_resources", [])
    assert "water" in resources
    assert "o2" in resources
    assert "power" in resources
    assert "food" in resources


def test_mars_simulate_dome():
    """Dome simulation returns expected structure."""
    from src.paths.mars import simulate_dome

    result = simulate_dome(crew=50, duration_days=365)

    assert result["stub_mode"] is True
    assert result["crew"] == 50
    assert result["duration_days"] == 365
    assert "resources_required" in result
    assert "isru_projected" in result
    assert "isru_closure_projected" in result


def test_mars_compute_isru_closure():
    """ISRU closure computation works."""
    from src.paths.mars import compute_isru_closure

    resources = {
        "water": (85, 100),
        "o2": (90, 100),
        "power": (95, 100),
        "food": (30, 100),
    }

    closure = compute_isru_closure(resources)

    # (85+90+95+30) / (100+100+100+100) = 300/400 = 0.75
    assert abs(closure - 0.75) < 0.01


def test_mars_compute_sovereignty():
    """Sovereignty computation works."""
    from src.paths.mars import compute_sovereignty

    # High crew, should be sovereign
    is_sovereign = compute_sovereignty(crew=100, bandwidth_mbps=100, latency_s=1200)
    assert is_sovereign is True

    # Very low crew, should not be sovereign
    is_sovereign = compute_sovereignty(crew=1, bandwidth_mbps=1000, latency_s=10)
    assert is_sovereign is False


def test_mars_dependencies():
    """Dependencies are specified."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")

    assert "dependencies" in spec
    assert "fractal_layers" in spec["dependencies"]


def test_mars_receipts_defined():
    """Receipt types are defined."""
    from src.paths import load_path_spec

    spec = load_path_spec("mars")

    receipts = spec.get("receipts", [])
    assert "mars_status" in receipts
    assert "mars_dome" in receipts
    assert "mars_isru" in receipts
    assert "mars_sovereignty" in receipts


def test_mars_cli_status():
    """CLI status command works."""
    from src.paths.mars.cli import cmd_mars_status

    result = cmd_mars_status()

    assert result["ready"] is True
    assert "version" in result
