"""Tests for pruning v4 module."""

from src.pruning_v4 import (
    load_pruning_config,
    prune_v4,
    identify_holes_v4,
    eliminate_holes_v4,
    iterative_prune,
    compute_persistence_diagram,
    measure_compression_v4,
    validate_pruning_target,
    compare_to_v3,
    get_pruning_status,
)


def create_test_tree():
    """Create test tree for pruning."""
    import random

    random.seed(42)  # Reproducible
    nodes = []
    for i in range(100):
        node = {
            "id": i,
            "value": random.random(),
            "children": [{"sub_id": j, "data": random.random()} for j in range(5)],
        }
        nodes.append(node)

    return {
        "root": "test_tree",
        "nodes": nodes,
        "depth": 3,
        "metadata": {"created": "test", "version": "1.0"},
    }


class TestPruningConfig:
    """Tests for pruning configuration."""

    def test_pruning_config_loads(self):
        """Config loads successfully."""
        config = load_pruning_config()
        assert config is not None
        assert "enabled" in config
        assert "compression_target" in config

    def test_pruning_v4_enabled(self):
        """Enabled flag works."""
        config = load_pruning_config()
        assert config["enabled"] is True

    def test_pruning_compression_target(self):
        """Target is 99.6%."""
        config = load_pruning_config()
        assert config["compression_target"] == 0.996


class TestPruningHoleDetection:
    """Tests for hole detection."""

    def test_pruning_hole_detection(self):
        """Holes detected."""
        tree = create_test_tree()
        holes = identify_holes_v4(tree, depth=3)
        assert isinstance(holes, list)

    def test_pruning_hole_dimensions(self):
        """Holes have dimension info."""
        tree = create_test_tree()
        holes = identify_holes_v4(tree, depth=3)
        if holes:
            assert all(hasattr(h, "dimension") for h in holes)


class TestPruningHoleElimination:
    """Tests for hole elimination."""

    def test_pruning_hole_elimination(self):
        """Holes eliminated."""
        tree = create_test_tree()
        holes = identify_holes_v4(tree, depth=3)
        result = eliminate_holes_v4(tree, holes)
        assert "tree" in result
        assert "holes_eliminated" in result


class TestPruningIterative:
    """Tests for iterative pruning."""

    def test_pruning_iterative(self):
        """Iterative passes work."""
        tree = create_test_tree()
        result = iterative_prune(tree, passes=3)
        assert "tree" in result
        assert result["passes_completed"] <= 3
        assert "final_compression" in result


class TestPruningCompression:
    """Tests for compression target."""

    def test_pruning_compression_target(self):
        """Compression >= 0.995."""
        tree = create_test_tree()
        result = prune_v4(tree)
        assert result["compression"] >= 0.995

    def test_pruning_target_met(self):
        """Target met flag correct."""
        tree = create_test_tree()
        result = prune_v4(tree)
        assert result["target_met"] is True


class TestPruningPersistence:
    """Tests for persistence diagram."""

    def test_pruning_persistence(self):
        """Persistence computed."""
        tree = create_test_tree()
        result = compute_persistence_diagram(tree, depth=3)
        assert "dimensions" in result
        assert "diagrams" in result
        assert "H0" in result["diagrams"]


class TestPruningComparison:
    """Tests for v3 vs v4 comparison."""

    def test_pruning_compare_v3(self):
        """v4 > v3."""
        tree = create_test_tree()
        result = compare_to_v3(tree)
        assert result["v4_better"] is True
        assert result["v4_compression"] > result["v3_compression"]


class TestPruningReceipts:
    """Tests for receipt emission."""

    def test_pruning_receipt(self, capsys):
        """Receipt emitted."""
        load_pruning_config()
        captured = capsys.readouterr()
        assert "pruning_v4_config_receipt" in captured.out


class TestPruningMeasurement:
    """Tests for compression measurement."""

    def test_pruning_measure(self):
        """Measure compression works."""
        original = {"nodes": [1, 2, 3, 4, 5]}
        pruned = {"nodes": [1]}
        ratio = measure_compression_v4(original, pruned)
        assert 0.0 <= ratio <= 1.0


class TestPruningValidation:
    """Tests for target validation."""

    def test_pruning_validate_target(self):
        """Target validation works."""
        assert validate_pruning_target(0.996, 0.995) is True
        assert validate_pruning_target(0.990, 0.995) is False


class TestPruningStatus:
    """Tests for status queries."""

    def test_pruning_status(self):
        """Status query works."""
        status = get_pruning_status()
        assert "enabled" in status
        assert "compression_target" in status
        assert "persistence_depth" in status
        assert "method" in status
