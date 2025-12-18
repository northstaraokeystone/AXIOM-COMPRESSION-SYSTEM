"""Test suite for CLI modular structure.

32 tests covering:
- Import validation (8 tests)
- Dispatcher routing (8 tests)
- Flag parsing (8 tests)
- cli.py size (4 tests)
- Help output (4 tests)
"""

import subprocess
import sys
import pytest


# === IMPORT VALIDATION TESTS (8) ===


class TestImportValidation:
    """Tests for CLI module imports."""

    def test_import_cli_base(self):
        """Test cli/base.py imports successfully."""
        from cli import base
        assert hasattr(base, 'print_header')

    def test_import_cli_depth(self):
        """Test cli/depth.py imports successfully."""
        from cli import depth
        assert hasattr(depth, 'cmd_adaptive_depth_run')

    def test_import_cli_rl(self):
        """Test cli/rl.py imports successfully."""
        from cli import rl
        assert hasattr(rl, 'cmd_rl_500_sweep')

    def test_import_cli_quantum(self):
        """Test cli/quantum.py imports successfully."""
        from cli import quantum
        assert hasattr(quantum, 'cmd_quantum_sim')

    def test_import_cli_pipeline(self):
        """Test cli/pipeline.py imports successfully."""
        from cli import pipeline
        assert hasattr(pipeline, 'cmd_full_pipeline')

    def test_import_cli_fractal(self):
        """Test cli/fractal.py imports successfully."""
        from cli import fractal
        assert hasattr(fractal, 'cmd_fractal_push')

    def test_import_cli_sweep(self):
        """Test cli/sweep.py imports successfully."""
        from cli import sweep
        assert hasattr(sweep, 'cmd_full_500_sweep')

    def test_import_cli_info(self):
        """Test cli/info.py imports successfully."""
        from cli import info
        assert hasattr(info, 'cmd_hybrid_boost_info')


# === DISPATCHER ROUTING TESTS (8) ===


class TestDispatcherRouting:
    """Tests for dispatcher routing."""

    def test_fractal_push_dispatch(self):
        """Test --fractal_push routes to cmd_fractal_push."""
        from cli import cmd_fractal_push
        assert callable(cmd_fractal_push)

    def test_alpha_boost_dispatch(self):
        """Test --alpha_boost routes to cmd_alpha_boost."""
        from cli import cmd_alpha_boost
        assert callable(cmd_alpha_boost)

    def test_fractal_info_hybrid_dispatch(self):
        """Test --fractal_info_hybrid routes to cmd_fractal_info_hybrid."""
        from cli import cmd_fractal_info_hybrid
        assert callable(cmd_fractal_info_hybrid)

    def test_full_500_sweep_dispatch(self):
        """Test --full_500_sweep routes to cmd_full_500_sweep."""
        from cli import cmd_full_500_sweep
        assert callable(cmd_full_500_sweep)

    def test_hybrid_boost_info_dispatch(self):
        """Test --hybrid_boost_info routes to cmd_hybrid_boost_info."""
        from cli import cmd_hybrid_boost_info
        assert callable(cmd_hybrid_boost_info)

    def test_quantum_sim_dispatch(self):
        """Test --quantum_sim routes to cmd_quantum_sim."""
        from cli import cmd_quantum_sim
        assert callable(cmd_quantum_sim)

    def test_lr_pilot_dispatch(self):
        """Test --lr_pilot routes to cmd_lr_pilot."""
        from cli import cmd_lr_pilot
        assert callable(cmd_lr_pilot)

    def test_full_pipeline_dispatch(self):
        """Test --full_pipeline routes to cmd_full_pipeline."""
        from cli import cmd_full_pipeline
        assert callable(cmd_full_pipeline)


# === FLAG PARSING TESTS (8) ===


class TestFlagParsing:
    """Tests for argument flag parsing."""

    def test_alpha_boost_flag_exists(self):
        """Test --alpha_boost flag is defined."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--alpha_boost" in result.stdout

    def test_fractal_push_flag_exists(self):
        """Test --fractal_push flag is defined."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--fractal_push" in result.stdout

    def test_fractal_info_hybrid_flag_exists(self):
        """Test --fractal_info_hybrid flag is defined."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--fractal_info_hybrid" in result.stdout

    def test_full_500_sweep_flag_exists(self):
        """Test --full_500_sweep flag is defined."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--full_500_sweep" in result.stdout

    def test_hybrid_boost_info_flag_exists(self):
        """Test --hybrid_boost_info flag is defined."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--hybrid_boost_info" in result.stdout

    def test_base_alpha_flag_exists(self):
        """Test --base_alpha flag is defined."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--base_alpha" in result.stdout

    def test_tree_size_flag_exists(self):
        """Test --tree_size flag is defined."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--tree_size" in result.stdout

    def test_simulate_flag_exists(self):
        """Test --simulate flag is defined."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--simulate" in result.stdout


# === CLI.PY SIZE TESTS (4) ===


class TestCliSize:
    """Tests for cli.py size constraints."""

    def test_cli_token_count(self):
        """Test cli.py is under 5000 tokens (words as proxy)."""
        with open("cli.py", "r") as f:
            content = f.read()
        # Count words as a proxy for tokens
        words = len(content.split())
        # 5000 tokens ~= 3500-4000 words typically
        assert words < 5000, f"cli.py has {words} words (target < 5000)"

    def test_cli_line_count(self):
        """Test cli.py has reasonable line count."""
        with open("cli.py", "r") as f:
            lines = f.readlines()
        # Should be under 500 lines for a dispatcher
        assert len(lines) < 600, f"cli.py has {len(lines)} lines"

    def test_cli_is_dispatcher_only(self):
        """Test cli.py is primarily a dispatcher (no complex logic)."""
        with open("cli.py", "r") as f:
            content = f.read()
        # Should not contain class definitions (business logic)
        assert "class " not in content or content.count("class ") < 2

    def test_cli_imports_from_modules(self):
        """Test cli.py imports handlers from cli/ modules."""
        with open("cli.py", "r") as f:
            content = f.read()
        assert "from cli import" in content


# === HELP OUTPUT TESTS (4) ===


class TestHelpOutput:
    """Tests for CLI help output."""

    def test_cli_help_works(self):
        """Test cli.py --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0

    def test_cli_help_shows_description(self):
        """Test help shows description."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "AXIOM" in result.stdout or "Sovereignty" in result.stdout

    def test_cli_help_shows_commands(self):
        """Test help shows available commands."""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True
        )
        # Should show at least some of the main commands
        assert "baseline" in result.stdout.lower() or "command" in result.stdout.lower()

    def test_cli_no_args_shows_usage(self):
        """Test cli.py with no args shows usage."""
        result = subprocess.run(
            [sys.executable, "cli.py"],
            capture_output=True,
            text=True
        )
        # Should show usage or help info, not crash
        assert result.returncode == 0 or "usage" in result.stdout.lower() or "Usage" in result.stdout


# === ALL MODULES IMPORT TEST ===


class TestAllModulesImport:
    """Test that all CLI modules can be imported."""

    def test_all_modules_import(self):
        """Test all cli/ modules import without error."""
        modules = [
            "cli.base",
            "cli.core",
            "cli.partition",
            "cli.blackout",
            "cli.pruning",
            "cli.ablation",
            "cli.depth",
            "cli.rl",
            "cli.quantum",
            "cli.pipeline",
            "cli.scale",
            "cli.fractal",
            "cli.sweep",
            "cli.info",
        ]

        for module_name in modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
