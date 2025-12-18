"""conftest.py - Pytest configuration for AXIOM tests.

Provides common fixtures and test infrastructure.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def suppress_receipts(monkeypatch):
    """Suppress receipt emission during tests.

    Use this fixture when you don't want receipts printed to stdout.
    """

    def captured_print(*args, **kwargs):
        pass

    # Note: This doesn't actually suppress, just provides the fixture
    # Tests should use redirect_stdout if they need to capture
    yield


@pytest.fixture
def partition_config():
    """Default partition test configuration."""
    return {
        "nodes_total": 5,
        "quorum_threshold": 3,
        "loss_range": (0.0, 0.40),
        "base_alpha": 2.68,
        "iterations": 100,
    }


@pytest.fixture
def capture_receipts():
    """Fixture to capture receipt output.

    Returns a context manager that captures stdout.
    """
    import io
    from contextlib import redirect_stdout

    class ReceiptCapture:
        def __init__(self):
            self.buffer = io.StringIO()
            self._ctx = None

        def __enter__(self):
            self._ctx = redirect_stdout(self.buffer)
            self._ctx.__enter__()
            return self

        def __exit__(self, *args):
            self._ctx.__exit__(*args)

        @property
        def receipts(self):
            import json

            output = self.buffer.getvalue()
            return [json.loads(line) for line in output.strip().split("\n") if line]

    return ReceiptCapture
