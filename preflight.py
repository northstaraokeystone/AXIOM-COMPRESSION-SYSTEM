#!/usr/bin/env python3
"""
Preflight validation for SpaceProof.

Runs all checks locally before push to prevent CI failures.

Usage:
    python preflight.py          # Full validation
    python preflight.py --quick  # Syntax + imports only
"""

import subprocess
import sys
import py_compile
from pathlib import Path
from typing import Tuple, List


# AUTO-DETECTED from repo structure
SOURCE_DIRS = ["spaceproof", "benchmark", "data", "script"]
CLI_PATH = "cli.py"
TEST_DIR = "test"
LINTERS = {
    "flake8": ["flake8", "spaceproof/", "benchmark/", "--max-line-length=120",
               "--ignore=E501,W503,W504,E128,E226,F841,F811,F824,F401,E203,E741"],
    "black": ["black", "--check", "--line-length=120", "spaceproof/", "benchmark/"]
}

# CLAUDEME compliance
EMIT_RECEIPT = True


def print_header(step: str, desc: str):
    """Print check header."""
    print(f"\n[{step}] {desc}...")


def print_result(success: bool, message: str):
    """Print check result."""
    symbol = "✓" if success else "✗"
    print(f"{symbol} {message}")


def check_syntax() -> Tuple[bool, str]:
    """Check all .py files parse correctly."""
    print_header("SYNTAX", "Checking Python syntax")

    py_files: List[Path] = []
    for source_dir in SOURCE_DIRS + [TEST_DIR, "."]:
        if Path(source_dir).exists():
            py_files.extend(Path(source_dir).rglob("*.py"))

    # Deduplicate and filter out virtual environments
    py_files = [f for f in set(py_files) if "venv" not in str(f) and ".venv" not in str(f)]

    if not py_files:
        return False, "No Python files found"

    errors = []
    for py_file in py_files:
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"{py_file}: {e}")

    if errors:
        return False, "Syntax errors:\n  " + "\n  ".join(errors)

    return True, f"All {len(py_files)} Python files parse correctly"


def check_imports() -> Tuple[bool, str]:
    """Check main modules can be imported."""
    print_header("IMPORT", "Checking imports")

    # Core modules to test
    import_tests = [
        "from spaceproof import core, compress, witness, anchor, detect, ledger, loop",
        "from spaceproof.domain import galaxy, colony, telemetry",
        "from spaceproof.sovereignty import mars",
    ]

    for test_import in import_tests:
        result = subprocess.run(
            [sys.executable, "-c", test_import],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown import error"
            return False, f"Import failed: {error_msg}"

    return True, "All core modules import successfully"


def check_cli() -> Tuple[bool, str]:
    """Check CLI responds to --help."""
    print_header("CLI", "Checking CLI")

    if not Path(CLI_PATH).exists():
        return True, "No CLI found (skipped)"

    result = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )

    if result.returncode != 0:
        return False, f"CLI failed (exit code {result.returncode})"

    return True, "CLI responds to --help (exit 0)"


def check_tests() -> Tuple[bool, str]:
    """Run pytest suite."""
    print_header("TEST", "Running tests")

    if not Path(TEST_DIR).exists():
        return True, "No tests directory found (skipped)"

    # Run pytest with -x (stop on first failure) and -q (quiet)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", TEST_DIR, "-x", "-q"],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        # Extract failure info from pytest output
        output_lines = result.stdout.split('\n')
        failure_line = next((line for line in output_lines if "FAILED" in line), "Test failure")
        return False, f"Tests failed: {failure_line}"

    # Extract pass count from pytest output
    output = result.stdout.strip()
    if "passed" in output:
        pass_info = [line for line in output.split('\n') if "passed" in line]
        if pass_info:
            return True, pass_info[-1]

    return True, "Tests passed"


def check_lint() -> Tuple[bool, str]:
    """Run configured linters."""
    print_header("LINT", "Running linters")

    errors = []

    # Check flake8
    if "flake8" in LINTERS:
        result = subprocess.run(
            LINTERS["flake8"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            errors.append(f"flake8: {result.stdout.strip()}")

    # Check black (note: black exits 1 if formatting needed)
    if "black" in LINTERS:
        result = subprocess.run(
            LINTERS["black"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            output_lines = result.stderr.split('\n') if result.stderr else result.stdout.split('\n')
            reformatted = [line for line in output_lines if "would reformat" in line.lower()]
            if reformatted:
                errors.append(f"black: {', '.join(reformatted)}")

    if errors:
        return False, "\n  ".join(errors)

    return True, "No linting errors"


def emit_preflight_receipt(passed: bool, checks: dict):
    """Emit CLAUDEME-compliant receipt."""
    if not EMIT_RECEIPT:
        return

    import json
    import hashlib
    from datetime import datetime

    receipt = {
        "receipt_type": "preflight_validation",
        "ts": datetime.utcnow().isoformat() + "Z",
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "tenant_id": "spaceproof",
    }

    # Dual-hash (CLAUDEME standard)
    receipt_json = json.dumps(receipt, sort_keys=True)
    receipt["hash_sha256"] = hashlib.sha256(receipt_json.encode()).hexdigest()

    try:
        import blake3
        receipt["hash_blake3"] = blake3.blake3(receipt_json.encode()).hexdigest()
    except ImportError:
        receipt["hash_blake3"] = None

    # Write to receipt directory
    receipt_dir = Path("receipt")
    receipt_dir.mkdir(exist_ok=True)

    with open(receipt_dir / "preflight_receipts.jsonl", "a") as f:
        f.write(json.dumps(receipt) + "\n")


def main() -> int:
    """Run all checks in sequence."""
    quick_mode = "--quick" in sys.argv

    print("━" * 60)
    print("[PREFLIGHT] Starting validation...")
    print("━" * 60)

    checks = {}

    # Syntax check (always)
    success, message = check_syntax()
    checks["syntax"] = {"passed": success, "message": message}
    print_result(success, message)
    if not success:
        print("\n" + "━" * 60)
        print("✗ PREFLIGHT FAILED - DO NOT PUSH")
        print("━" * 60)
        print("\nFix the error above before pushing.")
        emit_preflight_receipt(False, checks)
        return 1

    # Import check (always)
    success, message = check_imports()
    checks["imports"] = {"passed": success, "message": message}
    print_result(success, message)
    if not success:
        print("\n" + "━" * 60)
        print("✗ PREFLIGHT FAILED - DO NOT PUSH")
        print("━" * 60)
        print("\nFix the error above before pushing.")
        emit_preflight_receipt(False, checks)
        return 1

    if quick_mode:
        print("\n" + "━" * 60)
        print("✓ QUICK CHECK PASSED")
        print("━" * 60)
        emit_preflight_receipt(True, checks)
        return 0

    # CLI check
    success, message = check_cli()
    checks["cli"] = {"passed": success, "message": message}
    print_result(success, message)
    if not success:
        print("\n" + "━" * 60)
        print("✗ PREFLIGHT FAILED - DO NOT PUSH")
        print("━" * 60)
        print("\nFix the error above before pushing.")
        emit_preflight_receipt(False, checks)
        return 1

    # Test check
    success, message = check_tests()
    checks["tests"] = {"passed": success, "message": message}
    print_result(success, message)
    if not success:
        print("\n" + "━" * 60)
        print("✗ PREFLIGHT FAILED - DO NOT PUSH")
        print("━" * 60)
        print("\nFix the error above before pushing.")
        emit_preflight_receipt(False, checks)
        return 1

    # Lint check
    success, message = check_lint()
    checks["lint"] = {"passed": success, "message": message}
    print_result(success, message)
    if not success:
        print("\n" + "━" * 60)
        print("✗ PREFLIGHT FAILED - DO NOT PUSH")
        print("━" * 60)
        print("\nFix the error above before pushing.")
        emit_preflight_receipt(False, checks)
        return 1

    # All checks passed
    print("\n" + "━" * 60)
    print("✓ ALL CHECKS PASSED - Safe to push")
    print("━" * 60)
    emit_preflight_receipt(True, checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
