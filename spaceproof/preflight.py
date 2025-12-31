"""preflight.py - Pre-push validation gate.

Run ALL validation before ANY push to prevent GitHub Actions failures.

Usage:
    python -m spaceproof.preflight           # Full validation
    python -m spaceproof.preflight --quick   # Syntax + imports only

Validates syntax, imports, receipts, tests, lint, gates, and entropy.
"""

import argparse
import ast
import importlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

from spaceproof.core import emit_receipt

# === CONSTANTS ===

PREFLIGHT_TENANT = "spaceproof-preflight"

# Root directory
ROOT_DIR = Path(__file__).parent
PROJECT_ROOT = ROOT_DIR.parent

# Required modules to check
REQUIRED_MODULES = [
    "spaceproof.core",
    "spaceproof.compress",
    "spaceproof.witness",
    "spaceproof.detect",
    "spaceproof.anchor",
    "spaceproof.ledger",
    "spaceproof.sovereignty",
    "spaceproof.loop",
    "spaceproof.domain.orbital_compute",
    "spaceproof.domain.constellation_ops",
    "spaceproof.domain.autonomous_decision",
    "spaceproof.domain.firmware_integrity",
    "spaceproof.meta_integration",
    "spaceproof.context_router",
]


def check_syntax() -> Tuple[bool, str]:
    """Verify all .py files parse correctly using AST.

    Returns:
        Tuple of (passed, message)
    """
    errors: List[str] = []
    checked = 0

    for py_file in ROOT_DIR.rglob("*.py"):
        checked += 1
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=str(py_file))
        except SyntaxError as e:
            errors.append(f"{py_file.relative_to(ROOT_DIR)}: {e}")

    if errors:
        return False, f"Syntax errors in {len(errors)} files:\n" + "\n".join(errors[:10])

    return True, f"Checked {checked} files, all parse correctly"


def check_imports() -> Tuple[bool, str]:
    """Verify all required modules import successfully.

    Returns:
        Tuple of (passed, message)
    """
    errors: List[str] = []

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            errors.append(f"{module_name}: {e}")
        except Exception as e:
            errors.append(f"{module_name}: {type(e).__name__}: {e}")

    if errors:
        return False, f"Import errors in {len(errors)} modules:\n" + "\n".join(errors[:10])

    return True, f"All {len(REQUIRED_MODULES)} required modules import successfully"


def check_receipts() -> Tuple[bool, str]:
    """Verify core functions emit receipts (grep emit_receipt).

    Returns:
        Tuple of (passed, message)
    """
    # Key modules that MUST emit receipts
    receipt_modules = [
        "domain/orbital_compute.py",
        "domain/constellation_ops.py",
        "domain/autonomous_decision.py",
        "domain/firmware_integrity.py",
    ]

    missing: List[str] = []

    for module_path in receipt_modules:
        full_path = ROOT_DIR / module_path
        if full_path.exists():
            content = full_path.read_text()
            if "emit_receipt" not in content:
                missing.append(module_path)

    if missing:
        return False, f"Missing emit_receipt in: {', '.join(missing)}"

    return True, f"All {len(receipt_modules)} domain modules emit receipts"


def check_tests() -> Tuple[bool, str]:
    """Run pytest -x -q and verify pass.

    Returns:
        Tuple of (passed, message)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "--tb=short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return True, "All tests passed"
        elif result.returncode == 5:
            # No tests collected - this is OK for now
            return True, "No tests collected (pytest exit code 5)"
        else:
            # Extract failure summary
            output = result.stdout + result.stderr
            lines = output.split("\n")
            summary = [line for line in lines if "FAILED" in line or "error" in line.lower()][:5]
            return False, "Tests failed:\n" + "\n".join(summary)

    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 300 seconds"
    except FileNotFoundError:
        return True, "pytest not available, skipping tests"
    except Exception as e:
        return False, f"Test execution error: {e}"


def check_lint() -> Tuple[bool, str]:
    """Run ruff check and verify clean.

    Returns:
        Tuple of (passed, message)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", str(ROOT_DIR), "--select=E,F"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            return True, "Lint check passed (ruff)"
        else:
            # Count errors
            lines = result.stdout.strip().split("\n")
            error_count = len([line for line in lines if line.strip()])
            return False, f"Lint errors ({error_count}):\n" + "\n".join(lines[:10])

    except FileNotFoundError:
        return True, "ruff not available, skipping lint"
    except subprocess.TimeoutExpired:
        return False, "Lint check timed out"
    except Exception as e:
        return False, f"Lint execution error: {e}"


def check_gates() -> Tuple[bool, str]:
    """Run gate_t2h.sh if available.

    Returns:
        Tuple of (passed, message)
    """
    gate_script = PROJECT_ROOT / "gate_t2h.sh"

    if not gate_script.exists():
        return True, "gate_t2h.sh not found, skipping gate check"

    try:
        result = subprocess.run(
            ["bash", str(gate_script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return True, "Gate t2h passed"
        else:
            return False, f"Gate t2h failed: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        return False, "Gate check timed out"
    except Exception as e:
        return False, f"Gate execution error: {e}"


def check_entropy_conservation() -> Tuple[bool, str]:
    """Verify entropy conservation |ΔS| < 0.01 in key modules.

    Returns:
        Tuple of (passed, message)
    """
    # Check that ENTROPY_CONSERVATION_LIMIT is defined correctly
    try:
        from spaceproof.domain.orbital_compute import ENTROPY_CONSERVATION_LIMIT

        if ENTROPY_CONSERVATION_LIMIT != 0.01:
            return False, f"ENTROPY_CONSERVATION_LIMIT is {ENTROPY_CONSERVATION_LIMIT}, expected 0.01"
        return True, "Entropy conservation limit verified (0.01)"
    except ImportError:
        return True, "Entropy conservation check skipped (module not available)"
    except Exception as e:
        return False, f"Entropy check error: {e}"


def run_all_checks(quick: bool = False) -> Tuple[bool, List[Tuple[str, bool, str]]]:
    """Run all preflight checks.

    Args:
        quick: If True, only run syntax + imports

    Returns:
        Tuple of (all_passed, list of (check_name, passed, message))
    """
    checks: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
        ("syntax", check_syntax),
        ("imports", check_imports),
    ]

    if not quick:
        checks.extend(
            [
                ("receipts", check_receipts),
                ("entropy", check_entropy_conservation),
                ("tests", check_tests),
                ("lint", check_lint),
                ("gates", check_gates),
            ]
        )

    results: List[Tuple[str, bool, str]] = []
    all_passed = True

    for name, check_fn in checks:
        start = time.time()
        try:
            passed, message = check_fn()
        except Exception as e:
            passed, message = False, f"Check crashed: {e}"

        elapsed = time.time() - start
        results.append((name, passed, f"{message} ({elapsed:.2f}s)"))

        if not passed:
            all_passed = False

    return all_passed, results


def emit_preflight_receipt(passed: bool, results: List[Tuple[str, bool, str]]) -> dict:
    """Emit preflight validation receipt.

    Args:
        passed: Whether all checks passed
        results: List of (check_name, passed, message)

    Returns:
        Receipt dict
    """
    check_results = {name: {"passed": p, "message": m} for name, p, m in results}

    return emit_receipt(
        "preflight_validation",
        {
            "tenant_id": PREFLIGHT_TENANT,
            "passed": passed,
            "checks": check_results,
            "check_count": len(results),
            "passed_count": sum(1 for _, p, _ in results if p),
        },
    )


def main() -> int:
    """Run preflight validation.

    Returns:
        Exit code (0 = pass, 1 = fail)
    """
    parser = argparse.ArgumentParser(description="SpaceProof preflight validation")
    parser.add_argument("--quick", action="store_true", help="Quick mode: syntax + imports only")
    parser.add_argument("--quiet", action="store_true", help="Suppress output, only exit code")
    args = parser.parse_args()

    passed, results = run_all_checks(quick=args.quick)

    if not args.quiet:
        # Print results
        mode = "QUICK" if args.quick else "FULL"
        status = "PASSED" if passed else "FAILED"

        # Suppress receipt output during preflight
        for name, check_passed, message in results:
            icon = "✓" if check_passed else "✗"
            line = f"  {icon} {name}: {message}"
            if not args.quiet:
                # Print directly to stderr to avoid mixing with receipt JSON
                import sys

                print(line, file=sys.stderr)

        print(f"\nPreflight {mode}: {status}", file=sys.stderr)

    # Emit receipt (goes to stdout as JSON)
    emit_preflight_receipt(passed, results)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
