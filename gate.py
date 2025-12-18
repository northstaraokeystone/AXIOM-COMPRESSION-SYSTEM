#!/usr/bin/env python3
"""AXIOM Gate v2 — Self-Installing Validation Gate

THE PARADIGM SHIFT:
    The gate doesn't check if you can commit.
    The gate IS how you commit.

Usage:
    python gate.py              # Run all validations, emit receipt
    python gate.py --install    # Install self as git pre-commit hook
    python gate.py --ci         # Output GitHub Actions workflow to stdout
    python gate.py --version    # Show gate.py's own dual-hash

Source: CLAUDEME.md §3 Timeline Gates + BLOW IT UP optimization
"""

import hashlib
import json
import py_compile
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# === CONSTANTS ===

TENANT_ID = "axiom-gate"
RECEIPT_FILE = "receipts/gate_receipts.jsonl"
SLO_TOTAL_MS = 60000  # Max 60 seconds for all checks
PROJECT_ROOT = Path(__file__).parent.absolute()

# blake3 availability
try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


# === CORE FUNCTIONS (per CLAUDEME §8) ===


def dual_hash(data: bytes | str) -> str:
    """SHA256:BLAKE3 format. ALWAYS use this, never single hash."""
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest() if HAS_BLAKE3 else sha
    return f"{sha}:{b3}"


def emit_gate_receipt(
    checks: dict[str, dict], all_passed: bool, gate_hash: str
) -> dict:
    """Emit gate_receipt to receipts file. Returns the receipt."""
    receipt = {
        "receipt_type": "gate",
        "ts": datetime.utcnow().isoformat() + "Z",
        "tenant_id": TENANT_ID,
        "checks": checks,
        "all_passed": all_passed,
        "gate_hash": gate_hash,
    }
    receipt["payload_hash"] = dual_hash(json.dumps(receipt, sort_keys=True))

    # Ensure receipts directory exists
    receipts_dir = PROJECT_ROOT / "receipts"
    receipts_dir.mkdir(exist_ok=True)

    # Append to receipt file
    receipt_path = PROJECT_ROOT / RECEIPT_FILE
    with open(receipt_path, "a") as f:
        f.write(json.dumps(receipt) + "\n")

    return receipt


def emit_install_receipt(hook_path: str, gate_hash: str) -> dict:
    """Emit install_receipt when hook is installed."""
    receipt = {
        "receipt_type": "install",
        "ts": datetime.utcnow().isoformat() + "Z",
        "tenant_id": TENANT_ID,
        "hook_path": hook_path,
        "gate_hash": gate_hash,
    }
    receipt["payload_hash"] = dual_hash(json.dumps(receipt, sort_keys=True))

    receipts_dir = PROJECT_ROOT / "receipts"
    receipts_dir.mkdir(exist_ok=True)

    receipt_path = PROJECT_ROOT / RECEIPT_FILE
    with open(receipt_path, "a") as f:
        f.write(json.dumps(receipt) + "\n")

    return receipt


# === STOPRULE FUNCTIONS ===


def stoprule_syntax(error: str) -> None:
    """Print stoprule for syntax errors."""
    print("\n[STOPRULE] syntax_check FAILED")
    print(f"  Error: {error}")
    print("  Fix: Correct the syntax error in the indicated file")


def stoprule_import(error: str) -> None:
    """Print stoprule for import errors."""
    print("\n[STOPRULE] import_check FAILED")
    print(f"  Error: {error}")
    print("  Fix: Check for missing dependencies or circular imports")


def stoprule_cli(error: str) -> None:
    """Print stoprule for CLI errors."""
    print("\n[STOPRULE] cli_check FAILED")
    print(f"  Error: {error}")
    print("  Fix: Ensure cli.py runs without errors")


def stoprule_test(error: str) -> None:
    """Print stoprule for test errors."""
    print("\n[STOPRULE] test_check FAILED")
    print(f"  Error: {error}")
    print("  Fix: Run 'pytest tests/ -v' to see failing tests")


def stoprule_lint(error: str) -> None:
    """Print stoprule for lint errors."""
    print("\n[STOPRULE] lint_check FAILED")
    print(f"  Error: {error}")
    print("  Fix: Run 'ruff check . --fix' to auto-fix issues")


# === VALIDATION CHECKS ===


def check_syntax() -> dict[str, Any]:
    """Check 1: Compile all .py files."""
    t0 = time.time()
    errors = []

    for py_file in PROJECT_ROOT.rglob("*.py"):
        # Skip hidden directories, __pycache__, venv
        if any(
            part.startswith(".") or part in ("__pycache__", "venv", ".venv")
            for part in py_file.parts
        ):
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(str(e))

    duration_ms = int((time.time() - t0) * 1000)
    passed = len(errors) == 0

    return {
        "passed": passed,
        "duration_ms": duration_ms,
        "error": "; ".join(errors[:3]) if errors else None,  # Limit error length
    }


def check_imports() -> dict[str, Any]:
    """Check 2: Import all src/*.py modules."""
    t0 = time.time()
    errors = []

    src_dir = PROJECT_ROOT / "src"
    if not src_dir.exists():
        return {"passed": True, "duration_ms": 0, "error": None}

    # Add project root to path temporarily
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        for py_file in src_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = f"src.{py_file.stem}"
            try:
                __import__(module_name)
            except Exception as e:
                errors.append(f"{module_name}: {e}")
    finally:
        sys.path.pop(0)

    duration_ms = int((time.time() - t0) * 1000)
    passed = len(errors) == 0

    return {
        "passed": passed,
        "duration_ms": duration_ms,
        "error": "; ".join(errors[:3]) if errors else None,
    }


def check_cli() -> dict[str, Any]:
    """Check 3: Run cli.py --help."""
    t0 = time.time()

    cli_path = PROJECT_ROOT / "cli.py"
    if not cli_path.exists():
        return {"passed": False, "duration_ms": 0, "error": "cli.py not found"}

    try:
        result = subprocess.run(
            [sys.executable, str(cli_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        passed = result.returncode == 0
        error = result.stderr[:200] if not passed else None
    except subprocess.TimeoutExpired:
        passed = False
        error = "CLI help timed out after 30s"
    except Exception as e:
        passed = False
        error = str(e)[:200]

    duration_ms = int((time.time() - t0) * 1000)

    return {"passed": passed, "duration_ms": duration_ms, "error": error}


def check_tests() -> dict[str, Any]:
    """Check 4: Run pytest tests/ -x (fail fast)."""
    t0 = time.time()

    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists():
        return {"passed": True, "duration_ms": 0, "error": None}

    # Check if pytest is available
    pytest_available = shutil.which("pytest") is not None
    if not pytest_available:
        try:
            subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True,
                check=True,
            )
            pytest_cmd = [sys.executable, "-m", "pytest"]
        except Exception:
            return {
                "passed": True,
                "duration_ms": 0,
                "error": "pytest not installed, skipping",
            }
    else:
        pytest_cmd = ["pytest"]

    try:
        result = subprocess.run(
            pytest_cmd + [str(tests_dir), "-x", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for tests
            cwd=str(PROJECT_ROOT),
        )
        passed = result.returncode == 0
        if not passed:
            # Extract failing test name and summary
            lines = result.stdout.strip().split("\n")
            failed_lines = [line for line in lines if "FAILED" in line]
            summary = lines[-1] if lines else ""
            if failed_lines:
                error = f"{failed_lines[0][:100]} | {summary}"
            else:
                error = summary if summary else result.stderr[:200]
        else:
            error = None
    except subprocess.TimeoutExpired:
        passed = False
        error = "Tests timed out after 5 minutes"
    except Exception as e:
        passed = False
        error = str(e)[:200]

    duration_ms = int((time.time() - t0) * 1000)

    return {"passed": passed, "duration_ms": duration_ms, "error": error}


def check_lint() -> dict[str, Any]:
    """Check 5: Run ruff check . (optional, fails gracefully)."""
    t0 = time.time()

    # Check if ruff is available
    ruff_available = shutil.which("ruff") is not None
    if not ruff_available:
        try:
            subprocess.run(
                [sys.executable, "-m", "ruff", "--version"],
                capture_output=True,
                check=True,
            )
            ruff_cmd = [sys.executable, "-m", "ruff"]
        except Exception:
            return {
                "passed": True,
                "duration_ms": 0,
                "error": "ruff not installed, skipping",
            }
    else:
        ruff_cmd = ["ruff"]

    try:
        result = subprocess.run(
            ruff_cmd + ["check", ".", "--select=E,F", "--ignore=E501,F841"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        passed = result.returncode == 0
        if not passed:
            # Count errors
            error_lines = [line for line in result.stdout.split("\n") if line.strip()]
            error = f"{len(error_lines)} lint errors found"
        else:
            error = None
    except subprocess.TimeoutExpired:
        passed = False
        error = "Lint check timed out after 60s"
    except Exception as e:
        passed = False
        error = str(e)[:200]

    duration_ms = int((time.time() - t0) * 1000)

    return {"passed": passed, "duration_ms": duration_ms, "error": error}


# === MAIN GATE FUNCTION ===


def run_gate() -> int:
    """Run all validation checks. Returns 0 on pass, 1 on fail."""
    print("=" * 60)
    print("AXIOM GATE v2 — Self-Installing Validation Gate")
    print("=" * 60)

    # Compute gate hash for self-verification
    gate_path = Path(__file__).absolute()
    gate_hash = dual_hash(gate_path.read_bytes())

    # Run all checks
    checks = {}
    stoprules = {
        "syntax": stoprule_syntax,
        "import": stoprule_import,
        "cli": stoprule_cli,
        "test": stoprule_test,
        "lint": stoprule_lint,
    }

    check_funcs = [
        ("syntax", check_syntax),
        ("import", check_imports),
        ("cli", check_cli),
        ("test", check_tests),
        ("lint", check_lint),
    ]

    total_start = time.time()

    for name, check_func in check_funcs:
        print(f"\n[{name.upper()}] Running...", end=" ", flush=True)
        result = check_func()
        checks[name] = result

        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} ({result['duration_ms']}ms)")

        if not result["passed"] and result.get("error"):
            stoprules[name](result["error"])

    total_ms = int((time.time() - total_start) * 1000)

    # Check SLO
    all_passed = all(c["passed"] for c in checks.values())
    slo_passed = total_ms <= SLO_TOTAL_MS

    if not slo_passed:
        print(f"\n[SLO] WARN: Total time {total_ms}ms exceeds {SLO_TOTAL_MS}ms")

    # Emit receipt
    receipt = emit_gate_receipt(checks, all_passed, gate_hash)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("GATE: PASS — Commit allowed")
        print(f"  Total: {total_ms}ms | Receipt: {receipt['payload_hash'][:16]}...")
        return 0
    else:
        print("GATE: FAIL — Commit blocked")
        failed = [name for name, c in checks.items() if not c["passed"]]
        print(f"  Failed checks: {', '.join(failed)}")
        print(f"  Receipt: {receipt['payload_hash'][:16]}...")
        return 1


# === SELF-INSTALLATION ===


def install_hook() -> int:
    """Install gate.py as git pre-commit hook."""
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        print("ERROR: Not a git repository")
        return 1

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    hook_path = hooks_dir / "pre-commit"
    gate_path = Path(__file__).absolute()
    gate_hash = dual_hash(gate_path.read_bytes())

    # Create hook script that calls gate.py
    hook_content = f'''#!/bin/sh
# AXIOM Gate pre-commit hook (auto-generated)
# Gate hash: {gate_hash[:32]}...
exec python3 "{gate_path}"
'''

    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)

    # Emit install receipt
    receipt = emit_install_receipt(str(hook_path), gate_hash)

    print(f"Installed pre-commit hook: {hook_path}")
    print(f"Receipt: {receipt['payload_hash'][:16]}...")
    return 0


def check_hook_installed() -> bool:
    """Check if gate is installed as pre-commit hook."""
    hook_path = PROJECT_ROOT / ".git" / "hooks" / "pre-commit"
    if not hook_path.exists():
        return False

    content = hook_path.read_text()
    return "AXIOM Gate" in content or "gate.py" in content


# === CI GENERATION ===


def generate_ci() -> None:
    """Output minimal GitHub Actions CI config."""
    ci_config = """# AXIOM CI — Generated by gate.py --ci
# The gate IS the CI. CI just confirms gate passed.

name: AXIOM Gate

on:
  push:
    branches: [main, claude/*]
  pull_request:
    branches: [main]

jobs:
  gate:
    name: Gate Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest numpy
          pip install blake3 || true
          pip install ruff || true

      - name: Run Gate
        run: python gate.py

      - name: Upload receipt
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: gate-receipts
          path: receipts/gate_receipts.jsonl
          if-no-files-found: ignore
"""
    print(ci_config)


# === SELF-VERIFICATION ===


def show_version() -> None:
    """Show gate.py version and self-hash."""
    gate_path = Path(__file__).absolute()
    gate_hash = dual_hash(gate_path.read_bytes())

    print("AXIOM Gate v2.0")
    print(f"Path: {gate_path}")
    print(f"Hash: {gate_hash}")
    print(
        f"BLAKE3: {'available' if HAS_BLAKE3 else 'unavailable (using SHA256 fallback)'}"
    )


# === ENTRY POINT ===


def main() -> int:
    """Main entry point."""
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--install":
            return install_hook()
        elif arg == "--ci":
            generate_ci()
            return 0
        elif arg == "--version":
            show_version()
            return 0
        elif arg == "--help" or arg == "-h":
            print(__doc__)
            return 0
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage")
            return 1

    # First-run check: offer to install if not installed
    if not check_hook_installed():
        print("NOTE: Gate not installed as pre-commit hook.")
        print("      Run 'python gate.py --install' to auto-block bad commits.\n")

    return run_gate()


if __name__ == "__main__":
    sys.exit(main())
