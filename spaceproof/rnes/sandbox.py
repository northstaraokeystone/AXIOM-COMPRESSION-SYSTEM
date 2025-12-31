"""sandbox.py - Isolated execution environment.

Execute operations in sandboxed environment for safety.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from spaceproof.core import emit_receipt

# === CONSTANTS ===

RNES_TENANT = "spaceproof-rnes"


@dataclass
class SandboxConfig:
    """Sandbox configuration."""

    sandbox_id: str
    max_memory_mb: int
    max_cpu_seconds: float
    allowed_operations: List[str]
    network_access: bool
    filesystem_access: bool
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
            "allowed_operations": self.allowed_operations,
            "network_access": self.network_access,
            "filesystem_access": self.filesystem_access,
            "created_at": self.created_at,
        }


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""

    sandbox_id: str
    execution_id: str
    success: bool
    output: Any
    memory_used_mb: float
    cpu_seconds_used: float
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sandbox_id": self.sandbox_id,
            "execution_id": self.execution_id,
            "success": self.success,
            "memory_used_mb": self.memory_used_mb,
            "cpu_seconds_used": self.cpu_seconds_used,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# Active sandboxes
_sandboxes: Dict[str, SandboxConfig] = {}
_sandbox_state: Dict[str, Dict[str, Any]] = {}


def create_sandbox(
    max_memory_mb: int = 512,
    max_cpu_seconds: float = 60.0,
    allowed_operations: Optional[List[str]] = None,
    network_access: bool = False,
    filesystem_access: bool = False,
) -> SandboxConfig:
    """Create a new sandbox environment.

    Args:
        max_memory_mb: Maximum memory in MB
        max_cpu_seconds: Maximum CPU time
        allowed_operations: List of allowed operations
        network_access: Allow network access
        filesystem_access: Allow filesystem access

    Returns:
        SandboxConfig
    """
    sandbox_id = str(uuid.uuid4())

    config = SandboxConfig(
        sandbox_id=sandbox_id,
        max_memory_mb=max_memory_mb,
        max_cpu_seconds=max_cpu_seconds,
        allowed_operations=allowed_operations or ["read", "compute", "emit_receipt"],
        network_access=network_access,
        filesystem_access=filesystem_access,
    )

    _sandboxes[sandbox_id] = config
    _sandbox_state[sandbox_id] = {
        "memory_used": 0,
        "cpu_used": 0,
        "executions": 0,
    }

    return config


def _check_operation_allowed(sandbox_id: str, operation: str) -> bool:
    """Check if operation is allowed in sandbox.

    Args:
        sandbox_id: Sandbox identifier
        operation: Operation name

    Returns:
        True if allowed
    """
    config = _sandboxes.get(sandbox_id)
    if not config:
        return False
    return operation in config.allowed_operations


def _check_resource_limits(sandbox_id: str) -> tuple[bool, str]:
    """Check resource limits for sandbox.

    Args:
        sandbox_id: Sandbox identifier

    Returns:
        Tuple of (within_limits, reason)
    """
    config = _sandboxes.get(sandbox_id)
    state = _sandbox_state.get(sandbox_id)

    if not config or not state:
        return False, "Sandbox not found"

    if state["memory_used"] >= config.max_memory_mb:
        return False, "Memory limit exceeded"

    if state["cpu_used"] >= config.max_cpu_seconds:
        return False, "CPU time limit exceeded"

    return True, ""


def execute_in_sandbox(
    sandbox_id: str,
    operation: str,
    func: Callable,
    args: tuple = (),
    kwargs: Optional[Dict] = None,
) -> SandboxResult:
    """Execute function in sandbox.

    Args:
        sandbox_id: Sandbox identifier
        operation: Operation name
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        SandboxResult
    """
    import time

    execution_id = str(uuid.uuid4())
    kwargs = kwargs or {}

    # Check sandbox exists
    if sandbox_id not in _sandboxes:
        return SandboxResult(
            sandbox_id=sandbox_id,
            execution_id=execution_id,
            success=False,
            output=None,
            memory_used_mb=0,
            cpu_seconds_used=0,
            error="Sandbox not found",
            completed_at=datetime.utcnow().isoformat() + "Z",
        )

    # Check operation allowed
    if not _check_operation_allowed(sandbox_id, operation):
        return SandboxResult(
            sandbox_id=sandbox_id,
            execution_id=execution_id,
            success=False,
            output=None,
            memory_used_mb=0,
            cpu_seconds_used=0,
            error=f"Operation '{operation}' not allowed in sandbox",
            completed_at=datetime.utcnow().isoformat() + "Z",
        )

    # Check resource limits
    within_limits, reason = _check_resource_limits(sandbox_id)
    if not within_limits:
        return SandboxResult(
            sandbox_id=sandbox_id,
            execution_id=execution_id,
            success=False,
            output=None,
            memory_used_mb=0,
            cpu_seconds_used=0,
            error=reason,
            completed_at=datetime.utcnow().isoformat() + "Z",
        )

    # Execute with resource tracking
    time.time()
    start_cpu = time.process_time()

    try:
        output = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        output = None
        success = False
        error = str(e)

    time.time()
    end_cpu = time.process_time()

    cpu_used = end_cpu - start_cpu
    # Estimate memory (simplified - in production would use resource module)
    memory_used = 0.1  # Placeholder

    # Update sandbox state
    state = _sandbox_state[sandbox_id]
    state["cpu_used"] += cpu_used
    state["memory_used"] += memory_used
    state["executions"] += 1

    return SandboxResult(
        sandbox_id=sandbox_id,
        execution_id=execution_id,
        success=success,
        output=output,
        memory_used_mb=memory_used,
        cpu_seconds_used=cpu_used,
        error=error,
        completed_at=datetime.utcnow().isoformat() + "Z",
    )


def cleanup_sandbox(sandbox_id: str) -> bool:
    """Clean up sandbox resources.

    Args:
        sandbox_id: Sandbox identifier

    Returns:
        True if cleaned up
    """
    if sandbox_id not in _sandboxes:
        return False

    del _sandboxes[sandbox_id]
    if sandbox_id in _sandbox_state:
        del _sandbox_state[sandbox_id]

    return True


def emit_sandbox_receipt(result: SandboxResult) -> Dict[str, Any]:
    """Emit sandbox execution receipt.

    Args:
        result: SandboxResult to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "sandbox_execution",
        {
            "tenant_id": RNES_TENANT,
            **result.to_dict(),
        },
    )


def get_sandbox_state(sandbox_id: str) -> Optional[Dict[str, Any]]:
    """Get current sandbox state.

    Args:
        sandbox_id: Sandbox identifier

    Returns:
        State dict or None
    """
    config = _sandboxes.get(sandbox_id)
    state = _sandbox_state.get(sandbox_id)

    if not config or not state:
        return None

    return {
        "config": config.to_dict(),
        **state,
    }


def clear_sandboxes() -> None:
    """Clear all sandboxes (for testing)."""
    global _sandboxes, _sandbox_state
    _sandboxes = {}
    _sandbox_state = {}
