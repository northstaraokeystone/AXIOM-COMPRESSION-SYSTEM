"""audit_trail.py - Generate compliance audit trails.

Complete audit trail generation for compliance requirements.
SLO: < 5 seconds for trail generation.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from spaceproof.core import emit_receipt, merkle

# === CONSTANTS ===

COMPLIANCE_TENANT = "spaceproof-compliance"
AUDIT_TRAIL_GENERATION_TIMEOUT = 5.0  # 5 seconds max per GOVERNANCE scenario


@dataclass
class AuditTrailEntry:
    """Single entry in audit trail."""

    entry_id: str
    timestamp: str
    event_type: str
    actor_id: str
    actor_role: str
    action: str
    target_id: Optional[str]
    details: Dict[str, Any]
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "actor_role": self.actor_role,
            "action": self.action,
            "target_id": self.target_id,
            "details": self.details,
            "receipt_hash": self.receipt_hash,
        }


@dataclass
class AuditTrailReport:
    """Complete audit trail report."""

    report_id: str
    generated_at: str
    time_range_start: str
    time_range_end: str
    entries: List[AuditTrailEntry]
    entry_count: int
    merkle_root: str
    generation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "entry_count": self.entry_count,
            "merkle_root": self.merkle_root,
            "generation_time_ms": self.generation_time_ms,
            "entries": [e.to_dict() for e in self.entries],
        }


# In-memory audit trail storage
_audit_entries: List[AuditTrailEntry] = []


def add_audit_entry(
    event_type: str,
    actor_id: str,
    actor_role: str,
    action: str,
    target_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    receipt_hash: str = "",
) -> AuditTrailEntry:
    """Add entry to audit trail.

    Args:
        event_type: Type of event
        actor_id: Actor identifier
        actor_role: Actor's role
        action: Action performed
        target_id: Target of action (optional)
        details: Additional details (optional)
        receipt_hash: Associated receipt hash

    Returns:
        Created AuditTrailEntry
    """
    entry = AuditTrailEntry(
        entry_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat() + "Z",
        event_type=event_type,
        actor_id=actor_id,
        actor_role=actor_role,
        action=action,
        target_id=target_id,
        details=details or {},
        receipt_hash=receipt_hash,
    )

    _audit_entries.append(entry)
    return entry


def query_audit_trail(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    actor_id: Optional[str] = None,
    limit: int = 1000,
) -> List[AuditTrailEntry]:
    """Query audit trail with filters.

    Args:
        start_time: Filter entries after this time
        end_time: Filter entries before this time
        event_types: Filter by event types
        actor_id: Filter by actor
        limit: Maximum entries to return

    Returns:
        List of matching AuditTrailEntry objects
    """
    results = []

    for entry in _audit_entries:
        # Time filters
        if start_time and entry.timestamp < start_time:
            continue
        if end_time and entry.timestamp > end_time:
            continue

        # Event type filter
        if event_types and entry.event_type not in event_types:
            continue

        # Actor filter
        if actor_id and entry.actor_id != actor_id:
            continue

        results.append(entry)

        if len(results) >= limit:
            break

    return results


def generate_audit_trail(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_types: Optional[List[str]] = None,
) -> AuditTrailReport:
    """Generate audit trail report.

    Must complete within 5 seconds per GOVERNANCE scenario.

    Args:
        start_time: Filter entries after this time
        end_time: Filter entries before this time
        event_types: Filter by event types

    Returns:
        AuditTrailReport
    """
    start = time.time()

    # Default time range
    if not end_time:
        end_time = datetime.utcnow().isoformat() + "Z"
    if not start_time:
        start_time = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"

    # Query entries
    entries = query_audit_trail(
        start_time=start_time,
        end_time=end_time,
        event_types=event_types,
    )

    # Compute Merkle root
    merkle_root = merkle([e.to_dict() for e in entries]) if entries else ""

    generation_time = (time.time() - start) * 1000

    report = AuditTrailReport(
        report_id=str(uuid.uuid4()),
        generated_at=datetime.utcnow().isoformat() + "Z",
        time_range_start=start_time,
        time_range_end=end_time,
        entries=entries,
        entry_count=len(entries),
        merkle_root=merkle_root,
        generation_time_ms=generation_time,
    )

    # Emit receipt
    emit_receipt(
        "audit_trail",
        {
            "tenant_id": COMPLIANCE_TENANT,
            "report_id": report.report_id,
            "entry_count": report.entry_count,
            "merkle_root": report.merkle_root,
            "generation_time_ms": report.generation_time_ms,
            "meets_slo": generation_time < AUDIT_TRAIL_GENERATION_TIMEOUT * 1000,
        },
    )

    return report


def export_audit_trail(
    report: AuditTrailReport,
    output_path: Optional[Path] = None,
    format: str = "json",
) -> str:
    """Export audit trail to file.

    Args:
        report: AuditTrailReport to export
        output_path: Output file path
        format: Export format (json, csv)

    Returns:
        Path to exported file
    """
    if output_path is None:
        output_path = Path(f"audit_trail_{report.report_id[:8]}.{format}")

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
    elif format == "csv":
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["entry_id", "timestamp", "event_type", "actor_id", "action", "target_id"])
            for entry in report.entries:
                writer.writerow([
                    entry.entry_id,
                    entry.timestamp,
                    entry.event_type,
                    entry.actor_id,
                    entry.action,
                    entry.target_id,
                ])

    return str(output_path)


def clear_audit_trail() -> None:
    """Clear audit trail (for testing)."""
    global _audit_entries
    _audit_entries = []


def get_audit_entry_count() -> int:
    """Get total audit entry count."""
    return len(_audit_entries)


# Aliases for API compatibility
def get_audit_trail(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    actor_id: Optional[str] = None,
    limit: int = 1000,
) -> List[AuditTrailEntry]:
    """Alias for query_audit_trail for API compatibility."""
    return query_audit_trail(start_time, end_time, event_types, actor_id, limit)


def emit_audit_receipt(
    event_type: str,
    actor_id: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit an audit receipt for an action.

    Args:
        event_type: Type of audit event
        actor_id: Actor performing the action
        action: Action being audited
        details: Additional audit details

    Returns:
        Receipt dictionary
    """
    receipt = emit_receipt(
        "audit",
        {
            "tenant_id": COMPLIANCE_TENANT,
            "event_type": event_type,
            "actor_id": actor_id,
            "action": action,
            "details": details or {},
        },
    )
    return receipt
