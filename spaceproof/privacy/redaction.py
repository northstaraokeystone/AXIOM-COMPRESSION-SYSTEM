"""redaction.py - PII redaction with cryptographic proof.

Redact personally identifiable information while maintaining audit trail.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from spaceproof.core import dual_hash, emit_receipt

# === CONSTANTS ===

PRIVACY_TENANT = "spaceproof-privacy"
CONFIG_DIR = Path(__file__).parent.parent / "config"
PRIVACY_CONFIG_FILE = CONFIG_DIR / "privacy_policies.json"


@dataclass
class PIIMatch:
    """A detected PII match."""

    pattern_name: str
    start: int
    end: int
    original: str
    replacement: str
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "start": self.start,
            "end": self.end,
            "severity": self.severity,
            # Note: original is NOT included to prevent leakage
            "replacement": self.replacement,
        }


@dataclass
class RedactionResult:
    """Result of redaction operation."""

    redaction_id: str
    original_hash: str
    redacted_text: str
    redacted_hash: str
    matches: List[PIIMatch]
    pii_count: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "redaction_id": self.redaction_id,
            "original_hash": self.original_hash,
            "redacted_hash": self.redacted_hash,
            "pii_count": self.pii_count,
            "matches": [m.to_dict() for m in self.matches],
            "timestamp": self.timestamp,
        }


# Cache for privacy config
_privacy_config: Optional[Dict[str, Any]] = None

# Statistics
_redaction_stats = {
    "total_redactions": 0,
    "pii_by_type": {},
    "total_pii_found": 0,
}


def load_privacy_config() -> Dict[str, Any]:
    """Load privacy configuration.

    Returns:
        Privacy config dict
    """
    global _privacy_config

    if _privacy_config is not None:
        return _privacy_config

    if not PRIVACY_CONFIG_FILE.exists():
        # Default patterns
        _privacy_config = {
            "pii_patterns": {
                "email": {
                    "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                    "replacement": "[REDACTED_EMAIL]",
                    "severity": "HIGH",
                },
                "ssn": {
                    "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
                    "replacement": "[REDACTED_SSN]",
                    "severity": "CRITICAL",
                },
                "phone": {
                    "pattern": r"\b(?:\+1)?[-.]?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
                    "replacement": "[REDACTED_PHONE]",
                    "severity": "MEDIUM",
                },
            }
        }
        return _privacy_config

    with open(PRIVACY_CONFIG_FILE, "r") as f:
        _privacy_config = json.load(f)

    return _privacy_config


def detect_pii(text: str) -> List[PIIMatch]:
    """Detect PII in text using configured patterns.

    Args:
        text: Text to scan

    Returns:
        List of PIIMatch objects
    """
    config = load_privacy_config()
    patterns = config.get("pii_patterns", {})
    matches = []

    for name, pattern_config in patterns.items():
        pattern = pattern_config.get("pattern")
        if not pattern:
            continue

        replacement = pattern_config.get("replacement", f"[REDACTED_{name.upper()}]")
        severity = pattern_config.get("severity", "MEDIUM")

        try:
            for match in re.finditer(pattern, text):
                matches.append(
                    PIIMatch(
                        pattern_name=name,
                        start=match.start(),
                        end=match.end(),
                        original=match.group(),
                        replacement=replacement,
                        severity=severity,
                    )
                )
        except re.error:
            continue

    # Sort by position (descending for safe replacement)
    matches.sort(key=lambda m: m.start, reverse=True)

    return matches


def redact_pii(
    text: str,
    additional_patterns: Optional[Dict[str, str]] = None,
) -> RedactionResult:
    """Redact PII from text.

    Args:
        text: Text to redact
        additional_patterns: Optional additional patterns to apply

    Returns:
        RedactionResult with redacted text and audit info
    """
    global _redaction_stats

    redaction_id = str(uuid.uuid4())
    original_hash = dual_hash(text)

    # Detect PII
    matches = detect_pii(text)

    # Apply additional patterns if provided
    if additional_patterns:
        for name, pattern in additional_patterns.items():
            try:
                for match in re.finditer(pattern, text):
                    matches.append(
                        PIIMatch(
                            pattern_name=name,
                            start=match.start(),
                            end=match.end(),
                            original=match.group(),
                            replacement=f"[REDACTED_{name.upper()}]",
                            severity="MEDIUM",
                        )
                    )
            except re.error:
                continue

    # Sort again after adding custom patterns
    matches.sort(key=lambda m: m.start, reverse=True)

    # Apply redactions (from end to start to preserve positions)
    redacted_text = text
    for match in matches:
        redacted_text = redacted_text[: match.start] + match.replacement + redacted_text[match.end :]

    redacted_hash = dual_hash(redacted_text)

    # Update stats
    _redaction_stats["total_redactions"] += 1
    _redaction_stats["total_pii_found"] += len(matches)
    for match in matches:
        _redaction_stats["pii_by_type"][match.pattern_name] = _redaction_stats["pii_by_type"].get(match.pattern_name, 0) + 1

    return RedactionResult(
        redaction_id=redaction_id,
        original_hash=original_hash,
        redacted_text=redacted_text,
        redacted_hash=redacted_hash,
        matches=matches,
        pii_count=len(matches),
    )


def emit_redaction_receipt(result: RedactionResult) -> Dict[str, Any]:
    """Emit redaction receipt.

    Args:
        result: RedactionResult to emit

    Returns:
        Receipt dict
    """
    return emit_receipt(
        "redaction",
        {
            "tenant_id": PRIVACY_TENANT,
            **result.to_dict(),
        },
    )


def get_redaction_stats() -> Dict[str, Any]:
    """Get redaction statistics.

    Returns:
        Stats dict
    """
    return _redaction_stats.copy()


def clear_redaction_stats() -> None:
    """Clear redaction stats (for testing)."""
    global _redaction_stats
    _redaction_stats = {
        "total_redactions": 0,
        "pii_by_type": {},
        "total_pii_found": 0,
    }


def clear_privacy_config() -> None:
    """Clear privacy config cache (for testing)."""
    global _privacy_config
    _privacy_config = None
