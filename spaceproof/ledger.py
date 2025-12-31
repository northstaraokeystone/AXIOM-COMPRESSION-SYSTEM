"""ledger.py - Append-Only Receipt Storage

D20 Production Evolution: Stakeholder-intuitive name for receipt ledger.

THE LEDGER INSIGHT:
    Receipts are the atoms of proof.
    An append-only ledger is tamper-evident.
    No receipt, not real.

Source: SpaceProof D20 Production Evolution

Stakeholder Value:
    - DOGE: Audit trail for all government payments
    - Defense: Chain of custody for decisions
    - NRO: Immutable decision lineage
"""

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import json
import os

from .core import emit_receipt, dual_hash, merkle

# === CONSTANTS ===

TENANT_ID = "spaceproof-ledger"


@dataclass
class LedgerEntry:
    """A single entry in the ledger."""

    receipt: Dict
    hash: str
    timestamp: str
    index: int


@dataclass
class Ledger:
    """Append-only receipt ledger."""

    entries: List[LedgerEntry] = field(default_factory=list)
    merkle_root: str = ""

    def append(self, receipt: Dict) -> LedgerEntry:
        """Append a receipt to the ledger.

        Args:
            receipt: Receipt dict to append

        Returns:
            LedgerEntry with hash and index
        """
        receipt_hash = dual_hash(json.dumps(receipt, sort_keys=True))
        timestamp = datetime.utcnow().isoformat() + "Z"
        index = len(self.entries)

        entry = LedgerEntry(
            receipt=receipt,
            hash=receipt_hash,
            timestamp=timestamp,
            index=index,
        )

        self.entries.append(entry)
        self._update_merkle_root()

        return entry

    def _update_merkle_root(self) -> None:
        """Update Merkle root after append."""
        if not self.entries:
            self.merkle_root = dual_hash(b"empty")
        else:
            self.merkle_root = merkle([e.receipt for e in self.entries])

    def verify(self) -> bool:
        """Verify ledger integrity.

        Returns:
            True if all entries are valid
        """
        for entry in self.entries:
            expected_hash = dual_hash(json.dumps(entry.receipt, sort_keys=True))
            if expected_hash != entry.hash:
                return False

        expected_root = merkle([e.receipt for e in self.entries]) if self.entries else dual_hash(b"empty")
        return expected_root == self.merkle_root

    def get_proof(self, index: int) -> Dict:
        """Get Merkle proof for entry at index.

        Args:
            index: Entry index

        Returns:
            Proof dict with path and root
        """
        if index < 0 or index >= len(self.entries):
            raise ValueError(f"Index {index} out of range")

        return {
            "entry_hash": self.entries[index].hash,
            "merkle_root": self.merkle_root,
            "index": index,
            "total_entries": len(self.entries),
        }


def create_ledger() -> Ledger:
    """Create a new empty ledger.

    Returns:
        Empty Ledger instance
    """
    ledger = Ledger()

    emit_receipt(
        "ledger_created",
        {
            "tenant_id": TENANT_ID,
            "merkle_root": ledger.merkle_root,
            "entry_count": 0,
        },
    )

    return ledger


def append_to_ledger(ledger: Ledger, receipt: Dict) -> LedgerEntry:
    """Append receipt to ledger.

    Args:
        ledger: Ledger instance
        receipt: Receipt to append

    Returns:
        New LedgerEntry
    """
    entry = ledger.append(receipt)

    emit_receipt(
        "ledger_append",
        {
            "tenant_id": TENANT_ID,
            "entry_hash": entry.hash,
            "entry_index": entry.index,
            "merkle_root": ledger.merkle_root,
        },
    )

    return entry


def verify_ledger(ledger: Ledger) -> Dict:
    """Verify ledger integrity.

    Args:
        ledger: Ledger to verify

    Returns:
        Verification result dict
    """
    is_valid = ledger.verify()

    result = {
        "valid": is_valid,
        "entry_count": len(ledger.entries),
        "merkle_root": ledger.merkle_root,
    }

    emit_receipt(
        "ledger_verify",
        {
            "tenant_id": TENANT_ID,
            **result,
        },
    )

    return result


def save_ledger(ledger: Ledger, path: str) -> None:
    """Save ledger to JSONL file.

    Args:
        ledger: Ledger to save
        path: File path
    """
    with open(path, "w") as f:
        for entry in ledger.entries:
            line = json.dumps(entry.receipt)
            f.write(line + "\n")


def load_ledger(path: str) -> Ledger:
    """Load ledger from JSONL file.

    Args:
        path: File path

    Returns:
        Ledger instance
    """
    ledger = Ledger()

    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    receipt = json.loads(line)
                    ledger.append(receipt)

    return ledger
