"""Parallel testnet infrastructure for cross-chain operations.

Implements parallel testnet support for Ethereum and Solana with
cross-chain bridge and transaction validation.

Receipt Types:
    - testnet_config_receipt: Configuration loaded
    - testnet_ethereum_receipt: Ethereum testnet status
    - testnet_solana_receipt: Solana testnet status
    - testnet_bridge_receipt: Bridge created
    - cross_chain_tx_receipt: Cross-chain transaction
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core import TENANT_ID, dual_hash, emit_receipt

# Testnet constants
TESTNET_PARALLEL_ENABLED = True
TESTNET_CHAIN_ETHEREUM = True
TESTNET_CHAIN_SOLANA = True
TESTNET_BRIDGE_ENABLED = True

# Confirmation block requirements
ETHEREUM_CONFIRMATIONS = 12
SOLANA_CONFIRMATIONS = 32


@dataclass
class TestnetChain:
    """Represents a testnet chain."""

    name: str
    enabled: bool
    block_time_sec: float
    current_block: int = 0
    transactions: int = 0
    status: str = "active"
    last_block_time: Optional[str] = None


@dataclass
class CrossChainTransaction:
    """Represents a cross-chain transaction."""

    tx_id: str
    from_chain: str
    to_chain: str
    data: Dict[str, Any]
    status: str  # "pending", "confirmed", "failed"
    confirmations: int = 0
    created_at: str = ""
    confirmed_at: Optional[str] = None


@dataclass
class TestnetState:
    """Current testnet state."""

    chains: Dict[str, TestnetChain] = field(default_factory=dict)
    bridge_enabled: bool = False
    pending_transactions: List[CrossChainTransaction] = field(default_factory=list)
    completed_transactions: List[CrossChainTransaction] = field(default_factory=list)
    tx_counter: int = 0


# Global testnet state
_testnet_state = TestnetState()


def load_testnet_config() -> Dict[str, Any]:
    """Load testnet configuration from spec file.

    Returns:
        dict: Testnet configuration.

    Receipt:
        testnet_config_receipt
    """
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "swarm_testnet_spec.json"
    )
    with open(spec_path, "r") as f:
        spec = json.load(f)

    testnet_chains = spec.get("testnet_chains", {})
    bridge_config = spec.get("bridge_config", {})

    config = {
        "parallel_enabled": TESTNET_PARALLEL_ENABLED,
        "ethereum": testnet_chains.get(
            "ethereum",
            {"enabled": TESTNET_CHAIN_ETHEREUM, "rpc_simulation": True, "block_time_sec": 12},
        ),
        "solana": testnet_chains.get(
            "solana",
            {"enabled": TESTNET_CHAIN_SOLANA, "rpc_simulation": True, "block_time_sec": 0.4},
        ),
        "bridge": bridge_config,
    }

    emit_receipt(
        "testnet_config_receipt",
        {
            "receipt_type": "testnet_config_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "parallel_enabled": config["parallel_enabled"],
            "ethereum_enabled": config["ethereum"]["enabled"],
            "solana_enabled": config["solana"]["enabled"],
            "bridge_enabled": config["bridge"].get("enabled", TESTNET_BRIDGE_ENABLED),
            "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
        },
    )
    return config


def init_ethereum_testnet() -> Dict[str, Any]:
    """Initialize Ethereum testnet.

    Returns:
        dict: Ethereum testnet initialization result.

    Receipt:
        testnet_ethereum_receipt
    """
    global _testnet_state

    config = load_testnet_config()
    eth_config = config["ethereum"]

    chain = TestnetChain(
        name="ethereum",
        enabled=eth_config.get("enabled", True),
        block_time_sec=eth_config.get("block_time_sec", 12),
        current_block=random.randint(1000000, 2000000),
        status="active",
        last_block_time=datetime.utcnow().isoformat() + "Z",
    )

    _testnet_state.chains["ethereum"] = chain

    result = {
        "initialized": True,
        "chain": "ethereum",
        "block_time_sec": chain.block_time_sec,
        "current_block": chain.current_block,
        "status": chain.status,
    }

    emit_receipt(
        "testnet_ethereum_receipt",
        {
            "receipt_type": "testnet_ethereum_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "init",
            "initialized": True,
            "current_block": chain.current_block,
            "status": chain.status,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def init_solana_testnet() -> Dict[str, Any]:
    """Initialize Solana testnet.

    Returns:
        dict: Solana testnet initialization result.

    Receipt:
        testnet_solana_receipt
    """
    global _testnet_state

    config = load_testnet_config()
    sol_config = config["solana"]

    chain = TestnetChain(
        name="solana",
        enabled=sol_config.get("enabled", True),
        block_time_sec=sol_config.get("block_time_sec", 0.4),
        current_block=random.randint(100000000, 200000000),
        status="active",
        last_block_time=datetime.utcnow().isoformat() + "Z",
    )

    _testnet_state.chains["solana"] = chain

    result = {
        "initialized": True,
        "chain": "solana",
        "block_time_sec": chain.block_time_sec,
        "current_block": chain.current_block,
        "status": chain.status,
    }

    emit_receipt(
        "testnet_solana_receipt",
        {
            "receipt_type": "testnet_solana_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "init",
            "initialized": True,
            "current_block": chain.current_block,
            "status": chain.status,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def create_bridge() -> Dict[str, Any]:
    """Create cross-chain bridge.

    Returns:
        dict: Bridge creation result.

    Receipt:
        testnet_bridge_receipt
    """
    global _testnet_state

    config = load_testnet_config()
    bridge_config = config["bridge"]

    # Initialize chains if not already done
    if "ethereum" not in _testnet_state.chains:
        init_ethereum_testnet()
    if "solana" not in _testnet_state.chains:
        init_solana_testnet()

    _testnet_state.bridge_enabled = bridge_config.get("enabled", TESTNET_BRIDGE_ENABLED)

    result = {
        "bridge_created": True,
        "chains_connected": list(_testnet_state.chains.keys()),
        "confirmation_blocks": bridge_config.get("confirmation_blocks", 6),
        "timeout_sec": bridge_config.get("timeout_sec", 3600),
    }

    emit_receipt(
        "testnet_bridge_receipt",
        {
            "receipt_type": "testnet_bridge_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "create",
            "bridge_created": True,
            "chains_connected": result["chains_connected"],
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def send_cross_chain(
    from_chain: str, to_chain: str, data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Send cross-chain transaction.

    Args:
        from_chain: Source chain.
        to_chain: Destination chain.
        data: Transaction data.

    Returns:
        dict: Transaction result.

    Receipt:
        cross_chain_tx_receipt
    """
    global _testnet_state

    if not _testnet_state.bridge_enabled:
        create_bridge()

    if from_chain not in _testnet_state.chains:
        return {"error": f"Chain {from_chain} not initialized"}
    if to_chain not in _testnet_state.chains:
        return {"error": f"Chain {to_chain} not initialized"}

    _testnet_state.tx_counter += 1
    tx_id = f"cross_tx_{_testnet_state.tx_counter:08d}"

    if data is None:
        data = {"type": "transfer", "amount": random.randint(1, 1000)}

    tx = CrossChainTransaction(
        tx_id=tx_id,
        from_chain=from_chain,
        to_chain=to_chain,
        data=data,
        status="pending",
        created_at=datetime.utcnow().isoformat() + "Z",
    )

    _testnet_state.pending_transactions.append(tx)

    # Update chain transaction counts
    _testnet_state.chains[from_chain].transactions += 1

    result = {
        "tx_id": tx_id,
        "from_chain": from_chain,
        "to_chain": to_chain,
        "status": "pending",
        "data": data,
    }

    emit_receipt(
        "cross_chain_tx_receipt",
        {
            "receipt_type": "cross_chain_tx_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "send",
            "tx_id": tx_id,
            "from_chain": from_chain,
            "to_chain": to_chain,
            "status": "pending",
            "payload_hash": dual_hash(json.dumps(result, default=str)),
        },
    )
    return result


def validate_cross_chain(tx_id: str) -> Dict[str, Any]:
    """Validate and confirm cross-chain transaction.

    Args:
        tx_id: Transaction ID.

    Returns:
        dict: Validation result.

    Receipt:
        cross_chain_tx_receipt
    """
    global _testnet_state

    config = load_testnet_config()
    required_confirmations = config["bridge"].get("confirmation_blocks", 6)

    # Find transaction
    tx = None
    for pending in _testnet_state.pending_transactions:
        if pending.tx_id == tx_id:
            tx = pending
            break

    if tx is None:
        # Check completed
        for completed in _testnet_state.completed_transactions:
            if completed.tx_id == tx_id:
                return {
                    "validated": True,
                    "tx_id": tx_id,
                    "status": completed.status,
                    "confirmations": completed.confirmations,
                }
        return {"error": f"Transaction {tx_id} not found"}

    # Simulate confirmation process
    tx.confirmations = random.randint(required_confirmations, required_confirmations + 5)
    tx.status = "confirmed" if tx.confirmations >= required_confirmations else "pending"

    if tx.status == "confirmed":
        tx.confirmed_at = datetime.utcnow().isoformat() + "Z"
        _testnet_state.pending_transactions.remove(tx)
        _testnet_state.completed_transactions.append(tx)
        _testnet_state.chains[tx.to_chain].transactions += 1

    result = {
        "validated": tx.status == "confirmed",
        "tx_id": tx_id,
        "status": tx.status,
        "confirmations": tx.confirmations,
        "required_confirmations": required_confirmations,
    }

    emit_receipt(
        "cross_chain_tx_receipt",
        {
            "receipt_type": "cross_chain_tx_receipt",
            "tenant_id": TENANT_ID,
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": "validate",
            "tx_id": tx_id,
            "validated": result["validated"],
            "status": tx.status,
            "confirmations": tx.confirmations,
            "payload_hash": dual_hash(json.dumps(result)),
        },
    )
    return result


def get_testnet_status() -> Dict[str, Any]:
    """Get current testnet status.

    Returns:
        dict: Testnet status.
    """
    global _testnet_state

    chain_status = {}
    for name, chain in _testnet_state.chains.items():
        chain_status[name] = {
            "enabled": chain.enabled,
            "status": chain.status,
            "current_block": chain.current_block,
            "transactions": chain.transactions,
            "block_time_sec": chain.block_time_sec,
        }

    return {
        "chains": chain_status,
        "bridge_enabled": _testnet_state.bridge_enabled,
        "pending_transactions": len(_testnet_state.pending_transactions),
        "completed_transactions": len(_testnet_state.completed_transactions),
        "total_transactions": _testnet_state.tx_counter,
    }


def simulate_block_production(chain: str, blocks: int = 10) -> Dict[str, Any]:
    """Simulate block production on a chain.

    Args:
        chain: Chain name.
        blocks: Number of blocks to produce.

    Returns:
        dict: Block production result.
    """
    global _testnet_state

    if chain not in _testnet_state.chains:
        return {"error": f"Chain {chain} not initialized"}

    chain_state = _testnet_state.chains[chain]
    start_block = chain_state.current_block

    for _ in range(blocks):
        chain_state.current_block += 1
        chain_state.last_block_time = datetime.utcnow().isoformat() + "Z"
        time.sleep(0.001)  # Small delay

    return {
        "chain": chain,
        "blocks_produced": blocks,
        "start_block": start_block,
        "end_block": chain_state.current_block,
    }


def stress_test_bridge(transactions: int = 100) -> Dict[str, Any]:
    """Stress test the cross-chain bridge.

    Args:
        transactions: Number of transactions to send.

    Returns:
        dict: Stress test result.
    """
    global _testnet_state

    if not _testnet_state.bridge_enabled:
        create_bridge()

    start_time = time.time()
    successful = 0
    failed = 0

    for i in range(transactions):
        # Alternate between chains
        if i % 2 == 0:
            result = send_cross_chain("ethereum", "solana")
        else:
            result = send_cross_chain("solana", "ethereum")

        if "error" not in result:
            # Validate transaction
            validation = validate_cross_chain(result["tx_id"])
            if validation.get("validated", False):
                successful += 1
            else:
                failed += 1
        else:
            failed += 1

    total_time = time.time() - start_time

    return {
        "stress_passed": successful / max(1, transactions) >= 0.95,
        "transactions": transactions,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / max(1, transactions),
        "total_time_s": total_time,
        "throughput_tps": transactions / total_time,
    }


def sync_testnets() -> Dict[str, Any]:
    """Synchronize block heights across testnets.

    Returns:
        dict: Sync result.
    """
    global _testnet_state

    if not _testnet_state.chains:
        return {"synced": False, "error": "testnets not initialized"}

    # Sync all chains to highest block
    max_block = max(c.current_block for c in _testnet_state.chains.values())
    for chain in _testnet_state.chains.values():
        chain.current_block = max_block

    return {
        "synced": True,
        "block_height": max_block,
        "chains": list(_testnet_state.chains.keys()),
    }


# Backward-compatibility alias
stress_test_testnets = stress_test_bridge
