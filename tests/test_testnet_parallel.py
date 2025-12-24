"""Tests for parallel testnet module."""

from src.testnet_parallel import (
    load_testnet_config,
    init_ethereum_testnet,
    init_solana_testnet,
    create_bridge,
    send_cross_chain,
    sync_testnets,
    get_testnet_status,
    stress_test_testnets,
    ETHEREUM_CONFIRMATIONS,
    SOLANA_CONFIRMATIONS,
)


class TestTestnetConfig:
    """Tests for testnet configuration."""

    def test_testnet_config_loads(self):
        """Config loads successfully."""
        config = load_testnet_config()
        assert config is not None
        assert "ethereum" in config
        assert "solana" in config

    def test_confirmation_blocks(self):
        """Confirmation blocks configured."""
        assert ETHEREUM_CONFIRMATIONS == 12
        assert SOLANA_CONFIRMATIONS == 32


class TestEthereumTestnet:
    """Tests for Ethereum testnet."""

    def test_ethereum_init(self):
        """Ethereum testnet initializes."""
        result = init_ethereum_testnet()
        assert result["initialized"] is True
        assert result["chain"] == "ethereum"

    def test_ethereum_confirmations(self):
        """Confirmation count correct."""
        result = init_ethereum_testnet()
        assert result["confirmations"] == ETHEREUM_CONFIRMATIONS


class TestSolanaTestnet:
    """Tests for Solana testnet."""

    def test_solana_init(self):
        """Solana testnet initializes."""
        result = init_solana_testnet()
        assert result["initialized"] is True
        assert result["chain"] == "solana"

    def test_solana_confirmations(self):
        """Confirmation count correct."""
        result = init_solana_testnet()
        assert result["confirmations"] == SOLANA_CONFIRMATIONS


class TestCrossChainBridge:
    """Tests for cross-chain bridge."""

    def test_bridge_creation(self):
        """Bridge creates successfully."""
        init_ethereum_testnet()
        init_solana_testnet()
        result = create_bridge()
        assert result["bridge_created"] is True
        assert "ethereum" in result["chains_connected"]
        assert "solana" in result["chains_connected"]


class TestCrossChainTransactions:
    """Tests for cross-chain transactions."""

    def test_cross_chain_send(self):
        """Cross-chain tx succeeds."""
        init_ethereum_testnet()
        init_solana_testnet()
        create_bridge()
        result = send_cross_chain("ethereum", "solana", 100)
        assert result["success"] is True
        assert result["source_chain"] == "ethereum"
        assert result["target_chain"] == "solana"

    def test_cross_chain_eventual_consistency(self):
        """Eventual consistency achieved."""
        init_ethereum_testnet()
        init_solana_testnet()
        create_bridge()
        result = send_cross_chain("ethereum", "solana", 100)
        assert result["eventual_consistency"] is True


class TestTestnetSync:
    """Tests for testnet sync."""

    def test_testnet_sync(self):
        """Testnets sync successfully."""
        init_ethereum_testnet()
        init_solana_testnet()
        result = sync_testnets()
        assert result["synced"] is True
        assert result["chains_synced"] == 2


class TestTestnetStress:
    """Tests for testnet stress testing."""

    def test_testnet_stress(self):
        """Stress test passes."""
        init_ethereum_testnet()
        init_solana_testnet()
        create_bridge()
        result = stress_test_testnets(10)  # 10 iterations
        assert result["stress_passed"] is True
        assert result["iterations"] == 10


class TestTestnetStatus:
    """Tests for status queries."""

    def test_testnet_status(self):
        """Status query works."""
        init_ethereum_testnet()
        init_solana_testnet()
        status = get_testnet_status()
        assert "ethereum_initialized" in status
        assert "solana_initialized" in status
        assert "bridge_active" in status


class TestTestnetReceipts:
    """Tests for receipt emission."""

    def test_testnet_receipt(self, capsys):
        """Receipt emitted."""
        init_ethereum_testnet()
        captured = capsys.readouterr()
        assert "testnet_receipt" in captured.out
