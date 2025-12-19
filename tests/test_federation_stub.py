"""Tests for multi-star federation stub.

Tests:
- Federation configuration loading
- Federation initialization
- Consensus-with-lag protocol
- Autonomous arbitration
- Federation status
"""

from src.paths.multiplanet.federation.stub import (
    load_federation_config,
    initialize_federation,
    consensus_with_lag,
    autonomous_arbitration,
    federation_status,
    INITIAL_SYSTEMS,
    FEDERATION_PROTOCOL,
    GOVERNANCE_MODEL,
)


class TestFederationConfig:
    """Tests for federation configuration."""

    def test_load_federation_config(self):
        """Config loads correctly."""
        config = load_federation_config()

        assert config is not None
        assert "enabled" in config
        assert "initial_systems" in config
        assert "federation_protocol" in config
        assert "governance_model" in config

    def test_initial_systems(self):
        """Initial systems are Sol and Proxima."""
        assert INITIAL_SYSTEMS == ["sol", "proxima_centauri"]

    def test_federation_protocol(self):
        """Protocol is consensus_with_lag."""
        assert FEDERATION_PROTOCOL == "consensus_with_lag"

    def test_governance_model(self):
        """Governance is autonomous_with_arbitration."""
        assert GOVERNANCE_MODEL == "autonomous_with_arbitration"


class TestFederationInitialization:
    """Tests for federation initialization."""

    def test_initialize_federation(self):
        """Federation initializes correctly."""
        federation = initialize_federation()

        assert "federation_id" in federation
        assert "members" in federation
        assert "member_count" in federation
        assert "protocol" in federation
        assert "status" in federation

    def test_member_count(self):
        """Member count matches initial systems."""
        federation = initialize_federation()
        assert federation["member_count"] == len(INITIAL_SYSTEMS)

    def test_member_details(self):
        """Members have required details."""
        federation = initialize_federation()

        for member in federation["members"]:
            assert "system_name" in member
            assert "joined_ts" in member
            assert "status" in member
            assert "autonomy_level" in member
            assert "vote_weight" in member

    def test_member_autonomy(self):
        """Members have high autonomy."""
        federation = initialize_federation()

        for member in federation["members"]:
            assert member["autonomy_level"] >= 0.99

    def test_custom_systems(self):
        """Custom systems can be added."""
        systems = ["sol", "proxima_centauri", "alpha_centauri_a"]
        federation = initialize_federation(systems=systems)

        assert federation["member_count"] == 3


class TestConsensusWithLag:
    """Tests for consensus-with-lag protocol."""

    def test_consensus_with_lag(self):
        """Consensus executes correctly."""
        proposal = {"action": "resource_allocation", "amount": 1000}
        result = consensus_with_lag(proposal)

        assert "proposal_hash" in result
        assert "lag_years" in result
        assert "round_trip_years" in result
        assert "predicted_votes" in result
        assert "consensus_reached" in result

    def test_lag_years_default(self):
        """Default lag is 4.24 years."""
        proposal = {"action": "test"}
        result = consensus_with_lag(proposal)

        assert result["lag_years"] == 4.24

    def test_round_trip_years(self):
        """Round trip is 2x lag."""
        proposal = {"action": "test"}
        result = consensus_with_lag(proposal, lag_years=4.24)

        assert result["round_trip_years"] == 8.48

    def test_consensus_reached(self):
        """Consensus is reached."""
        proposal = {"action": "test"}
        result = consensus_with_lag(proposal)

        assert result["consensus_reached"] is True

    def test_predicted_votes(self):
        """Predicted votes are present."""
        proposal = {"action": "test"}
        result = consensus_with_lag(proposal)

        assert "sol" in result["predicted_votes"]
        assert "proxima_centauri" in result["predicted_votes"]


class TestAutonomousArbitration:
    """Tests for autonomous arbitration."""

    def test_autonomous_arbitration(self):
        """Arbitration executes correctly."""
        dispute = {
            "type": "resource_allocation",
            "parties": ["sol", "proxima_centauri"],
            "amount": 1000,
        }
        result = autonomous_arbitration(dispute)

        assert "dispute_type" in result
        assert "parties" in result
        assert "rule_applied" in result
        assert "resolution" in result
        assert "binding" in result

    def test_arbitration_binding(self):
        """Arbitration is binding."""
        dispute = {"type": "resource_allocation"}
        result = autonomous_arbitration(dispute)

        assert result["binding"] is True

    def test_appeal_window(self):
        """Appeal window is round-trip to Proxima."""
        dispute = {"type": "resource_allocation"}
        result = autonomous_arbitration(dispute)

        assert result["appeal_window_years"] == 8.48

    def test_resource_allocation_rule(self):
        """Resource allocation uses proportional rule."""
        dispute = {"type": "resource_allocation"}
        result = autonomous_arbitration(dispute)

        assert result["rule_applied"] == "proportional_to_population"

    def test_territory_rule(self):
        """Territory uses first claim priority."""
        dispute = {"type": "territory"}
        result = autonomous_arbitration(dispute)

        assert result["rule_applied"] == "first_claim_priority"

    def test_defense_rule(self):
        """Defense uses mutual assistance."""
        dispute = {"type": "defense"}
        result = autonomous_arbitration(dispute)

        assert result["rule_applied"] == "mutual_assistance"


class TestFederationStatus:
    """Tests for federation status."""

    def test_federation_status(self):
        """Status retrieves correctly."""
        status = federation_status()

        assert "enabled" in status
        assert "member_count" in status
        assert "members" in status
        assert "protocol" in status
        assert "governance" in status
        assert "status" in status

    def test_federation_operational(self):
        """Federation is operational."""
        status = federation_status()
        assert status["status"] == "operational"

    def test_federation_enabled(self):
        """Federation is enabled."""
        status = federation_status()
        assert status["enabled"] is True
