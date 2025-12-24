"""Tests for lag-tolerant consensus module."""

from src.lag_consensus import (
    load_consensus_config,
    initialize_modified_raft,
    send_heartbeat,
    handle_heartbeat,
    start_election,
    request_vote,
    append_entries_batch,
    commit_entries,
    modified_raft_consensus,
    measure_consensus_latency,
    validate_quorum,
    snapshot_log,
    get_consensus_status,
    LogEntry,
)


class TestConsensusConfig:
    """Tests for consensus configuration."""

    def test_consensus_config_loads(self):
        """Config loads successfully."""
        config = load_consensus_config()
        assert config is not None
        assert "algorithm" in config
        assert "heartbeat_ms" in config
        assert "election_timeout_ms" in config

    def test_consensus_algorithm(self):
        """Config has modified_raft algorithm."""
        config = load_consensus_config()
        assert config["algorithm"] == "modified_raft"

    def test_consensus_heartbeat(self):
        """Config has 60s heartbeat."""
        config = load_consensus_config()
        assert config["heartbeat_ms"] == 60000

    def test_consensus_election_timeout(self):
        """Config has 5min election timeout."""
        config = load_consensus_config()
        assert config["election_timeout_ms"] == 300000


class TestConsensusInit:
    """Tests for consensus initialization."""

    def test_consensus_init(self):
        """Initialization works."""
        nodes = ["node_0", "node_1", "node_2"]
        cluster = initialize_modified_raft(nodes)
        assert cluster["initialized"] is True
        assert len(cluster["nodes"]) == 3
        assert cluster["leader"] is not None


class TestConsensusHeartbeat:
    """Tests for heartbeat mechanism."""

    def test_consensus_heartbeat(self):
        """Heartbeat works."""
        nodes = ["node_0", "node_1", "node_2"]
        cluster = initialize_modified_raft(nodes)

        leader = cluster["leader"]
        followers = [n for n in nodes if n != leader]

        result = send_heartbeat(cluster, leader, followers)
        assert result["sent"] is True
        assert result["leader"] == leader
        assert len(result["responses"]) == 2

    def test_consensus_handle_heartbeat(self):
        """Handle heartbeat works."""
        nodes = ["node_0", "node_1", "node_2"]
        cluster = initialize_modified_raft(nodes)

        heartbeat = {
            "term": 1,
            "leader_id": cluster["leader"],
            "prev_log_index": 0,
            "prev_log_term": 0,
            "leader_commit": 0,
            "entries": [],
        }

        result = handle_heartbeat(cluster, "node_1", heartbeat)
        assert result["success"] is True


class TestConsensusElection:
    """Tests for leader election."""

    def test_consensus_election(self):
        """Election works."""
        nodes = ["node_0", "node_1", "node_2", "node_3", "node_4"]
        cluster = initialize_modified_raft(nodes)

        result = start_election(cluster, "node_1", 2)
        assert result["election_started"] is True
        assert result["term"] == 2

    def test_consensus_vote(self):
        """Voting works."""
        nodes = ["node_0", "node_1", "node_2"]
        cluster = initialize_modified_raft(nodes)

        vote = request_vote(cluster, "node_1", "node_2", 2)
        assert "vote_granted" in vote


class TestConsensusAppend:
    """Tests for append entries."""

    def test_consensus_append(self):
        """Append entries works."""
        nodes = ["node_0", "node_1", "node_2"]
        cluster = initialize_modified_raft(nodes)

        entries = [{"command": "write", "data": {"key": "val"}}]
        result = append_entries_batch(cluster, cluster["leader"], entries)
        assert result["success"] is True
        assert result["entries_appended"] == 1


class TestConsensusCommit:
    """Tests for commit entries."""

    def test_consensus_commit(self):
        """Commit works."""
        nodes = ["node_0", "node_1", "node_2"]
        cluster = initialize_modified_raft(nodes)

        entries = [{"command": "write", "data": {"key": "val"}}]
        result = commit_entries(cluster, entries)
        assert result["success"] is True
        assert result["committed_count"] == 1


class TestConsensusQuorum:
    """Tests for quorum validation."""

    def test_consensus_quorum(self):
        """Quorum validation works."""
        votes = [True, True, False]
        result = validate_quorum(votes, 0.51)
        assert result is True

    def test_consensus_quorum_fail(self):
        """Quorum fails when not met."""
        votes = [True, False, False, False]
        result = validate_quorum(votes, 0.51)
        assert result is False


class TestConsensus6300xLatency:
    """Tests for 6300x latency scenario."""

    def test_consensus_6300x_latency(self):
        """Works at 6300x latency."""
        nodes = ["node_0", "node_1", "node_2", "node_3", "node_4"]
        result = modified_raft_consensus(nodes, latency_multiplier=6300)
        assert result["consensus_achieved"] is True
        assert result["latency_multiplier"] == 6300


class TestConsensusSnapshot:
    """Tests for log snapshot."""

    def test_consensus_snapshot(self):
        """Snapshot works."""
        nodes = ["node_0", "node_1", "node_2"]
        cluster = initialize_modified_raft(nodes)

        # Create some log entries
        log = [
            LogEntry(term=1, index=1, command="write", data={"k": "v1"}),
            LogEntry(term=1, index=2, command="write", data={"k": "v2"}),
        ]

        result = snapshot_log(cluster, log, 1)
        assert result["success"] is True
        assert result["snapshot_index"] == 1


class TestConsensusReceipts:
    """Tests for receipt emission."""

    def test_consensus_receipt(self, capsys):
        """Receipt emitted."""
        load_consensus_config()
        captured = capsys.readouterr()
        assert "lag_consensus_config_receipt" in captured.out


class TestConsensusStatus:
    """Tests for status queries."""

    def test_consensus_status(self):
        """Status query works."""
        status = get_consensus_status()
        assert "algorithm" in status
        assert "heartbeat_ms" in status
        assert "election_timeout_ms" in status


class TestConsensusMeasure:
    """Tests for latency measurement."""

    def test_consensus_measure_latency(self):
        """Latency measurement works."""
        nodes = ["node_0", "node_1", "node_2"]
        result = measure_consensus_latency(nodes, iterations=3)
        assert result["iterations"] == 3
        assert "avg_latency_ms" in result
