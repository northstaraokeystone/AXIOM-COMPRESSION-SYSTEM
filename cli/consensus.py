"""Lag-tolerant consensus CLI commands.

Provides CLI commands for modified Raft consensus testing.
"""

from argparse import Namespace
from typing import Any, Dict


def cmd_consensus_info(args: Namespace) -> Dict[str, Any]:
    """Show consensus configuration.

    Args:
        args: CLI arguments.

    Returns:
        dict: Configuration info.
    """
    from src.lag_consensus import load_consensus_config

    config = load_consensus_config()

    print("\n=== LAG-TOLERANT CONSENSUS CONFIGURATION ===")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Heartbeat: {config['heartbeat_ms']}ms ({config['heartbeat_ms']/1000:.0f}s)")
    print(f"Election Timeout: {config['election_timeout_ms']}ms ({config['election_timeout_ms']/1000:.0f}s)")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Quorum Fraction: {config['quorum_fraction']*100:.0f}%")
    print(f"Max Latency: {config['max_latency_years']} years (one-way)")
    print(f"Log Replication: {config['log_replication']}")
    print(f"Snapshot Interval: {config['snapshot_interval']} entries")

    return config


def cmd_consensus_init(args: Namespace) -> Dict[str, Any]:
    """Initialize consensus cluster.

    Args:
        args: CLI arguments.

    Returns:
        dict: Initialization result.
    """
    from src.lag_consensus import initialize_modified_raft

    nodes = getattr(args, "nodes", 5)
    node_ids = [f"node_{i}" for i in range(nodes)]

    print(f"\n=== INITIALIZING CONSENSUS CLUSTER ({nodes} nodes) ===")
    cluster = initialize_modified_raft(node_ids)

    print(f"Initialized: {cluster['initialized']}")
    print(f"Node Count: {len(cluster['nodes'])}")
    print(f"Leader: {cluster['leader']}")
    print(f"Term: {cluster['term']}")
    print(f"Algorithm: {cluster['config']['algorithm']}")

    return {
        "initialized": cluster["initialized"],
        "node_count": len(cluster["nodes"]),
        "leader": cluster["leader"],
        "term": cluster["term"],
    }


def cmd_consensus_simulate(args: Namespace) -> Dict[str, Any]:
    """Run consensus simulation.

    Args:
        args: CLI arguments.

    Returns:
        dict: Simulation result.
    """
    from src.lag_consensus import modified_raft_consensus

    nodes = getattr(args, "nodes", 5)
    latency = getattr(args, "latency", 6300)
    node_ids = [f"node_{i}" for i in range(nodes)]

    print("\n=== RUNNING CONSENSUS SIMULATION ===")
    print(f"Nodes: {nodes}")
    print(f"Latency Multiplier: {latency}x")

    result = modified_raft_consensus(node_ids, latency_multiplier=latency)

    print(f"\nConsensus Achieved: {result['consensus_achieved']}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Leader: {result['leader']}")
    print(f"Term: {result['term']}")
    print(f"Entries Committed: {result['entries_committed']}")
    print(f"Effective Heartbeat: {result['effective_heartbeat_ms']:.0f}ms")
    print(f"Effective Election: {result['effective_election_ms']:.0f}ms")
    print(f"Quorum: {result['quorum_fraction']*100:.0f}%")

    return result


def cmd_consensus_election(args: Namespace) -> Dict[str, Any]:
    """Trigger leader election.

    Args:
        args: CLI arguments.

    Returns:
        dict: Election result.
    """
    from src.lag_consensus import initialize_modified_raft, start_election, request_vote, handle_vote

    nodes = getattr(args, "nodes", 5)
    node_ids = [f"node_{i}" for i in range(nodes)]

    print("\n=== TRIGGERING LEADER ELECTION ===")

    cluster = initialize_modified_raft(node_ids)

    # Simulate leader failure - node_1 starts election
    candidate = "node_1"
    new_term = cluster["term"] + 1

    election = start_election(cluster, candidate, new_term)
    print(f"Election Started: {election['election_started']}")
    print(f"Candidate: {election['candidate']}")
    print(f"Term: {election['term']}")

    # Request votes from others
    votes = []
    for voter_id in node_ids:
        if voter_id != candidate:
            vote = request_vote(cluster, candidate, voter_id, new_term)
            votes.append(vote)
            print(f"Vote from {voter_id}: {vote['vote_granted']}")

    # Handle votes
    for vote in votes:
        handle_vote(cluster, candidate, vote)

    candidate_node = cluster["nodes"][candidate]
    print(f"\nFinal Leader: {cluster.get('leader', 'none')}")
    print(f"Votes Received: {candidate_node.votes_received}")
    print(f"Is Leader: {candidate_node.state.value == 'leader'}")

    return {
        "election_triggered": True,
        "candidate": candidate,
        "term": new_term,
        "votes_received": candidate_node.votes_received,
        "became_leader": candidate_node.state.value == "leader",
    }


def cmd_consensus_status(args: Namespace) -> Dict[str, Any]:
    """Show consensus status.

    Args:
        args: CLI arguments.

    Returns:
        dict: Status info.
    """
    from src.lag_consensus import get_consensus_status

    status = get_consensus_status()

    print("\n=== CONSENSUS STATUS ===")
    print(f"Algorithm: {status['algorithm']}")
    print(f"Heartbeat: {status['heartbeat_ms']}ms")
    print(f"Election Timeout: {status['election_timeout_ms']}ms")
    print(f"Batch Size: {status['batch_size']}")
    print(f"Quorum: {status['quorum_fraction']*100:.0f}%")
    print(f"Max Latency: {status['max_latency_years']} years")
    print(f"Latency Multiplier: {status['latency_multiplier']}x")

    return status
