"""PLONK ZK proof CLI commands.

Commands for PLONK succinct ZK proof operations.
"""

from argparse import Namespace
from typing import Any, Dict


def cmd_plonk_info(args: Namespace) -> Dict[str, Any]:
    """Show PLONK configuration.

    Args:
        args: CLI arguments

    Returns:
        Dict with PLONK info
    """
    from src.plonk_zk_upgrade import get_plonk_info

    info = get_plonk_info()

    print("\n=== PLONK ZK CONFIGURATION ===")
    print(f"Proof system: {info.get('proof_system', 'plonk')}")
    print(f"Circuit size: {info.get('circuit_size', 0):,} constraints")

    print("\nTiming:")
    print(f"  Proof time: {info.get('proof_time_ms', 0)} ms")
    print(f"  Verify time: {info.get('verify_time_ms', 0)} ms")

    print("\nFeatures:")
    print(f"  Universal setup: {info.get('universal_setup', False)}")
    print(f"  Recursion capable: {info.get('recursion_capable', False)}")
    print(f"  Privacy preserving: {info.get('privacy_preserving', False)}")

    print("\nSecurity:")
    print(f"  Resilience target: {info.get('resilience_target', 0):.0%}")
    print(f"  Trusted setup participants: {info.get('trusted_setup_participants', 0)}")

    print("\nAttestation Claims:")
    for claim in info.get("attestation_claims", []):
        print(f"  - {claim}")

    return info


def cmd_plonk_setup(args: Namespace) -> Dict[str, Any]:
    """Run universal trusted setup for PLONK.

    Args:
        args: CLI arguments

    Returns:
        Dict with setup results
    """
    from src.plonk_zk_upgrade import universal_setup, PLONK_TRUSTED_SETUP_PARTICIPANTS

    participants = getattr(args, "participants", PLONK_TRUSTED_SETUP_PARTICIPANTS)
    result = universal_setup(participants)

    print("\n=== PLONK UNIVERSAL SETUP ===")
    print(f"Setup type: {result.get('setup_type', 'universal')}")
    print(f"Max circuit size: {result.get('max_circuit_size', 0):,}")
    print(f"Participants: {result.get('participants', 0)}")

    print("\nKeys:")
    print(f"  URS hash: {result.get('urs_hash', '')[:32]}...")
    print(f"  Verification key hash: {result.get('verification_key_hash', '')[:32]}...")

    print("\nSetup Status:")
    print(f"  Setup time: {result.get('setup_time_ms', 0):.2f} ms")
    print(f"  Toxic waste destroyed: {result.get('toxic_waste_destroyed', False)}")
    print(
        f"  Universal setup complete: {result.get('universal_setup_complete', False)}"
    )
    print(f"  Any circuit supported: {result.get('any_circuit_supported', False)}")

    return result


def cmd_plonk_prove(args: Namespace) -> Dict[str, Any]:
    """Generate a PLONK proof.

    Args:
        args: CLI arguments

    Returns:
        Dict with proof
    """
    from src.plonk_zk_upgrade import generate_plonk_circuit, generate_plonk_proof
    import secrets

    circuit = generate_plonk_circuit()

    witness = {
        "enclave_id_private": getattr(args, "enclave_id", secrets.token_hex(16)),
        "enclave_id_commitment_public": secrets.token_hex(32),
        "code_hash_private": getattr(args, "code_hash", secrets.token_hex(32)),
        "code_hash_commitment_public": secrets.token_hex(32),
        "config_hash_private": getattr(args, "config_hash", secrets.token_hex(32)),
        "config_hash_commitment_public": secrets.token_hex(32),
        "timestamp_public": "2024-01-01T00:00:00Z",
        "recursion_depth_public": "0",
    }

    result = generate_plonk_proof(circuit, witness)

    print("\n=== PLONK PROOF GENERATION ===")
    print(f"Proof system: {result.get('proof_system', 'plonk')}")
    print(f"Circuit constraints: {result.get('circuit_constraints', 0):,}")

    print("\nProof Components:")
    print(f"  A: {result.get('proof_a', '')[:32]}...")
    print(f"  B: {result.get('proof_b', '')[:32]}...")
    print(f"  C: {result.get('proof_c', '')[:32]}...")
    print(f"  Z: {result.get('proof_z', '')[:32]}...")

    print("\nMetrics:")
    print(f"  Proof size: {result.get('proof_size_bytes', 0)} bytes")
    print(f"  Generation time: {result.get('generation_time_ms', 0):.2f} ms")
    print(f"  Valid format: {result.get('valid_format', False)}")

    return result


def cmd_plonk_verify(args: Namespace) -> Dict[str, Any]:
    """Verify a PLONK proof.

    Args:
        args: CLI arguments

    Returns:
        Dict with verification result
    """
    from src.plonk_zk_upgrade import verify_plonk

    result = verify_plonk()

    print("\n=== PLONK PROOF VERIFICATION ===")
    print(f"Proof verified: {result.get('valid', False)}")
    print(f"Proof system: {result.get('proof_system', 'plonk')}")
    print(f"Verification time: {result.get('verification_time_ms', 0):.2f} ms")

    return result


def cmd_plonk_recursive(args: Namespace) -> Dict[str, Any]:
    """Generate recursive proof (proof of proofs).

    Args:
        args: CLI arguments

    Returns:
        Dict with recursive proof
    """
    from src.plonk_zk_upgrade import (
        create_plonk_attestation,
        recursive_proof,
    )

    count = getattr(args, "count", 3)

    # Generate base proofs
    proofs = []
    for i in range(count):
        attestation = create_plonk_attestation(
            enclave_id=f"recursive_enclave_{i}",
            code_hash=f"recursive_code_{i}",
            config_hash=f"recursive_config_{i}",
        )
        proofs.append(
            {
                "proof_a": attestation.get("proof", {}).get("proof_a", ""),
                "proof_b": attestation.get("proof", {}).get("proof_b", ""),
            }
        )

    result = recursive_proof(proofs)

    print(f"\n=== PLONK RECURSIVE PROOF ({count} proofs) ===")
    print(f"Proofs aggregated: {result.get('proofs_aggregated', 0)}")
    print(f"Merkle root: {result.get('merkle_root', '')[:32]}...")

    print("\nRecursive Proof:")
    print(f"  Proof A: {result.get('recursive_proof_a', '')[:32]}...")
    print(f"  Proof Z: {result.get('recursive_proof_z', '')[:32]}...")

    print("\nMetrics:")
    print(f"  Proof size: {result.get('proof_size_bytes', 0)} bytes")
    print(f"  Generation time: {result.get('generation_time_ms', 0):.2f} ms")
    print(f"  Compression ratio: {result.get('compression_ratio', 0)}x")
    print(f"  Valid: {result.get('valid', False)}")

    return result


def cmd_plonk_attestation(args: Namespace) -> Dict[str, Any]:
    """Create a PLONK attestation.

    Args:
        args: CLI arguments

    Returns:
        Dict with attestation
    """
    from src.plonk_zk_upgrade import create_plonk_attestation

    enclave_id = getattr(args, "enclave_id", "test_enclave")
    code_hash = getattr(args, "code_hash", "test_code_hash")
    config_hash = getattr(args, "config_hash", "test_config_hash")
    recursion_depth = getattr(args, "recursion_depth", 0)

    result = create_plonk_attestation(
        enclave_id, code_hash, config_hash, recursion_depth
    )

    print("\n=== PLONK ATTESTATION ===")
    print(f"Version: {result.get('version', '')}")
    print(f"Proof system: {result.get('proof_system', '')}")

    print("\nCircuit:")
    circuit = result.get("circuit", {})
    print(f"  Claims: {circuit.get('claims', [])}")
    print(f"  Constraints: {circuit.get('constraints', 0):,}")

    print("\nPublic Inputs:")
    pub = result.get("public_inputs", {})
    print(f"  Enclave ID commitment: {pub.get('enclave_id_commitment', '')[:32]}...")
    print(f"  Timestamp: {pub.get('timestamp', '')}")
    print(f"  Recursion depth: {pub.get('recursion_depth', 0)}")

    print("\nMetadata:")
    meta = result.get("metadata", {})
    print(f"  Created at: {meta.get('created_at', '')}")
    print(f"  Proof time: {meta.get('proof_time_ms', 0):.2f} ms")
    print(f"  Universal setup: {meta.get('universal_setup', False)}")
    print(f"  Recursion capable: {meta.get('recursion_capable', False)}")

    print(f"\nAttestation hash: {result.get('attestation_hash', '')[:32]}...")

    return result


def cmd_plonk_audit(args: Namespace) -> Dict[str, Any]:
    """Run full PLONK audit.

    Args:
        args: CLI arguments

    Returns:
        Dict with audit results
    """
    from src.plonk_zk_upgrade import run_plonk_audit

    count = getattr(args, "count", 5)
    result = run_plonk_audit(attestation_count=count)

    print(f"\n=== PLONK AUDIT ({count} attestations) ===")
    print(f"Proof system: {result.get('proof_system', '')}")
    print(f"Universal setup: {result.get('universal_setup_complete', False)}")

    print("\nAttestation Results:")
    print(f"  Created: {result.get('attestations_created', 0)}")
    print(f"  Verified: {result.get('verifications_passed', 0)}")
    print(f"  Verification rate: {result.get('verification_rate', 0):.2%}")

    print("\nResilience:")
    print(f"  Level: {result.get('resilience', 0):.2%}")
    print(f"  Target: {result.get('resilience_target', 0):.2%}")
    print(f"  Target met: {result.get('resilience_target_met', False)}")

    print("\nRecursive Proofs:")
    print(f"  Valid: {result.get('recursive_proof_valid', False)}")
    print(f"  Compression ratio: {result.get('recursive_compression_ratio', 0)}x")

    print("\nBenchmark:")
    bench = result.get("benchmark", {})
    proof_time = bench.get("proof_time_ms", {})
    verify_time = bench.get("verify_time_ms", {})
    print(f"  Avg proof time: {proof_time.get('avg', 0):.2f} ms")
    print(f"  Avg verify time: {verify_time.get('avg', 0):.2f} ms")

    print(f"\nOverall validated: {result.get('overall_validated', False)}")

    return result


def cmd_plonk_benchmark(args: Namespace) -> Dict[str, Any]:
    """Benchmark PLONK proof system.

    Args:
        args: CLI arguments

    Returns:
        Dict with benchmark results
    """
    from src.plonk_zk_upgrade import benchmark_plonk

    iterations = getattr(args, "iterations", 10)
    result = benchmark_plonk(iterations)

    print(f"\n=== PLONK BENCHMARK ({iterations} iterations) ===")
    print(f"Proof system: {result.get('proof_system', '')}")
    print(f"Circuit size: {result.get('circuit_size', 0):,}")

    print("\nProof Generation:")
    proof_time = result.get("proof_time_ms", {})
    print(f"  Min: {proof_time.get('min', 0):.2f} ms")
    print(f"  Max: {proof_time.get('max', 0):.2f} ms")
    print(f"  Avg: {proof_time.get('avg', 0):.2f} ms")

    print("\nVerification:")
    verify_time = result.get("verify_time_ms", {})
    print(f"  Min: {verify_time.get('min', 0):.2f} ms")
    print(f"  Max: {verify_time.get('max', 0):.2f} ms")
    print(f"  Avg: {verify_time.get('avg', 0):.2f} ms")

    print("\nThroughput:")
    print(f"  Proofs/sec: {result.get('throughput_proofs_per_sec', 0):.2f}")
    print(f"  Verifies/sec: {result.get('throughput_verifies_per_sec', 0):.2f}")

    return result


def cmd_plonk_compare(args: Namespace) -> Dict[str, Any]:
    """Compare PLONK vs Groth16.

    Args:
        args: CLI arguments

    Returns:
        Dict with comparison results
    """
    from src.plonk_zk_upgrade import compare_to_groth16

    result = compare_to_groth16()

    print("\n=== PLONK vs GROTH16 COMPARISON ===")

    print("\nPLONK:")
    plonk = result.get("plonk", {})
    print(f"  Circuit size: {plonk.get('circuit_size', 0):,}")
    print(f"  Proof time: {plonk.get('proof_time_ms', 0)} ms")
    print(f"  Verify time: {plonk.get('verify_time_ms', 0)} ms")
    print(f"  Universal setup: {plonk.get('universal_setup', False)}")
    print(f"  Recursion capable: {plonk.get('recursion_capable', False)}")

    print("\nGroth16:")
    groth16 = result.get("groth16", {})
    print(f"  Circuit size: {groth16.get('circuit_size', 0):,}")
    print(f"  Proof time: {groth16.get('proof_time_ms', 0)} ms")
    print(f"  Verify time: {groth16.get('verify_time_ms', 0)} ms")
    print(f"  Per-circuit setup: {groth16.get('per_circuit_setup', True)}")

    print("\nComparison:")
    comparison = result.get("comparison", {})
    print(f"  Proof speedup: {comparison.get('proof_speedup', 0)}x")
    print(f"  Verify speedup: {comparison.get('verify_speedup', 0)}x")
    print(f"  Circuit size increase: {comparison.get('circuit_size_increase', 0)}x")

    print("\nPLONK Advantages:")
    for adv in comparison.get("plonk_advantages", []):
        print(f"  - {adv}")

    print(f"\nRecommendation: {result.get('recommendation', '')}")

    return result
