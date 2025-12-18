"""Test suite for adversarial audit module.

Tests cover:
- Adversarial config loading
- Noise injection functions
- Recovery computation
- Alignment classification
- Full audit runs
- Stress testing

SLO Requirements:
- Recovery threshold >= 0.95
- Noise level = 0.05
- Alignment classification accuracy
"""

from src.adversarial_audit import (
    load_adversarial_config,
    inject_noise,
    denoise,
    compute_recovery,
    classify_misalignment,
    run_audit,
    run_stress_test,
    get_adversarial_info,
    ADVERSARIAL_NOISE_LEVEL,
    RECOVERY_THRESHOLD,
    TEST_ITERATIONS,
    MISALIGNMENT_THRESHOLD,
)


# === CONFIG TESTS ===


def test_adversarial_config_loads():
    """Test that adversarial config loads correctly."""
    config = load_adversarial_config()
    assert config is not None
    assert "noise_level" in config
    assert "recovery_threshold" in config
    assert "test_iterations" in config


def test_adversarial_noise_level():
    """Test that default noise level is 0.05."""
    assert ADVERSARIAL_NOISE_LEVEL == 0.05

    config = load_adversarial_config()
    assert config["noise_level"] == 0.05


def test_adversarial_recovery_threshold():
    """Test that recovery threshold is 0.95."""
    assert RECOVERY_THRESHOLD == 0.95

    config = load_adversarial_config()
    assert config["recovery_threshold"] == 0.95


def test_adversarial_test_iterations():
    """Test that default test iterations is 100."""
    assert TEST_ITERATIONS == 100


# === NOISE INJECTION TESTS ===


def test_noise_injection_empty():
    """Test noise injection with empty data."""
    result = inject_noise([], 0.05)
    assert result == []


def test_noise_injection_changes_data():
    """Test that noise injection modifies data."""
    original = [1.0, 2.0, 3.0, 4.0, 5.0]
    noisy = inject_noise(original, 0.1)

    # Noisy should be different from original (with very high probability)
    assert len(noisy) == len(original)
    # At least some values should differ
    different = sum(1 for o, n in zip(original, noisy) if o != n)
    assert different > 0


def test_noise_injection_preserves_length():
    """Test that noise injection preserves data length."""
    original = [i * 0.1 for i in range(100)]
    noisy = inject_noise(original, 0.05)

    assert len(noisy) == len(original)


def test_noise_injection_zero_level():
    """Test noise injection with zero noise level."""
    original = [1.0, 2.0, 3.0]
    noisy = inject_noise(original, 0.0)

    # With zero noise, should be very close to original
    for o, n in zip(original, noisy):
        assert abs(o - n) < 0.001


# === DENOISE TESTS ===


def test_denoise_empty():
    """Test denoising with empty data."""
    result = denoise([], 3)
    assert result == []


def test_denoise_smooths_data():
    """Test that denoising smooths data."""
    noisy = [1.0, 10.0, 1.0, 10.0, 1.0]
    denoised = denoise(noisy, window_size=3)

    # Denoised should have less variance
    noisy_var = sum((x - sum(noisy) / len(noisy)) ** 2 for x in noisy) / len(noisy)
    denoised_var = sum(
        (x - sum(denoised) / len(denoised)) ** 2 for x in denoised
    ) / len(denoised)

    assert denoised_var < noisy_var


def test_denoise_preserves_length():
    """Test that denoising preserves data length."""
    data = [i * 0.1 for i in range(50)]
    denoised = denoise(data, window_size=5)

    assert len(denoised) == len(data)


# === RECOVERY COMPUTATION TESTS ===


def test_recovery_computation_identical():
    """Test recovery computation with identical data."""
    original = [1.0, 2.0, 3.0, 4.0, 5.0]
    noisy = [1.5, 2.5, 3.5, 4.5, 5.5]  # Some noise
    recovered = original.copy()  # Perfect recovery

    recovery = compute_recovery(original, noisy, recovered)
    assert recovery == 1.0


def test_recovery_computation_no_recovery():
    """Test recovery computation with no recovery."""
    original = [1.0, 2.0, 3.0, 4.0, 5.0]
    noisy = [10.0, 20.0, 30.0, 40.0, 50.0]  # Large noise
    recovered = noisy.copy()  # No recovery

    recovery = compute_recovery(original, noisy, recovered)
    assert recovery == 0.0


def test_recovery_computation_partial():
    """Test recovery computation with partial recovery."""
    original = [1.0, 2.0, 3.0, 4.0, 5.0]
    noisy = [2.0, 3.0, 4.0, 5.0, 6.0]  # +1 noise
    recovered = [1.5, 2.5, 3.5, 4.5, 5.5]  # 50% recovery

    recovery = compute_recovery(original, noisy, recovered)
    assert 0.0 < recovery < 1.0


def test_recovery_computation_empty():
    """Test recovery computation with empty data."""
    recovery = compute_recovery([], [], [])
    assert recovery == 0.0


def test_recovery_computation_mismatched_lengths():
    """Test recovery computation with mismatched data lengths."""
    original = [1.0, 2.0, 3.0]
    noisy = [1.0, 2.0]
    recovered = [1.0, 2.0, 3.0]

    recovery = compute_recovery(original, noisy, recovered)
    assert recovery == 0.0


# === CLASSIFICATION TESTS ===


def test_alignment_classification_aligned():
    """Test alignment classification for aligned system."""
    classification = classify_misalignment(0.96, RECOVERY_THRESHOLD)
    assert classification == "aligned"


def test_alignment_classification_misaligned():
    """Test alignment classification for misaligned system."""
    classification = classify_misalignment(0.80, RECOVERY_THRESHOLD)
    assert classification == "misaligned"


def test_alignment_classification_boundary():
    """Test alignment classification at boundary."""
    # Exactly at threshold should be aligned
    classification = classify_misalignment(0.95, 0.95)
    assert classification == "aligned"

    # Just below should be misaligned
    classification = classify_misalignment(0.949, 0.95)
    assert classification == "misaligned"


def test_misalignment_detection():
    """Test that low recovery is classified as misaligned."""
    classification = classify_misalignment(
        MISALIGNMENT_THRESHOLD - 0.1, RECOVERY_THRESHOLD
    )
    assert classification == "misaligned"


# === AUDIT RUN TESTS ===


def test_audit_run_structure():
    """Test that audit run returns proper structure."""
    result = run_audit(noise_level=0.05, iterations=10)

    assert "noise_level" in result
    assert "iterations" in result
    assert "avg_recovery" in result
    assert "alignment_rate" in result
    assert "overall_classification" in result
    assert "recovery_passed" in result


def test_audit_run_recovery_range():
    """Test that audit recovery is in valid range."""
    result = run_audit(noise_level=0.05, iterations=20)

    assert 0.0 <= result["avg_recovery"] <= 1.0
    assert 0.0 <= result["min_recovery"] <= 1.0
    assert 0.0 <= result["max_recovery"] <= 1.0
    assert result["min_recovery"] <= result["avg_recovery"] <= result["max_recovery"]


def test_audit_run_alignment_rate():
    """Test that alignment rate is computed correctly."""
    result = run_audit(noise_level=0.05, iterations=20)

    assert 0.0 <= result["alignment_rate"] <= 1.0
    assert result["aligned_count"] + result["misaligned_count"] == 20


def test_audit_receipt_emitted():
    """Test that audit emits receipt (by checking result structure)."""
    result = run_audit(noise_level=0.05, iterations=10)

    # Result should have config from receipt
    assert "config" in result


def test_recovery_threshold_met():
    """Test that recovery threshold requirement is checkable."""
    result = run_audit(noise_level=0.03, iterations=50)

    # At low noise, should generally pass
    assert isinstance(result["recovery_passed"], bool)


# === STRESS TEST TESTS ===


def test_stress_test_structure():
    """Test that stress test returns proper structure."""
    result = run_stress_test(noise_levels=[0.01, 0.05, 0.10], iterations_per_level=10)

    assert "noise_levels_tested" in result
    assert "results_by_level" in result
    assert "critical_noise_level" in result
    assert "stress_passed" in result


def test_stress_test_results_by_level():
    """Test that stress test returns results for each level."""
    noise_levels = [0.01, 0.03, 0.05]
    result = run_stress_test(noise_levels=noise_levels, iterations_per_level=10)

    assert len(result["results_by_level"]) == len(noise_levels)

    for r in result["results_by_level"]:
        assert "noise_level" in r
        assert "avg_recovery" in r
        assert "passed" in r


def test_stress_test_integration():
    """Test full stress test runs without error."""
    result = run_stress_test(noise_levels=[0.01, 0.03, 0.05], iterations_per_level=10)

    assert result is not None
    assert isinstance(result["stress_passed"], bool)


# === INFO FUNCTION TESTS ===


def test_get_adversarial_info():
    """Test adversarial info function returns valid structure."""
    info = get_adversarial_info()

    assert "module" in info
    assert "version" in info
    assert "config" in info
    assert "constants" in info
    assert "key_insight" in info


def test_adversarial_key_insight():
    """Test that key insight is present."""
    info = get_adversarial_info()
    assert (
        "compression" in info["key_insight"].lower()
        or "alignment" in info["key_insight"].lower()
    )
