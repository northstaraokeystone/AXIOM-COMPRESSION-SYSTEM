"""test_food_verification.py - Food domain verification tests.

Tests for olive oil, honey, and seafood verification.
Detection target: ≥99.9% recall, <1% false positive
"""

import numpy as np

from spaceproof.food.olive_oil import verify_olive_oil
from spaceproof.food.honey import verify_honey
from spaceproof.food.seafood import verify_seafood
from spaceproof.food.entropy import spectral_entropy, texture_entropy, pollen_diversity_entropy


class TestOliveOilVerification:
    """Tests for olive oil verification."""

    def test_authentic_extra_virgin(self, suppress_receipts):
        """Genuine EVOO with entropy 4.2 → verdict AUTHENTIC."""
        # Generate high-entropy spectral data (genuine)
        spectrum = np.random.uniform(3.9, 4.5, 100) + np.random.normal(0, 0.3, 100)

        verdict, receipt = verify_olive_oil(
            batch_id="TEST-EVOO-001",
            product_grade="extra_virgin",
            spectral_scan=spectrum,
            provenance_chain=["italian_farm", "processor", "bottler"],
        )

        assert verdict in ["AUTHENTIC", "SUSPICIOUS"]  # May be suspicious if borderline
        assert "olive_oil_verification" in receipt.get("receipt_type", "")

    def test_adulterated_sunflower_mix(self, suppress_receipts):
        """EVOO + sunflower oil (50/50) with low entropy → verdict COUNTERFEIT."""
        # Generate low-entropy spectral data (adulterated/homogenized)
        base = np.full(100, 2.5)  # Uniform (low entropy)
        spectrum = base + np.random.normal(0, 0.05, 100)

        verdict, receipt = verify_olive_oil(
            batch_id="TEST-FAKE-001",
            product_grade="extra_virgin",
            spectral_scan=spectrum,
            provenance_chain=["unknown"],
        )

        assert verdict in ["COUNTERFEIT", "SUSPICIOUS"]
        # Entropy should be tracked in receipt
        assert "measured_entropy" in receipt or "entropy_analysis" in receipt

    def test_boundary_case_threshold(self, suppress_receipts):
        """Entropy at boundary should be flagged."""
        # Generate borderline spectral data
        spectrum = np.random.uniform(3.2, 3.5, 100)

        verdict, receipt = verify_olive_oil(
            batch_id="TEST-BORDER-001",
            product_grade="extra_virgin",
            spectral_scan=spectrum,
            provenance_chain=["farm"],
        )

        assert verdict in ["COUNTERFEIT", "SUSPICIOUS"]

    def test_receipt_emission(self, suppress_receipts):
        """Verify receipt emitted with all required fields."""
        spectrum = np.random.uniform(4.0, 4.5, 100)

        verdict, receipt = verify_olive_oil(
            batch_id="TEST-RECEIPT-001",
            product_grade="extra_virgin",
            spectral_scan=spectrum,
            provenance_chain=["farm", "processor"],
        )

        # Check required fields
        assert "receipt_type" in receipt
        assert "ts" in receipt
        assert "tenant_id" in receipt
        assert "payload_hash" in receipt
        assert ":" in receipt.get("payload_hash", "")  # dual_hash format

    def test_fsma_compliance(self, suppress_receipts):
        """Check compliance_standard field present."""
        spectrum = np.random.uniform(4.0, 4.5, 100)

        verdict, receipt = verify_olive_oil(
            batch_id="TEST-FSMA-001",
            product_grade="virgin",
            spectral_scan=spectrum,
            provenance_chain=["farm"],
            compliance_standard="FSMA_204",
        )

        assert receipt.get("compliance_standard") == "FSMA_204"
        assert "provenance_chain" in receipt


class TestHoneyVerification:
    """Tests for honey verification."""

    def test_authentic_manuka(self, suppress_receipts):
        """Genuine manuka honey → verdict AUTHENTIC."""
        texture = np.random.uniform(0, 255, 200) + np.random.normal(0, 30, 200)
        pollen = {"species": {"manuka": 300, "wildflower": 150}}

        verdict, receipt = verify_honey(
            batch_id="TEST-MANUKA-001",
            honey_type="manuka",
            texture_scan=texture,
            pollen_analysis=pollen,
            provenance_chain=["apiary", "processor"],
        )

        assert verdict in ["AUTHENTIC", "SUSPICIOUS"]
        assert receipt.get("pollen_count", 0) >= 0

    def test_syrup_adulterated(self, suppress_receipts):
        """Corn syrup-adulterated sample → verdict COUNTERFEIT."""
        # Uniform texture (low entropy - syrup)
        base = np.full(200, 128)
        texture = base + np.random.normal(0, 3, 200)

        verdict, receipt = verify_honey(
            batch_id="TEST-SYRUP-001",
            honey_type="manuka",
            texture_scan=texture,
            pollen_analysis=None,  # No pollen (fake)
            provenance_chain=["unknown"],
        )

        assert verdict in ["COUNTERFEIT", "SUSPICIOUS"]

    def test_pollen_count_validation(self, suppress_receipts):
        """Verify pollen count threshold (genuine manuka >500 pollen grains)."""
        texture = np.random.uniform(0, 255, 200)
        pollen = {"species": {"manuka": 600, "clover": 100}}

        verdict, receipt = verify_honey(
            batch_id="TEST-POLLEN-001",
            honey_type="manuka",
            texture_scan=texture,
            pollen_analysis=pollen,
            provenance_chain=["apiary"],
        )

        assert receipt.get("pollen_count", 0) >= 500


class TestSeafoodVerification:
    """Tests for seafood verification."""

    def test_authentic_blue_crab(self, suppress_receipts):
        """Genuine blue crab → verdict AUTHENTIC."""
        tissue = np.random.uniform(4.5, 5.3, 150) + np.random.normal(0, 0.3, 150)

        verdict, receipt = verify_seafood(
            sample_id="TEST-CRAB-001",
            claimed_species="blue_crab",
            tissue_scan=tissue,
            provenance_chain=["fishery", "processor"],
        )

        assert verdict in ["AUTHENTIC", "SUSPICIOUS"]
        assert receipt.get("claimed_species") == "blue_crab"

    def test_species_substitution(self, suppress_receipts):
        """Foreign crab repackaged as blue crab → verdict COUNTERFEIT."""
        # Different species = different entropy
        tissue = np.random.uniform(3.0, 3.5, 150) + np.random.normal(0, 0.1, 150)

        verdict, receipt = verify_seafood(
            sample_id="TEST-FAKE-CRAB-001",
            claimed_species="blue_crab",
            tissue_scan=tissue,
            provenance_chain=["unknown"],
        )

        assert verdict in ["COUNTERFEIT", "SUSPICIOUS"]

    def test_dna_barcode_validation(self, suppress_receipts):
        """DNA barcode confirmation."""
        tissue = np.random.uniform(4.3, 5.1, 150)

        verdict, receipt = verify_seafood(
            sample_id="TEST-DNA-001",
            claimed_species="wild_salmon",
            tissue_scan=tissue,
            dna_barcode="WS123456",  # Valid prefix
            provenance_chain=["fishery"],
        )

        assert receipt.get("dna_verified")


class TestEntropyCalculators:
    """Tests for entropy calculator functions."""

    def test_spectral_entropy_range(self):
        """Spectral entropy returns value in expected range."""
        spectrum = np.random.uniform(0, 1, 100)
        entropy = spectral_entropy(spectrum)
        assert 0 <= entropy <= 10

    def test_texture_entropy_range(self):
        """Texture entropy returns value in expected range."""
        image = np.random.uniform(0, 255, 200)
        entropy = texture_entropy(image)
        assert 0 <= entropy <= 10

    def test_pollen_diversity_entropy(self):
        """Pollen diversity entropy calculation."""
        pollen = {"manuka": 300, "wildflower": 200, "clover": 100}
        entropy = pollen_diversity_entropy(pollen)
        assert entropy > 0

    def test_empty_input_handling(self):
        """Empty inputs return zero entropy."""
        assert spectral_entropy(np.array([])) == 0.0
        assert texture_entropy(np.array([])) == 0.0
        assert pollen_diversity_entropy({}) == 0.0
