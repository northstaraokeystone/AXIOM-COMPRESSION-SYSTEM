"""test_api_endpoints.py - REST API endpoint tests."""

import pytest

# Try to import fastapi
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    TestClient = None

# Import app for testing
try:
    from api.server import app
    HAS_API = True
except ImportError:
    HAS_API = False
    app = None


pytestmark = pytest.mark.skipif(
    not HAS_FASTAPI or not HAS_API,
    reason="FastAPI or API module not available"
)


@pytest.fixture
def client():
    """Create test client."""
    if not HAS_FASTAPI or not HAS_API:
        pytest.skip("API dependencies not available")
    return TestClient(app)


@pytest.fixture
def suppress_receipts():
    """Suppress receipt output during tests."""
    yield


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_200(self, client):
        """GET /api/v1/health → 200 response."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "domains" in data
        assert "version" in data

    def test_health_returns_domains(self, client):
        """Health check includes available domains."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "aerospace" in data.get("domains", [])


class TestAerospaceEndpoint:
    """Tests for aerospace verification endpoint."""

    def test_aerospace_endpoint_200(self, client, suppress_receipts):
        """POST to /api/v1/verify/aerospace → 200 response."""
        payload = {
            "component_id": "CAP-TEST-001",
            "component_type": "capacitor",
            "sensor_data": {"entropy": 0.85},
            "provenance_chain": ["manufacturer", "distributor"],
        }

        response = client.post("/api/v1/verify/aerospace", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "verdict" in data
        assert "receipt" in data


class TestFoodEndpoints:
    """Tests for food verification endpoints."""

    def test_olive_oil_endpoint_200(self, client, suppress_receipts):
        """POST to /api/v1/verify/food/olive-oil → 200 response."""
        payload = {
            "batch_id": "OO-TEST-001",
            "product_grade": "extra_virgin",
            "spectral_scan": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            "provenance_chain": ["farm", "processor"],
            "compliance_standard": "FSMA_204",
        }

        response = client.post("/api/v1/verify/food/olive-oil", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "verdict" in data
        assert data.get("domain") == "food"

    def test_honey_endpoint_200(self, client, suppress_receipts):
        """POST to /api/v1/verify/food/honey → 200 response."""
        payload = {
            "batch_id": "HN-TEST-001",
            "honey_type": "manuka",
            "texture_scan": [0.5, 0.6, 0.7, 0.8, 0.9] * 20,
            "provenance_chain": ["apiary"],
        }

        response = client.post("/api/v1/verify/food/honey", json=payload)
        assert response.status_code == 200

    def test_seafood_endpoint_200(self, client, suppress_receipts):
        """POST to /api/v1/verify/food/seafood → 200 response."""
        payload = {
            "sample_id": "SF-TEST-001",
            "claimed_species": "blue_crab",
            "tissue_scan": [0.4, 0.5, 0.6, 0.7, 0.8] * 20,
            "provenance_chain": ["fishery"],
        }

        response = client.post("/api/v1/verify/food/seafood", json=payload)
        assert response.status_code == 200


class TestMedicalEndpoints:
    """Tests for medical verification endpoints."""

    def test_glp1_endpoint_200(self, client, suppress_receipts):
        """POST to /api/v1/verify/medical/glp1 → 200 response."""
        payload = {
            "serial_number": "OZP-TEST-12345",
            "device_type": "ozempic_0.5mg",
            "fill_measurements": {"fill_level": 0.95, "compression": 0.88},
            "lot_number": "OZP-2025-00001",
            "provenance_chain": ["novo_nordisk", "mckesson", "cvs"],
        }

        response = client.post("/api/v1/verify/medical/glp1", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data.get("risk_level") == "CRITICAL"

    def test_botox_endpoint_200(self, client, suppress_receipts):
        """POST to /api/v1/verify/medical/botox → 200 response."""
        payload = {
            "vial_id": "BTX-TEST-001",
            "unit_count": 100,
            "surface_scan": [0.3, 0.4, 0.5, 0.6, 0.7] * 20,
            "provenance_chain": ["allergan"],
        }

        response = client.post("/api/v1/verify/medical/botox", json=payload)
        assert response.status_code == 200

    def test_cancer_drug_endpoint_200(self, client, suppress_receipts):
        """POST to /api/v1/verify/medical/cancer-drug → 200 response."""
        payload = {
            "drug_id": "IMFINZI-TEST-001",
            "drug_name": "imfinzi_120mg",
            "raman_map": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 20,
            "provenance_chain": ["astrazeneca"],
        }

        response = client.post("/api/v1/verify/medical/cancer-drug", json=payload)
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_request_400(self, client):
        """POST with malformed JSON → 400 or 422 Bad Request."""
        response = client.post(
            "/api/v1/verify/food/olive-oil",
            json={"invalid": "data"},  # Missing required fields
        )
        assert response.status_code in [400, 422]

    def test_invalid_product_grade(self, client, suppress_receipts):
        """Invalid product_grade → 422 Unprocessable Entity."""
        payload = {
            "batch_id": "OO-BAD-001",
            "product_grade": "super_extra_virgin",  # Invalid
            "spectral_scan": [0.1, 0.2, 0.3],
            "provenance_chain": [],
        }

        response = client.post("/api/v1/verify/food/olive-oil", json=payload)
        assert response.status_code == 422
