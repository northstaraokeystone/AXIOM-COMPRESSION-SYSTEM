"""server.py - FastAPI REST API for SpaceProof Multi-Domain Verification.

Exposes verification endpoints for:
- Aerospace (hardware counterfeit detection)
- Food (adulteration detection)
- Medical (counterfeit device/drug detection)

Target: Unblock Jay Lewis test bench integration.
Response time: <500ms per verification (99th percentile)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .routes import aerospace_router, food_router, medical_router
from .models.response import HealthResponse

# Create FastAPI app
app = FastAPI(
    title="SpaceProof Multi-Domain Verification API",
    description="""
    Universal entropy-based verification API for:
    - **Aerospace**: Hardware counterfeit detection (Jay Lewis use case)
    - **Food**: Olive oil, honey, seafood adulteration detection
    - **Medical**: GLP-1, Botox, cancer drug counterfeit detection

    All verifications emit cryptographic receipts for audit trail.

    Detection targets:
    - Aerospace: 100% counterfeit detection
    - Food: ≥99.9% recall, <1% false positive
    - Medical: ≥99.9% recall (life-critical), <0.5% false positive
    """,
    version="2.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

# Enable CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(aerospace_router)
app.include_router(food_router)
app.include_router(medical_router)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and available domains.

    Validates:
    - Receipt ledger is writable
    - Configuration files are readable
    - Domain modules are loadable
    """
    # Check config directory
    config_dir = Path(__file__).parent.parent / "config"
    config_readable = (
        (config_dir / "food_baselines.json").exists() and
        (config_dir / "medical_baselines.json").exists()
    )

    # Check ledger (receipts.jsonl)
    ledger_path = Path(__file__).parent.parent / "receipts.jsonl"
    try:
        # Try to touch the file
        ledger_path.touch(exist_ok=True)
        ledger_writable = True
    except Exception:
        ledger_writable = False

    # Check domain imports
    try:
        from spaceproof.food import verify_olive_oil
        from spaceproof.medical import verify_glp1_pen
        from spaceproof.detect import detect_hardware_fraud
        domains_available = ["aerospace", "food", "medical"]
    except ImportError as e:
        domains_available = ["aerospace"]  # Minimal fallback

    # Determine overall health
    if config_readable and ledger_writable and len(domains_available) == 3:
        status = "healthy"
    elif len(domains_available) >= 1:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        domains=domains_available,
        version="2.0.0",
        ledger_writable=ledger_writable,
        config_readable=config_readable,
    )


@app.get("/api/v1/receipt/{receipt_id}")
async def get_receipt(receipt_id: str) -> dict:
    """Retrieve a verification receipt by ID.

    Searches receipts.jsonl for matching payload_hash.
    """
    import json

    ledger_path = Path(__file__).parent.parent / "receipts.jsonl"

    if not ledger_path.exists():
        raise HTTPException(status_code=404, detail=f"Receipt not found: {receipt_id}")

    try:
        with open(ledger_path) as f:
            for line in f:
                if line.strip():
                    receipt = json.loads(line)
                    if receipt.get("payload_hash", "").startswith(receipt_id):
                        return receipt
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading ledger: {e}")

    raise HTTPException(status_code=404, detail=f"Receipt not found: {receipt_id}")


@app.get("/")
async def root():
    """API root - redirect to docs."""
    return {
        "message": "SpaceProof Multi-Domain Verification API",
        "version": "2.0.0",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health",
        "domains": {
            "aerospace": "/api/v1/verify/aerospace",
            "food": {
                "olive_oil": "/api/v1/verify/food/olive-oil",
                "honey": "/api/v1/verify/food/honey",
                "seafood": "/api/v1/verify/food/seafood",
            },
            "medical": {
                "glp1": "/api/v1/verify/medical/glp1",
                "botox": "/api/v1/verify/medical/botox",
                "cancer_drug": "/api/v1/verify/medical/cancer-drug",
            },
        },
    }


# Entry point for hypercorn
def create_app():
    """Factory function for ASGI server."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
