"""food.py - Food verification endpoints."""

from datetime import datetime
from fastapi import APIRouter, HTTPException
import numpy as np

from spaceproof.core import emit_receipt
from spaceproof.food.olive_oil import verify_olive_oil
from spaceproof.food.honey import verify_honey
from spaceproof.food.seafood import verify_seafood

from ..models.request import OliveOilRequest, HoneyRequest, SeafoodRequest
from ..models.response import VerificationResponse, EntropyAnalysisResponse

router = APIRouter(prefix="/api/v1/verify/food", tags=["food"])


@router.post("/olive-oil", response_model=VerificationResponse)
async def verify_olive_oil_endpoint(request: OliveOilRequest) -> VerificationResponse:
    """Verify olive oil authenticity via spectral entropy.

    Detects adulteration in olive oil by analyzing:
    - Spectral entropy from NIR/hyperspectral scans
    - Polyphenol/fatty acid gradients

    Genuine oils: HIGH spectral entropy (3.8-4.6)
    Adulterants: LOW spectral entropy (<3.2) due to homogenization
    """
    try:
        verdict, receipt = verify_olive_oil(
            batch_id=request.batch_id,
            product_grade=request.product_grade,
            spectral_scan=np.array(request.spectral_scan),
            provenance_chain=request.provenance_chain,
            compliance_standard=request.compliance_standard,
        )

        entropy_analysis = EntropyAnalysisResponse(
            measured_entropy=receipt.get("measured_entropy", 0.0),
            baseline_min=receipt.get("baseline_min", 3.8),
            baseline_max=receipt.get("baseline_max", 4.6),
            deviation_pct=receipt.get("deviation_pct", 0.0),
            flags=receipt.get("flags", []),
            comparison=f"Spectral entropy for {request.product_grade}: {receipt.get('measured_entropy', 0):.3f}",
        )

        return VerificationResponse(
            verdict=verdict,
            confidence=receipt.get("confidence", 0.8),
            risk_score=1.0 - receipt.get("confidence", 0.8),
            entropy_analysis=entropy_analysis,
            receipt=receipt,
            receipt_id=receipt.get("payload_hash", f"olive-oil-{request.batch_id}"),
            domain="food",
            item_id=request.batch_id,
        )

    except Exception as e:
        error_receipt = emit_receipt(
            "verification_error",
            {
                "tenant_id": "spaceproof",
                "domain": "food",
                "product_type": "olive_oil",
                "item_id": request.batch_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/honey", response_model=VerificationResponse)
async def verify_honey_endpoint(request: HoneyRequest) -> VerificationResponse:
    """Verify honey authenticity via texture entropy + pollen analysis.

    Detects syrup adulteration in honey by analyzing:
    - Microstructural texture entropy
    - Pollen distribution entropy

    Genuine honey: HIGH texture entropy (4.2-5.1)
    Syrup-adulterated: LOW texture entropy (<3.5)
    """
    try:
        verdict, receipt = verify_honey(
            batch_id=request.batch_id,
            honey_type=request.honey_type,
            texture_scan=np.array(request.texture_scan),
            pollen_analysis=request.pollen_analysis,
            provenance_chain=request.provenance_chain,
            compliance_standard=request.compliance_standard,
        )

        entropy_analysis = EntropyAnalysisResponse(
            measured_entropy=receipt.get("measured_entropy", 0.0),
            baseline_min=receipt.get("baseline_min", 4.2),
            baseline_max=receipt.get("baseline_max", 5.1),
            deviation_pct=receipt.get("deviation_pct", 0.0),
            flags=receipt.get("flags", []),
            comparison=f"Texture entropy for {request.honey_type}: {receipt.get('measured_entropy', 0):.3f}",
        )

        return VerificationResponse(
            verdict=verdict,
            confidence=receipt.get("confidence", 0.8),
            risk_score=1.0 - receipt.get("confidence", 0.8),
            entropy_analysis=entropy_analysis,
            receipt=receipt,
            receipt_id=receipt.get("payload_hash", f"honey-{request.batch_id}"),
            domain="food",
            item_id=request.batch_id,
        )

    except Exception as e:
        error_receipt = emit_receipt(
            "verification_error",
            {
                "tenant_id": "spaceproof",
                "domain": "food",
                "product_type": "honey",
                "item_id": request.batch_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seafood", response_model=VerificationResponse)
async def verify_seafood_endpoint(request: SeafoodRequest) -> VerificationResponse:
    """Verify seafood species via tissue entropy analysis.

    Detects species substitution by analyzing:
    - Tissue entropy signature (species-specific)
    - Optional DNA barcode confirmation

    Each species has unique tissue entropy range.
    """
    try:
        verdict, receipt = verify_seafood(
            sample_id=request.sample_id,
            claimed_species=request.claimed_species,
            tissue_scan=np.array(request.tissue_scan),
            dna_barcode=request.dna_barcode,
            provenance_chain=request.provenance_chain,
            compliance_standard=request.compliance_standard,
        )

        entropy_analysis = EntropyAnalysisResponse(
            measured_entropy=receipt.get("measured_entropy", 0.0),
            baseline_min=receipt.get("baseline_min", 4.5),
            baseline_max=receipt.get("baseline_max", 5.3),
            deviation_pct=receipt.get("deviation_pct", 0.0),
            flags=receipt.get("flags", []),
            comparison=f"Tissue entropy for {request.claimed_species}: {receipt.get('measured_entropy', 0):.3f}",
        )

        return VerificationResponse(
            verdict=verdict,
            confidence=receipt.get("confidence", 0.8),
            risk_score=1.0 - receipt.get("confidence", 0.8),
            entropy_analysis=entropy_analysis,
            receipt=receipt,
            receipt_id=receipt.get("payload_hash", f"seafood-{request.sample_id}"),
            domain="food",
            item_id=request.sample_id,
        )

    except Exception as e:
        error_receipt = emit_receipt(
            "verification_error",
            {
                "tenant_id": "spaceproof",
                "domain": "food",
                "product_type": "seafood",
                "item_id": request.sample_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))
