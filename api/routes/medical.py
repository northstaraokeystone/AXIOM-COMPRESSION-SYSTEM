"""medical.py - Medical verification endpoints.

All medical verifications are risk_level=CRITICAL.
Detection target: ≥99.9% recall (life-threatening if missed).
"""

from fastapi import APIRouter, HTTPException
import numpy as np

from spaceproof.core import emit_receipt
from spaceproof.medical.glp1 import verify_glp1_pen
from spaceproof.medical.botox import verify_botox_vial
from spaceproof.medical.cancer_drugs import verify_cancer_drug

from ..models.request import GLP1Request, BotoxRequest, CancerDrugRequest
from ..models.response import VerificationResponse, EntropyAnalysisResponse

router = APIRouter(prefix="/api/v1/verify/medical", tags=["medical"])


@router.post("/glp1", response_model=VerificationResponse)
async def verify_glp1_endpoint(request: GLP1Request) -> VerificationResponse:
    """Verify GLP-1 pen authenticity (Ozempic, Wegovy).

    Detects counterfeit GLP-1 pens by analyzing:
    - Fill variance entropy
    - Lot number format validation

    Genuine pens: Controlled entropy (2.8-3.4)
    Counterfeits: Abnormal uniformity (<2.5) OR material deviations (>3.8)

    CRITICAL: Detection target ≥99.9% recall (life-threatening if missed)
    """
    try:
        verdict, receipt = verify_glp1_pen(
            serial_number=request.serial_number,
            device_type=request.device_type,
            fill_imaging=request.fill_measurements,
            lot_number=request.lot_number,
            provenance_chain=request.provenance_chain,
            compliance_standard=request.compliance_standard,
        )

        entropy_analysis = EntropyAnalysisResponse(
            measured_entropy=receipt.get("measured_entropy", 0.0),
            baseline_min=receipt.get("baseline_min", 2.8),
            baseline_max=receipt.get("baseline_max", 3.4),
            deviation_pct=receipt.get("deviation_pct", 0.0),
            flags=receipt.get("flags", []),
            comparison=f"Fill entropy for {request.device_type}: {receipt.get('measured_entropy', 0):.3f}",
        )

        return VerificationResponse(
            verdict=verdict,
            confidence=receipt.get("confidence", 0.8),
            risk_score=1.0 - receipt.get("confidence", 0.8),
            entropy_analysis=entropy_analysis,
            receipt=receipt,
            receipt_id=receipt.get("payload_hash", f"glp1-{request.serial_number}"),
            domain="medical",
            item_id=request.serial_number,
            risk_level="CRITICAL",
        )

    except Exception as e:
        emit_receipt(
            "verification_error",
            {
                "tenant_id": "spaceproof",
                "domain": "medical",
                "product_type": "glp1_pen",
                "item_id": request.serial_number,
                "error": str(e),
                "risk_level": "CRITICAL",
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/botox", response_model=VerificationResponse)
async def verify_botox_endpoint(request: BotoxRequest) -> VerificationResponse:
    """Verify Botox vial authenticity.

    Detects counterfeit Botox vials by analyzing:
    - Surface topography entropy
    - Solution concentration entropy

    Genuine vials: Controlled entropy (3.2-3.9)
    Counterfeits: Cheap vial packaging with different entropy

    CRITICAL: Detection target ≥99.9% recall (botulism risk if fake)
    """
    try:
        verdict, receipt = verify_botox_vial(
            vial_id=request.vial_id,
            unit_count=request.unit_count,
            surface_scan=np.array(request.surface_scan),
            solution_analysis=request.solution_analysis,
            provenance_chain=request.provenance_chain,
            compliance_standard=request.compliance_standard,
        )

        entropy_analysis = EntropyAnalysisResponse(
            measured_entropy=receipt.get("measured_entropy", 0.0),
            baseline_min=receipt.get("baseline_min", 3.2),
            baseline_max=receipt.get("baseline_max", 3.9),
            deviation_pct=receipt.get("deviation_pct", 0.0),
            flags=receipt.get("flags", []),
            comparison=f"Surface entropy for {request.unit_count}U vial: {receipt.get('measured_entropy', 0):.3f}",
        )

        return VerificationResponse(
            verdict=verdict,
            confidence=receipt.get("confidence", 0.8),
            risk_score=1.0 - receipt.get("confidence", 0.8),
            entropy_analysis=entropy_analysis,
            receipt=receipt,
            receipt_id=receipt.get("payload_hash", f"botox-{request.vial_id}"),
            domain="medical",
            item_id=request.vial_id,
            risk_level="CRITICAL",
        )

    except Exception as e:
        emit_receipt(
            "verification_error",
            {
                "tenant_id": "spaceproof",
                "domain": "medical",
                "product_type": "botox_vial",
                "item_id": request.vial_id,
                "error": str(e),
                "risk_level": "CRITICAL",
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancer-drug", response_model=VerificationResponse)
async def verify_cancer_drug_endpoint(request: CancerDrugRequest) -> VerificationResponse:
    """Verify cancer drug authenticity via API distribution entropy.

    Detects counterfeit cancer drugs by analyzing:
    - API (active pharmaceutical ingredient) distribution via Raman
    - No-API detection (immediate rejection)

    Genuine tablets: API distribution entropy (4.1-4.9)
    Counterfeits: No API (near-zero entropy) OR wrong distribution

    CRITICAL: Detection target ≥99.9% recall (treatment failure = death)
    """
    try:
        verdict, receipt = verify_cancer_drug(
            drug_id=request.drug_id,
            drug_name=request.drug_name,
            raman_map=np.array(request.raman_map),
            tablet_structure=request.tablet_structure,
            provenance_chain=request.provenance_chain,
            compliance_standard=request.compliance_standard,
        )

        entropy_analysis = EntropyAnalysisResponse(
            measured_entropy=receipt.get("measured_entropy", 0.0),
            baseline_min=receipt.get("baseline_min", 4.1),
            baseline_max=receipt.get("baseline_max", 4.9),
            deviation_pct=receipt.get("deviation_pct", 0.0),
            flags=receipt.get("flags", []),
            comparison=f"API distribution entropy for {request.drug_name}: {receipt.get('measured_entropy', 0):.3f}",
        )

        return VerificationResponse(
            verdict=verdict,
            confidence=receipt.get("confidence", 0.8),
            risk_score=1.0 - receipt.get("confidence", 0.8),
            entropy_analysis=entropy_analysis,
            receipt=receipt,
            receipt_id=receipt.get("payload_hash", f"cancer-drug-{request.drug_id}"),
            domain="medical",
            item_id=request.drug_id,
            risk_level="CRITICAL",
        )

    except Exception as e:
        emit_receipt(
            "verification_error",
            {
                "tenant_id": "spaceproof",
                "domain": "medical",
                "product_type": "cancer_drug",
                "item_id": request.drug_id,
                "error": str(e),
                "risk_level": "CRITICAL",
            },
        )
        raise HTTPException(status_code=500, detail=str(e))
