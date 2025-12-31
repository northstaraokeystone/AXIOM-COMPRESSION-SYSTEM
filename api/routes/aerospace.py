"""aerospace.py - Aerospace verification endpoint.

Jay Lewis's primary use case: Hardware counterfeit detection.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException

from spaceproof.core import emit_receipt
from spaceproof.detect import detect_hardware_fraud

from ..models.request import AerospaceRequest
from ..models.response import VerificationResponse, EntropyAnalysisResponse

router = APIRouter(prefix="/api/v1/verify", tags=["aerospace"])


@router.post("/aerospace", response_model=VerificationResponse)
async def verify_aerospace_component(request: AerospaceRequest) -> VerificationResponse:
    """Verify aerospace component authenticity.

    Uses entropy-based counterfeit detection to analyze:
    - Component entropy signature
    - Rework accumulation
    - Supply chain provenance

    This is Jay Lewis's primary use case for test bench integration.
    """
    try:
        # Prepare component data for detect_hardware_fraud
        component = {
            "id": request.component_id,
            "component_id": request.component_id,
            "component_type": request.component_type,
            "sensor_data": request.sensor_data,
            "provenance_chain": [
                {"receipt_type": "handoff", "entity": entity}
                for entity in request.provenance_chain
            ],
            "manufacturer_baseline": {
                "entropy": request.sensor_data.get("baseline_entropy", 0.30),
                "manufacturer": request.sensor_data.get("manufacturer", "Unknown"),
            },
        }

        # Add visual/electrical hashes if provided
        if "visual_hash" in request.sensor_data:
            component["visual_hash"] = request.sensor_data["visual_hash"]
        if "electrical_hash" in request.sensor_data:
            component["electrical_hash"] = request.sensor_data["electrical_hash"]

        # Run hardware fraud detection
        result = detect_hardware_fraud(
            component=component,
            baseline=component.get("manufacturer_baseline"),
            rework_history=request.sensor_data.get("rework_history", []),
            provenance_chain=component.get("provenance_chain", []),
        )

        # Map result to verdict
        if result.get("reject", False):
            verdict = "COUNTERFEIT"
        elif result.get("risk_score", 0) > 0.5:
            verdict = "SUSPICIOUS"
        else:
            verdict = "AUTHENTIC"

        # Build entropy analysis
        counterfeit_data = result.get("counterfeit", {})
        measured_entropy = counterfeit_data.get("entropy", 0.30)
        baseline_min = 0.0
        baseline_max = 0.35

        if measured_entropy > baseline_max:
            deviation_pct = ((measured_entropy - baseline_max) / baseline_max) * 100
        elif measured_entropy < baseline_min:
            deviation_pct = ((baseline_min - measured_entropy) / (baseline_max or 1)) * 100
        else:
            deviation_pct = 0.0

        entropy_analysis = EntropyAnalysisResponse(
            measured_entropy=measured_entropy,
            baseline_min=baseline_min,
            baseline_max=baseline_max,
            deviation_pct=deviation_pct,
            flags=result.get("reject_reasons", []),
            comparison=f"Component entropy {measured_entropy:.3f}, classification: {counterfeit_data.get('classification', 'unknown')}",
        )

        # Build receipt
        receipt = {
            "receipt_type": "aerospace_verification",
            "ts": datetime.utcnow().isoformat() + "Z",
            "tenant_id": request.tenant_id or "spaceproof",
            "component_id": request.component_id,
            "component_type": request.component_type,
            "verdict": verdict,
            "risk_score": result.get("risk_score", 0.0),
            "counterfeit_analysis": counterfeit_data,
            "rework_analysis": result.get("rework", {}),
            "provenance_analysis": result.get("provenance", {}),
        }

        return VerificationResponse(
            verdict=verdict,
            confidence=counterfeit_data.get("confidence", 0.8),
            risk_score=result.get("risk_score", 0.0),
            entropy_analysis=entropy_analysis,
            receipt=receipt,
            receipt_id=receipt.get("payload_hash", f"aerospace-{request.component_id}"),
            domain="aerospace",
            item_id=request.component_id,
        )

    except Exception as e:
        # Emit error receipt
        emit_receipt(
            "verification_error",
            {
                "tenant_id": request.tenant_id or "spaceproof",
                "domain": "aerospace",
                "item_id": request.component_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise HTTPException(status_code=500, detail=str(e))
