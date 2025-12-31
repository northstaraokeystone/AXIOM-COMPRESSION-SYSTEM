"""request.py - Pydantic request models for API endpoints."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class AerospaceRequest(BaseModel):
    """Request for aerospace verification."""

    component_id: str = Field(..., description="Unique component identifier", example="CAP-12345")
    component_type: str = Field(..., description="Component type", example="capacitor")
    sensor_data: Dict = Field(
        ...,
        description="Sensor measurements",
        example={"entropy": 0.85, "visual_hash": "abc123", "electrical_hash": "def456"},
    )
    provenance_chain: List[str] = Field(
        default=[],
        description="Chain of custody",
        example=["manufacturer", "distributor", "integrator"],
    )
    tenant_id: Optional[str] = Field(default="default", description="Tenant identifier")


class OliveOilRequest(BaseModel):
    """Request for olive oil verification."""

    batch_id: str = Field(..., description="Unique batch identifier", example="EVOO-2025-001")
    product_grade: Literal["extra_virgin", "virgin", "pure"] = Field(
        ..., description="Olive oil grade"
    )
    spectral_scan: List[float] = Field(
        ...,
        description="NIR/hyperspectral wavelength readings",
        example=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    provenance_chain: List[str] = Field(
        default=[],
        description="Chain of custody (farm → processor → bottler → retailer)",
        example=["italian_farm", "processor", "bottler", "retailer"],
    )
    compliance_standard: Literal["FSMA_204", "CFR_Part_11"] = Field(
        default="FSMA_204", description="Compliance standard"
    )


class HoneyRequest(BaseModel):
    """Request for honey verification."""

    batch_id: str = Field(..., description="Unique batch identifier", example="HONEY-2025-001")
    honey_type: Literal["manuka", "wildflower", "clover"] = Field(
        ..., description="Honey type"
    )
    texture_scan: List[float] = Field(
        ...,
        description="Optical microscopy or ultrasound imaging data",
        example=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    pollen_analysis: Optional[Dict] = Field(
        default=None,
        description="Pollen count and species distribution",
        example={"pollen_count": 500, "species": {"manuka": 300, "wildflower": 200}},
    )
    provenance_chain: List[str] = Field(default=[], description="Chain of custody")
    compliance_standard: Literal["FSMA_204", "CFR_Part_11"] = Field(
        default="FSMA_204", description="Compliance standard"
    )


class SeafoodRequest(BaseModel):
    """Request for seafood verification."""

    sample_id: str = Field(..., description="Unique sample identifier", example="CRAB-2025-001")
    claimed_species: Literal["blue_crab", "wild_salmon", "cod", "tuna"] = Field(
        ..., description="Claimed species"
    )
    tissue_scan: List[float] = Field(
        ...,
        description="Optical or ultrasound tissue scan data",
        example=[0.4, 0.5, 0.6, 0.7, 0.8],
    )
    dna_barcode: Optional[str] = Field(
        default=None,
        description="DNA barcode sequence (if available)",
        example="BC123456",
    )
    provenance_chain: List[str] = Field(default=[], description="Chain of custody")
    compliance_standard: Literal["FSMA_204", "FSMA_204_high_risk", "CFR_Part_11"] = Field(
        default="FSMA_204", description="Compliance standard"
    )


class GLP1Request(BaseModel):
    """Request for GLP-1 pen verification (Ozempic, Wegovy)."""

    serial_number: str = Field(
        ..., description="Pen serial number", example="OZP-TEST-12345"
    )
    device_type: Literal["ozempic_0.5mg", "ozempic_1mg", "wegovy_1.7mg", "wegovy_2.4mg"] = Field(
        ..., description="Device type and dosage"
    )
    fill_measurements: Dict = Field(
        ...,
        description="Fill level and compression measurements",
        example={"fill_level": 0.95, "compression": 0.88, "uniformity_score": 0.92},
    )
    lot_number: str = Field(
        ..., description="Manufacturer lot number", example="OZP-2025-00001"
    )
    provenance_chain: List[str] = Field(
        default=[],
        description="Chain of custody (manufacturer → distributor → pharmacy)",
        example=["novo_nordisk", "mckesson", "cvs"],
    )
    compliance_standard: Literal["21_CFR_Part_820_QSR", "ISO_13485"] = Field(
        default="21_CFR_Part_820_QSR", description="Compliance standard"
    )


class BotoxRequest(BaseModel):
    """Request for Botox vial verification."""

    vial_id: str = Field(..., description="Vial identifier", example="BTX-2025-001")
    unit_count: Literal[50, 100, 200] = Field(..., description="Unit count")
    surface_scan: List[float] = Field(
        ...,
        description="Laser speckle or optical surface scan data",
        example=[0.3, 0.4, 0.5, 0.6, 0.7],
    )
    solution_analysis: Optional[Dict] = Field(
        default=None,
        description="Solution concentration and particulate data",
        example={"concentration": 0.95, "particulate_count": 5},
    )
    provenance_chain: List[str] = Field(default=[], description="Chain of custody")
    compliance_standard: Literal["21_CFR_Part_820_QSR", "ISO_13485"] = Field(
        default="21_CFR_Part_820_QSR", description="Compliance standard"
    )


class CancerDrugRequest(BaseModel):
    """Request for cancer drug verification."""

    drug_id: str = Field(..., description="Drug identifier/lot number", example="IMFINZI-2025-001")
    drug_name: Literal["imfinzi_120mg", "keytruda_100mg", "opdivo_240mg"] = Field(
        ..., description="Drug name and dosage"
    )
    raman_map: List[float] = Field(
        ...,
        description="3D Raman spectroscopy spatial mapping data",
        example=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    tablet_structure: Optional[Dict] = Field(
        default=None,
        description="Micro-CT or cross-section analysis",
        example={"coating_thickness": 0.12, "core_density": 0.95},
    )
    provenance_chain: List[str] = Field(default=[], description="Chain of custody")
    compliance_standard: Literal["21_CFR_Part_820_QSR", "ISO_13485"] = Field(
        default="21_CFR_Part_820_QSR", description="Compliance standard"
    )
