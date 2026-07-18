"""FHIR R4 compliance models and standard LOINC mappings for lab values."""

from typing import List, Optional
from pydantic import BaseModel, Field

# Standard LOINC (Logical Observation Identifiers Names and Codes) dictionary
# Maps common test names to their international standardized LOINC code & display name.
LOINC_MAP = {
    "hemoglobin": {"code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood"},
    "hb": {"code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood"},
    "haemoglobin": {"code": "718-7", "display": "Hemoglobin [Mass/volume] in Blood"},
    "rbc": {"code": "26453-1", "display": "Erythrocytes [#/volume] in Blood"},
    "rbc count": {"code": "26453-1", "display": "Erythrocytes [#/volume] in Blood"},
    "red blood cell count": {"code": "26453-1", "display": "Erythrocytes [#/volume] in Blood"},
    "wbc": {"code": "6690-2", "display": "Leukocytes [#/volume] in Blood"},
    "wbc count": {"code": "6690-2", "display": "Leukocytes [#/volume] in Blood"},
    "white blood cell count": {"code": "6690-2", "display": "Leukocytes [#/volume] in Blood"},
    "total wbc count": {"code": "6690-2", "display": "Leukocytes [#/volume] in Blood"},
    "platelet count": {"code": "777-3", "display": "Platelets [#/volume] in Blood"},
    "platelets": {"code": "777-3", "display": "Platelets [#/volume] in Blood"},
    "hematocrit": {"code": "4544-3", "display": "Hematocrit [Volume Fraction] of Blood"},
    "hct": {"code": "4544-3", "display": "Hematocrit [Volume Fraction] of Blood"},
    "pcv": {"code": "4544-3", "display": "Hematocrit [Volume Fraction] of Blood"},
    "packed cell volume": {"code": "4544-3", "display": "Hematocrit [Volume Fraction] of Blood"},
    
    "hba1c": {"code": "4548-4", "display": "Hemoglobin A1c/Hemoglobin.total in Blood"},
    "glycated hemoglobin": {"code": "4548-4", "display": "Hemoglobin A1c/Hemoglobin.total in Blood"},
    "fasting blood sugar": {"code": "1558-6", "display": "Glucose [Mass/volume] in Blood --fasting"},
    "fasting glucose": {"code": "1558-6", "display": "Glucose [Mass/volume] in Blood --fasting"},
    "fbs": {"code": "1558-6", "display": "Glucose [Mass/volume] in Blood --fasting"},
    "random blood sugar": {"code": "2345-7", "display": "Glucose [Mass/volume] in Blood"},
    "rbs": {"code": "2345-7", "display": "Glucose [Mass/volume] in Blood"},
    
    "total cholesterol": {"code": "2093-3", "display": "Cholesterol [Mass/volume] in Serum or Plasma"},
    "cholesterol": {"code": "2093-3", "display": "Cholesterol [Mass/volume] in Serum or Plasma"},
    "hdl cholesterol": {"code": "2085-9", "display": "Cholesterol in HDL [Mass/volume] in Serum or Plasma"},
    "hdl": {"code": "2085-9", "display": "Cholesterol in HDL [Mass/volume] in Serum or Plasma"},
    "ldl cholesterol": {"code": "18262-6", "display": "Cholesterol in LDL [Mass/volume] in Serum or Plasma"},
    "ldl": {"code": "18262-6", "display": "Cholesterol in LDL [Mass/volume] in Serum or Plasma"},
    "triglycerides": {"code": "2571-8", "display": "Triglyceride [Mass/volume] in Serum or Plasma"},
    
    "creatinine": {"code": "2160-0", "display": "Creatinine [Mass/volume] in Serum or Plasma"},
    "serum creatinine": {"code": "2160-0", "display": "Creatinine [Mass/volume] in Serum or Plasma"},
    "urea": {"code": "3094-0", "display": "Urea nitrogen [Mass/volume] in Serum or Plasma"},
    "blood urea": {"code": "3094-0", "display": "Urea nitrogen [Mass/volume] in Serum or Plasma"},
    "bun": {"code": "3094-0", "display": "Urea nitrogen [Mass/volume] in Serum or Plasma"},
    
    "tsh": {"code": "89579-7", "display": "Thyrotropin [Units/volume] in Serum or Plasma"},
    "vitamin d": {"code": "62292-8", "display": "25-hydroxyvitamin D3 [Mass/volume] in Serum or Plasma"},
    "vitamin d3": {"code": "62292-8", "display": "25-hydroxyvitamin D3 [Mass/volume] in Serum or Plasma"},
    "vitamin b12": {"code": "14685-2", "display": "Cobalamin [Mass/volume] in Serum or Plasma"},
    
    "sodium": {"code": "2951-2", "display": "Sodium [Moles/volume] in Serum or Plasma"},
    "potassium": {"code": "2823-3", "display": "Potassium [Moles/volume] in Serum or Plasma"},
    "chloride": {"code": "2075-0", "display": "Chloride [Moles/volume] in Serum or Plasma"},
    "calcium": {"code": "17861-6", "display": "Calcium [Mass/volume] in Serum or Plasma"},
}


class FHIRCoding(BaseModel):
    system: str = "http://loinc.org"
    code: str
    display: str


class FHIRCodeableConcept(BaseModel):
    coding: List[FHIRCoding]
    text: str


class FHIRReference(BaseModel):
    reference: str


class FHIRQuantity(BaseModel):
    value: float
    unit: str
    system: str = "http://unitsofmeasure.org"
    code: str


class FHIRReferenceRange(BaseModel):
    low: Optional[FHIRQuantity] = None
    high: Optional[FHIRQuantity] = None
    text: Optional[str] = None


class FHIRObservation(BaseModel):
    resourceType: str = "Observation"
    id: str
    status: str = "final"
    category: List[FHIRCodeableConcept] = Field(default_factory=list)
    code: FHIRCodeableConcept
    subject: FHIRReference
    valueQuantity: FHIRQuantity
    referenceRange: List[FHIRReferenceRange] = Field(default_factory=list)
    interpretation: List[FHIRCodeableConcept] = Field(default_factory=list)


class FHIRDiagnosticReport(BaseModel):
    resourceType: str = "DiagnosticReport"
    id: str
    status: str = "final"
    code: FHIRCodeableConcept
    subject: FHIRReference
    result: List[FHIRReference] = Field(default_factory=list)


def create_fhir_observation_from_flagged_value(fv, report_id: str, patient_id: str) -> dict:
    """Helper to construct a FHIR R4 Observation resource from a FlaggedValue."""
    import uuid

    normalized_name = fv.test_name.lower().strip()
    loinc_info = LOINC_MAP.get(normalized_name, {"code": "unknown", "display": fv.test_name})
    
    # 1. Category: laboratory
    category = [
        FHIRCodeableConcept(
            coding=[
                FHIRCoding(
                    system="http://terminology.hl7.org/CodeSystem/observation-category",
                    code="laboratory",
                    display="Laboratory"
                )
            ],
            text="Laboratory"
        )
    ]
    
    # 2. Code: LOINC mapping
    code = FHIRCodeableConcept(
        coding=[
            FHIRCoding(
                system="http://loinc.org",
                code=loinc_info["code"],
                display=loinc_info["display"]
            )
        ],
        text=fv.test_name
    )
    
    # 3. Subject: Patient Reference
    subject = FHIRReference(reference=f"Patient/{patient_id}")
    
    # 4. Value Quantity
    value_quantity = FHIRQuantity(
        value=fv.value,
        unit=fv.unit,
        system="http://unitsofmeasure.org",
        code=fv.unit if fv.unit else "1"
    )
    
    # 5. Reference Range
    ref_ranges = []
    if fv.reference_low is not None or fv.reference_high is not None:
        low_qty = None
        high_qty = None
        if fv.reference_low is not None:
            low_qty = FHIRQuantity(value=fv.reference_low, unit=fv.unit, code=fv.unit if fv.unit else "1")
        if fv.reference_high is not None:
            high_qty = FHIRQuantity(value=fv.reference_high, unit=fv.unit, code=fv.unit if fv.unit else "1")
        
        ref_ranges.append(
            FHIRReferenceRange(
                low=low_qty,
                high=high_qty,
                text=f"{fv.reference_low} - {fv.reference_high} {fv.unit}"
            )
        )
    
    # 6. Interpretation (H, L, N, or Crit)
    interpretation = []
    if fv.flag != "UNKNOWN":
        code_str = "N"
        display_str = "Normal"
        if fv.flag == "HIGH":
            code_str = "H"
            display_str = "High"
        elif fv.flag == "LOW":
            code_str = "L"
            display_str = "Low"
        elif fv.flag == "CRITICAL_HIGH":
            code_str = "HU"
            display_str = "High Alert"
        elif fv.flag == "CRITICAL_LOW":
            code_str = "LU"
            display_str = "Low Alert"
        
        interpretation.append(
            FHIRCodeableConcept(
                coding=[
                    FHIRCoding(
                        system="http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        code=code_str,
                        display=display_str
                    )
                ],
                text=fv.flag
            )
        )
    
    obs = FHIRObservation(
        id=f"obs-{uuid.uuid4().hex[:12]}",
        category=category,
        code=code,
        subject=subject,
        valueQuantity=value_quantity,
        referenceRange=ref_ranges,
        interpretation=interpretation
    )
    
    return obs.dict()

