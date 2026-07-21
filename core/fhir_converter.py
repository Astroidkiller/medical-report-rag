"""
Google Cloud Healthcare API FHIR R4 Resource Generator.

Transforms parsed lab test observations into official FHIR R4 Bundle
and Observation resources compatible with GCP Cloud Healthcare API FHIR Stores.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any


def create_fhir_observation(
    test_name: str,
    value: float,
    unit: str,
    flag: str,
    loinc_code: str = "29463-7",
    patient_id: str = None,
    effective_datetime: str = None
) -> Dict[str, Any]:
    """
    Construct a FHIR R4 Observation resource conforming to GCP Healthcare API specifications.
    """
    obs_id = f"obs-{uuid.uuid4().hex[:12]}"
    now_iso = effective_datetime or datetime.now(timezone.utc).isoformat()
    
    # Interpretation mapping
    interpretation_code = "N"
    interpretation_display = "Normal"
    if flag in ("HIGH", "CRITICAL_HIGH"):
        interpretation_code = "H"
        interpretation_display = "High"
    elif flag in ("LOW", "CRITICAL_LOW"):
        interpretation_code = "L"
        interpretation_display = "Low"

    resource = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                        "display": "Laboratory"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": loinc_code,
                    "display": test_name
                }
            ],
            "text": test_name
        },
        "subject": {
            "reference": f"Patient/{patient_id or 'anon-patient'}"
        },
        "effectiveDateTime": now_iso,
        "valueQuantity": {
            "value": float(value),
            "unit": unit or "",
            "system": "http://unitsofmeasure.org",
            "code": unit or ""
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": interpretation_code,
                        "display": interpretation_display
                    }
                ]
            }
        ]
    }

    return resource


def build_fhir_bundle(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Wrap multiple FHIR R4 Observation resources into a FHIR transaction/collection Bundle.
    """
    bundle_id = f"bundle-{uuid.uuid4().hex[:12]}"
    entries = []

    for obs in observations:
        entries.append({
            "fullUrl": f"urn:uuid:{obs['id']}",
            "resource": obs,
            "request": {
                "method": "POST",
                "url": "Observation"
            }
        })

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "transaction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry": entries
    }
