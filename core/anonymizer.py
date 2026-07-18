"""
TRIP-RAG inspired Context-Aware Entity Anonymization.
Selectively redacts Protected Health Information (PHI) like patient names,
doctor names, phone numbers, emails, addresses, and identifiers from raw text
before it is chunked and embedded in the vector store.
"""

import re

def anonymize_text(text: str) -> str:
    """
    Scrub PHI and sensitive entities from unstructured text.
    
    Args:
        text: Raw extracted report text.
        
    Returns:
        De-identified text string with placeholder tokens.
    """
    if not text:
        return ""

    lines = text.split("\n")
    anonymized_lines = []

    # Regex for standard fields
    name_re = re.compile(r"(Name\s*:\s*)(?:Mr\.|Mrs\.|Ms\.|Dr\.)?\s*([A-Za-z\s\.]+)", re.IGNORECASE)
    ref_by_re = re.compile(r"(Ref\s*By\s*:\s*)(?:Dr\.)?\s*([A-Za-z\s\.]+)", re.IGNORECASE)
    lab_no_re = re.compile(r"(Lab\s*No\.?\s*:\s*)\d+", re.IGNORECASE)
    age_re = re.compile(r"(Age\s*:\s*)\d+\s*(Years|Yrs|Y)?", re.IGNORECASE)
    gender_re = re.compile(r"(Gender\s*:\s*)(Male|Female|M|F|Other)", re.IGNORECASE)
    phone_re = re.compile(r"\+?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4,6}", re.IGNORECASE)
    email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.IGNORECASE)
    
    # Specific address keywords
    address_keywords = [
        "ROAD", "STREET", "COMPLEX", "OUTER RING ROAD", "KADUBEESANAHALLI", 
        "BANGALORE", "DELHI", "MUMBAI", "CHENNAI", "HYDERABAD", "KOLKATA", "BENGALURU",
        "ADDRESS", "S.O,", "SHOP NO"
    ]

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            anonymized_lines.append("")
            continue

        # Check if line looks like an address line containing key landmarks/locations
        is_address = any(keyword in cleaned_line.upper() for keyword in address_keywords)
        # Avoid redacting lines that contain actual clinical test results
        if is_address and not any(flag in cleaned_line.upper() for flag in ["RESULTS", "UNITS", "BIO. REF. INTERVAL", "TACROLIMUS", "HEMOGLOBIN"]):
            anonymized_lines.append("[LAB_LOCATION_ADDRESS]")
            continue

        # Apply specific field redactions
        line = name_re.sub(r"\1[PATIENT_NAME]", line)
        line = ref_by_re.sub(r"\1[REFERRING_DOCTOR]", line)
        line = lab_no_re.sub(r"\1[LAB_NUMBER]", line)
        line = age_re.sub(r"\1[PATIENT_AGE]", line)
        line = gender_re.sub(r"\1[PATIENT_GENDER]", line)
        line = phone_re.sub("[PHONE_NUMBER]", line)
        line = email_re.sub("[EMAIL_ADDRESS]", line)

        anonymized_lines.append(line)

    return "\n".join(anonymized_lines)
