"""
Structured lab value extraction from medical report text.
Uses regex patterns to identify test names, values, units, and reference ranges.
Falls back to LLM-assisted extraction for non-standard formats.
"""

import re
from dataclasses import dataclass


@dataclass
class ExtractedLabValue:
    """A single extracted lab value from a medical report."""
    test_name: str
    value: float
    unit: str
    reference_low: float | None
    reference_high: float | None
    raw_line: str  # Original text line for traceability


# FIX #21: "to" must be matched as a word boundary, not character class.
# Use alternation: (?:[-–—]|\bto\b) instead of [-–—to]
_RANGE_SEP = r"(?:[-–—]|\bto\b)"

# Regex patterns for common lab report formats
# Pattern 1: "Test Name    14.2    g/dL    13.0 - 17.0"
# Pattern 2: "Test Name    14.2 g/dL   (13.0-17.0)"
# Pattern 3: "Test Name: 14.2 g/dL  Ref: 13.0-17.0"
_PATTERNS = [
    # Pattern: Name ... value unit ... low - high
    re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9\s/\-\(\)\.]{2,50}?)\s+"
        r"(?P<value>\d+\.?\d*)\s*"
        r"(?P<unit>[A-Za-z%/µμ\.\-]+(?:/[A-Za-z%µμ\.\-]+)?)\s+"
        r"(?P<low>\d+\.?\d*)\s*" + _RANGE_SEP + r"\s*(?P<high>\d+\.?\d*)",
        re.IGNORECASE,
    ),
    # Pattern: Name ... value ... low - high ... unit
    re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9\s/\-\(\)\.]{2,50}?)\s+"
        r"(?P<value>\d+\.?\d*)\s+"
        r"(?P<low>\d+\.?\d*)\s*" + _RANGE_SEP + r"\s*(?P<high>\d+\.?\d*)\s*"
        r"(?P<unit>[A-Za-z%/µμ\.\-]+(?:/[A-Za-z%µμ\.\-]+)?)",
        re.IGNORECASE,
    ),
    # Pattern: Name: value unit (low-high)
    re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9\s/\-\(\)\.]{2,50}?)\s*[:=]\s*"
        r"(?P<value>\d+\.?\d*)\s*"
        r"(?P<unit>[A-Za-z%/µμ\.\-]+(?:/[A-Za-z%µμ\.\-]+)?)?\s*"
        r"[\(\[]\s*(?P<low>\d+\.?\d*)\s*" + _RANGE_SEP + r"\s*(?P<high>\d+\.?\d*)\s*[\)\]]",
        re.IGNORECASE,
    ),
    # Pattern: Name ... value (no ref range, just name and number)
    re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9\s/\-\(\)\.]{2,50}?)\s+"
        r"(?P<value>\d+\.?\d*)\s*"
        r"(?P<unit>[A-Za-z%/µμ\.\-]+(?:/[A-Za-z%µμ\.\-]+)?)",
        re.IGNORECASE,
    ),
]

# Words that should NOT be treated as test names
_NOISE_WORDS = {
    "page", "date", "time", "name", "age", "sex", "gender", "patient",
    "doctor", "dr", "report", "lab", "laboratory", "hospital", "clinic",
    "sample", "collected", "received", "reported", "method", "department",
    "ref", "reference", "range", "normal", "unit", "result", "test",
    "sr", "no", "s.no", "sl", "serial",
}


def _is_noise(name: str) -> bool:
    """Check if extracted name is likely noise/header text."""
    cleaned = name.strip().lower()
    # Too short or too long
    if len(cleaned) < 2 or len(cleaned) > 60:
        return True
    # Starts with a noise word
    first_word = cleaned.split()[0] if cleaned.split() else ""
    if first_word in _NOISE_WORDS:
        return True
    # All digits or purely numeric
    if cleaned.replace(".", "").replace(" ", "").isdigit():
        return True
    return False


def parse_lab_values(text: str) -> list[ExtractedLabValue]:
    """
    Parse structured lab values from medical report text.

    Tries multiple regex patterns against each line. Deduplicates by
    test name (keeps the match with the most complete data).

    Args:
        text: Full text of the medical report.

    Returns:
        List of ExtractedLabValue objects.
    """
    seen_tests = {}  # test_name_lower -> ExtractedLabValue

    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 5:
            continue

        for pattern in _PATTERNS:
            match = pattern.match(line)
            if not match:
                continue

            groups = match.groupdict()
            name = groups.get("name", "").strip()

            if _is_noise(name):
                continue

            try:
                value = float(groups["value"])
            except (ValueError, KeyError):
                continue

            unit = groups.get("unit", "").strip() if groups.get("unit") else ""

            ref_low = None
            ref_high = None
            try:
                if groups.get("low") is not None:
                    ref_low = float(groups["low"])
                if groups.get("high") is not None:
                    ref_high = float(groups["high"])
            except (ValueError, TypeError):
                pass

            lab_val = ExtractedLabValue(
                test_name=name,
                value=value,
                unit=unit,
                reference_low=ref_low,
                reference_high=ref_high,
                raw_line=line,
            )

            # Keep the most complete match per test name
            key = name.lower().strip()
            if key not in seen_tests:
                seen_tests[key] = lab_val
            else:
                existing = seen_tests[key]
                # Prefer the one with reference ranges
                if (ref_low is not None and existing.reference_low is None):
                    seen_tests[key] = lab_val

            break  # First matching pattern wins for this line

    return list(seen_tests.values())


def parse_lab_values_from_tables(tables: list[list[list[str]]]) -> list[ExtractedLabValue]:
    """
    Parse lab values from extracted PDF tables.

    Attempts to identify columns for test name, value, unit, and reference range
    by analyzing header rows.

    Args:
        tables: List of tables from pdf_extractor.extract_tables_from_pdf.

    Returns:
        List of ExtractedLabValue objects.
    """
    results = []
    _table_range_re = re.compile(
        r"(\d+\.?\d*)\s*(?:[-–—]|\bto\b)\s*(\d+\.?\d*)",
        re.IGNORECASE,
    )

    for table in tables:
        if len(table) < 2:
            continue

        # Try to identify column layout from headers
        header = [cell.lower().strip() for cell in table[0]]

        name_col = None
        value_col = None
        unit_col = None
        ref_col = None

        for i, h in enumerate(header):
            if any(w in h for w in ["test", "investigation", "parameter", "analyte"]):
                name_col = i
            elif any(w in h for w in ["result", "value", "observed"]):
                value_col = i
            elif any(w in h for w in ["unit", "units"]):
                unit_col = i
            elif any(w in h for w in ["reference", "ref", "range", "normal", "biological"]):
                ref_col = i

        # Fallback: assume first col = name, second = value
        if name_col is None and len(header) >= 2:
            name_col = 0
        if value_col is None and len(header) >= 2:
            value_col = 1

        if name_col is None or value_col is None:
            continue

        for row in table[1:]:
            if len(row) <= max(name_col, value_col):
                continue

            name = row[name_col].strip()
            if _is_noise(name) or not name:
                continue

            try:
                value = float(row[value_col].strip().replace(",", ""))
            except (ValueError, IndexError):
                continue

            unit = ""
            if unit_col is not None and unit_col < len(row):
                unit = row[unit_col].strip()

            ref_low, ref_high = None, None
            if ref_col is not None and ref_col < len(row):
                ref_text = row[ref_col].strip()
                ref_match = _table_range_re.search(ref_text)
                if ref_match:
                    ref_low = float(ref_match.group(1))
                    ref_high = float(ref_match.group(2))

            raw_line = " | ".join(row)
            results.append(ExtractedLabValue(
                test_name=name,
                value=value,
                unit=unit,
                reference_low=ref_low,
                reference_high=ref_high,
                raw_line=raw_line,
            ))

    return results
