"""
PDF text extraction module.
Extracts raw text from medical report PDFs using pdfplumber.
"""

import pdfplumber


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Full text content of the PDF, pages separated by newlines.
    """
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def extract_tables_from_pdf(file_path: str) -> list[list[list[str]]]:
    """
    Extract tables from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of tables, where each table is a list of rows,
        and each row is a list of cell strings.
    """
    all_tables = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                # Clean None cells
                cleaned = []
                for table in tables:
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [cell.strip() if cell else "" for cell in row]
                        cleaned_table.append(cleaned_row)
                    cleaned.append(cleaned_table)
                all_tables.extend(cleaned)
    return all_tables
