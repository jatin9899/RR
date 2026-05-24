"""
resume_parser.py
----------------
Extract plain text from a resume file.
Supported formats: PDF, DOCX, TXT
"""

import io
from PyPDF2 import PdfReader
import docx

def _read_pdf(file) -> str:
    raw_bytes = file.read()
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    full_text = "\n".join(pages).strip()
    if not full_text:
        raise ValueError(
            "No readable text found in the PDF. "
            "The file might be scanned or image-based."
        )
    return full_text

def _read_docx(file) -> str:
    raw_bytes = file.read()
    document = docx.Document(io.BytesIO(raw_bytes))
    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
    full_text = "\n".join(paragraphs).strip()
    if not full_text:
        raise ValueError(
            "No readable text found in the DOCX file. "
            "The file might be empty or contain only images."
        )
    return full_text

def _read_txt(file) -> str:
    raw_bytes = file.read()
    full_text = raw_bytes.decode("utf-8", errors="ignore").strip()
    if not full_text:
        raise ValueError("No readable text found in the TXT file.")
    return full_text

def extract_text_from_resume(uploaded_file) -> str:
    """
    Extract text from a Streamlit uploaded file (PDF, DOCX, or TXT).
    """
    filename = uploaded_file.name.lower()
    if filename.endswith('.pdf'):
        return _read_pdf(uploaded_file)
    elif filename.endswith('.docx'):
        return _read_docx(uploaded_file)
    elif filename.endswith('.txt'):
        return _read_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
