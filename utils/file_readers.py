import io
from typing import List
from pathlib import Path

def read_text_file(file_obj) -> str:
    """Read uploaded file into text (utf-8)."""
    try:
        raw = file_obj.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    except Exception:
        file_obj.seek(0)
        try:
            return file_obj.getvalue().decode("utf-8", errors="replace")
        except Exception:
            return ""

def read_pdf(file_obj) -> str:
    """Extract text from PDF using PyPDF2 (lightweight)."""
    try:
        from PyPDF2 import PdfReader
        file_obj.seek(0)
        reader = PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"[PDF extraction failed: {e}]"

def read_docx(file_obj) -> str:
    """Extract text from Word documents."""
    try:
        import docx
        file_obj.seek(0)
        doc = docx.Document(file_obj)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX extraction failed: {e}]"

def read_uploaded_file(file_obj) -> str:
    """Smart reader: detect file type by extension and route to correct reader."""
    name = Path(file_obj.name).suffix.lower()
    if name in [".txt", ".md", ".json", ".csv"]:
        return read_text_file(file_obj)
    elif name == ".pdf":
        return read_pdf(file_obj)
    elif name == ".docx":
        return read_docx(file_obj)
    else:
        return read_text_file(file_obj)
