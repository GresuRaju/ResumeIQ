"""
parser.py — Resume text extraction
Supports PDF (via PyMuPDF / pdfplumber fallback) and plain TXT files.
"""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes. Tries PyMuPDF first, then pdfplumber."""
    text = ""

    # Try PyMuPDF (fitz) — fastest
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
        if text.strip():
            return text.strip()
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: pdfplumber
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text.strip()
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: pypdf
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
        if text.strip():
            return text.strip()
    except ImportError:
        pass
    except Exception:
        pass

    return text.strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode TXT file bytes to string."""
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(encoding).strip()
        except (UnicodeDecodeError, AttributeError):
            continue
    return ""
