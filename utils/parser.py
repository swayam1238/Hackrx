from PyPDF2 import PdfReader
import docx
import email
from typing import Tuple, List

def parse_document_from_bytes(file_bytes: bytes, filename: str) -> Tuple[str, List[str]]:
    filename = filename.lower()
    if filename.endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif filename.endswith(".docx"):
        return parse_docx(file_bytes)
    elif filename.endswith(".eml"):
        return parse_email(file_bytes)
    else:
        raise ValueError("Unsupported file type")

def parse_pdf(file_bytes: bytes) -> Tuple[str, List[str]]:
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    return text, chunks

def parse_docx(file_bytes: bytes) -> Tuple[str, List[str]]:
    import io
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs])
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    return text, chunks

def parse_email(file_bytes: bytes) -> Tuple[str, List[str]]:
    msg = email.message_from_bytes(file_bytes)
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_payload(decode=True).decode(errors="ignore"))
        text = "\n".join(parts)
    else:
        text = msg.get_payload(decode=True).decode(errors="ignore")
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    return text, chunks
