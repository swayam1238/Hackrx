from PyPDF2 import PdfReader
import docx
import email
import re
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
    chunks = create_semantic_chunks(text)
    return text, chunks

def parse_docx(file_bytes: bytes) -> Tuple[str, List[str]]:
    import io
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs])
    chunks = create_semantic_chunks(text)
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
    chunks = create_semantic_chunks(text)
    return text, chunks

def create_semantic_chunks(text: str, max_chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Create semantic chunks based on sentences and paragraphs rather than fixed character counts
    Optimized for large files while maintaining performance
    """
    # Split into sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 5:  # Allow shorter sentences for large files
            continue
            
        # If adding this sentence would exceed max size, save current chunk and start new one
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous chunk
            words = current_chunk.split()
            if len(words) > 8:  # Balanced overlap
                overlap_words = words[-8:]  # Last 8 words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks but allow more chunks for large files
    chunks = [chunk for chunk in chunks if len(chunk) > 20]
    
    # Adaptive chunk limit based on text size
    text_size = len(text)
    if text_size > 100000:  # Large file (>100KB)
        max_chunks = 150
    elif text_size > 50000:  # Medium file (50-100KB)
        max_chunks = 100
    else:  # Small file (<50KB)
        max_chunks = 75
    
    return chunks[:max_chunks]
