from PyPDF2 import PdfReader
import docx
import email
import re
from typing import Tuple, List
import nltk
from nltk.tokenize import sent_tokenize

# Initialize NLTK data at module level
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: NLTK punkt download failed: {e}")

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
    """Enhanced semantic chunking with fallback tokenization"""
    # First split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Initialize all_sentences list
    all_sentences = []
    
    # Try NLTK tokenization first, fallback to simple regex if it fails
    for para in paragraphs:
        try:
            sentences = sent_tokenize(para)
        except LookupError:
            # Fallback to simple regex-based sentence splitting
            sentences = [s.strip() for s in re.split(r'[.!?]+', para) if s.strip()]
        except Exception as e:
            print(f"Warning: Sentence tokenization failed: {e}")
            sentences = [para]  # Use whole paragraph as one sentence
        
        all_sentences.extend(sentences)

    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in all_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_size = len(sentence)
        
        # Check if adding this sentence would exceed max size
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Join current chunk and add to chunks
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= 50:  # Minimum chunk size
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_size = 0
            new_chunk = []
            for prev_sent in reversed(current_chunk):
                if overlap_size + len(prev_sent) <= overlap:
                    new_chunk.insert(0, prev_sent)
                    overlap_size += len(prev_sent)
                else:
                    break
            
            current_chunk = new_chunk
            current_size = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
