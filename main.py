from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from utils.parser import parse_document_from_bytes
from utils.embedder import get_embeddings, build_faiss_index, search_similar_chunks
from utils.llm_gemini import ask_question
from utils.formatter import format_response

app = FastAPI(title="HackRx LLM-Powered Query System", version="1.0.0")

# Health check endpoint
@app.get("/")
def home():
    return {"status": "ok", "service": "HackRx LLM-Powered Query System"}

class QueryRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

# Main API endpoint as per requirements
@app.post("/api/v1/hackrx/run")
async def run_query(
    request: Request,
    body: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    # Authentication check
    required_token = "Bearer 3dea41115332ec6960807ebc546a0244e3bf91529888d3949b484cd22f0de72f"
    if authorization != required_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Download the document from the URL
    try:
        response = requests.get(body.documents)
        response.raise_for_status()
        file_bytes = response.content
        filename = body.documents.split("?")[0].split("/")[-1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

    # Step 1: Parse file content
    raw_text, chunks = parse_document_from_bytes(file_bytes, filename)
    
    # Log document size for monitoring
    file_size_kb = len(file_bytes) / 1024
    text_size_kb = len(raw_text) / 1024
    print(f"📄 Document: {filename} | File: {file_size_kb:.1f}KB | Text: {text_size_kb:.1f}KB | Chunks: {len(chunks)}")

    # Step 2: Embeddings + FAISS index (with caching)
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings, cache_key=filename)

    # Step 3: Define parallel question processor
    async def process_question(question: str):
        # Adaptive search based on document size
        document_size = len(raw_text)
        if document_size > 100000:
            k = 4
        elif document_size > 50000:
            k = 3
        else:
            k = 2
            
        # Move CPU-bound tasks to thread pool executor
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            top_chunks = await loop.run_in_executor(pool, search_similar_chunks, question, chunks, index, k)
            result = await loop.run_in_executor(pool, ask_question, question, top_chunks)
            answer, reasoning, relevant_clauses, confidence = result
            return format_response(answer, reasoning, relevant_clauses, confidence)

    # Step 4: Run questions in parallel
    answers = await asyncio.gather(*[process_question(q) for q in body.questions])

    return {"answers": answers}