from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import requests

from utils.parser import parse_document_from_bytes
from utils.embedder import get_embeddings, build_faiss_index, search_similar_chunks
from utils.llm_gemini import ask_question
from utils.formatter import format_response

app = FastAPI()
@app.get("/")
def home():
    return {"status": "ok"}
class QueryRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

@app.post("/hackrx/run")
async def run_query(
    request: Request,
    body: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    # Optional: Check Bearer token
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

    # Step 2: Embeddings + FAISS index
    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)

    # Step 3: Process each question
    answers = []
    for question in body.questions:
        top_chunks = search_similar_chunks(question, chunks, index)
        answer = ask_question(question, top_chunks)
        answers.append(format_response(answer))

    return {"answers": answers}
