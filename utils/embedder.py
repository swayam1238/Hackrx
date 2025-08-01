
import numpy as np
import faiss
from typing import List
import numpy.typing as npt
import google.generativeai as genai

# --- Configuration ---
# WARNING: Hardcoding API keys is a major security risk. 
# It's highly recommended to use environment variables instead.
genai.configure(api_key="AIzaSyDg6eoWSyMxwzycFHbnGRll9xRP_777PyY")

# The Gemini model for embeddings
EMBEDDING_MODEL = "models/embedding-001"

# The dimension of the embeddings produced by the model
EMBEDDING_DIM = 768

_index_cache = {}

def get_embeddings(chunks: List[str]) -> npt.NDArray:
    """Get embeddings with caching to avoid reprocessing identical content."""
    try:
        # Build content signature to identify similar text
        content_signature = ":".join([
            f"{len(chunk)}:{hash(chunk)}" for chunk in chunks[:5]  # First 5 chunks for faster comparison
        ])[:200]  # Keep short for efficiency

        # Check cache first
        if hasattr(get_embeddings, "_cache"):
            if content_signature in get_embeddings._cache:
                return get_embeddings._cache[content_signature]

        # Get new embeddings
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=chunks,
            task_type="retrieval_document"
        )
        embeddings = np.array(response['embedding'])
        
        # Save to cache
        if not hasattr(get_embeddings, "_cache"):
            get_embeddings._cache = {}
        get_embeddings._cache[content_signature] = embeddings
        
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return np.array([]).reshape(0, EMBEDDING_DIM)

def build_faiss_index(embeddings: npt.NDArray, cache_key: str = None):
    """Builds an optimized FAISS index from embeddings with HNSW for faster search."""
    if cache_key and cache_key in _index_cache:
        return _index_cache[cache_key]

    if embeddings.shape[0] == 0:
        print("Cannot build index from empty embeddings.")
        return None

    dim = embeddings.shape[1]
    # HNSW offers significant speed improvements over brute-force search
    index = faiss.IndexHNSWFlat(dim, 32)  # 32 is the HNSW constant (can be tuned)
    index.add(embeddings)

    if cache_key:
        _index_cache[cache_key] = index

    return index

def search_similar_chunks(question: str, chunks: List[str], index: faiss.Index, k: int = 5) -> List[str]:
    """Search for chunks similar to the question using Gemini embeddings."""
    if not index:
        print("Search failed: FAISS index is not valid.")
        return chunks[:k]
        
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=question,
            task_type="retrieval_query"
        )
        q_embed = np.array([response['embedding']])
        distances, indices = index.search(q_embed, k)
        return [chunks[i] for i in indices[0]]
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return chunks[:k]
    