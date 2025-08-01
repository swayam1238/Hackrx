from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple

# Global model instance for reuse
_model = None
_index_cache = {}

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

# Get embeddings for each chunk with caching and adaptive batching
def get_embeddings(chunks):
    model = get_model()
    # Adaptive batch size based on number of chunks
    if len(chunks) > 100:  # Large document
        batch_size = 64
    elif len(chunks) > 50:  # Medium document
        batch_size = 48
    else:  # Small document
        batch_size = 32
    
    return model.encode(chunks, show_progress_bar=False, batch_size=batch_size)

# Build FAISS index with caching
def build_faiss_index(embeddings, cache_key=None):
    if cache_key and cache_key in _index_cache:
        return _index_cache[cache_key]
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    
    if cache_key:
        _index_cache[cache_key] = index
    
    return index

# Search top-k chunks with optimized search
def search_similar_chunks(question, chunks, index, k=3):
    model = get_model()
    q_embed = model.encode([question], show_progress_bar=False)
    D, I = index.search(q_embed, k)
    return [chunks[i] for i in I[0]]