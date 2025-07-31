from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Get embeddings for each chunk
def get_embeddings(chunks):
    return model.encode(chunks)

# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# Search top-k chunks
def search_similar_chunks(question, chunks, index, k=3):
    q_embed = model.encode([question])
    D, I = index.search(q_embed, k)
    return [chunks[i] for i in I[0]]