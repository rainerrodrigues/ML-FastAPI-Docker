import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("rag/index.faiss")
docs = np.load("rag/docs.npy", allow_pickle=True)

def retrieve(query, k=2):
    q_emb = model.encode([query])
    _, I = index.search(np.array(q_emb), k)
    return [docs[i] for i in I[0]]

