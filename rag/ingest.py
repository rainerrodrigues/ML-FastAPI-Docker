import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/docs/docs.txt") as f:
    docs = f.readlines()

embeddings = model.encode(docs)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "rag/index.faiss")
np.save("rag/docs.npy", np.array(docs))

print("✅ Vector index created")

