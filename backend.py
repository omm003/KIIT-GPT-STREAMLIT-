import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
try:
    index = faiss.read_index("kiit_faiss.index")
except:
    index = None

# Load dataset
try:
    with open("kiit_full_dataset.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
except:
    texts = []

def search(query, top_k=3):
    if index is None or not texts:
        return []

    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx != -1 and idx < len(texts):
            results.append(texts[idx])

    return results