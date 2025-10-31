import pandas as pd, faiss, pickle
from sentence_transformers import SentenceTransformer

print("[BUILD] Loading evidence...")
df = pd.read_csv("data/evidence.csv")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("[BUILD] Encoding evidence embeddings...")
embs = embedder.encode(df["text"].tolist(), convert_to_numpy=True)
faiss.normalize_L2(embs)

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

faiss.write_index(index, "data/faiss_index.bin")
with open("data/faiss_meta.pkl", "wb") as f:
    pickle.dump(df.to_dict(), f)

print("[OK] FAISS index successfully built with", len(df), "entries.")
