# ================================================================
# src/live_retriever.py
# ================================================================
"""
LiveRetriever: Combines Wikipedia, NewsAPI, and FAISS
-----------------------------------------------------
Retrieves factual evidence in real-time for RAG pipelines.
"""

import os
import wikipedia
import requests
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# -------------------------
# CONFIGURATION
# -------------------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # Set your key in .env or system env
EMB_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX = "data/faiss_index.bin"
FAISS_META = "data/faiss_meta.pkl"


class LiveRetriever:
    def __init__(self):
        print("[INIT] Initializing LiveRetriever...")
        self.embedder = SentenceTransformer(EMB_MODEL)
        self.index, self.meta = self._load_faiss()

    # ------------------------------------------------------------
    # Load FAISS Index if available
    # ------------------------------------------------------------
    def _load_faiss(self):
        if os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_META):
            try:
                index = faiss.read_index(FAISS_INDEX)
                with open(FAISS_META, "rb") as f:
                    meta = pickle.load(f)
                print("[OK] FAISS index loaded.")
                return index, meta
            except Exception as e:
                print("[WARN] Could not load FAISS index:", e)
        return None, None

    # ------------------------------------------------------------
    # Search within FAISS
    # ------------------------------------------------------------
    def _search_faiss(self, query: str, top_k: int = 5):
        if self.index is None or self.meta is None:
            return []
        try:
            emb = self.embedder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(emb)
            D, I = self.index.search(emb, top_k)
            results = []
            for idx, score in zip(I[0], D[0]):
                results.append({
                    "source": self.meta.get("source_url", ["local_FAISS"])[0],
                    "text": self.meta["text"][idx],
                    "score": float(score)
                })
            return results
        except Exception as e:
            print("[FAISS Error]", e)
            return []

    # ------------------------------------------------------------
    # Wikipedia Search
    # ------------------------------------------------------------
    def _wikipedia_search(self, query: str, top_k: int = 2):
        results = []
        try:
            search_results = wikipedia.search(query, results=top_k)
            for title in search_results:
                try:
                    summary = wikipedia.summary(title, sentences=2)
                    results.append({
                        "source": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "text": summary,
                        "score": None
                    })
                except Exception:
                    continue
        except Exception as e:
            print("[Wikipedia Error]", e)
        return results

    # ------------------------------------------------------------
    # NewsAPI Search
    # ------------------------------------------------------------
    def _newsapi_search(self, query: str, top_k: int = 3):
        if not NEWSAPI_KEY:
            return []
        try:
            url = "https://newsapi.org/v2/everything"
            params = {"q": query, "language": "en", "pageSize": top_k, "apiKey": NEWSAPI_KEY}
            r = requests.get(url, params=params, timeout=8)
            articles = r.json().get("articles", [])
            results = []
            for a in articles:
                snippet = (a.get("title", "") + ". " + a.get("description", ""))[:1000]
                results.append({
                    "source": a.get("url", "newsapi"),
                    "text": snippet,
                    "score": None
                })
            return results
        except Exception as e:
            print("[NewsAPI Error]", e)
            return []

    # ------------------------------------------------------------
    # Combine all sources
    # ------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5):
        if not query:
            return []

        print(f"[RETRIEVE] Searching evidence for: {query}")
        evidence = []

        # 1. FAISS (local evidence)
        evidence += self._search_faiss(query, top_k=top_k)

        # 2. Wikipedia (general knowledge)
        wiki_evidence = self._wikipedia_search(query, top_k=2)
        evidence += wiki_evidence

        # 3. NewsAPI (optional live news)
        news_evidence = self._newsapi_search(query, top_k=3)
        evidence += news_evidence

        # 4. Score using sentence embeddings (cosine similarity)
        if not evidence:
            print("[INFO] No evidence sources found.")
            return []

        try:
            texts = [e["text"] for e in evidence]
            query_emb = self.embedder.encode([query], convert_to_numpy=True)
            text_embs = self.embedder.encode(texts, convert_to_numpy=True)
            faiss.normalize_L2(query_emb)
            faiss.normalize_L2(text_embs)
            scores = np.dot(text_embs, query_emb.T).flatten()

            for i, score in enumerate(scores):
                evidence[i]["score"] = float(score)

            evidence = sorted(evidence, key=lambda x: x["score"], reverse=True)[:top_k]
            print(f"[OK] Retrieved {len(evidence)} relevant documents.")
            return evidence

        except Exception as e:
            print("[Scoring Error]", e)
            return evidence
