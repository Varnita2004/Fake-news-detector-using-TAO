# ================================================================
# src/rag_pipeline.py
# ================================================================
"""
RAG + TAO Pipeline (Final Version)
----------------------------------
Connects:
    • LiveRetriever – Retrieves factual evidence
    • Generator – Generates explanations & verdicts
    • TAOOptimizer – Continuously fine-tunes and optimizes model behavior
"""

import os
import sys
import time
import traceback

# ------------------------------------------------------------
# Ensure project root is on sys.path
# ------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
try:
    from src.live_retriever import LiveRetriever
    print("[OK] LiveRetriever imported successfully.")
except Exception as e:
    print(f"[ERROR] Could not import LiveRetriever: {e}")
    LiveRetriever = None

try:
    from src.generator import Generator
    print("[OK] Generator imported successfully.")
except Exception as e:
    print(f"[ERROR] Could not import Generator: {e}")
    Generator = None

try:
    from src.tao_optimizer import TAOOptimizer
    print("[OK] TAOOptimizer imported successfully.")
except Exception as e:
    print(f"[ERROR] Could not import TAOOptimizer: {e}")
    TAOOptimizer = None


# ================================================================
# RAGPipeline CLASS
# ================================================================
class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline with TAO Optimization."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        print(f"[INIT] Initializing RAGPipeline with model: {model_name}")

        # ---------- Retriever ----------
        self.ret = None
        if LiveRetriever:
            try:
                self.ret = LiveRetriever()
                print("[OK] LiveRetriever initialized.")
            except Exception as e:
                print("[ERROR] LiveRetriever init failed:", e)

        # ---------- Generator ----------
        self.gen = None
        if Generator:
            try:
                self.gen = Generator(model_name)
                print("[OK] Generator initialized.")
            except Exception as e:
                print("[ERROR] Generator init failed:", e)

        # ---------- TAO Optimizer ----------
        self.tao = None
        if TAOOptimizer:
            try:
                self.tao = TAOOptimizer(model_name=model_name)
                print("[OK] TAO Optimizer initialized.")
            except Exception as e:
                print("[ERROR] TAO Optimizer init failed:", e)

    # ------------------------------------------------------------
    # MAIN PIPELINE FUNCTION
    # ------------------------------------------------------------
    def analyze(self, text: str):
        """Retrieve evidence, generate explanation, and apply TAO optimization."""
        if not text or not text.strip():
            return {
                "label": "Uncertain",
                "confidence": 0.0,
                "explanation": "Empty input.",
                "_evidence": [],
                "tao_status": "No input provided."
            }

        print(f"\n[ANALYZE] Processing claim: {text[:80]}...")

        # ---------- Step 1: Retrieve ----------
        evidence_docs, evidence_texts = [], []
        if self.ret:
            try:
                evidence_docs = self.ret.retrieve(text, top_k=6)
                evidence_texts = [d["text"] for d in evidence_docs if d.get("text")]
                print(f"[INFO] Retrieved {len(evidence_docs)} evidence documents.")
            except Exception as e:
                print("[Retriever failed]", e)
                traceback.print_exc()
        else:
            print("[WARN] Retriever not initialized.")

        # ---------- Step 2: TAO Optimization (before generation) ----------
        decoding_cfg = None
        if self.tao:
            try:
                decoding_cfg = self.tao.optimize_generation()
                print(f"[TAO] Decoding params set: {decoding_cfg}")
            except Exception as e:
                print("[TAO optimize_generation failed]", e)
        else:
            print("[WARN] TAO module not loaded (decoding params default).")

        # ---------- Step 3: Generate ----------
        if not self.gen:
            return {
                "label": "Uncertain",
                "confidence": 0.5,
                "explanation": "Generator not available.",
                "_evidence": evidence_docs,
                "tao_status": "Skipped (no generator)"
            }

        try:
            out = self.gen.generate_explanation(text, evidence_texts, decoding_params=decoding_cfg)
            if not out or not isinstance(out, dict):
                out = {"label": "Uncertain", "confidence": 0.5, "explanation": "No output"}
        except Exception as e:
            print("[Generator failed]", e)
            traceback.print_exc()
            out = {"label": "Uncertain", "confidence": 0.5, "explanation": str(e)}

        # ---------- Step 4: TAO Continual Training ----------
        if self.tao:
            try:
                sample = {"text": text, "label": out.get("label", "Uncertain")}
                status = self.tao.continual_train([sample])
                out["tao_status"] = status
                out["last_update"] = time.ctime(self.tao.last_update)
                print(f"[TAO] {status}")
            except Exception as e:
                print("[TAO continual_train failed]", e)
                traceback.print_exc()
                out["tao_status"] = f"TAO failed: {e}"
        else:
            out["tao_status"] = "TAO module not loaded."

        # ---------- Step 5: Attach Evidence ----------
        out["_evidence"] = evidence_docs
        return out
