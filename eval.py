# src/eval.py
import pandas as pd
from rag_pipeline import RAGPipeline
from sklearn.metrics import precision_recall_fscore_support

def evaluate(df_test):
    rag = RAGPipeline()
    preds, golds = [], []
    for _, r in df_test.iterrows():
        out = rag.analyze(r['text'])
        # parse label
        lab = out.get('label', out.get('predicted_label', None))
        if lab is None and 'raw_output' in out:
            # fallback - simple heuristic
            txt = out['raw_output'].lower()
            if 'fake' in txt: lab='Fake'
            elif 'true' in txt: lab='True'
            else: lab='Uncertain'
        preds.append(lab)
        golds.append(r['label'])
    p,r,f,_ = precision_recall_fscore_support(golds, preds, average='macro', zero_division=0)
    print("Precision:", p, "Recall:", r, "F1:", f)
    return p,r,f
