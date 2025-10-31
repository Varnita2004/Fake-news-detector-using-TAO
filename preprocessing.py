# src/preprocessing.py
import re
import json
import pandas as pd
from typing import List

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_liar(path_csv: str) -> pd.DataFrame:
    # Expected format: columns include 'label' and 'statement'
    df = pd.read_csv(path_csv)
    df['statement'] = df['statement'].astype(str).apply(clean_text)
    # Normalize labels (e.g., pants-fire, false -> Fake ; true -> True ; else Uncertain)
    def map_label(l):
        l = str(l).lower()
        if 'true' in l and 'mostly' not in l: return 'True'
        if 'false' in l or 'pants-fire' in l or 'barely-true' in l: return 'Fake'
        return 'Uncertain'
    df['label_norm'] = df['label'].apply(map_label)
    return df[['statement', 'label_norm', 'source']].rename(columns={'statement':'text','label_norm':'label'})

def create_stream_csv(df: pd.DataFrame, out_path='data/stream_sim.csv'):
    # Simulate a stream by adding timestamp & platform columns
    import time
    from datetime import datetime, timedelta
    start = datetime.utcnow()
    rows = []
    for i, r in df.sample(frac=1).iterrows():
        ts = start + timedelta(seconds=int(i % 1000))
        rows.append({'id': i, 'text': r['text'], 'label': r['label'], 'timestamp': ts.isoformat(), 'platform':'sim'})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Stream created at", out_path)
