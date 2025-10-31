# src/make_evidence_demo.py
import pandas as pd
docs = [
    {"id":0, "text":"The moon landing occurred on July 20, 1969, with Apollo 11 landing on the lunar surface.", "source_url":"https://nasa.gov/apollo11"},
    {"id":1, "text":"Vaccines undergo safety trials before approval; multiple agencies monitor safety.", "source_url":"https://who.int/vaccines"},
    {"id":2, "text":"Climate change is driven by greenhouse gas emissions from human activity.", "source_url":"https://ipcc.ch"}
]
pd.DataFrame(docs).to_csv("../data/evidence.csv", index=False)

print("Demo evidence created.")
