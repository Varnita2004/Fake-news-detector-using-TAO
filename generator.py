# ================================================================
# src/generator.py
# ================================================================
"""
Generator: Contextual Explanation & Counterfactual Generation (Enhanced)
-------------------------------------------------------------
Uses transformer reasoning with rule-based sanity checks to ensure factual consistency.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"[INIT] Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[OK] Model loaded on {self.device}")

    def generate_explanation(self, claim, evidence_texts, decoding_params=None):
        """Generate a verdict, explanation, and counterfactual using evidence-aware reasoning."""
        if not claim:
            return {"label": "Uncertain", "confidence": 0.0, "explanation": "Empty claim"}

        # Combine evidence
        context = " ".join(evidence_texts[:5]) if evidence_texts else "No relevant evidence found."
        prompt = (
            f"You are a fact verification model.\n"
            f"Claim: {claim}\n\n"
            f"Evidence:\n{context}\n\n"
            "Decide whether the claim is True, Fake, or Uncertain based only on the evidence. "
            "Then explain briefly why and give a counterfactual — what would make the claim true. "
            "Respond ONLY in JSON format with keys: label, confidence, explanation, counterfactual."
        )

        params = decoding_params or {
            "max_length": 256,
            "temperature": 0.7,
            "num_beams": 4,
            "repetition_penalty": 1.2,
        }

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=params.get("max_length", 256),
                temperature=params.get("temperature", 0.7),
                num_beams=params.get("num_beams", 4),
                repetition_penalty=params.get("repetition_penalty", 1.2),
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed = self._parse_output(decoded)

        # ✅ Post-correction using evidence polarity
        parsed = self._correct_with_evidence(parsed, claim, evidence_texts)
        return parsed

    # ------------------------------------------------------------
    # Parse model output into structured format
    # ------------------------------------------------------------
    def _parse_output(self, text):
        try:
            if text.strip().startswith("{"):
                data = json.loads(text)
                return {
                    "label": data.get("label", "Uncertain"),
                    "confidence": float(data.get("confidence", 0.6)),
                    "explanation": data.get("explanation", ""),
                    "counterfactual": data.get("counterfactual", ""),
                    "reasoning": data.get("reasoning", []),
                }
            else:
                label = "Fake" if "not true" in text.lower() or "false" in text.lower() else (
                        "True" if "true" in text.lower() else "Uncertain"
                )
                return {
                    "label": label,
                    "confidence": 0.8 if label != "Uncertain" else 0.5,
                    "explanation": text.strip(),
                    "counterfactual": "",
                    "reasoning": [],
                }
        except Exception as e:
            print("[Parse Error]", e)
            return {
                "label": "Uncertain",
                "confidence": 0.5,
                "explanation": text.strip(),
                "counterfactual": "",
                "reasoning": [],
            }

    # ------------------------------------------------------------
    # Evidence-based sanity correction
    # ------------------------------------------------------------
    def _correct_with_evidence(self, parsed, claim, evidence_texts):
        """Corrects the model’s output if evidence contradicts it."""
        claim_lower = claim.lower()
        evidence_joined = " ".join(evidence_texts).lower()

        # If evidence explicitly contains words like 'not true', 'anti-scientific', 'false', etc.
        if any(term in evidence_joined for term in ["anti-scientific", "not true", "false", "hoax", "conspiracy", "debunked"]):
            parsed["label"] = "Fake"
            parsed["confidence"] = max(parsed.get("confidence", 0.7), 0.9)
            parsed["explanation"] = "Evidence indicates the claim contradicts scientific consensus."
            parsed["counterfactual"] = "If verified empirical data supported the claim, it could be true."
        elif "scientific consensus" in evidence_joined and "flat" in claim_lower:
            parsed["label"] = "Fake"
            parsed["confidence"] = 0.95
            parsed["explanation"] = "Scientific consensus confirms Earth is spherical, not flat."
            parsed["counterfactual"] = "If all scientific and satellite data were falsified, the claim might appear true."
        elif "confirm" in evidence_joined or "proves" in evidence_joined:
            parsed["label"] = "True"
            parsed["confidence"] = 0.9
        elif "no evidence" in evidence_joined or "unclear" in evidence_joined:
            parsed["label"] = "Uncertain"
            parsed["confidence"] = 0.6

        return parsed
