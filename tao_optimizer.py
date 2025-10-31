# ================================================================
# src/tao_optimizer.py
# ================================================================
"""
TAO Optimizer: Training, Adaptation, and Optimization Engine
------------------------------------------------------------
This module simulates real-time model adaptation and dynamic decoding optimization
to make the Fake News Detector more context-aware and self-improving.
"""

import random
import time
import torch

class TAOOptimizer:
    """Implements dynamic generation parameter optimization and continual fine-tuning."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "google/flan-t5-base"
        self.training_stats = {"steps": 0, "updates": 0, "loss": 0.0}
        self.last_update = time.time()
        print(f"[INIT] TAO Optimizer initialized for model: {self.model_name}")

    # ------------------------------------------------------------
    # Step 1: Dynamic Decoding Parameter Optimization
    # ------------------------------------------------------------
    def optimize_generation(self):
        """
        Dynamically adjusts generation parameters based on pseudo-online conditions.
        Returns a dictionary with decoding parameters only (1 value).
        """
        try:
            # Simulate adaptive parameter tuning using randomization
            temperature = round(random.uniform(0.6, 0.9), 2)
            num_beams = random.choice([2, 4, 6])
            repetition_penalty = round(random.uniform(1.0, 1.3), 2)

            decoding_params = {
                "temperature": temperature,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty
            }

            print(f"[TAO] Optimized decoding params: {decoding_params}")
            return decoding_params  # ✅ Return only one dictionary, not multiple values

        except Exception as e:
            print("[TAO Error] optimize_generation:", e)
            return {"temperature": 0.7, "num_beams": 4, "repetition_penalty": 1.2}

    # ------------------------------------------------------------
    # Step 2: Continual Fine-tuning Simulation
    # ------------------------------------------------------------
    def continual_train(self, data_batch):
        """
        Simulates continual fine-tuning on small batches of labeled claims.
        Each update slightly improves internal state without full retraining.
        """
        try:
            if not data_batch or not isinstance(data_batch, list):
                return "No data provided for TAO training."

            # Simulate light-weight online adaptation
            self.training_stats["steps"] += len(data_batch)
            self.training_stats["updates"] += 1
            self.training_stats["loss"] = max(
                0.01, 0.5 - 0.05 * (self.training_stats["updates"])
            )
            self.last_update = time.time()

            # Mock GPU operation check
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[TAO] Continual training on {len(data_batch)} samples ({device})")

            return f"TAO adapted: {self.training_stats['updates']} updates, loss={self.training_stats['loss']:.3f}"

        except Exception as e:
            print("[TAO Error] continual_train:", e)
            return f"TAO failed: {e}"
