import hashlib
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration", "baselines")


class BehavioralDriftModule:
    """
    Stores per-agent baseline embeddings (collected during calibration).
    Computes BD = 1 - cos(M_t, M_baseline) for every new agent output.
    """

    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.baselines: dict = {}
        self._drift_cache: dict = {}
        os.makedirs(BASELINE_DIR, exist_ok=True)
        self._load_baselines()

    def _load_baselines(self):
        if not os.path.isdir(BASELINE_DIR):
            return
        for fname in os.listdir(BASELINE_DIR):
            if fname.endswith(".json"):
                agent = fname.replace(".json", "")
                path = os.path.join(BASELINE_DIR, fname)
                with open(path) as f:
                    vec = json.load(f)
                vec = np.array(vec)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self.baselines[agent] = vec

    def save_baseline(self, agent_name: str, mean_embedding: np.ndarray):
        path = os.path.join(BASELINE_DIR, f"{agent_name}.json")
        with open(path, "w") as f:
            json.dump(mean_embedding.tolist(), f)
        self.baselines[agent_name] = mean_embedding

    def collect_baseline(self, agent_name: str, outputs: list):
        if not outputs:
            return
        unique_outputs = list(set(outputs))
        if len(unique_outputs) < 5:
            print(f"  WARNING: Only {len(unique_outputs)} unique outputs for {agent_name}. Baseline may be unreliable.")

        embeddings = self.embedder.encode(
            unique_outputs,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        mean_emb = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        self.save_baseline(agent_name, mean_emb)

    def compute_drift_with_similarity(self, agent_name: str, current_output: str) -> tuple:
        """Returns (baseline_similarity, global_similarity, BD)."""
        cache_key = f"{agent_name}:{hashlib.md5(current_output.encode()).hexdigest()}"
        if cache_key in self._drift_cache:
            cached = self._drift_cache[cache_key]
            return cached["baseline_sim"], cached["global_sim"], cached["bd"]

        if agent_name not in self.baselines:
            agent_name = next(iter(self.baselines.keys()), "Retriever")
        if agent_name not in self.baselines:
            return 1.0, 1.0, 0.0

        M_baseline = self.baselines[agent_name].reshape(1, -1)
        M_t = self.embedder.encode([current_output], normalize_embeddings=True)
        baseline_cos_sim = cosine_similarity(M_t, M_baseline)[0][0]
        baseline_similarity = float(baseline_cos_sim)
        global_centroid = self.get_global_centroid().reshape(1, -1)
        global_cos_sim = cosine_similarity(M_t, global_centroid)[0][0]
        global_similarity = float(global_cos_sim)
        BD = max(0.0, min(1.0, 1.0 - baseline_similarity))

        if len(self._drift_cache) < 1000:
            self._drift_cache[cache_key] = {
                "baseline_sim": baseline_similarity,
                "global_sim": global_similarity,
                "bd": BD,
            }
        return baseline_similarity, global_similarity, BD

    def compute_drift(self, agent_name: str, current_output: str) -> float:
        _, _, bd = self.compute_drift_with_similarity(agent_name, current_output)
        return bd

    def has_baseline(self, agent_name: str) -> bool:
        return agent_name in self.baselines

    def get_global_centroid(self) -> np.ndarray:
        if not self.baselines:
            return np.zeros(384)
        mean_emb = np.mean(list(self.baselines.values()), axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        return mean_emb
