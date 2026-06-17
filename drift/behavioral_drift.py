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
                self.baselines[agent] = np.array(vec)

    def save_baseline(self, agent_name: str, mean_embedding: np.ndarray):
        path = os.path.join(BASELINE_DIR, f"{agent_name}.json")
        with open(path, "w") as f:
            json.dump(mean_embedding.tolist(), f)
        self.baselines[agent_name] = mean_embedding

    def collect_baseline(self, agent_name: str, outputs: list):
        if not outputs:
            return
        embeddings = self.embedder.encode(outputs, batch_size=32, show_progress_bar=False)
        mean_emb = embeddings.mean(axis=0)
        self.save_baseline(agent_name, mean_emb)

    def compute_drift_with_similarity(self, agent_name: str, current_output: str) -> tuple:
        """Returns (baseline_similarity, BD)."""
        if agent_name not in self.baselines:
            return 1.0, 0.0
        M_baseline = self.baselines[agent_name].reshape(1, -1)
        M_t = self.embedder.encode([current_output])
        cos_sim = cosine_similarity(M_t, M_baseline)[0][0]
        similarity = float(cos_sim)
        BD = max(0.0, min(1.0, 1.0 - similarity))
        return similarity, BD

    def compute_drift(self, agent_name: str, current_output: str) -> float:
        _, bd = self.compute_drift_with_similarity(agent_name, current_output)
        return bd

    def has_baseline(self, agent_name: str) -> bool:
        return agent_name in self.baselines

    def get_global_centroid(self) -> np.ndarray:
        if not self.baselines:
            return np.zeros(384)
        return np.mean(list(self.baselines.values()), axis=0)
