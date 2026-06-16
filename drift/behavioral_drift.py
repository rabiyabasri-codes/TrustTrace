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
    BD ∈ [0, 1]: 0 = identical to baseline, 1 = completely different.
    """

    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.baselines: dict = {}   # agent_name → mean embedding
        os.makedirs(BASELINE_DIR, exist_ok=True)
        self._load_baselines()

    # ── Baseline management ────────────────────────────────────────────────────

    def _load_baselines(self):
        """Load any saved baselines from disk."""
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
        """Persist a baseline embedding to disk."""
        path = os.path.join(BASELINE_DIR, f"{agent_name}.json")
        with open(path, "w") as f:
            json.dump(mean_embedding.tolist(), f)
        self.baselines[agent_name] = mean_embedding

    def collect_baseline(self, agent_name: str, outputs: list):
        """
        Called during Trust Calibration phase (Change 8).
        Embeds all clean outputs for an agent and stores the mean as its baseline.
        """
        if not outputs:
            return
        print(f"  Computing baseline for {agent_name} from {len(outputs)} outputs...")
        embeddings = self.embedder.encode(outputs, batch_size=32, show_progress_bar=False)
        mean_emb = embeddings.mean(axis=0)
        self.save_baseline(agent_name, mean_emb)
        print(f"  Baseline saved for {agent_name}.")

    # ── Runtime drift computation ──────────────────────────────────────────────

    def compute_drift(self, agent_name: str, current_output: str) -> float:
        """
        BD = 1 - cos(M_t, M_baseline)
        Returns 0.0 if no baseline exists for this agent (fail-safe).
        """
        if agent_name not in self.baselines:
            return 0.0  # no baseline = no drift signal

        M_baseline = self.baselines[agent_name].reshape(1, -1)
        M_t = self.embedder.encode([current_output])
        cos_sim = cosine_similarity(M_t, M_baseline)[0][0]
        BD = 1.0 - float(cos_sim)
        return max(0.0, min(1.0, BD))  # clamp to [0,1]

    def has_baseline(self, agent_name: str) -> bool:
        return agent_name in self.baselines
