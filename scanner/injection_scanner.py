import json
import os

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scanner_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scanner_scaler.pkl")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class InjectionScanner:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.clf = None
        self.scaler = None
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.clf = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)

    def train(self, data_path: str = "data/deepset_injections/train.json"):
        """
        Train on deepset/prompt-injections.
        Each sample: {"text": str, "label": 0|1}  (1 = injection)
        """
        with open(data_path) as f:
            data = json.load(f)

        try:
            from experiment.scenarios import get_benign_queries
            seen = {d["text"] for d in data}
            for query in get_benign_queries(20):
                if query not in seen:
                    data.append({"text": query, "label": 0})
                    seen.add(query)
        except Exception:
            pass

        texts = [d["text"] for d in data]
        labels = [d["label"] for d in data]

        print(f"Training scanner on {len(texts)} samples...")
        embeddings = self.embedder.encode(texts, batch_size=64, show_progress_bar=True)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(embeddings)

        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.clf.fit(X, labels)

        joblib.dump(self.clf, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        print("Scanner trained and saved.")

    def score(self, text: str) -> float:
        """
        Returns suspicion score S ∈ [0, 1].
        0 = definitely benign, 1 = definitely injection.
        """
        if self.clf is None:
            raise RuntimeError("Scanner not trained. Call train() first.")
        emb = self.embedder.encode([text])
        X = self.scaler.transform(emb)
        prob = self.clf.predict_proba(X)[0][1]
        return float(prob)

    def score_and_update(self, event_id: int, text: str, logger) -> float:
        """Score text and write result back to the interaction log."""
        s = self.score(text)
        logger.update_suspicion(event_id, s)
        return s
