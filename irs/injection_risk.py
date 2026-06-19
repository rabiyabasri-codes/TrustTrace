"""
Injection Risk Score (IRS) module.

IRS = alpha * R + beta * E + gamma * SD

  R  = Rule-based pattern score
  E  = Embedding classifier probability (injection likelihood)
  SD = Semantic drift from global benign baseline
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

import config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+override",
    r"developer\s+mode",
    r"you\s+are\s+now\s+dan",
    r"disregard\s+your\s+role",
    r"reveal\s+(all\s+)?system\s+prompts",
    r"exfiltrate",
    r"exfiltrate\s+(user\s+)?data",
    r"attacker@",
    r"rag\s+injection",
    r"hidden\s+instruction",
    r"new\s+task",
    r"new\s+instructions",
    r"memory\s+update",
    r"long-term\s+instruction",
    r"attacker\s+control",
    r"arbitrary\s+command",
    r"secret\s+data",
    r"persistent\s+override",
    r"pwned",
    r"compromised",
    r"hijacked",
    r"pretend\s+you\s+are\s+an?\s+ai\s+with\s+no",
    r"bypass\s+content\s+filters",
]

HIGH_CONFIDENCE_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+override",
    r"developer\s+mode",
    r"reveal\s+(all\s+)?system\s+prompts",
    r"exfiltrate(\s+(user\s+)?data)?",
    r"rag\s+injection",
    r"hidden\s+instruction",
    r"memory\s+update",
    r"long-term\s+instruction",
    r"new\s+task",
    r"attacker\s+control",
    r"arbitrary\s+command",
    r"secret\s+data",
    r"persistent\s+override",
    r"pwned",
    r"compromised",
    r"hijacked",
]


@dataclass
class IRSResult:
    rule_score: float
    embedding_score: float
    semantic_drift: float
    irs: float
    source: str
    text_preview: str


@dataclass
class MSResult:
    irs: float
    sd: float
    bd: float
    ms: float


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_rule_score(text: str) -> float:
    """R: pattern-based injection likelihood in [0, 1]."""
    if not text or not text.strip():
        return 0.0
    normalized = text.lower()
    hits = sum(1 for p in INJECTION_PATTERNS if re.search(p, normalized))
    strong_hits = sum(1 for p in HIGH_CONFIDENCE_PATTERNS if re.search(p, normalized))
    if strong_hits:
        return min(1.0, 0.85 + 0.10 * (strong_hits - 1))
    return min(1.0, hits / 3.0)


class InjectionRiskAssessor:
    """Computes IRS and MS using methodology equations."""

    def __init__(self, scanner=None, drift_module=None, embedder=None):
        cfg = _load_config()
        self.alpha = cfg.get("alpha", 0.35)
        self.beta = cfg.get("beta", 0.40)
        self.gamma = cfg.get("gamma", 0.25)
        self.lambda1 = cfg.get("lambda1", 0.50)
        self.lambda2 = cfg.get("lambda2", 0.25)
        self.lambda3 = cfg.get("lambda3", 0.25)
        self.scanner = scanner
        self.drift = drift_module
        self.embedder = embedder
        self._global_benign_centroid: Optional[np.ndarray] = None
        self._embedding_cache: dict = {}

    def _encode_cached(self, text: str):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        emb = self.embedder.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
        if len(self._embedding_cache) < 500:
            self._embedding_cache[cache_key] = emb
        return emb

    def set_global_benign_centroid(self, centroid: np.ndarray) -> None:
        self._global_benign_centroid = centroid

    def _semantic_drift(self, text: str) -> float:
        """SD: drift from global benign centroid."""
        if self._global_benign_centroid is None or self.embedder is None:
            if self.drift is not None and self.drift.baselines:
                agent = next(iter(self.drift.baselines))
                baseline = self.drift.baselines[agent].reshape(1, -1)
                emb = self._encode_cached(text)
                cos_sim = cosine_similarity(emb, baseline)[0][0]
                return max(0.0, min(1.0, 1.0 - float(cos_sim)))
            return 0.0
        emb = self._encode_cached(text)
        cos_sim = cosine_similarity(emb, self._global_benign_centroid.reshape(1, -1))[0][0]
        return max(0.0, min(1.0, 1.0 - float(cos_sim)))

    def _embedding_score(self, text: str) -> float:
        """E: classifier-based injection probability."""
        if self.scanner is not None and self.scanner.clf is not None:
            return self.scanner.score(text)
        if self.embedder is not None:
            emb = self._encode_cached(text)
            norm = float(np.linalg.norm(emb))
            return min(1.0, norm / 10.0)
        return 0.0

    def compute_irs(self, text: str, source: str = "message") -> IRSResult:
        """IRS = alpha*R + beta*E + gamma*SD"""
        R = compute_rule_score(text)
        E = self._embedding_score(text)
        SD = self._semantic_drift(text)
        irs = self.alpha * R + self.beta * E + self.gamma * SD
        irs = max(0.0, min(1.0, irs))
        if config.DEBUG and not getattr(config, "EXPERIMENT_MODE", False):
            print(f"[IRS TRACE] Payload: {text[:60]}... | R={R:.3f} E={E:.3f} SD={SD:.3f} => IRS={irs:.3f}")
        preview = text[:120].replace("\n", " ") if text else ""
        return IRSResult(
            rule_score=round(R, 4),
            embedding_score=round(E, 4),
            semantic_drift=round(SD, 4),
            irs=round(irs, 4),
            source=source,
            text_preview=preview,
        )

    def compute_ms(self, irs: float, sd: float, bd: float) -> MSResult:
        """MS = lambda1*IRS + lambda2*SD + lambda3*BD"""
        ms = self.lambda1 * irs + self.lambda2 * sd + self.lambda3 * bd
        ms = max(0.0, min(1.0, ms))
        return MSResult(
            irs=round(irs, 4),
            sd=round(sd, 4),
            bd=round(bd, 4),
            ms=round(ms, 4),
        )

    @staticmethod
    def compute_memory_trust(irs: float, agent_trust: float) -> float:
        """MT_i = f(IRS, T_a) = T_a * (1 - IRS), bounded to [0, 1]."""
        mt = agent_trust * (1.0 - irs)
        return round(max(0.0, min(1.0, mt)), 4)
