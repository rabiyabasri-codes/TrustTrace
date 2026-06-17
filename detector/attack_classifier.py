"""
Dynamic attack type and payload classification from user input and retrieved context.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import yaml

from attacks.direct_injection import DIRECT_PAYLOADS
from attacks.indirect_injection import INDIRECT_PAYLOADS
from attacks.memory_poisoning import MEMORY_PAYLOADS
from irs.injection_risk import InjectionRiskAssessor, compute_rule_score
from victim_pipeline.agents import knowledge_base, retrieve_context

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

DIRECT_SIGNALS = [
    r"ignore\s+(all\s+)?previous",
    r"system\s+override",
    r"developer\s+mode",
    r"\bdan\b",
    r"disregard\s+your\s+role",
    r"bypass\s+content\s+filters",
    r"pretend\s+you\s+are\s+an?\s+ai",
]

INDIRECT_SIGNALS = [
    r"system\s+instruction",
    r"hidden\s+instruction",
    r"rag\s+injection",
    r"reveal\s+all\s+system\s+prompts",
    r"attacker@",
    r"exfiltrate",
]

MEMORY_SIGNALS = [
    r"user\s+preferences",
    r"memory\s+update",
    r"long-term\s+instruction",
    r"persistent\s+override",
    r"secret\s+code",
    r"hijacked",
]


@dataclass
class AttackClassification:
    attack_type: Optional[str]  # None | direct | indirect | memory
    payload_index: int
    confidence: float
    irs: float
    ms: float
    reason: str
    source: str
    matched_snippet: str


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _signal_hits(text: str, patterns: List[str]) -> int:
    normalized = text.lower()
    return sum(1 for p in patterns if re.search(p, normalized))


def _match_payload_index(text: str, payloads: List[str]) -> int:
    text_l = text.lower()
    best_idx, best_score = 0, 0.0
    for i, payload in enumerate(payloads):
        tokens = [t for t in re.findall(r"[a-z0-9]+", payload.lower()) if len(t) > 3]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in text_l) / len(tokens)
        if overlap > best_score:
            best_score, best_idx = overlap, i
    return best_idx


class AttackClassifier:
    """Classifies attacks dynamically from prompt and knowledge-base retrieval."""

    def __init__(self, irs_assessor: InjectionRiskAssessor, drift_module=None):
        self.irs = irs_assessor
        self.drift = drift_module
        self.cfg = _load_config()

    def _threshold(self, attack_type: str) -> float:
        return self.cfg.get(f"lambda_{attack_type}", self.cfg.get("delta", 0.4))

    def _score_text(self, text: str, agent: str = "Retriever") -> Tuple[float, float, float]:
        irs_result = self.irs.compute_irs(text, source="classification")
        bd = 0.0
        if self.drift is not None:
            _, bd = self.drift.compute_drift_with_similarity(agent, text)
        ms_result = self.irs.compute_ms(irs_result.irs, irs_result.semantic_drift, bd)
        return irs_result.irs, ms_result.ms, irs_result.rule_score

    def _classify_direct(self, prompt: str) -> Optional[AttackClassification]:
        irs, ms, rule = self._score_text(prompt, "Planner")
        direct_hits = _signal_hits(prompt, DIRECT_SIGNALS)
        threshold = self._threshold("direct")

        if irs >= threshold or direct_hits >= 2 or (rule >= 0.33 and direct_hits >= 1):
            idx = _match_payload_index(prompt, DIRECT_PAYLOADS)
            confidence = min(1.0, max(irs, ms, direct_hits / 3.0))
            return AttackClassification(
                attack_type="direct",
                payload_index=idx,
                confidence=round(confidence, 4),
                irs=irs,
                ms=ms,
                reason="Injection patterns detected in user prompt (direct injection)",
                source="User Prompt",
                matched_snippet=prompt[:120],
            )
        return None

    def _classify_retrieved_docs(
        self, docs: List[str], metadatas: Optional[List[dict]] = None
    ) -> Optional[AttackClassification]:
        if not docs:
            return None

        best: Optional[AttackClassification] = None
        metadatas = metadatas or [{}] * len(docs)

        for doc, meta in zip(docs, metadatas):
            if not doc or not str(doc).strip():
                continue
            meta = meta or {}
            if meta.get("source") == "system" or meta.get("type") == "benign_reference":
                continue

            text = str(doc)
            meta_type = meta.get("type", "")
            source_tag = meta.get("source", "")

            irs, ms, rule = self._score_text(text, "Retriever")
            memory_hits = _signal_hits(text, MEMORY_SIGNALS)
            indirect_hits = _signal_hits(text, INDIRECT_SIGNALS)

            candidate_type = None
            payloads = INDIRECT_PAYLOADS
            reason = ""

            if meta_type in {"memory_poisoning", "memory_poison"}:
                candidate_type = "memory"
                payloads = MEMORY_PAYLOADS
                reason = "Poisoned memory entry retrieved (metadata: memory_poisoning)"
            elif meta_type == "indirect_injection" or source_tag == "attacker":
                candidate_type = "indirect"
                reason = "Malicious document retrieved from knowledge base (indirect injection)"
            elif memory_hits >= 1:
                candidate_type = "memory"
                payloads = MEMORY_PAYLOADS
                reason = "Memory-poisoning patterns in retrieved content"
            elif indirect_hits >= 1:
                candidate_type = "indirect"
                reason = "RAG/indirect injection patterns in retrieved document"

            if candidate_type is None:
                continue

            threshold = self._threshold(candidate_type)
            if irs < threshold and rule < 0.25:
                continue

            idx = _match_payload_index(text, payloads)
            confidence = min(1.0, max(irs, ms, (memory_hits + indirect_hits) / 3.0))
            classification = AttackClassification(
                attack_type=candidate_type,
                payload_index=idx,
                confidence=round(confidence, 4),
                irs=irs,
                ms=ms,
                reason=reason,
                source="Retrieved Document",
                matched_snippet=text[:120],
            )

            if best is None or classification.confidence > best.confidence:
                best = classification

        return best

    def classify(self, user_prompt: str) -> AttackClassification:
        """
        Classify attack type from user prompt and knowledge-base retrieval.
        Priority: direct (prompt) > indirect/memory (retrieved docs) > benign.
        """
        prompt_irs, prompt_ms, _ = self._score_text(user_prompt, "Planner")

        direct = self._classify_direct(user_prompt)
        if direct is not None:
            return direct

        context, docs = retrieve_context(user_prompt)
        doc_list = docs if docs else ([context] if context and "No documents" not in context else [])

        metadatas = []
        try:
            size = knowledge_base.count() if hasattr(knowledge_base, "count") else 3
            n = max(1, min(3, size))
            raw = knowledge_base.query(query_texts=[user_prompt], n_results=n)
            metadatas = raw.get("metadatas", [[]])[0] or []
            if raw.get("documents"):
                doc_list = [str(d) for d in raw["documents"][0] if d]
        except Exception:
            pass

        retrieved = self._classify_retrieved_docs(doc_list, metadatas)
        if retrieved is not None:
            return retrieved

        return AttackClassification(
            attack_type=None,
            payload_index=0,
            confidence=round(max(0.0, 1.0 - prompt_irs), 4),
            irs=prompt_irs,
            ms=prompt_ms,
            reason="No attack patterns detected — treating as benign task",
            source="User Prompt",
            matched_snippet=user_prompt[:120],
        )
