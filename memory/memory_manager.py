import os
import time
import uuid
from typing import Dict, List, Optional, Set

from irs.injection_risk import InjectionRiskAssessor


class MemoryManager:
    """ChromaDB memory with Memory Trust tracking: MT = T_a * (1 - IRS)."""

    def __init__(self, collection_name: str = "pipeline_memory", collection=None, path: Optional[str] = None):
        if collection is not None:
            self.collection = collection
            self.client = collection._client if hasattr(collection, "_client") else None
        else:
            from victim_pipeline.chroma_store import chroma_client
            self.client = chroma_client
            self.collection = self.client.get_or_create_collection(collection_name)
        self.write_log: List[Dict] = []
        self.checkpoint_counter = 0

    def write(
        self,
        content: str,
        metadata: Dict,
        agent: str,
        trust_at_write: float,
        irs: float = 0.0,
        memory_trust: Optional[float] = None,
    ) -> str:
        entry_id = str(uuid.uuid4())
        ts = time.time()
        if memory_trust is None:
            memory_trust = InjectionRiskAssessor.compute_memory_trust(irs, trust_at_write)
        full_meta = {
            **metadata,
            "agent": agent,
            "timestamp": ts,
            "trust_at_write": trust_at_write,
            "irs": irs,
            "memory_trust": memory_trust,
            "entry_id": entry_id,
        }
        self.collection.add(documents=[content], metadatas=[full_meta], ids=[entry_id])
        self.write_log.append({
            "id": entry_id,
            "timestamp": ts,
            "agent": agent,
            "irs": irs,
            "trust_at_write": trust_at_write,
            "memory_trust": memory_trust,
        })
        return entry_id

    def read(self, query: str, n_results: int = 3) -> list:
        try:
            size = int(self.collection.count()) if hasattr(self.collection, "count") else n_results
            n = max(1, min(n_results, size if size else n_results))
        except Exception:
            n = n_results
        results = self.collection.query(query_texts=[query], n_results=n)
        return [str(d) for d in results.get("documents", [[]])[0] if d]

    def checkpoint(self) -> str:
        self.checkpoint_counter += 1
        return f"checkpoint_{self.checkpoint_counter}"

    def rollback_after(self, timestamp: float, compromised_agents: Set[str]) -> int:
        to_delete = [
            entry["id"]
            for entry in self.write_log
            if entry["timestamp"] >= timestamp and entry["agent"] in compromised_agents
        ]
        if to_delete:
            self.collection.delete(ids=to_delete)
            self.write_log = [e for e in self.write_log if e["id"] not in to_delete]
        return len(to_delete)

    def get_last_clean_checkpoint(self, before_timestamp: float) -> Optional[float]:
        clean = [e["timestamp"] for e in self.write_log if e["timestamp"] < before_timestamp]
        return max(clean) if clean else None
