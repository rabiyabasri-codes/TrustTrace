import time
import uuid
from typing import Dict, List, Optional, Set

import chromadb


class MemoryManager:
    """
    Wraps ChromaDB with checkpoint versioning.
    Tracks which agent wrote each entry and when.
    Supports rollback: remove all entries written after a compromise timestamp.
    """

    def __init__(self, collection_name: str = "pipeline_memory", path: Optional[str] = None):
        if path is not None:
            client = chromadb.PersistentClient(path=path)
        else:
            client = chromadb.Client()
        self.client = client
        self.collection = self.client.get_or_create_collection(collection_name)
        self.write_log: List[Dict] = []

    def write(self, content: str, metadata: Dict, agent: str, trust_at_write: float) -> str:
        """Write an entry. Records agent identity and trust score at time of write."""
        entry_id = str(uuid.uuid4())
        ts = time.time()
        full_meta = {
            **metadata,
            "agent": agent,
            "timestamp": ts,
            "trust_at_write": trust_at_write,
            "entry_id": entry_id,
        }
        self.collection.add(
            documents=[content],
            metadatas=[full_meta],
            ids=[entry_id],
        )
        self.write_log.append({"id": entry_id, "timestamp": ts, "agent": agent})
        return entry_id

    def read(self, query: str, n_results: int = 3) -> list:
        """Semantic search over memory."""
        # Clamp requested results to the number of elements in the collection
        try:
            size = None
            if hasattr(self.collection, "count") and callable(self.collection.count):
                size = int(self.collection.count())
            else:
                info = self.collection.get(include=["ids"]) if hasattr(self.collection, "get") else {}
                ids = info.get("ids", []) if isinstance(info, dict) else []
                size = len(ids)
            n = max(1, min(n_results, size if size is not None else n_results))
        except Exception:
            n = n_results

        results = self.collection.query(query_texts=[query], n_results=n)
        docs = results.get("documents", [[]])[0]
        # Normalize results to strings and filter out None/empty
        docs = [str(d) for d in docs if d is not None and str(d).strip()]
        return docs

    def checkpoint(self) -> float:
        """Return current timestamp as a checkpoint marker."""
        return time.time()

    def rollback_after(self, timestamp: float, compromised_agents: Set[str]) -> int:
        """
        Remove all entries written after `timestamp` by `compromised_agents`.
        This is the memory rollback step in the recovery sequence.
        """
        to_delete = [
            entry["id"]
            for entry in self.write_log
            if entry["timestamp"] >= timestamp and entry["agent"] in compromised_agents
        ]
        if to_delete:
            self.collection.delete(ids=to_delete)
            self.write_log = [e for e in self.write_log if e["id"] not in to_delete]
            print(f"  Memory rollback: removed {len(to_delete)} entries.")
        return len(to_delete)

    def get_last_clean_checkpoint(self, before_timestamp: float) -> Optional[float]:
        """Return the most recent write timestamp before a given time."""
        clean = [e["timestamp"] for e in self.write_log if e["timestamp"] < before_timestamp]
        return max(clean) if clean else None

