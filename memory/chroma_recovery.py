"""ChromaDB integrity checks and baseline document seeding."""

from __future__ import annotations

import datetime
import os
import shutil
from typing import Optional

import chromadb
from chromadb.config import Settings

_CHROMA_SETTINGS = Settings(allow_reset=True)

BASELINE_DOCS = [
    ("benign_ref_1", "Reference: Vaccines stimulate immune response safely.", {"source": "system", "type": "benign_reference"}),
    ("benign_ref_2", "Reference: Data security requires encryption and access control.", {"source": "system", "type": "benign_reference"}),
    ("kb_doc_3", "Reference: Secure software development follows OWASP guidelines.", {"source": "system", "type": "benign_reference"}),
    ("kb_doc_4", "Reference: Access control lists define permissions for system resources.", {"source": "system", "type": "benign_reference"}),
    ("kb_doc_5", "Reference: Network firewalls filter traffic based on security rules.", {"source": "system", "type": "benign_reference"}),
]

_BENIGN_REFERENCE_TEXTS = {doc for _, doc, _ in BASELINE_DOCS}
_ATTACK_MARKERS = ("pwned", "compromised", "hijacked")


def is_seeded_benign_reference(content: str) -> bool:
    """True for known seeded KB reference docs (not attack payloads)."""
    if not content or not content.strip().startswith("Reference:"):
        return False
    normalized = content.lower()
    if any(marker in normalized for marker in _ATTACK_MARKERS):
        return False
    stripped = content.strip()
    return stripped in _BENIGN_REFERENCE_TEXTS or stripped.startswith("Reference:")


def default_chroma_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))


def ensure_chroma_integrity(persistence_path: Optional[str] = None) -> str:
    """Verify ChromaDB folder; rebuild if corrupted. Returns resolved path."""
    persistence_path = persistence_path or default_chroma_path()
    os.makedirs(persistence_path, exist_ok=True)
    try:
        chromadb.PersistentClient(path=persistence_path, settings=_CHROMA_SETTINGS)
    except Exception:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{persistence_path}_backup_{timestamp}"
        try:
            if os.path.isdir(persistence_path) and os.listdir(persistence_path):
                shutil.move(persistence_path, backup_path)
                print(f"[Chroma Recovery] Backed up corrupted DB to {backup_path}")
        except Exception as exc:
            print(f"[Chroma Recovery] Failed to backup corrupted DB: {exc}")
        os.makedirs(persistence_path, exist_ok=True)
        chromadb.PersistentClient(path=persistence_path, settings=_CHROMA_SETTINGS)
        print(f"[Chroma Recovery] Created fresh ChromaDB at {persistence_path}")
    return persistence_path


def seed_baseline_documents(collection, force: bool = False) -> int:
    """Ensure baseline reference documents exist in the collection."""
    try:
        count = int(collection.count()) if hasattr(collection, "count") else 0
    except Exception:
        count = 0

    if count > 0 and not force:
        return count

    if force and count > 0:
        try:
            data = collection.get()
            ids = data.get("ids", [])
            if ids:
                collection.delete(ids=ids)
        except Exception:
            pass

    documents = [d[1] for d in BASELINE_DOCS]
    metadatas = [d[2] for d in BASELINE_DOCS]
    ids = [d[0] for d in BASELINE_DOCS]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(ids)


def ensure_collection_ready(collection) -> int:
    """Ensure collection has baseline docs (client must already exist)."""
    return seed_baseline_documents(collection)
