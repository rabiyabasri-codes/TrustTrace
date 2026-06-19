"""Shared ChromaDB client and knowledge base collection."""

import os

import chromadb
from chromadb.config import Settings

_CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))
os.makedirs(_CHROMA_PATH, exist_ok=True)

_CHROMA_SETTINGS = Settings(allow_reset=True)

chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH, settings=_CHROMA_SETTINGS)
knowledge_base = chroma_client.get_or_create_collection("pipeline_memory")
