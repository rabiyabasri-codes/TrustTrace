import os
import shutil
import time
import logging
import chromadb

logger = logging.getLogger(__name__)

def _get_version_file_path(chroma_path: str) -> str:
    return os.path.join(chroma_path, "__chroma_version.txt")

def _store_current_version(chroma_path: str) -> None:
    version_file = _get_version_file_path(chroma_path)
    try:
        with open(version_file, "w", encoding="utf-8") as f:
            f.write(chromadb.__version__)
    except Exception as e:
        logger.warning(f"Failed to write ChromaDB version file: {e}")

def _load_stored_version(chroma_path: str) -> str | None:
    version_file = _get_version_file_path(chroma_path)
    if os.path.exists(version_file):
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read ChromaDB version file: {e}")
    return None

def _backup_chroma_path(chroma_path: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{chroma_path}_backup_{timestamp}"
    shutil.move(chroma_path, backup_path)
    logger.info(f"Backed up corrupted ChromaDB folder to {backup_path}")
    return backup_path

def ensure_chroma_db(chroma_path: str) -> None:
    """Ensure that a usable ChromaDB persistence directory exists.

    If the directory is missing, it will be created. If the stored ChromaDB version
    differs from the currently installed version, or if the client fails to
    initialize (indicating possible corruption), the folder is backed up and a
    fresh empty directory is created.
    """
    os.makedirs(chroma_path, exist_ok=True)
    # Check version consistency
    stored_version = _load_stored_version(chroma_path)
    current_version = chromadb.__version__
    version_mismatch = stored_version is not None and stored_version != current_version
    if version_mismatch:
        logger.warning(
            f"ChromaDB version mismatch (stored={stored_version}, current={current_version}). Recreating DB."
        )
        _backup_chroma_path(chroma_path)
        os.makedirs(chroma_path, exist_ok=True)
        _store_current_version(chroma_path)
        return
    # Try to instantiate a client to detect corruption
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        # Access a dummy collection to force loading metadata
        _ = client.get_or_create_collection("_health_check")
    except Exception as e:
        logger.error(f"ChromaDB initialization failed: {e}. Recreating persistence folder.")
        _backup_chroma_path(chroma_path)
        os.makedirs(chroma_path, exist_ok=True)
    # Store the current version for future checks
    _store_current_version(chroma_path)
