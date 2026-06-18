import os
import shutil
import datetime
import chromadb

def ensure_chroma_integrity(persistence_path: str = None) -> None:
    """Check ChromaDB persistence folder integrity.

    If the folder is missing or cannot be opened (e.g., version mismatch or corruption),
    it is backed up to `chroma_db_backup_<timestamp>` and a fresh PersistentClient is
    initialized at the original location.
    """
    # Determine default path consistent with MemoryManager
    if persistence_path is None:
        # Default relative to this file's directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"))
        persistence_path = base_dir
    # Ensure the directory exists
    os.makedirs(persistence_path, exist_ok=True)
    try:
        # Attempt to create a client; this will raise if the DB is corrupted
        _ = chromadb.PersistentClient(path=persistence_path)
    except Exception as e:
        # Backup current folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{persistence_path}_backup_{timestamp}"
        try:
            shutil.move(persistence_path, backup_path)
            print(f"[Chroma Recovery] Backed up corrupted DB to {backup_path}")
        except Exception as be:
            print(f"[Chroma Recovery] Failed to backup corrupted DB: {be}")
        # Recreate fresh folder
        os.makedirs(persistence_path, exist_ok=True)
        # Initialize new client to create necessary files
        _ = chromadb.PersistentClient(path=persistence_path)
        print(f"[Chroma Recovery] Created new empty ChromaDB at {persistence_path}")
    else:
        # No exception, DB appears healthy
        if isinstance(persistence_path, str) and os.listdir(persistence_path):
            # Optionally could verify version files here
            pass
        # Optionally print status in debug mode
        from config import DEBUG
        if DEBUG:
            print(f"[Chroma Recovery] ChromaDB at {persistence_path} is healthy.")
