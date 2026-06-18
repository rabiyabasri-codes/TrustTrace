import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum
from config import DEBUG  # central debug flag

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "interactions.db")

@dataclass
class InteractionEvent:
    sender: str
    receiver: str
    timestamp: float
    message_content: str
    event_type: str           # "message" | "memory_write" | "memory_read" | "tool_call"
    tool_name: Optional[str]  # populated for tool_call events
    memory_key: Optional[str] # populated for memory events
    suspicion_score: float = 0.0  # filled in by scanner later
    run_id: str = ""

class InteractionLogger:
    """
    Middleware that sits between the victim pipeline and all TrustTrace modules.
    Records every inter-agent event to SQLite.
    No classification here — only recording.
    """

    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT,
                    sender      TEXT NOT NULL,
                    receiver    TEXT NOT NULL,
                    timestamp   REAL NOT NULL,
                    message_content TEXT,
                    event_type  TEXT NOT NULL,
                    tool_name   TEXT,
                    memory_key  TEXT,
                    suspicion_score REAL DEFAULT 0.0
                )
            """)
            conn.commit()

    @staticmethod
    def serialize_sqlite(value):
        """Convert unsupported types to SQLite‑compatible representations."""
        if isinstance(value, (dict, list, set)):
            return json.dumps(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return str(value.value)
        if value is None:
            return None
        # Preserve basic types; otherwise stringify
        return value if isinstance(value, (str, int, float, bool)) else str(value)

    def log(self, event: InteractionEvent) -> int:
        """Insert event, return its event_id. Includes optional debug prints."""
        raw_params = [
            event.run_id,
            event.sender,
            event.receiver,
            event.timestamp,
            event.message_content,
            event.event_type,
            event.tool_name,
            event.memory_key,
            event.suspicion_score,
        ]
        if DEBUG:
            for i, value in enumerate(raw_params):
                print(f"[SQL Debug] Param {i}: type={type(value)} value={repr(value)[:200]}")
        params = [self.serialize_sqlite(v) for v in raw_params]
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO interactions
                  (run_id, sender, receiver, timestamp, message_content,
                   event_type, tool_name, memory_key, suspicion_score)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                params,
            )
            conn.commit()
            return cur.lastrowid

    def update_suspicion(self, event_id: int, score: float):
        """Called by the scanner to fill in the suspicion score after classification."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE interactions SET suspicion_score=? WHERE event_id=?",
                (score, event_id)
            )
            conn.commit()

    def get_events_for_run(self, run_id: str) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM interactions WHERE run_id=? ORDER BY timestamp",
                (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_events_after(self, timestamp: float) -> list:
        """Used by recovery manager to find entries to roll back."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM interactions WHERE timestamp >= ? ORDER BY timestamp",
                (timestamp,)
            ).fetchall()
        return [dict(r) for r in rows]

# ── Wrapper that hooks into the pipeline ───────────────────────────────────────

def log_pipeline_run(pipeline_outputs: dict, run_id: str, logger: InteractionLogger):
    """
    Convert pipeline_outputs (dict of agent_name → output_text)
    into sequential InteractionEvents and log them.
    Call this immediately after run_pipeline() returns.
    """
    agent_order = ["Retriever", "Planner", "Executor", "Generator"]
    ts = time.time()
    for i, sender in enumerate(agent_order[:-1]):
        receiver = agent_order[i + 1]
        content = pipeline_outputs.get(sender, "")
        event = InteractionEvent(
            sender=sender,
            receiver=receiver,
            timestamp=ts + i * 0.001,
            message_content=content,
            event_type="message",
            tool_name=None,
            memory_key=None,
            run_id=run_id,
        )
        logger.log(event)
