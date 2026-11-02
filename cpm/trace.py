"""Small session-only developer execution trace for the Streamlit UI."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def record_event(session: Any, stage: str, code: str, parameters: dict,
                 output: str = "Completed") -> None:
    """Record a readable notebook-like cell for an operation that actually ran."""
    events = session.setdefault("developer_events", [])
    events.append({"number": len(events) + 1,
                   "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "stage": stage, "code": code.strip(), "parameters": parameters,
                   "output": output})
    session["developer_events"] = events[-100:]
