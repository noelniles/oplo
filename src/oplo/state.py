# src/oplo/state.py
from typing import Any, Dict

import uuid
from flask import has_request_context, session

class Store:
    def __init__(self):
        # per-session state keyed by a generated UUID stored in Flask session
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _get_session_id(self) -> str | None:
        if not has_request_context():
            return None
        sid = session.get("oplo_session_id")
        if sid is None:
            sid = uuid.uuid4().hex
            session["oplo_session_id"] = sid
        if sid not in self._sessions:
            self._sessions[sid] = {"bundle": None, "disp": None, "scale": 1.0}
        return sid

    def set(self, **kwargs):
        sid = self._get_session_id()
        if sid is None:
            raise RuntimeError("State access requires a request context")
        self._sessions[sid].update(kwargs)

    def get(self, key: str, default=None):
        sid = self._get_session_id()
        if sid is None:
            return default
        return self._sessions.get(sid, {}).get(key, default)

    def clear(self):
        sid = self._get_session_id()
        if sid is None:
            return
        self._sessions[sid] = {"bundle": None, "disp": None, "scale": 1.0}

STORE = Store()
