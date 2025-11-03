# src/oplo/state.py
from typing import Any, Dict

class Store:
    def __init__(self):
        # bundle: ImageBundle (raw, meta, calib)
        # disp: np.ndarray (float32 [0,1] downsampled for display)
        # history: list of (bundle, label) tuples for stepping through processing stages
        # history_index: current position in history
        self.data: Dict[str, Any] = {
            "bundle": None, 
            "disp": None,
            "history": [],
            "history_index": -1,
        }

    def set(self, **kwargs):
        self.data.update(kwargs)

    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def push_history(self, bundle, label: str):
        """Add a bundle to history stack with a label."""
        history = self.data.get("history", [])
        idx = self.data.get("history_index", -1)
        # Truncate future history if we're not at the end
        if idx >= 0 and idx < len(history) - 1:
            history = history[:idx + 1]
        history.append((bundle, label))
        self.data["history"] = history
        self.data["history_index"] = len(history) - 1
        self.data["bundle"] = bundle
    
    def get_history_length(self) -> int:
        return len(self.data.get("history", []))
    
    def get_history_index(self) -> int:
        return self.data.get("history_index", -1)
    
    def set_history_index(self, idx: int):
        """Navigate to a specific history index."""
        history = self.data.get("history", [])
        if 0 <= idx < len(history):
            self.data["history_index"] = idx
            self.data["bundle"] = history[idx][0]
            return True
        return False
    
    def get_history_labels(self):
        """Get list of labels for all history items."""
        history = self.data.get("history", [])
        return [label for _, label in history]

STORE = Store()
