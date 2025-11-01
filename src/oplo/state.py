# src/oplo/state.py
from typing import Any, Dict

class Store:
    def __init__(self):
        # bundle: ImageBundle (raw, meta, calib)
        # disp: np.ndarray (float32 [0,1] downsampled for display)
        self.data: Dict[str, Any] = {"bundle": None, "disp": None}

    def set(self, **kwargs):
        self.data.update(kwargs)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

STORE = Store()
