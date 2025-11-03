"""Simple plugin registry for readers/processors/viewers.

This keeps a small, testable surface for registering reader classes
so the core `image_io` module can be extended without editing it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List
import logging

# configure a logger for the registry; avoid configuring root logger
logger = logging.getLogger("oplo.registry")
if not logger.handlers:
	# Add a simple StreamHandler at INFO level by default
	h = logging.StreamHandler()
	h.setFormatter(logging.Formatter("[oplo.registry] %(message)s"))
	logger.addHandler(h)
	logger.setLevel(logging.INFO)


@dataclass(order=True)
class _RegEntry:
	priority: int
	plugin: Any


class Registry:
	def __init__(self) -> None:
		# readers / plugins are stored as _RegEntry so we can sort by priority
		self._readers: List[_RegEntry] = []

	def register_reader(self, reader: Any, priority: int = 100) -> None:
		"""Register a reader object (class or callable with .accepts/.load).

		reader: object implementing .accepts(path: Path)->bool and
				.load(path: Path)->(np.ndarray, ImageMeta)
		priority: lower numbers run first
		"""
		self._readers.append(_RegEntry(priority=priority, plugin=reader))
		# keep readers sorted by priority (smallest first)
		self._readers.sort()
		# Log registration for easier debugging
		try:
			name = getattr(reader, "name", getattr(reader, "__name__", str(reader)))
		except Exception:
			name = str(reader)
		logger.info(f"registered reader: {name} (priority={priority})")

	def get_readers(self) -> List[Any]:
		"""Return registered reader plugins in priority order."""
		return [e.plugin for e in self._readers]

	def get_reader_names(self) -> List[str]:
		"""Return registered reader names for diagnostics."""
		out = []
		for e in self._readers:
			try:
				out.append(getattr(e.plugin, "name", getattr(e.plugin, "__name__", str(e.plugin))))
			except Exception:
				out.append(str(e.plugin))
		return out

	def find_reader_for_path(self, path) -> Any | None:
		for e in self._readers:
			try:
				if e.plugin.accepts(path):
					return e.plugin
			except Exception:
				# be robust to broken acceptors
				continue
		return None


# module-level registry instance used by the app
registry = Registry()

# Attempt to auto-import the `oplo.plugins` package so builtin plugins
# (readers/processors/writers) can register themselves at import time.
try:
	import importlib

	importlib.import_module("oplo.plugins")
except Exception:
	# Non-fatal: plugins are optional and may not be importable in minimal envs
	pass

