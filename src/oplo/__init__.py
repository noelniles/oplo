"""Top-level package for oplo.

Auto-import the `oplo.plugins` package if present so plugin modules have a
chance to register themselves when the package is imported.
"""
try:
	import importlib

	importlib.import_module("oplo.plugins")
except Exception:
	# non-fatal: plugins are optional
	pass
