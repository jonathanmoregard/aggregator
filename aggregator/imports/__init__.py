"""Import port + adapters.

One structural seam (``port.ImportAdapter``) that every source is adapted
onto, and one runner (``runner.run_imports``) that drives N adapters
concurrently with per-adapter failure isolation. See ``port.py`` for the
design rationale.
"""
