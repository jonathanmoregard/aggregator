"""Fixtures shared by the `tests/core` module tests.

``fresh_scrub_state`` exists because ``aggregator.core.scrub`` now decides
ONCE per process whether Presidio is usable, and a test of that decision has to
be able to un-decide it. It does not use ``importlib.reload``: reload re-executes
the module in the SAME namespace dict, so it works, but on a machine that has the
spaCy model installed the next ``scrub()`` then pays a fresh ~40 s engine build
for every test that reloaded. Saving and restoring five attributes costs nothing
and leaves the process exactly as it was found.
"""
import pytest


@pytest.fixture
def fresh_scrub_state():
    """Yield ``aggregator.core.scrub`` with its init state reset, then restore."""
    import aggregator.core.scrub as mod

    saved_analyzer = mod._analyzer
    saved_anonymizer = mod._anonymizer
    saved_ok = mod._PRESIDIO_OK
    saved_done = mod._INIT_DONE.is_set()
    saved_thread = mod._warm_thread

    mod._analyzer = None
    mod._anonymizer = None
    mod._PRESIDIO_OK = False
    mod._INIT_DONE.clear()
    mod._warm_thread = None
    try:
        yield mod
    finally:
        mod._analyzer = saved_analyzer
        mod._anonymizer = saved_anonymizer
        mod._PRESIDIO_OK = saved_ok
        mod._warm_thread = saved_thread
        if saved_done:
            mod._INIT_DONE.set()
        else:
            mod._INIT_DONE.clear()
