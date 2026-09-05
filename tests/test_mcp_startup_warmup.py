"""``main()`` answers ``initialize`` first and loads Presidio in parallel.

ORDER IS THE ENTIRE PROPERTY. Presidio is ~50 s of model loading on this host
and every result path needs it (scrub-on-return, `aggregator/mcp.py` lines 2333,
3535, 4818, 4882, 4888). Starting the warm-up BEFORE ``build_server()`` would put
lock contention in front of the cache read that builds the tool descriptions;
starting it AFTER ``server.run()`` would never happen at all, because ``run()``
serves stdio and does not return. Between the two, on a daemon thread, is the
only placement that answers the handshake in seconds and still has warm engines
by the time the first query returns.

No engine is built here: ``start_background_init`` is monkeypatched, so this test
costs milliseconds and touches no model.
"""


def test_main_builds_then_warms_then_serves(monkeypatch):
    import aggregator.mcp as mcp_mod

    order: list[str] = []

    class _FakeServer:
        def run(self, **_kwargs):
            order.append("run")

    def _fake_build_server():
        order.append("build")
        return _FakeServer()

    monkeypatch.setattr(mcp_mod, "build_server", _fake_build_server)
    monkeypatch.setattr(
        mcp_mod, "start_background_init", lambda: order.append("warm")
    )

    mcp_mod.main()

    assert order == ["build", "warm", "run"]


def test_main_does_not_wait_for_the_warmup(monkeypatch):
    """A blocking warm-up would reintroduce the exact 30 s timeout this branch
    exists to remove. ``main()`` must not join the thread it starts."""
    import threading

    import aggregator.mcp as mcp_mod

    release = threading.Event()
    served = threading.Event()

    class _FakeServer:
        def run(self, **_kwargs):
            served.set()

    def _slow_warm():
        thread = threading.Thread(target=release.wait, daemon=True)
        thread.start()
        return thread

    monkeypatch.setattr(mcp_mod, "build_server", lambda: _FakeServer())
    monkeypatch.setattr(mcp_mod, "start_background_init", _slow_warm)

    mcp_mod.main()

    assert served.is_set(), "main() served stdio without waiting on the warm-up"
    release.set()


def test_mcp_reexports_the_warmup_entry_point():
    """The test above patches ``aggregator.mcp.start_background_init``. That only
    works while mcp.py binds the name into its own namespace with
    ``from aggregator.core.scrub import …``. Pin it, so a refactor to
    ``import aggregator.core.scrub as scrub_mod`` fails here rather than turning
    the order test into a no-op."""
    import aggregator.core.scrub as scrub_mod
    import aggregator.mcp as mcp_mod

    assert mcp_mod.start_background_init is scrub_mod.start_background_init
