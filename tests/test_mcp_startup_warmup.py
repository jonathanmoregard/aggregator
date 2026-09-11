"""``main()`` answers ``initialize`` first and loads Presidio in parallel.

ORDER IS THE ENTIRE PROPERTY. Presidio is ~50 s of model loading on this host
and every result path needs it — scrub-on-return, called from ``mcp.py``'s
``_scrub_record``, ``_observation_to_item``, ``_first_user_prompt`` and
``_session_body_preview`` (twice). Named rather than numbered: the line numbers
that used to stand here had already drifted onto unrelated comments and a blank
line, and a citation that points at the wrong place is worse than none, because
it is checked once and believed after that. Starting the warm-up BEFORE
``build_server()`` would put
lock contention in front of the cache read that builds the tool descriptions;
starting it AFTER ``server.run()`` would never happen at all, because ``run()``
serves stdio and does not return. Between the two, on a daemon thread, is the
only placement that answers the handshake in seconds and still has warm engines
by the time the first query returns.

No engine is built here: ``start_background_init`` is monkeypatched, so this test
costs milliseconds and touches no model.
"""

import pytest


@pytest.mark.parametrize("backend_url", [None, "", "   "])
def test_main_builds_then_warms_then_serves(monkeypatch, backend_url):
    import aggregator.mcp as mcp_mod

    if backend_url is None:
        monkeypatch.delenv("AGGREGATOR_MCP_BACKEND_URL", raising=False)
    else:
        monkeypatch.setenv("AGGREGATOR_MCP_BACKEND_URL", backend_url)

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


def test_main_proxies_without_building_or_warming(monkeypatch):
    import fastmcp.server

    import aggregator.mcp as mcp_mod

    backend_url = "http://127.0.0.1:8765/mcp"
    calls: list[tuple[str, object]] = []

    class _FakeProxy:
        def run(self, **kwargs):
            calls.append(("run", kwargs))

    def _fake_create_proxy(target):
        calls.append(("proxy", target))
        return _FakeProxy()

    def _unexpected_local_server():
        pytest.fail("proxy mode must not build a local aggregator server")

    def _unexpected_warmup():
        pytest.fail("proxy mode must not initialize local Presidio state")

    monkeypatch.setenv("AGGREGATOR_MCP_BACKEND_URL", backend_url)
    monkeypatch.setattr(fastmcp.server, "create_proxy", _fake_create_proxy)
    monkeypatch.setattr(mcp_mod, "build_server", _unexpected_local_server)
    monkeypatch.setattr(mcp_mod, "start_background_init", _unexpected_warmup)

    mcp_mod.main()

    assert calls == [
        ("proxy", backend_url),
        ("run", {"show_banner": False}),
    ]


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
    """The tests above patch ``aggregator.mcp.start_background_init``. That only
    works while mcp.py binds the name into its own namespace with
    ``from aggregator.core.scrub import …``. Pin it, so a refactor to
    ``import aggregator.core.scrub as scrub_mod`` fails here rather than turning
    the order test into a no-op.

    ASSERTED BY NAME-BINDING, NOT BY IDENTITY, AND THAT IS NOT PEDANTRY.
    ``tests/core/test_scrub.py`` calls ``importlib.reload`` on
    ``aggregator.core.scrub`` four times, which rebinds that module's attributes
    to NEW function objects while ``aggregator.mcp`` keeps the one it imported.
    So ``mcp_mod.x is scrub_mod.x`` answers "was aggregator.mcp imported before
    or after those reloads" — it passes on this file alone and fails in the full
    suite. What the monkeypatching actually needs is that the name lives in
    mcp.py's own ``__dict__`` and came from the scrub module, which no reload
    can perturb.
    """
    import aggregator.mcp as mcp_mod

    warm = mcp_mod.__dict__.get("start_background_init")
    assert warm is not None, "mcp.py must bind the name in its own namespace"
    assert warm.__module__ == "aggregator.core.scrub"
    assert warm.__name__ == "start_background_init"
