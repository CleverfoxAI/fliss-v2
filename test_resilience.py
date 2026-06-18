"""
Resilience regression tests (offline, no server, no network, no API key).

These lock in Fliss's "bulletproof" guarantees so they can't silently regress:
  1. /api/query NEVER raises to the user — any downstream failure degrades to a
     graceful fallback response.
  2. The DB query layer retries transient connection failures.
  3. The search handlers never crash on a missing/empty location.

Run in CI / the deploy gate:  python test_resilience.py   (or: pytest)
"""
from __future__ import annotations

import asyncio


async def _no_sleep(*_a, **_k):
    """Async no-op to skip retry backoff delays in tests."""
    return


# ── 1. The API boundary can never hand the user a hard error ──────────────────

def test_api_query_degrades_gracefully_when_engine_raises():
    import main

    class _BoomEngine:
        def __init__(self, frontend_type):  # noqa: D401
            self.frontend_type = frontend_type

        async def chat(self, message, conversation_history):
            raise RuntimeError("simulated total downstream failure")

    original = main.ConversationEngine
    main.ConversationEngine = _BoomEngine
    try:
        req = main.QueryRequest(
            query="i need a care home",
            context=main.QueryContext(session_id="resilience-test"),
            type="CAREHOME",
        )
        resp = asyncio.run(main.query(req))
    finally:
        main.ConversationEngine = original

    # No exception escaped, and the user got a valid, polite response.
    assert resp.answer == main.BACKEND_FALLBACK_REPLY
    assert resp.intent == "clarify"
    assert resp.results == []


# ── 2. DB layer retries transient connection failures ────────────────────────

def test_run_query_retries_transient_then_succeeds():
    import tools.search as search

    class _FakeConn:
        def __init__(self, fail_times):
            self.calls = 0
            self.fail_times = fail_times

        async def fetch(self, query, *params):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise ConnectionError("stale connection")
            return ["row1", "row2"]

    class _FakeAcquire:
        def __init__(self, conn):
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, *exc):
            return False

    class _FakePool:
        def __init__(self, conn):
            self.conn = conn

        def acquire(self):
            return _FakeAcquire(self.conn)

    conn = _FakeConn(fail_times=2)  # fail twice, succeed on the 3rd attempt

    async def _fake_get_pool():
        return _FakePool(conn)

    orig_pool, orig_sleep = search.get_pool, search.asyncio.sleep
    search.get_pool = _fake_get_pool
    search.asyncio.sleep = _no_sleep  # skip real backoff delay
    try:
        rows = asyncio.run(search._run_query("SELECT 1", []))
    finally:
        search.get_pool, search.asyncio.sleep = orig_pool, orig_sleep

    assert rows == ["row1", "row2"]
    assert conn.calls == 3  # 2 failures + 1 success


def test_run_query_raises_after_exhausting_retries():
    import tools.search as search

    class _AlwaysFailConn:
        async def fetch(self, query, *params):
            raise ConnectionError("db down")

    class _Acquire:
        async def __aenter__(self):
            return _AlwaysFailConn()

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def acquire(self):
            return _Acquire()

    async def _fake_get_pool():
        return _Pool()

    orig_pool, orig_sleep = search.get_pool, search.asyncio.sleep
    search.get_pool = _fake_get_pool
    search.asyncio.sleep = _no_sleep
    try:
        raised = False
        try:
            asyncio.run(search._run_query("SELECT 1", [], attempts=3))
        except ConnectionError:
            raised = True
        assert raised, "should re-raise after exhausting retries"
    finally:
        search.get_pool, search.asyncio.sleep = orig_pool, orig_sleep


# ── 3. Search handlers never crash on missing location ───────────────────────

def test_search_handlers_no_crash_on_missing_location():
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    from chat.engine import ConversationEngine

    eng = ConversationEngine("CAREHOME")

    async def main_():
        for bad in ({}, {"location": ""}, {"location": "   "}):
            j, raw, geo = await eng._handle_search(bad)
            assert raw == [] and geo is None
        j, raw, geo = await eng._handle_job_search({})
        assert raw == [] and geo is None

    asyncio.run(main_())


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    raise SystemExit(1 if failures else 0)
