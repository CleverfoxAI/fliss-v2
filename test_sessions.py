"""
Session-store tests (offline, no real Redis).

Lock in the persistence behaviour:
  1. In-memory store: get/set + LRU eviction bound.
  2. build_session_store() defaults to in-memory when REDIS_URL is unset.
  3. Redis store round-trips history (incl. nested filters_used/results) via JSON.
  4. Redis store degrades to in-memory on any backend error — never raises.

Run: python test_sessions.py   (or: pytest)
"""
from __future__ import annotations

import asyncio
import os

from chat.sessions import InMemorySessionStore, RedisSessionStore, build_session_store


def test_inmemory_get_set_roundtrip():
    store = InMemorySessionStore()

    async def go():
        assert await store.get("missing") is None
        hist = [{"role": "user", "content": "hi"}]
        await store.set("k", hist)
        assert await store.get("k") == hist

    asyncio.run(go())


def test_inmemory_lru_eviction_is_bounded():
    store = InMemorySessionStore(max_sessions=3)

    async def go():
        for k in ["a", "b", "c"]:
            await store.set(k, [k])
        await store.get("a")          # touch 'a' so it's most-recently-used
        await store.set("d", ["d"])   # over cap -> evicts least-recently-used ('b')
        keys = list(store._data.keys())
        assert len(keys) == 3
        assert "b" not in keys and "a" in keys and "d" in keys

    asyncio.run(go())


def test_build_session_store_defaults_to_inmemory(monkeypatch=None):
    saved = {k: os.environ.pop(k, None) for k in ("REDIS_URL", "REDIS_PRIVATE_URL")}
    try:
        store = build_session_store()
        assert isinstance(store, InMemorySessionStore)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


class _FakeRedis:
    """Minimal async Redis stand-in backed by a dict."""

    def __init__(self):
        self.store = {}

    async def get(self, k):
        return self.store.get(k)

    async def set(self, k, v, ex=None):
        self.store[k] = v

    async def aclose(self):
        pass


class _RaisingRedis:
    async def get(self, k):
        raise ConnectionError("redis down")

    async def set(self, k, v, ex=None):
        raise ConnectionError("redis down")

    async def aclose(self):
        pass


def _redis_store_with(client):
    # Build without touching real Redis: bypass __init__'s redis import.
    store = RedisSessionStore.__new__(RedisSessionStore)
    store._redis = client
    store._ttl = 3600
    store._fallback = InMemorySessionStore()
    return store


def test_redis_store_json_roundtrip_preserves_structure():
    store = _redis_store_with(_FakeRedis())

    async def go():
        history = [
            {"role": "user", "content": "care home in Chichester for my mum"},
            {
                "role": "assistant",
                "content": "Here are some options",
                "filters_used": {"location": "Chichester", "keywords": ["dementia"], "radius_km": 25},
                "results": [{"organisationName": "Sunrise Manor", "distance_km": 1.2}],
                "title": "Results near Chichester",
                "center_lat": 50.8376,
                "center_lng": -0.7749,
            },
        ]
        await store.set("sess", history)
        out = await store.get("sess")
        assert out == history  # full structure survives JSON round-trip

    asyncio.run(go())


def test_redis_store_falls_back_on_error():
    store = _redis_store_with(_RaisingRedis())

    async def go():
        hist = [{"role": "user", "content": "hi"}]
        # set must not raise even though the backend errors
        await store.set("k", hist)
        # get must not raise; serves from the in-memory fallback set above
        assert await store.get("k") == hist

    asyncio.run(go())


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
