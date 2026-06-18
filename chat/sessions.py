"""Pluggable conversation-session store.

Two backends, same async interface:

* ``InMemorySessionStore`` — the default. Bounded (LRU eviction); lost on restart.
* ``RedisSessionStore`` — used when ``REDIS_URL`` is set. Survives restarts and
  redeploys and is shared across replicas. Every operation degrades to an
  in-memory fallback on any Redis error, so a Redis outage can never break the
  chat endpoint.

``build_session_store()`` picks the backend from the environment and never
raises — if Redis can't be initialised it logs and returns the in-memory store.
"""
from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SESSIONS = 2000
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days


class InMemorySessionStore:
    """Bounded in-process store. Lost on restart; LRU-evicts the oldest when full."""

    label = "memory"

    def __init__(self, max_sessions: int = _DEFAULT_MAX_SESSIONS):
        self._data: "OrderedDict[str, list]" = OrderedDict()
        self._max = max_sessions

    async def get(self, key: str) -> list | None:
        history = self._data.get(key)
        if history is not None:
            self._data.move_to_end(key)
        return history

    async def set(self, key: str, history: list) -> None:
        self._data[key] = history
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    async def close(self) -> None:
        pass


class RedisSessionStore:
    """Redis-backed store. Survives restarts/redeploys; shared across replicas.

    Any Redis error falls back to an in-memory copy for that operation and logs a
    ``FLISS-ERROR step=session_*`` line — the chat endpoint never fails because of
    the session store.
    """

    label = "redis"

    def __init__(self, url: str, ttl_seconds: int = _DEFAULT_TTL_SECONDS):
        import redis.asyncio as redis  # imported only when Redis is configured

        self._redis = redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds
        self._fallback = InMemorySessionStore()

    @staticmethod
    def _k(key: str) -> str:
        return f"fliss:session:{key}"

    async def get(self, key: str) -> list | None:
        try:
            raw = await self._redis.get(self._k(key))
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("FLISS-ERROR step=session_get err=%s: %s", type(exc).__name__, exc)
            return await self._fallback.get(key)

    async def set(self, key: str, history: list) -> None:
        try:
            await self._redis.set(
                self._k(key), json.dumps(history, default=str), ex=self._ttl
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FLISS-ERROR step=session_set err=%s: %s", type(exc).__name__, exc)
            await self._fallback.set(key, history)

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception:  # noqa: BLE001
            pass


def build_session_store():
    """Select a backend from the environment. Never raises."""
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_PRIVATE_URL")
    try:
        ttl = int(os.getenv("SESSION_TTL_SECONDS", str(_DEFAULT_TTL_SECONDS)))
    except ValueError:
        ttl = _DEFAULT_TTL_SECONDS
    if url:
        try:
            store = RedisSessionStore(url, ttl_seconds=ttl)
            logger.info("Session store: Redis (conversations persist across deploys)")
            return store
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FLISS-ERROR step=session_store_init err=%s: %s; using in-memory",
                type(exc).__name__, exc,
            )
    logger.info("Session store: in-memory (set REDIS_URL to persist across deploys)")
    return InMemorySessionStore()
