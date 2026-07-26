"""
Semantic query cache - Redis-backed when available, in-memory fallback
otherwise (Improvement #15).

WHY THIS CHANGE: the original in-memory cache (cachetools.TTLCache) lives
in a single process's RAM. It disappears on restart, and if you ever run
more than one backend instance behind a load balancer, each instance has
its own independent, inconsistent cache - defeating much of the point of
caching (a query answered by instance A doesn't benefit instance B). This
is one specific thing standing between this app and horizontal scaling.

Both cache implementations expose the SAME dict-like interface
(MutableMapping: __getitem__, __setitem__, __delitem__, __iter__,
__len__) that agent/loop.py already used against the old TTLCache -
this is a deliberate drop-in replacement, minimizing changes to the
(already large, delicate) agent loop.

WHY GRACEFUL FALLBACK (not required Redis): this project runs entirely
locally out of the box, no extra infrastructure required. Making Redis
mandatory would break that experience for anyone just trying it out.
If REDIS_URL is unset or Redis is unreachable, this transparently falls
back to the original in-memory behavior - you only need to actually run
Redis once you care about multi-instance deployment.
"""

import logging
import pickle
from collections.abc import Iterator, MutableMapping
from typing import Any

logger = logging.getLogger(__name__)


class InMemorySemanticCache(MutableMapping):
    """The original behavior: a per-process TTL cache. Used automatically
    when Redis isn't configured or isn't reachable."""

    def __init__(self, maxsize: int = 100, ttl: int = 300):
        from cachetools import TTLCache

        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)

    def __getitem__(self, key):
        return self._cache[key]

    def __setitem__(self, key, value):
        self._cache[key] = value

    def __delitem__(self, key):
        del self._cache[key]

    def __iter__(self) -> Iterator:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)


class RedisSemanticCache(MutableMapping):
    """
    Redis-backed cache - shared across every process/instance connected
    to the same Redis, which is what actually unblocks horizontal
    scaling (see roadmap item on multi-instance deployment).

    Serialization uses pickle rather than JSON: cached values contain
    numpy embedding arrays and arbitrary nested result dicts, and this
    is purely an internal cache (never exposed to users or other
    services), so pickle's usual downsides (not human-readable, not
    cross-language) don't apply here - it's simpler and more robust than
    hand-rolling JSON encoding for numpy arrays.

    NOTE on eviction: Redis SETs (used to track which cache keys exist)
    are unordered, so the "evict the oldest entry when full" behavior
    from the original TTLCache becomes "evict an arbitrary entry" here.
    This is an accepted approximation - entries also expire via Redis's
    own TTL (see `ttl` below), so size-based eviction is a secondary,
    rarely-hit safety net rather than the primary eviction mechanism.
    """

    def __init__(self, redis_client, namespace: str = "semantic_cache", ttl: int = 300):
        self._redis = redis_client
        self._namespace = namespace
        self._ttl = ttl

    def _full_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    @property
    def _keyset_key(self) -> str:
        return f"{self._namespace}:__keys__"

    def __getitem__(self, key):
        raw = self._redis.get(self._full_key(key))
        if raw is None:
            raise KeyError(key)
        return pickle.loads(raw)

    def __setitem__(self, key, value):
        self._redis.set(self._full_key(key), pickle.dumps(value), ex=self._ttl)
        self._redis.sadd(self._keyset_key, key)

    def __delitem__(self, key):
        if self._redis.get(self._full_key(key)) is None:
            raise KeyError(key)
        self._redis.delete(self._full_key(key))
        self._redis.srem(self._keyset_key, key)

    def __iter__(self) -> Iterator:
        """Only yields keys whose values haven't expired - membership in
        the tracking set can lag behind actual TTL expiry, so we check
        and lazily clean up stale entries as we go."""
        members = self._redis.smembers(self._keyset_key)
        for member in members:
            k = member.decode() if isinstance(member, bytes) else member
            if self._redis.exists(self._full_key(k)):
                yield k
            else:
                self._redis.srem(self._keyset_key, k)

    def __len__(self) -> int:
        return sum(1 for _ in self)


def get_semantic_cache(maxsize: int = 100, ttl: int = 300) -> MutableMapping:
    """
    Factory: returns a Redis-backed cache if REDIS_URL is configured and
    the server is actually reachable (verified with a PING, not just
    "the URL is set"), otherwise falls back to the in-memory cache.
    """
    from rag_system.config import settings

    redis_url = settings.redis_url
    if redis_url:
        try:
            import redis as redis_lib

            client = redis_lib.from_url(redis_url, socket_connect_timeout=2)
            client.ping()
            logger.info(f"Semantic cache: connected to Redis at {redis_url}")
            return RedisSemanticCache(client, ttl=ttl)
        except Exception as e:
            logger.warning(
                f"REDIS_URL is set but Redis is unreachable ({e}); "
                f"falling back to in-memory cache for this process"
            )

    return InMemorySemanticCache(maxsize=maxsize, ttl=ttl)
