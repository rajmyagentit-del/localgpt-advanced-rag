"""
Unit tests for rag_system/agent/semantic_cache.py (Improvement #15).

RedisSemanticCache is tested against fakeredis (a real, in-memory
implementation of the Redis protocol) rather than mocks - this exercises
real serialization/deserialization round-trips and real Redis command
semantics (SETEX, SADD, SMEMBERS, etc), not just "did we call the right
method." A live Redis server isn't available in this environment, but
fakeredis gives genuine behavioral coverage rather than a shallow mock.
"""

import numpy as np
import pytest

from rag_system.agent.semantic_cache import (
    InMemorySemanticCache,
    RedisSemanticCache,
    get_semantic_cache,
)


@pytest.fixture
def redis_cache():
    fakeredis = pytest.importorskip("fakeredis")
    fake_client = fakeredis.FakeStrictRedis()
    return RedisSemanticCache(fake_client, namespace="test_cache", ttl=300)


class TestRedisSemanticCache:
    def test_set_and_get_roundtrip_with_numpy_array(self, redis_cache):
        redis_cache["q1"] = {"result": "answer", "embedding": np.array([1.0, 2.0, 3.0])}
        retrieved = redis_cache["q1"]
        assert retrieved["result"] == "answer"
        assert np.array_equal(retrieved["embedding"], np.array([1.0, 2.0, 3.0]))

    def test_missing_key_raises_keyerror(self, redis_cache):
        with pytest.raises(KeyError):
            redis_cache["nonexistent"]

    def test_len_and_iter_reflect_actual_contents(self, redis_cache):
        redis_cache["q1"] = {"v": 1}
        redis_cache["q2"] = {"v": 2}
        assert len(redis_cache) == 2
        assert set(redis_cache) == {"q1", "q2"}

    def test_items_mixin_works(self, redis_cache):
        redis_cache["q1"] = {"v": 1}
        items = dict(redis_cache.items())
        assert items["q1"] == {"v": 1}

    def test_delete_removes_from_keyset_too(self, redis_cache):
        redis_cache["q1"] = {"v": 1}
        redis_cache["q2"] = {"v": 2}
        del redis_cache["q1"]
        assert len(redis_cache) == 1
        assert "q1" not in dict(redis_cache.items())

    def test_delete_missing_key_raises_keyerror(self, redis_cache):
        with pytest.raises(KeyError):
            del redis_cache["nonexistent"]


class TestInMemorySemanticCache:
    def test_basic_set_get_delete(self):
        cache = InMemorySemanticCache(maxsize=10, ttl=300)
        cache["q1"] = {"v": 1}
        assert cache["q1"] == {"v": 1}
        del cache["q1"]
        with pytest.raises(KeyError):
            cache["q1"]

    def test_supports_same_interface_as_redis_version(self):
        """Both implementations must be interchangeable drop-ins for
        agent/loop.py - this checks they expose the same dict-like
        surface (MutableMapping)."""
        cache = InMemorySemanticCache()
        cache["q1"] = {"v": 1}
        assert len(cache) == 1
        assert list(cache) == ["q1"]
        assert dict(cache.items()) == {"q1": {"v": 1}}


class TestGetSemanticCacheFactory:
    def test_falls_back_to_in_memory_when_no_redis_url_configured(self, monkeypatch):
        import rag_system.config as config_module

        monkeypatch.setattr(config_module.settings, "redis_url", "")
        cache = get_semantic_cache()
        assert isinstance(cache, InMemorySemanticCache)

    def test_falls_back_gracefully_when_redis_unreachable(self, monkeypatch):
        """Setting REDIS_URL to something unreachable must NOT crash the
        app - it should log a warning and fall back, since Redis is
        optional infrastructure, not a hard requirement."""
        import rag_system.config as config_module

        monkeypatch.setattr(config_module.settings, "redis_url", "redis://nonexistent-host:6379")
        cache = get_semantic_cache()
        assert isinstance(cache, InMemorySemanticCache)
