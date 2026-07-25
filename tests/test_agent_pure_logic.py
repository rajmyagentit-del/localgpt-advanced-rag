"""
Unit tests for the pure, stateless logic inside rag_system/agent/loop.py.

These methods don't touch Ollama, LanceDB, or any external service, so we
can test them directly and fast (milliseconds, no mocking of I/O needed).
We bypass Agent.__init__ (which requires a real pipeline config, live LLM
client, etc.) via Agent.__new__, since these particular methods don't
depend on any instance state set up in __init__.
"""

import numpy as np
import pytest

from rag_system.agent.loop import Agent


@pytest.fixture
def agent():
    """A bare Agent instance, sufficient for testing methods that don't
    depend on __init__-configured state."""
    return Agent.__new__(Agent)


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self, agent):
        v = np.array([1.0, 2.0, 3.0])
        assert agent._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self, agent):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert agent._cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self, agent):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert agent._cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_not_nan(self, agent):
        """Real production bug class: without this guard, a zero vector
        causes a division-by-zero -> NaN, which then silently poisons any
        downstream comparison (NaN != NaN, NaN < threshold is always False)."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        result = agent._cosine_similarity(v1, v2)
        assert result == 0.0
        assert not np.isnan(result)

    def test_accepts_plain_lists_not_just_ndarray(self, agent):
        # The implementation explicitly converts non-ndarray input -
        # verify that contract holds.
        result = agent._cosine_similarity([1, 0], [1, 0])
        assert result == pytest.approx(1.0)

    def test_mismatched_shapes_raise_value_error(self, agent):
        with pytest.raises(ValueError):
            agent._cosine_similarity(np.array([1, 2]), np.array([1, 2, 3]))


class TestCacheKeyGeneration:
    def test_same_query_produces_same_key(self, agent):
        k1 = agent._get_cache_key("What is the refund policy?", "rag_query")
        k2 = agent._get_cache_key("What is the refund policy?", "rag_query")
        assert k1 == k2

    def test_key_is_case_insensitive(self, agent):
        """Users shouldn't get cache misses just because of capitalization -
        this is what makes the cache actually useful in practice."""
        k1 = agent._get_cache_key("What is the REFUND policy?", "rag_query")
        k2 = agent._get_cache_key("what is the refund policy?", "rag_query")
        assert k1 == k2

    def test_key_ignores_surrounding_whitespace(self, agent):
        k1 = agent._get_cache_key("  refund policy  ", "rag_query")
        k2 = agent._get_cache_key("refund policy", "rag_query")
        assert k1 == k2

    def test_different_query_types_produce_different_keys(self, agent):
        """Same text, different query_type, MUST be different cache entries -
        a direct_answer and a rag_query for the same string are not
        interchangeable and must never collide in the cache."""
        k1 = agent._get_cache_key("hello", "direct_answer")
        k2 = agent._get_cache_key("hello", "rag_query")
        assert k1 != k2
