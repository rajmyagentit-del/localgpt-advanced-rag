"""
Unit tests for the pure, stateless logic inside rag_system/agent/loop.py.

These methods don't touch Ollama, LanceDB, or any external service, so we
can test them directly and fast (milliseconds, no mocking of I/O needed).
We bypass Agent.__init__ (which requires a real pipeline config, live LLM
client, etc.) via Agent.__new__, since these particular methods don't
depend on any instance state set up in __init__.
"""

from unittest.mock import MagicMock

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


class TestRunGraphQuery:
    """
    Real end-to-end test of Agent._run_graph_query() (Improvement #19 -
    completing GraphRAG), using a GENUINE GraphRetriever against a real
    small graph - not a mock of the retrieval logic itself. Only the LLM
    calls (graph_query_translator.translate) are faked, since those
    genuinely need a live model.

    This is the test that would have caught all of the real bugs fixed
    in this feature (missing logger, dict-vs-str type mismatch, wrong
    result-key access, NodeView-vs-list fuzzy matching) if it had
    existed before - which is exactly the point of writing it now.
    """

    @pytest.fixture
    def graph_path(self, tmp_path):
        import networkx as nx

        G = nx.DiGraph()
        G.add_edge("Tim Cook", "Apple", label="IS_CEO_OF")
        G.add_edge("Apple", "Cupertino", label="HEADQUARTERED_IN")
        path = str(tmp_path / "test_graph.gml")
        nx.write_gml(G, path)
        return path

    @pytest.fixture
    def agent_with_graph(self, graph_path):
        from rag_system.retrieval.graph_retriever import GraphRetriever

        a = Agent.__new__(Agent)
        a.graph_retriever = GraphRetriever(graph_path)
        a.retrieval_pipeline = MagicMock()
        a.retrieval_pipeline.run.return_value = {
            "answer": "fallback answer",
            "source_documents": [],
        }
        return a

    def test_successful_graph_lookup_returns_graph_answer(self, agent_with_graph):
        agent_with_graph.graph_query_translator = MagicMock()
        agent_with_graph.graph_query_translator.translate.return_value = {
            "start_node": "Tim Cook",
            "edge_label": "IS_CEO_OF",
        }
        agent_with_graph._format_query_with_history = MagicMock(side_effect=lambda q, h: q)

        result = agent_with_graph._run_graph_query("Who is the CEO of Apple?", history=[])

        assert "Apple" in result["answer"]
        assert "From the knowledge graph" in result["answer"]
        assert len(result["source_documents"]) == 1
        agent_with_graph.retrieval_pipeline.run.assert_not_called()

    def test_no_start_node_falls_back_to_normal_retrieval(self, agent_with_graph):
        agent_with_graph.graph_query_translator = MagicMock()
        agent_with_graph.graph_query_translator.translate.return_value = {}  # no start_node
        agent_with_graph._format_query_with_history = MagicMock(side_effect=lambda q, h: q)

        result = agent_with_graph._run_graph_query("some vague query", history=[])

        assert result["answer"] == "fallback answer"
        agent_with_graph.retrieval_pipeline.run.assert_called_once()

    def test_unmatched_start_node_falls_back_to_normal_retrieval(self, agent_with_graph):
        """start_node is present but doesn't match anything in the real
        graph - should fall back gracefully, not crash or return a
        confusing empty answer."""
        agent_with_graph.graph_query_translator = MagicMock()
        agent_with_graph.graph_query_translator.translate.return_value = {
            "start_node": "Someone Not In The Graph",
            "edge_label": "IS_CEO_OF",
        }
        agent_with_graph._format_query_with_history = MagicMock(side_effect=lambda q, h: q)

        result = agent_with_graph._run_graph_query("who is the CEO of nowhere", history=[])

        assert result["answer"] == "fallback answer"
        agent_with_graph.retrieval_pipeline.run.assert_called_once()
