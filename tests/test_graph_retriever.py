"""
Tests for GraphRetriever (rag_system/retrieval/retrievers.py) -
Improvement #19 (completing GraphRAG).

GraphRetriever itself needs no LLM - it's pure graph traversal + fuzzy
string matching against an already-built graph. So unlike most of
rag_system's retrieval code, this can be tested with a REAL graph (not
mocked) and real assertions about what comes back, entirely offline.

The graph in these tests is built directly with networkx, saved to a
temp GML file, and loaded through the real GraphRetriever.__init__ -
this exercises the actual save/load round-trip GraphExtractor's output
goes through during indexing, not just the in-memory retrieval logic.
"""

import os
import tempfile

import networkx as nx
import pytest

from rag_system.retrieval.graph_retriever import GraphRetriever


@pytest.fixture
def sample_graph_path():
    """
    A small, realistic knowledge graph:
        Tim Cook -[IS_CEO_OF]-> Apple
        Apple -[HEADQUARTERED_IN]-> Cupertino
        Satya Nadella -[IS_CEO_OF]-> Microsoft
        Microsoft -[HEADQUARTERED_IN]-> Redmond
    """
    G = nx.DiGraph()
    G.add_edge("Tim Cook", "Apple", label="IS_CEO_OF")
    G.add_edge("Apple", "Cupertino", label="HEADQUARTERED_IN")
    G.add_edge("Satya Nadella", "Microsoft", label="IS_CEO_OF")
    G.add_edge("Microsoft", "Redmond", label="HEADQUARTERED_IN")

    fd, path = tempfile.mkstemp(suffix=".gml")
    os.close(fd)
    nx.write_gml(G, path)
    yield path
    os.unlink(path)


class TestGraphRetrieverPlainText:
    def test_exact_entity_name_finds_neighbors(self, sample_graph_path):
        retriever = GraphRetriever(sample_graph_path)
        results = retriever.retrieve("Tell me about Apple")
        assert len(results) >= 1
        texts = [r["text"] for r in results]
        assert any("Cupertino" in t for t in texts)

    def test_no_matching_entity_returns_empty(self, sample_graph_path):
        retriever = GraphRetriever(sample_graph_path)
        results = retriever.retrieve("completely unrelated gibberish xyzzy")
        assert results == []


class TestGraphRetrieverStructured:
    def test_exact_start_node_returns_correct_neighbor(self, sample_graph_path):
        retriever = GraphRetriever(sample_graph_path)
        results = retriever.retrieve_structured(start_node="Apple")
        assert len(results) == 1
        assert results[0]["details"]["node_id"] == "Cupertino"

    def test_fuzzy_start_node_matching_handles_case_differences(self, sample_graph_path):
        """LLM-extracted entity names won't always match the graph's
        exact stored casing/spelling - this is the whole reason fuzzy
        matching exists here."""
        retriever = GraphRetriever(sample_graph_path)
        results = retriever.retrieve_structured(start_node="apple")  # lowercase
        assert len(results) == 1
        assert results[0]["details"]["node_id"] == "Cupertino"

    def test_edge_label_filters_to_matching_relationship_only(self, sample_graph_path):
        """The key precision improvement over plain retrieve(): asking
        specifically about the IS_CEO_OF relationship from Tim Cook
        should NOT also return unrelated relationships."""
        G = nx.DiGraph()
        G.add_edge("Tim Cook", "Apple", label="IS_CEO_OF")
        G.add_edge("Tim Cook", "Auburn University", label="EDUCATED_AT")
        fd, path = tempfile.mkstemp(suffix=".gml")
        os.close(fd)
        nx.write_gml(G, path)
        try:
            retriever = GraphRetriever(path)
            results = retriever.retrieve_structured(start_node="Tim Cook", edge_label="IS_CEO_OF")
            assert len(results) == 1
            assert results[0]["details"]["node_id"] == "Apple"
        finally:
            os.unlink(path)

    def test_edge_label_fuzzy_matches_llm_phrasing(self, sample_graph_path):
        """The LLM won't phrase the relationship exactly as it's stored
        ('is the CEO of' vs 'IS_CEO_OF') - fuzzy matching on the label
        is what makes this actually usable with real LLM output."""
        retriever = GraphRetriever(sample_graph_path)
        results = retriever.retrieve_structured(start_node="Tim Cook", edge_label="is the CEO of")
        assert len(results) == 1
        assert results[0]["details"]["node_id"] == "Apple"

    def test_nonexistent_start_node_returns_empty_not_error(self, sample_graph_path):
        retriever = GraphRetriever(sample_graph_path)
        results = retriever.retrieve_structured(start_node="Completely Unknown Entity Xyzzy")
        assert results == []

    def test_edge_label_with_no_matching_relationship_returns_empty(self, sample_graph_path):
        """Tim Cook -[IS_CEO_OF]-> Apple exists, but asking about a
        relationship that doesn't exist from that node should return
        empty (so the caller falls back to normal retrieval), not
        silently return the wrong relationship."""
        retriever = GraphRetriever(sample_graph_path)
        results = retriever.retrieve_structured(start_node="Tim Cook", edge_label="MARRIED_TO")
        assert results == []

    def test_empty_start_node_returns_empty(self, sample_graph_path):
        retriever = GraphRetriever(sample_graph_path)
        assert retriever.retrieve_structured(start_node="") == []

    def test_result_shape_includes_both_text_and_details(self, sample_graph_path):
        """agent/loop.py's _run_graph_query needs BOTH the human-readable
        text (for building an answer) and details.node_id (for the
        structured summary) - this is exactly the shape mismatch that
        was the second real bug found in this feature."""
        retriever = GraphRetriever(sample_graph_path)
        results = retriever.retrieve_structured(start_node="Tim Cook")
        assert len(results) == 1
        result = results[0]
        assert "text" in result
        assert "details" in result
        assert "node_id" in result["details"]
        assert result["details"]["node_id"] == "Apple"
        assert result["details"]["relationship"] == "IS_CEO_OF"
