"""
GraphRetriever - lives in its own module (Improvement #19) because it
has ZERO dependency on the heavy multimodal/vision stack (torch,
transformers, CLIP) that rag_system/retrieval/retrievers.py's other
classes need. Coupling it to that import chain meant anyone wanting to
use, test, or reason about ONLY the knowledge-graph feature was forced
to pull in the entire vision ML stack for no reason - a real,
unnecessary coupling, not just a testing convenience. GraphRetriever is
re-exported from retrievers.py for backward compatibility with existing
imports (e.g. agent/loop.py).
"""

import logging
from typing import Any

import networkx as nx
from fuzzywuzzy import process

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    Retrieves information from a knowledge graph built by GraphExtractor
    during indexing (see rag_system/indexing/graph_extractor.py) and
    saved as a NetworkX-readable GML file.

    Supports two modes:
      - retrieve(query): fuzzy-matches entity names directly out of the
        raw query text, returns ALL neighbors of any matched entity.
        Simple, no LLM call needed, but imprecise - can't distinguish
        "who founded Apple" from "who is Apple's CEO".
      - retrieve_structured(start_node, edge_label): takes the output of
        GraphQueryTranslator (an LLM-produced structured query), fuzzy-
        matches start_node to an actual graph node (LLM extraction won't
        match stored casing/spelling exactly), then filters outgoing
        edges by edge_label (also fuzzy-matched, for the same reason) if
        one was given. This is the precise path - it actually answers
        "who is Apple's CEO" rather than just "what's connected to Apple".
    """

    def __init__(self, graph_path: str):
        self.graph = nx.read_gml(graph_path)
        logger.info(
            f"Loaded knowledge graph from {graph_path}: "
            f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )

    def retrieve(self, query: str, k: int = 5, score_cutoff: int = 80) -> list[dict[str, Any]]:
        """Plain-text fallback: fuzzy-match entity names directly out of
        the query, return all neighbors of any matched entity."""
        logger.info(f"Performing plain-text graph retrieval for query: '{query}'")

        # BUG FIX (Improvement #19): fuzzywuzzy's process.extractOne()
        # silently matches nothing when given a NetworkX NodeView
        # directly (as opposed to a plain list) - this was true in the
        # ORIGINAL code too, meaning plain-text graph retrieval has never
        # actually matched anything, ever, even before this rewrite.
        node_names = list(self.graph.nodes())

        query_parts = query.split()
        entities = []
        for part in query_parts:
            match = process.extractOne(part, node_names, score_cutoff=score_cutoff)
            if match and isinstance(match[0], str):
                entities.append(match[0])

        retrieved_docs = []
        for entity in set(entities):
            for neighbor in self.graph.neighbors(entity):
                edge_data = self.graph.get_edge_data(entity, neighbor) or {}
                label = edge_data.get("label", "related_to")
                retrieved_docs.append(self._format_result(entity, neighbor, label))

        logger.info(f"Retrieved {len(retrieved_docs)} documents from the graph.")
        return retrieved_docs[:k]

    def retrieve_structured(
        self, start_node: str, edge_label: str | None = None, k: int = 5, score_cutoff: int = 70
    ) -> list[dict[str, Any]]:
        """
        Precise retrieval using a structured query (see
        GraphQueryTranslator.translate() for how this gets produced from
        a natural-language question).

        Returns an empty list (not an error) if start_node can't be
        matched to any graph node, or if no outgoing edges match
        edge_label - callers should treat an empty result as "the graph
        doesn't have this information" and fall back to normal retrieval,
        exactly like the plain-text retrieve() path already does.
        """
        if not start_node or self.graph.number_of_nodes() == 0:
            return []

        node_match = process.extractOne(
            start_node, list(self.graph.nodes()), score_cutoff=score_cutoff
        )
        if not node_match:
            logger.info(
                f"No graph node matched start_node='{start_node}' (score_cutoff={score_cutoff})"
            )
            return []
        matched_node = node_match[0]

        candidate_edges = []
        for neighbor in self.graph.neighbors(matched_node):
            edge_data = self.graph.get_edge_data(matched_node, neighbor) or {}
            label = edge_data.get("label", "")
            candidate_edges.append((neighbor, label))

        if edge_label:
            # Fuzzy-match the requested relationship against actual edge
            # labels. Raw fuzzy distance between natural-language
            # phrasing ("is the CEO of") and a SNAKE_CASE stored label
            # ("IS_CEO_OF") scores too low to reliably match even when
            # they mean the same thing (verified: ~64 raw vs a 70+
            # cutoff) - normalizing both to "is the ceo of"-style text
            # first (underscores -> spaces, lowercase) fixes this
            # (~95 normalized) without loosening the cutoff and risking
            # false-positive matches between genuinely different labels.
            def _normalize_label(s: str) -> str:
                return s.replace("_", " ").lower().strip()

            labels_present = {label for _, label in candidate_edges if label}
            if labels_present:
                normalized_to_original = {
                    _normalize_label(label): label for label in labels_present
                }
                label_match = process.extractOne(
                    _normalize_label(edge_label),
                    list(normalized_to_original.keys()),
                    score_cutoff=score_cutoff,
                )
                if label_match:
                    matched_label = normalized_to_original[label_match[0]]
                    candidate_edges = [
                        (neighbor, label)
                        for neighbor, label in candidate_edges
                        if label == matched_label
                    ]
                else:
                    # Requested relationship doesn't match any real edge
                    # label from this node - genuinely no answer, not an
                    # error. Return empty so the caller can fall back.
                    logger.info(f"No edge from '{matched_node}' matched edge_label='{edge_label}'")
                    return []

        results = [
            self._format_result(matched_node, neighbor, label)
            for neighbor, label in candidate_edges
        ]
        logger.info(
            f"Structured graph retrieval: start_node='{start_node}' -> matched '{matched_node}', "
            f"edge_label='{edge_label}', {len(results)} result(s)"
        )
        return results[:k]

    @staticmethod
    def _format_result(source: str, target: str, label: str) -> dict[str, Any]:
        """One consistent result shape for both retrieve() and
        retrieve_structured() - includes both a human-readable `text`
        (for feeding into an LLM answer) and a structured `details`
        block (for programmatic access, e.g. agent/loop.py building a
        direct answer without another LLM call)."""
        readable_label = label.replace("_", " ").lower() if label else "related to"
        return {
            "chunk_id": f"graph_{source}_{target}",
            "text": f"{source} {readable_label} {target}.",
            "score": 1.0,
            "metadata": {"source": "graph"},
            "details": {
                "node_id": target,
                "source_node": source,
                "target_node": target,
                "relationship": label,
            },
        }
