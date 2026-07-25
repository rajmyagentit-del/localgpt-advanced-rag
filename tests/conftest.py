"""
Shared pytest fixtures and setup.

Some modules under test transitively import heavy ML libraries (torch,
transformers, ColPali, lancedb's native bindings, etc.) that aren't
needed to test the PURE LOGIC in this test suite (string formatting,
math, cache key generation). Rather than requiring multi-GB downloads
just to run unit tests, we stub the two heavy leaf modules here, at
session start, before anything under test gets imported.

This is a common and legitimate testing pattern: it isolates "logic we
own and want to verify" from "third-party ML runtime we don't need to
exercise for these particular tests." Tests that DO need real ML
behavior belong in a separate integration-test suite (not yet built -
see roadmap item for a full eval suite) that runs with the full stack.
"""

import sys
import types

# rag_system.agent.loop imports RetrievalPipeline and GraphRetriever, which
# transitively pull in torch/transformers/ColPali/docling (several GB).
# We only need the two NAMES to exist for the import statement to succeed -
# the pure-logic tests we run never call these classes.
fake_retrieval_pipeline = types.ModuleType("rag_system.pipelines.retrieval_pipeline")
fake_retrieval_pipeline.RetrievalPipeline = type("RetrievalPipeline", (), {})
sys.modules["rag_system.pipelines.retrieval_pipeline"] = fake_retrieval_pipeline

fake_retrievers = types.ModuleType("rag_system.retrieval.retrievers")
fake_retrievers.GraphRetriever = type("GraphRetriever", (), {})
fake_retrievers.MultiVectorRetriever = type("MultiVectorRetriever", (), {})
sys.modules["rag_system.retrieval.retrievers"] = fake_retrievers
