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

# --- Tracing test setup (Improvement #12) MUST come first ---
# OpenTelemetry's global TracerProvider can only be set ONCE per process
# (later calls to trace.set_tracer_provider() are silently ignored, with
# a warning). rag_system/__init__.py auto-calls the REAL setup_tracing()
# the moment ANYTHING imports the rag_system package - including the
# GraphRetriever import a few lines below this block. So this MUST be
# the very first thing in this file, before any other import that could
# transitively trigger rag_system/__init__.py, or the real
# console-exporting provider wins the race and this test provider is
# silently ignored (verified: this exact bug reappeared once already
# when a later import was accidentally added above this block).
from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor as _SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter as _InMemorySpanExporter,
)

TEST_SPAN_EXPORTER = _InMemorySpanExporter()
_test_provider = _TracerProvider()
_test_provider.add_span_processor(_SimpleSpanProcessor(TEST_SPAN_EXPORTER))
_otel_trace.set_tracer_provider(_test_provider)

import rag_system.observability as _obs

_obs._initialized = (
    True  # prevents the real setup_tracing() from trying (and failing) to override this
)


# --- Heavy-module stubs (must come after tracing setup, before test collection) ---
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
# GraphRetriever now lives in its own lightweight module (Improvement
# #19) with no torch/transformers dependency, so we use the REAL class
# here rather than a fake placeholder - any test exercising Agent's
# graph_query path gets genuine GraphRetriever behavior, not a stub.
from rag_system.retrieval.graph_retriever import GraphRetriever as _RealGraphRetriever

fake_retrievers.GraphRetriever = _RealGraphRetriever
fake_retrievers.MultiVectorRetriever = type("MultiVectorRetriever", (), {})
sys.modules["rag_system.retrieval.retrievers"] = fake_retrievers


import pytest


@pytest.fixture
def span_exporter():
    """
    Clears the shared in-memory exporter before each test that requests
    it, so tests only see spans created during that test.

    Deliberately defined HERE (not in test_observability.py) and used via
    pytest's automatic fixture injection - NOT an explicit
    `from tests.conftest import TEST_SPAN_EXPORTER`, which would re-import
    this file as a second, disconnected module (tests/ has no __init__.py,
    so `tests.conftest` and pytest's own `conftest` are two different
    module identities under Python's import system) and silently produce
    a second, empty exporter that never receives real spans.
    """
    TEST_SPAN_EXPORTER.clear()
    yield TEST_SPAN_EXPORTER
