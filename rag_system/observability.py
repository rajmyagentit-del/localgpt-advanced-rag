"""
Observability: distributed tracing for the RAG pipeline.

WHY THIS EXISTS
---------------
Before this change, the only way to answer "why did this specific query
get a bad or slow answer" was reading logs by hand and guessing. This
module traces every query end-to-end - triage, retrieval, rerank,
generation, verification - so each step's latency and key attributes
(query type, number of docs retrieved, cache hit/miss, etc.) are
recorded and queryable after the fact.

WHY OPENTELEMETRY (not Langfuse directly)
-------------------------------------------
This project's core value proposition is "100% local, private, no data
leaves your machine." Defaulting to a cloud SaaS tracing backend would
work against that. OpenTelemetry is vendor-neutral: by default, traces
are exported locally (console, for humans watching logs, or a local
JSON file for later analysis) with ZERO network calls. If you want to
send traces to Langfuse, Jaeger, or any other OTLP-compatible backend
later, set OTEL_EXPORTER_OTLP_ENDPOINT - the instrumentation code
(the @traced_span decorators used throughout the agent) never changes.

USAGE
-----
    from rag_system.observability import traced_span

    with traced_span("retrieval.dense_search", table_name=table_name) as span:
        results = do_search(...)
        span.set_attribute("num_results", len(results))

Configuration (environment variables):
    OTEL_TRACES_EXPORTER          "console" (default) | "file" | "otlp" | "none"
    OTEL_EXPORTER_OTLP_ENDPOINT   required only if OTEL_TRACES_EXPORTER=otlp
    OTEL_TRACE_FILE_PATH          used only if OTEL_TRACES_EXPORTER=file
                                  (default: ./traces.jsonl)
"""

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)

_initialized = False


class _JsonFileSpanExporter(SpanExporter):
    """Writes each finished span as one JSON line to a local file - no
    network calls, easy to grep/jq through after the fact."""

    def __init__(self, path: str):
        self.path = path

    def export(self, spans) -> SpanExportResult:
        try:
            with open(self.path, "a") as f:
                for span in spans:
                    f.write(span.to_json(indent=None) + "\n")
            return SpanExportResult.SUCCESS
        except OSError as e:
            logger.warning(f"Failed to write trace spans to {self.path}: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def setup_tracing(service_name: str = "localgpt-rag-system") -> None:
    """
    Configure the global TracerProvider once. Safe to call more than
    once - subsequent calls are no-ops, same pattern as
    logging_utils.setup_logging().
    """
    global _initialized
    if _initialized:
        return

    exporter_kind = os.getenv("OTEL_TRACES_EXPORTER", "console").lower()

    exporter: SpanExporter | None
    if exporter_kind == "console":
        exporter = ConsoleSpanExporter()
    elif exporter_kind == "file":
        file_path = os.getenv("OTEL_TRACE_FILE_PATH", "./traces.jsonl")
        exporter = _JsonFileSpanExporter(file_path)
    elif exporter_kind == "otlp":
        # Imported lazily - the otlp exporter package is an optional
        # extra, not required for the default local-only setup.
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "OTEL_TRACES_EXPORTER=otlp requires OTEL_EXPORTER_OTLP_ENDPOINT to be set"
            )
        exporter = OTLPSpanExporter(endpoint=endpoint)
    elif exporter_kind == "none":
        exporter = None
    else:
        raise ValueError(
            f"Unknown OTEL_TRACES_EXPORTER '{exporter_kind}' "
            f"(expected: console, file, otlp, or none)"
        )

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _initialized = True
    logger.info(f"Tracing initialized (exporter={exporter_kind})")


def get_tracer() -> trace.Tracer:
    if not _initialized:
        setup_tracing()
    return trace.get_tracer("rag_system")


@contextlib.contextmanager
def traced_span(name: str, **attributes: Any) -> Iterator[Span]:
    """
    Context manager wrapping a block of code in a traced span.

    Records success/failure automatically: if the wrapped block raises,
    the span is marked as an error and the exception is recorded on the
    span (then re-raised - this never swallows exceptions).

    Example:
        with traced_span("retrieval.rerank", num_candidates=len(docs)) as span:
            reranked = reranker.rerank(docs)
            span.set_attribute("num_reranked", len(reranked))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def traced(name: str):
    """
    Decorator version of traced_span, for wrapping an entire function or
    coroutine without needing to re-indent its body (useful for large
    existing functions where a `with` block would mean re-indenting
    hundreds of lines and risking an unrelated diff/bug).

    Works on both regular functions and async coroutines.

    Example:
        @traced("agent.run_query")
        async def _run_async(self, query, ...):
            ...  # body unchanged, no re-indent needed
    """
    import functools
    import inspect

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with traced_span(name):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with traced_span(name):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
