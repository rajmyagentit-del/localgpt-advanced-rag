"""Unit tests for rag_system/observability.py (Improvement #12)."""

import pytest

import rag_system.observability as obs

# Note: the `span_exporter` fixture used throughout this file is defined in
# conftest.py, not here - see that file's docstring for why (avoiding a
# module dual-import pitfall since tests/ has no __init__.py). pytest
# injects it automatically by parameter name, no import needed.


class TestTracedSpan:
    def test_creates_a_span_with_the_given_name(self, span_exporter):
        with obs.traced_span("my.operation"):
            pass
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "my.operation"

    def test_attributes_passed_as_kwargs_are_recorded(self, span_exporter):
        with obs.traced_span("op", query_type="rag_query", num_docs=5):
            pass
        span = span_exporter.get_finished_spans()[0]
        assert span.attributes["query_type"] == "rag_query"
        assert span.attributes["num_docs"] == 5

    def test_attributes_set_inside_the_block_are_recorded(self, span_exporter):
        with obs.traced_span("op") as span:
            span.set_attribute("result_count", 42)
        recorded = span_exporter.get_finished_spans()[0]
        assert recorded.attributes["result_count"] == 42

    def test_nested_spans_have_correct_parent_child_relationship(self, span_exporter):
        with obs.traced_span("parent"):
            with obs.traced_span("child"):
                pass
        spans = span_exporter.get_finished_spans()
        child = next(s for s in spans if s.name == "child")
        parent = next(s for s in spans if s.name == "parent")
        assert child.parent.span_id == parent.context.span_id

    def test_exception_marks_span_as_error_and_still_propagates(self, span_exporter):
        with pytest.raises(ValueError, match="boom"):
            with obs.traced_span("failing_op"):
                raise ValueError("boom")
        span = span_exporter.get_finished_spans()[0]
        assert span.status.status_code.name == "ERROR"

    def test_successful_span_has_unset_status_not_error(self, span_exporter):
        with obs.traced_span("ok_op"):
            pass
        span = span_exporter.get_finished_spans()[0]
        assert span.status.status_code.name != "ERROR"


class TestTracedDecorator:
    def test_wraps_sync_function_and_preserves_return_value(self, span_exporter):
        @obs.traced("sync.op")
        def double(x):
            return x * 2

        assert double(5) == 10
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "sync.op"

    def test_wraps_async_function_and_preserves_return_value(self, span_exporter):
        import asyncio

        @obs.traced("async.op")
        async def triple(x):
            return x * 3

        result = asyncio.run(triple(5))
        assert result == 15
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "async.op"

    def test_preserves_function_name_via_functools_wraps(self, span_exporter):
        @obs.traced("some.op")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"
