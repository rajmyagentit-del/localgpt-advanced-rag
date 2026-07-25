"""
Unit tests for rag_system/utils/logging_utils.py.

These formalize the manual verification done while building Improvement #1
(structured logging) into a real, repeatable test suite.
"""

import json
import logging

from rag_system.utils.logging_utils import JsonFormatter


class TestJsonFormatter:
    def _make_record(self, msg="hello world", level=logging.INFO):
        return logging.LogRecord(
            name="test.module",
            level=level,
            pathname=__file__,
            lineno=42,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_output_is_valid_json(self):
        formatter = JsonFormatter()
        output = formatter.format(self._make_record())
        parsed = json.loads(output)  # raises if not valid JSON
        assert parsed["message"] == "hello world"

    def test_includes_required_fields(self):
        formatter = JsonFormatter()
        parsed = json.loads(formatter.format(self._make_record()))
        for field in ("timestamp", "level", "logger", "message", "module", "line"):
            assert field in parsed

    def test_level_name_is_preserved(self):
        formatter = JsonFormatter()
        parsed = json.loads(formatter.format(self._make_record(level=logging.WARNING)))
        assert parsed["level"] == "WARNING"

    def test_exception_info_included_when_present(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test.module",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="something failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "boom" in parsed["exception"]
