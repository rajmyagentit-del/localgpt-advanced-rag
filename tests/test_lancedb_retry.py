"""
Unit tests for the scoped retry on LanceDBManager.get_table()
(Improvement #17).

Verifies the retry is correctly scoped to OSError (lock-contention-style
failures) and does NOT retry other exception types (real bugs like bad
table names or schema errors), which would otherwise waste time retrying
something that will never succeed.
"""

from unittest.mock import MagicMock

import pytest

from rag_system.indexing.embedders import LanceDBManager


@pytest.fixture
def manager():
    m = LanceDBManager.__new__(LanceDBManager)
    m.db = MagicMock()
    return m


class TestLanceDBGetTableRetry:
    def test_succeeds_immediately_when_no_error(self, manager):
        manager.db.open_table = MagicMock(return_value="a_table")
        result = manager.get_table("my_table")
        assert result == "a_table"
        assert manager.db.open_table.call_count == 1

    def test_retries_on_oserror_and_eventually_succeeds(self, manager):
        call_count = {"n": 0}

        def flaky(name):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise OSError("simulated lock contention")
            return "the_table"

        manager.db.open_table = flaky
        result = manager.get_table("my_table")
        assert result == "the_table"
        assert call_count["n"] == 3

    def test_gives_up_after_max_attempts_on_persistent_oserror(self, manager):
        call_count = {"n": 0}

        def always_fails(name):
            call_count["n"] += 1
            raise OSError("persistent lock issue")

        manager.db.open_table = always_fails
        with pytest.raises(OSError):
            manager.get_table("my_table")
        assert call_count["n"] == 3

    def test_non_oserror_exceptions_are_not_retried(self, manager):
        """A real bug (bad table name, schema mismatch) should fail fast,
        not waste time retrying something that will never succeed."""
        call_count = {"n": 0}

        def real_bug(name):
            call_count["n"] += 1
            raise ValueError("table does not exist")

        manager.db.open_table = real_bug
        with pytest.raises(ValueError):
            manager.get_table("my_table")
        assert call_count["n"] == 1
