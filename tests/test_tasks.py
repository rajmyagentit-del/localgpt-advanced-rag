"""
Unit tests for backend/tasks.py (Improvement #14).

Uses fakeredis (a real implementation of the Redis protocol) with RQ's
is_async=False mode, which runs enqueued jobs inline instead of needing
a real separate worker process - this gives genuine behavioral coverage
of the actual enqueue/execute/status-check flow, not just "was enqueue()
called."
"""

from unittest.mock import MagicMock, patch

import fakeredis
import requests
from rq import Queue

import backend.tasks as tasks_module
from backend.tasks import enqueue_or_run_sync, get_job_status, get_queue, index_documents_task


class TestIndexDocumentsTask:
    def test_successful_indexing_returns_completed_status(self):
        with patch("backend.tasks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"indexed": 3})
            result = index_documents_task(["/a.pdf"], "session-1")
        assert result["status"] == "completed"
        assert result["result"]["indexed"] == 3

    def test_rag_api_error_status_returns_failed(self):
        with patch("backend.tasks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=500, text="Internal error")
            result = index_documents_task(["/a.pdf"], "session-1")
        assert result["status"] == "failed"
        assert "500" in result["error"]

    def test_connection_error_handled_gracefully_not_raised(self):
        with patch(
            "backend.tasks.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            result = index_documents_task(["/a.pdf"], "session-1")
        assert result["status"] == "failed"
        assert "Could not reach" in result["error"]


class TestGetQueue:
    def test_returns_none_when_no_connection_available(self):
        with patch("backend.tasks._get_redis_connection", return_value=None):
            assert get_queue() is None

    def test_returns_queue_when_connection_provided(self):
        fake_conn = fakeredis.FakeStrictRedis()
        queue = get_queue(connection=fake_conn)
        assert queue is not None
        assert queue.name == "indexing"


class TestEnqueueOrRunSync:
    def test_falls_back_to_synchronous_execution_without_redis(self):
        with patch("backend.tasks._get_redis_connection", return_value=None):
            with patch("backend.tasks.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, json=lambda: {"indexed": 1})
                result = enqueue_or_run_sync(["/a.pdf"], "session-1")
        assert result["mode"] == "sync"
        assert result["status"] == "completed"

    def test_enqueues_via_real_queue_when_redis_available(self):
        fake_conn = fakeredis.FakeStrictRedis()
        sync_queue = Queue("indexing", connection=fake_conn, is_async=False)
        with patch("backend.tasks.get_queue", return_value=sync_queue):
            with patch("backend.tasks.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, json=lambda: {"indexed": 2})
                result = enqueue_or_run_sync(["/b.pdf"], "session-2")
        assert result["mode"] == "async"
        assert "job_id" in result


class TestGetJobStatus:
    def test_retrieves_real_job_status_after_execution(self):
        fake_conn = fakeredis.FakeStrictRedis()
        sync_queue = Queue("indexing", connection=fake_conn, is_async=False)
        with patch("backend.tasks.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"indexed": 5})
            job = sync_queue.enqueue(index_documents_task, ["/c.pdf"], "session-3")

        with patch("backend.tasks._get_redis_connection", return_value=fake_conn):
            status = get_job_status(job.id)

        assert status is not None
        assert status["job_id"] == job.id
        assert "FINISHED" in status["status"] or status["status"] == "finished"

    def test_nonexistent_job_returns_none(self):
        fake_conn = fakeredis.FakeStrictRedis()
        with patch("backend.tasks._get_redis_connection", return_value=fake_conn):
            assert get_job_status("this-job-does-not-exist") is None

    def test_returns_none_when_redis_unavailable(self):
        with patch("backend.tasks._get_redis_connection", return_value=None):
            assert get_job_status("any-id") is None
