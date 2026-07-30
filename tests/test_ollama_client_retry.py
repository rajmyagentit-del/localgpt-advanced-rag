"""
Unit tests for backend/ollama_client.py's retry/backoff logic (Improvement #17).

Uses a fake requests.post that fails a configurable number of times
before succeeding, so we can verify the retry actually happens (and
stops retrying once it succeeds, or gives up after 3 attempts) without
needing a real flaky network.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.ollama_client import OllamaClient


class FakeFlakyPost:
    """Callable that fails N times with a RequestException, then succeeds."""

    def __init__(self, fail_times: int, success_response=None):
        self.fail_times = fail_times
        self.call_count = 0
        self.success_response = success_response or MagicMock(status_code=200)

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise requests.exceptions.ConnectionError("simulated transient failure")
        return self.success_response


class TestOllamaClientRetry:
    def test_succeeds_immediately_with_no_failures(self):
        client = OllamaClient(base_url="http://fake-host:11434")
        fake_post = FakeFlakyPost(fail_times=0)
        with patch("backend.ollama_client.requests.post", fake_post):
            response = client._post("/chat", json={})
        assert fake_post.call_count == 1
        assert response.status_code == 200

    def test_retries_and_eventually_succeeds(self):
        client = OllamaClient(base_url="http://fake-host:11434")
        fake_post = FakeFlakyPost(fail_times=2)  # fails twice, succeeds on 3rd try
        with patch("backend.ollama_client.requests.post", fake_post):
            response = client._post("/chat", json={})
        assert fake_post.call_count == 3
        assert response.status_code == 200

    def test_gives_up_after_max_attempts_and_raises(self):
        client = OllamaClient(base_url="http://fake-host:11434")
        fake_post = FakeFlakyPost(fail_times=10)  # always fails
        with patch("backend.ollama_client.requests.post", fake_post):
            with pytest.raises(requests.exceptions.ConnectionError):
                client._post("/chat", json={})
        # Should have tried exactly 3 times (our configured stop_after_attempt), not more
        assert fake_post.call_count == 3

    def test_chat_method_still_returns_graceful_error_string_after_retries_exhausted(self):
        """The outer chat() method's existing behavior (return an error
        string instead of raising) must still work once retries are
        exhausted - this is a backward-compatibility guarantee for
        existing callers."""
        client = OllamaClient(base_url="http://fake-host:11434")
        fake_post = FakeFlakyPost(fail_times=10)
        with patch("backend.ollama_client.requests.post", fake_post):
            result = client.chat("hello")
        assert "Connection error" in result
        assert fake_post.call_count == 3
