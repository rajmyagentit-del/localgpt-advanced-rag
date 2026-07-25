"""Unit tests for backend/rate_limiter.py (Improvement #8)."""

import time

from backend.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_allows_requests_under_the_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            allowed, _ = limiter.is_allowed("client-a")
            assert allowed is True

    def test_blocks_requests_over_the_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.is_allowed("client-a")
        allowed, retry_after = limiter.is_allowed("client-a")
        assert allowed is False
        assert retry_after > 0

    def test_clients_are_tracked_independently(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("client-x")[0] is True
        assert limiter.is_allowed("client-y")[0] is True
        assert limiter.is_allowed("client-x")[0] is False

    def test_window_expiry_allows_requests_again(self):
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.is_allowed("client-z")[0] is True
        assert limiter.is_allowed("client-z")[0] is False
        time.sleep(1.1)
        assert limiter.is_allowed("client-z")[0] is True
