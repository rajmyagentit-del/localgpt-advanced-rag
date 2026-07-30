"""
Simple in-memory rate limiter.

WHY IN-MEMORY (for now): this backend currently runs as a single process
(see roadmap item on moving the semantic cache to Redis for the same
underlying reason - horizontal scaling). A single process's memory is a
perfectly fine place for rate-limit counters UNTIL you run more than one
instance behind a load balancer, at which point each instance would track
limits independently and the effective limit becomes (per-instance limit
x instance count). That's an explicit known limitation, not an oversight -
see the Redis-backed cache improvement in the roadmap for the natural next
step once this app is scaled horizontally.

Uses a fixed-window counter per client IP: simple, cheap, good enough to
stop a single bad client (or a buggy frontend retry loop) from hammering
the LLM/embedding pipeline.
"""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Returns (is_allowed, retry_after_seconds).
        retry_after_seconds is 0 when allowed.
        """
        now = time.monotonic()
        with self._lock:
            timestamps = self._requests[client_id]

            # Drop timestamps outside the current window
            while timestamps and timestamps[0] <= now - self.window_seconds:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = int(self.window_seconds - (now - timestamps[0])) + 1
                return False, max(retry_after, 1)

            timestamps.append(now)
            return True, 0


# Two separate limiters with different thresholds - chat is a lighter
# operation than a file upload feeding a full indexing pipeline, so it
# tolerates a higher request rate.
chat_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)
upload_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
