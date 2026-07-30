"""
Async background task queue for document indexing, using RQ (Redis
Queue) - Improvement #14.

WHY THIS CHANGE: indexing a large document (or many documents) can take
minutes - embedding, chunking, and writing to LanceDB are all slow
relative to a normal HTTP request/response cycle. Before this change,
POST /v1/sessions/{id}/index blocked the HTTP request for the entire
duration. A real production system needs uploads to be fire-and-forget:
return immediately with a job you can poll, not hold the connection open.

WHY RQ (not Celery): this project already added Redis for the semantic
cache (Improvement #15). RQ is built directly on Redis, with a much
smaller footprint and simpler mental model than Celery (which needs its
own broker abstraction, typically also backed by Redis or RabbitMQ
anyway). Given Redis is already a dependency here, RQ is the more
proportionate choice - not reaching for a heavier tool than the job needs.

WHY GRACEFUL FALLBACK (not required Redis): consistent with the
semantic cache's philosophy (see semantic_cache.py) - if Redis isn't
configured or isn't reachable, indexing falls back to running
SYNCHRONOUSLY (the original blocking behavior) rather than failing the
request outright. Async indexing is a scaling optimization, not a hard
requirement to use the app at all.
"""

import logging
from typing import Any

import requests
from rq import Queue
from rq.job import Job

logger = logging.getLogger(__name__)

QUEUE_NAME = "indexing"
_QUEUE_NAME = QUEUE_NAME  # internal alias, kept for readability within this file


def _get_redis_connection():
    """Returns a real Redis connection if REDIS_URL is configured and
    reachable, otherwise None (caller falls back to synchronous
    execution)."""
    from rag_system.config import settings

    if not settings.redis_url:
        return None
    try:
        import redis as redis_lib

        client = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception as e:
        logger.warning(
            f"REDIS_URL is set but Redis is unreachable ({e}); indexing will run synchronously"
        )
        return None


def get_queue(connection=None) -> Queue | None:
    """Returns an RQ Queue if Redis is available, otherwise None."""
    conn = connection or _get_redis_connection()
    if conn is None:
        return None
    return Queue(_QUEUE_NAME, connection=conn)


def index_documents_task(
    file_paths: list[str], session_or_index_id: str, table_name: str | None = None
) -> dict:
    """
    The actual indexing work - calls the RAG API's /index endpoint. This
    is the function that runs in the background worker process (or
    inline, synchronously, if no queue is available - see
    enqueue_or_run_sync below). Kept as a plain, picklable function (RQ
    requirement) rather than a method, with no dependency on any request
    object.
    """
    try:
        response = requests.post(
            "http://localhost:8001/index",
            json={
                "file_paths": file_paths,
                "session_id": session_or_index_id,
                **({"table_name": table_name} if table_name else {}),
            },
            timeout=600,  # indexing can genuinely take minutes for large documents
        )
        if response.status_code == 200:
            return {"status": "completed", "result": response.json()}
        return {
            "status": "failed",
            "error": f"RAG API returned {response.status_code}: {response.text}",
        }
    except requests.exceptions.RequestException as e:
        return {"status": "failed", "error": f"Could not reach the RAG API: {e}"}


def enqueue_or_run_sync(
    file_paths: list[str], session_or_index_id: str, table_name: str | None = None
) -> dict[str, Any]:
    """
    Tries to enqueue the indexing job for background execution. Falls
    back to running it synchronously (blocking, like the original
    behavior) if Redis/the queue isn't available.

    Returns either:
      {"mode": "async", "job_id": "...", "status": "queued"}
      {"mode": "sync", "status": "completed"|"failed", ...task result...}
    """
    queue = get_queue()
    if queue is not None:
        job = queue.enqueue(index_documents_task, file_paths, session_or_index_id, table_name)
        return {"mode": "async", "job_id": job.id, "status": "queued"}

    logger.info("No queue available - running indexing synchronously (blocking)")
    result = index_documents_task(file_paths, session_or_index_id, table_name)
    return {"mode": "sync", **result}


def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Looks up a job by ID. Returns None if the job doesn't exist (e.g.
    wrong ID, or Redis's job TTL has expired it)."""
    conn = _get_redis_connection()
    if conn is None:
        return None
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return None

    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.return_value,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }
