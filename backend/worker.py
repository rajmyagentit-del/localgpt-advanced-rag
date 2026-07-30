"""
Entrypoint for the RQ background worker process (Improvement #14).

Run with: python -m backend.worker

This is a SEPARATE process from the main FastAPI app (backend/app.py) -
in Docker, it runs as its own service (see docker-compose.yml's
`indexing-worker` service). The FastAPI app enqueues jobs; this process
picks them up and actually runs them.
"""

import logging

from rq import Worker

from backend.tasks import QUEUE_NAME, get_queue
from rag_system.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    queue = get_queue()
    if queue is None:
        logger.error(
            "Cannot start the indexing worker: REDIS_URL is not set or Redis is "
            "unreachable. The worker has nothing to connect to - set REDIS_URL "
            "and ensure Redis is running, then retry."
        )
        raise SystemExit(1)

    worker = Worker([QUEUE_NAME], connection=queue.connection)
    logger.info(f"Starting RQ worker, listening on queue '{QUEUE_NAME}'...")
    worker.work()


if __name__ == "__main__":
    main()
