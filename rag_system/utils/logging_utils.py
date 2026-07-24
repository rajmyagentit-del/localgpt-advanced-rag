"""
Centralized logging configuration for the localGPT application.

USAGE
-----
At the very top of an ENTRYPOINT (a script or server that actually gets
launched — e.g. backend/server.py's `if __name__ == "__main__":` block,
or rag_system/api_server.py), call once:

    from rag_system.utils.logging_utils import setup_logging
    setup_logging()

In every OTHER module, just use the standard Python pattern — get a
logger scoped to that module's name, and use it directly:

    import logging
    logger = logging.getLogger(__name__)

    logger.debug("Fine-grained detail, only shown when debugging")
    logger.info("Something normal and expected happened")
    logger.warning("Something looked off, but the app recovered")
    logger.error("Something failed", exc_info=True)

Do NOT call setup_logging() from every module — call it exactly ONCE,
at the entrypoint. Every other module should only ever call
logging.getLogger(__name__) and log through it.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from textwrap import shorten
from typing import Dict, List


class JsonFormatter(logging.Formatter):
    """
    Renders each log record as a single-line JSON object instead of plain
    text. Useful once logs are shipped somewhere that parses them
    automatically (e.g. Datadog, CloudWatch, an ELK stack) rather than
    just being read by a human scrolling a terminal.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_configured = False


def setup_logging(level: str | None = None) -> None:
    """
    Configure the ROOT logger once for the whole application.

    Controlled by environment variables so behavior can change between
    dev/staging/prod WITHOUT touching code:

      LOG_LEVEL   DEBUG | INFO | WARNING | ERROR   (default: INFO)
      LOG_FORMAT  text | json                       (default: text)

    Safe to call more than once — only the first call has any effect, so
    if two entrypoints both import a module that calls this, log lines
    won't get duplicated.
    """
    global _configured
    if _configured:
        return

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(handler)

    # Third-party libraries (httpx, urllib3, etc.) are chatty at INFO/DEBUG.
    # Keep them quiet unless we're specifically debugging the app itself.
    if resolved_level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    _configured = True


# --- Domain-specific helpers (kept from the original file, unchanged) ---

logger = logging.getLogger("rag-system")


def log_query(query: str, sub_queries: List[str] | None = None) -> None:
    """Emit a nicely-formatted block describing the incoming query and any
    decomposition."""
    border = "=" * 60
    logger.info("\n%s\nUSER QUERY: %s", border, query)
    if sub_queries:
        for i, q in enumerate(sub_queries, 1):
            logger.info("  sub-%d → %s", i, q)
    logger.info("%s", border)


def log_retrieval_results(results: List[Dict], k: int) -> None:
    """Show chunk_id, truncated text and score for the first *k* rows."""
    if not results:
        logger.info("Retrieval returned 0 documents.")
        return
    logger.info("Top %d results:", min(k, len(results)))
    header = f"{'chunk_id':<14} {'score':<7} preview"
    logger.info(header)
    logger.info("-" * len(header))
    for row in results[:k]:
        preview = shorten(row.get("text", ""), width=60, placeholder="…")
        logger.info("%s %-7.3f %s", str(row.get("chunk_id"))[:12], row.get("score", 0.0), preview) 