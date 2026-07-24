import logging
import os

# ---------------------------------------------------------
# Global logging setup for the entire `rag_system` package.
# ---------------------------------------------------------
# This is the ONE place logging gets configured. Every other module in
# this codebase should just call logging.getLogger(__name__) and log
# through it - see rag_system/utils/logging_utils.py for the full
# explanation and the setup_logging() implementation.
#
# RAG_LOG_LEVEL is kept as a legacy alias for LOG_LEVEL so existing
# deployments that set it don't break; LOG_LEVEL takes precedence if
# both are set.
# ---------------------------------------------------------
from rag_system.utils.logging_utils import setup_logging

_legacy_level = os.getenv("RAG_LOG_LEVEL")
if _legacy_level and not os.getenv("LOG_LEVEL"):
    os.environ["LOG_LEVEL"] = _legacy_level

setup_logging()

logging.getLogger(__name__).debug("Initialized rag_system logging")

# ---------------------------------------------------------
# Authenticate to Hugging Face Hub if a token is provided
# ---------------------------------------------------------
from typing import Optional


def _hf_auto_login() -> None:
    """Attempt to authenticate with Hugging Face Hub using an env token.

    We support both the new canonical env var name (HF_TOKEN) and the two
    historical variants to avoid breaking user setups. The login call is
    idempotent: if a cached token already exists, the hub library will simply
    reuse it, so it is safe to run on every import.
    """

    import os

    token: Optional[str] = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

    if not token:
        logging.getLogger(__name__).debug("No Hugging Face token found in env; proceeding anonymously.")
        return

    try:
        from huggingface_hub import login as hf_login

        hf_login(token=token, add_to_git_credential=False)  # type: ignore
        logging.getLogger(__name__).info("Authenticated to Hugging Face Hub via env token.")
    except Exception as exc:  # pragma: no cover – best-effort login
        logging.getLogger(__name__).warning(
            "Failed to login to Hugging Face Hub automatically: %s", exc
        )


# Run on module import
_hf_auto_login() 