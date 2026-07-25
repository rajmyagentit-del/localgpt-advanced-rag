"""
Shared input validation helpers for the backend API.

Centralizing these means every endpoint validates IDs and payloads the
same way, instead of each handler reinventing (or forgetting) its own
checks.
"""

import re
import uuid

# Session IDs and index IDs are created with uuid.uuid4() (see database.py) -
# so anything that isn't a valid UUID is either a bug in the caller or a
# malicious/malformed request, and should be rejected before it reaches
# any database query.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

MAX_MESSAGE_LENGTH = 8000  # characters - generous for real questions, blocks abuse
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB per file
MAX_TOTAL_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB per request across all files


def is_valid_id(value: str) -> bool:
    """True if `value` looks like a UUID4 our own code would have generated."""
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return bool(_UUID_RE.match(value))


def validate_chat_message(message) -> tuple[bool, str | None]:
    """
    Validate a chat message payload.

    Returns (is_valid, error_message). error_message is None when valid.
    """
    if not isinstance(message, str):
        return False, "Message must be a string"
    stripped = message.strip()
    if not stripped:
        return False, "Message is required"
    if len(stripped) > MAX_MESSAGE_LENGTH:
        return False, f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters"
    return True, None
