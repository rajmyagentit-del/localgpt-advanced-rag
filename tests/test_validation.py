"""Unit tests for backend/validation.py (Improvement #6)."""

import uuid

import pytest

from backend.validation import is_valid_id, validate_chat_message, MAX_MESSAGE_LENGTH


class TestIsValidId:
    def test_real_uuid4_is_valid(self):
        assert is_valid_id(str(uuid.uuid4())) is True

    @pytest.mark.parametrize(
        "bad_value",
        ["not-a-uuid", "", "12345", "'; DROP TABLE sessions;--", None, 12345, ["a", "b"]],
    )
    def test_rejects_malformed_or_wrong_type(self, bad_value):
        assert is_valid_id(bad_value) is False


class TestValidateChatMessage:
    def test_normal_message_is_valid(self):
        is_valid, error = validate_chat_message("What is the refund policy?")
        assert is_valid is True
        assert error is None

    def test_empty_string_rejected(self):
        is_valid, error = validate_chat_message("")
        assert is_valid is False
        assert error is not None

    def test_whitespace_only_rejected(self):
        is_valid, error = validate_chat_message("    \n\t  ")
        assert is_valid is False

    def test_non_string_type_rejected(self):
        is_valid, _ = validate_chat_message(12345)
        assert is_valid is False

    def test_message_at_max_length_is_valid(self):
        is_valid, _ = validate_chat_message("x" * MAX_MESSAGE_LENGTH)
        assert is_valid is True

    def test_message_over_max_length_rejected(self):
        is_valid, error = validate_chat_message("x" * (MAX_MESSAGE_LENGTH + 1))
        assert is_valid is False
        assert "maximum length" in error
