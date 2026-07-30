"""Unit tests for backend/auth.py (Improvement #10)."""

import time

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from backend.auth import (
    check_ownership,
    create_access_token,
    decode_access_token,
    get_current_user,
    get_current_user_optional,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_correct_password_verifies(self):
        password_hash, salt = hash_password("correct-horse-battery-staple")
        assert verify_password("correct-horse-battery-staple", password_hash, salt) is True

    def test_wrong_password_rejected(self):
        password_hash, salt = hash_password("correct-horse-battery-staple")
        assert verify_password("wrong-password", password_hash, salt) is False

    def test_different_calls_produce_different_salts(self):
        _, salt1 = hash_password("same-password")
        _, salt2 = hash_password("same-password")
        assert salt1 != salt2

    def test_different_salts_produce_different_hashes_for_same_password(self):
        hash1, _ = hash_password("same-password")
        hash2, _ = hash_password("same-password")
        assert hash1 != hash2

    def test_hash_from_one_salt_does_not_verify_against_another(self):
        password_hash, salt = hash_password("my-password")
        _, other_salt = hash_password("unrelated")
        assert verify_password("my-password", password_hash, other_salt) is False


class TestJWTTokens:
    def test_token_roundtrip_contains_correct_claims(self):
        token = create_access_token("user-123", "test@example.com")
        payload = decode_access_token(token)
        assert payload["sub"] == "user-123"
        assert payload["email"] == "test@example.com"

    def test_malformed_token_raises_invalid_token_error(self):
        with pytest.raises(pyjwt.InvalidTokenError):
            decode_access_token("not.a.valid.token")

    def test_tampered_token_raises_invalid_token_error(self):
        token = create_access_token("user-123", "test@example.com")
        tampered = token[:-4] + "xxxx"  # corrupt the signature
        with pytest.raises(pyjwt.InvalidTokenError):
            decode_access_token(tampered)

    def test_expired_token_raises_expired_signature_error(self, monkeypatch):
        import rag_system.config as config_module

        monkeypatch.setattr(config_module.settings, "jwt_expiry_minutes", -1)  # already expired
        token = create_access_token("user-123", "test@example.com")
        with pytest.raises(pyjwt.ExpiredSignatureError):
            decode_access_token(token)


class TestGetCurrentUser:
    def test_valid_bearer_token_returns_user_dict(self):
        token = create_access_token("user-123", "test@example.com")
        result = get_current_user(authorization=f"Bearer {token}")
        assert result == {"user_id": "user-123", "email": "test@example.com"}

    def test_missing_header_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(authorization=None)
        assert exc_info.value.status_code == 401

    def test_malformed_header_without_bearer_prefix_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(authorization="not-a-bearer-token")
        assert exc_info.value.status_code == 401

    def test_invalid_token_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(authorization="Bearer garbage.token.here")
        assert exc_info.value.status_code == 401


class TestGetCurrentUserOptional:
    def test_valid_token_returns_user(self):
        token = create_access_token("user-123", "test@example.com")
        result = get_current_user_optional(authorization=f"Bearer {token}")
        assert result == {"user_id": "user-123", "email": "test@example.com"}

    def test_missing_header_returns_none_not_raise(self):
        result = get_current_user_optional(authorization=None)
        assert result is None

    def test_invalid_token_returns_none_not_raise(self):
        result = get_current_user_optional(authorization="Bearer garbage")
        assert result is None


class TestCheckOwnership:
    def test_unowned_resource_accessible_to_anyone(self):
        check_ownership(None, None)  # should not raise
        check_ownership(None, {"user_id": "someone", "email": "a@b.com"})  # should not raise

    def test_owner_can_access_own_resource(self):
        check_ownership("user-123", {"user_id": "user-123", "email": "a@b.com"})  # should not raise

    def test_non_owner_denied_with_403(self):
        with pytest.raises(HTTPException) as exc_info:
            check_ownership("user-123", {"user_id": "user-999", "email": "x@y.com"})
        assert exc_info.value.status_code == 403

    def test_anonymous_user_denied_access_to_owned_resource(self):
        with pytest.raises(HTTPException) as exc_info:
            check_ownership("user-123", None)
        assert exc_info.value.status_code == 403
