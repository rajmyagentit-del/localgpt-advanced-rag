"""
Tests for backend/app.py (Improvement #9 - FastAPI migration).

Uses FastAPI's TestClient (in-process, no real socket/server needed -
this is itself an improvement over the old raw http.server backend,
which could only be tested by actually starting it on a port). A real
temporary SQLite database is used (not mocked) so these exercise real
CRUD behavior end-to-end. Routes that need a live Ollama or the RAG API
server (chat, session messages, index building) are tested for their
INPUT VALIDATION and ERROR HANDLING, which don't need live infra - the
underlying LLM/RAG behavior itself is out of scope here, consistent
with how the rest of this project's test suite handles ML-dependent
code (see retrieval_pipeline.py's docstrings for the same reasoning).
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """
    Fresh app + isolated temp SQLite DB per test, so tests never touch
    the real chat_data.db and don't leak state between tests.
    """
    tmp_db_fd, tmp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(tmp_db_fd)

    import backend.app as app_module
    from backend.database import ChatDatabase

    app_module.db = ChatDatabase(db_path=tmp_db_path)

    with TestClient(app_module.app) as test_client:
        yield test_client

    os.unlink(tmp_db_path)


class TestHealthEndpoint:
    def test_health_endpoint_returns_structured_response(self, client):
        response = client.get("/v1/health")
        # Status may be 200 or 503 depending on whether Ollama happens to
        # be running in this environment - either way, the SHAPE must be right.
        assert response.status_code in (200, 503)
        body = response.json()
        assert "status" in body
        assert "checks" in body
        assert "ollama" in body["checks"]
        assert "database" in body["checks"]

    def test_database_check_is_up_with_working_db(self, client):
        response = client.get("/v1/health")
        assert response.json()["checks"]["database"]["status"] == "up"


class TestSessionCRUD:
    def test_create_and_get_session(self, client):
        create_resp = client.post("/v1/sessions", json={"title": "Test Chat", "model": "llama3.2"})
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        get_resp = client.get(f"/v1/sessions/{session_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["session"]["title"] == "Test Chat"

    def test_get_nonexistent_session_returns_404(self, client):
        import uuid

        response = client.get(f"/v1/sessions/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_get_session_with_malformed_id_returns_400(self, client):
        response = client.get("/v1/sessions/not-a-real-uuid")
        assert response.status_code == 400

    def test_list_sessions_reflects_created_sessions(self, client):
        client.post("/v1/sessions", json={"title": "Session A"})
        client.post("/v1/sessions", json={"title": "Session B"})
        response = client.get("/v1/sessions")
        assert response.status_code == 200
        assert response.json()["total"] >= 2

    def test_delete_session(self, client):
        create_resp = client.post("/v1/sessions", json={"title": "To Delete"})
        session_id = create_resp.json()["session_id"]

        delete_resp = client.delete(f"/v1/sessions/{session_id}")
        assert delete_resp.status_code == 200

        get_resp = client.get(f"/v1/sessions/{session_id}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent_session_returns_404(self, client):
        import uuid

        response = client.delete(f"/v1/sessions/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_rename_session(self, client):
        create_resp = client.post("/v1/sessions", json={"title": "Old Title"})
        session_id = create_resp.json()["session_id"]

        rename_resp = client.put(f"/v1/sessions/{session_id}/rename", json={"title": "New Title"})
        assert rename_resp.status_code == 200
        assert rename_resp.json()["session"]["title"] == "New Title"

    def test_rename_session_with_empty_title_rejected(self, client):
        create_resp = client.post("/v1/sessions", json={"title": "Original"})
        session_id = create_resp.json()["session_id"]

        rename_resp = client.put(f"/v1/sessions/{session_id}/rename", json={"title": "   "})
        assert rename_resp.status_code == 400

    def test_create_session_uses_default_title_when_omitted(self, client):
        response = client.post("/v1/sessions", json={})
        assert response.status_code == 201
        session = client.get(f"/v1/sessions/{response.json()['session_id']}").json()["session"]
        assert session["title"] == "New Chat"


class TestIndexCRUD:
    def test_create_and_get_index(self, client):
        create_resp = client.post("/v1/indexes", json={"name": "My Index", "description": "test"})
        assert create_resp.status_code == 201
        index_id = create_resp.json()["index_id"]

        get_resp = client.get(f"/v1/indexes/{index_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "My Index"

    def test_create_index_without_name_rejected(self, client):
        response = client.post("/v1/indexes", json={"description": "no name given"})
        # Pydantic itself rejects this (name is a required field) with 422
        assert response.status_code == 422

    def test_list_indexes(self, client):
        client.post("/v1/indexes", json={"name": "Index A"})
        response = client.get("/v1/indexes")
        assert response.status_code == 200
        assert response.json()["total"] >= 1

    def test_delete_index(self, client):
        create_resp = client.post("/v1/indexes", json={"name": "Temp Index"})
        index_id = create_resp.json()["index_id"]

        delete_resp = client.delete(f"/v1/indexes/{index_id}")
        assert delete_resp.status_code == 200

        get_resp = client.get(f"/v1/indexes/{index_id}")
        assert get_resp.status_code == 404

    def test_get_nonexistent_index_returns_404(self, client):
        import uuid

        response = client.get(f"/v1/indexes/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_build_index_with_no_documents_returns_400(self, client):
        create_resp = client.post("/v1/indexes", json={"name": "Empty Index"})
        index_id = create_resp.json()["index_id"]

        build_resp = client.post(f"/v1/indexes/{index_id}/build", json={})
        assert build_resp.status_code == 400


class TestAsyncIndexing:
    """Improvement #14 - tests the endpoint's behavior for the
    no-documents case (needs no queue/Redis) and confirms the job status
    endpoint responds correctly for a nonexistent job. Full queue
    behavior itself is covered directly in tests/test_tasks.py."""

    def test_indexing_session_with_no_documents_returns_message(self, client):
        create_resp = client.post("/v1/sessions", json={"title": "Empty Session"})
        session_id = create_resp.json()["session_id"]

        response = client.post(f"/v1/sessions/{session_id}/index")
        assert response.status_code == 200
        assert "No documents" in response.json()["message"]

    def test_job_status_for_nonexistent_job_returns_404(self, client):
        response = client.get("/v1/jobs/some-fake-job-id")
        assert response.status_code == 404


class TestChatValidation:
    """These test input validation and error handling WITHOUT needing a
    live Ollama server - the actual LLM response content is out of scope."""

    def test_empty_message_rejected(self, client):
        response = client.post("/v1/chat", json={"message": "", "model": "llama3.2"})
        assert response.status_code == 400

    def test_missing_message_field_rejected_by_pydantic(self, client):
        response = client.post("/v1/chat", json={"model": "llama3.2"})
        assert response.status_code == 422

    def test_session_chat_on_nonexistent_session_returns_404(self, client):
        import uuid

        response = client.post(f"/v1/sessions/{uuid.uuid4()}/messages", json={"message": "hello"})
        assert response.status_code == 404

    def test_session_chat_with_empty_message_rejected(self, client):
        create_resp = client.post("/v1/sessions", json={"title": "Chat Test"})
        session_id = create_resp.json()["session_id"]

        response = client.post(f"/v1/sessions/{session_id}/messages", json={"message": ""})
        assert response.status_code == 400


class TestModelsEndpoint:
    def test_models_endpoint_returns_expected_shape(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert "generation_models" in body
        assert "embedding_models" in body
        # HuggingFace embedding models are always listed regardless of
        # whether Ollama is running
        assert "Qwen/Qwen3-Embedding-0.6B" in body["embedding_models"]


class TestOpenAPIDocs:
    """Confirms the automatic API documentation actually works - a
    concrete, checkable benefit of the FastAPI migration."""

    def test_openapi_schema_is_generated(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "LocalGPT API"
        assert "/v1/health" in schema["paths"]

    def test_docs_ui_is_served(self, client):
        response = client.get("/docs")
        assert response.status_code == 200


class TestAuthRegisterAndLogin:
    def test_register_creates_account_and_returns_token(self, client):
        response = client.post(
            "/v1/auth/register", json={"email": "alice@example.com", "password": "hunter22"}
        )
        assert response.status_code == 201
        body = response.json()
        assert "access_token" in body
        assert body["email"] == "alice@example.com"

    def test_register_duplicate_email_rejected(self, client):
        client.post("/v1/auth/register", json={"email": "bob@example.com", "password": "hunter22"})
        response = client.post(
            "/v1/auth/register", json={"email": "bob@example.com", "password": "different1"}
        )
        assert response.status_code == 409

    def test_register_short_password_rejected(self, client):
        response = client.post(
            "/v1/auth/register", json={"email": "carol@example.com", "password": "short"}
        )
        assert response.status_code == 400

    def test_register_invalid_email_rejected(self, client):
        response = client.post(
            "/v1/auth/register", json={"email": "not-an-email", "password": "hunter22"}
        )
        assert response.status_code == 400

    def test_login_with_correct_credentials_succeeds(self, client):
        client.post(
            "/v1/auth/register", json={"email": "dave@example.com", "password": "correct-pw1"}
        )
        response = client.post(
            "/v1/auth/login", json={"email": "dave@example.com", "password": "correct-pw1"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_with_wrong_password_rejected(self, client):
        client.post(
            "/v1/auth/register", json={"email": "eve@example.com", "password": "correct-pw1"}
        )
        response = client.post(
            "/v1/auth/login", json={"email": "eve@example.com", "password": "wrong-password"}
        )
        assert response.status_code == 401

    def test_login_with_nonexistent_email_rejected(self, client):
        response = client.post(
            "/v1/auth/login", json={"email": "nobody@example.com", "password": "whatever1"}
        )
        assert response.status_code == 401


class TestOwnership:
    def _register_and_get_token(self, client, email):
        resp = client.post("/v1/auth/register", json={"email": email, "password": "hunter22222"})
        return resp.json()["access_token"]

    def test_session_created_with_auth_is_owned(self, client):
        token = self._register_and_get_token(client, "owner1@example.com")
        resp = client.post(
            "/v1/sessions",
            json={"title": "My Session"},
            headers={"Authorization": f"Bearer {token}"},
        )
        session_id = resp.json()["session_id"]
        get_resp = client.get(
            f"/v1/sessions/{session_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["session"]["user_id"] is not None

    def test_other_user_cannot_access_owned_session(self, client):
        token_a = self._register_and_get_token(client, "usera@example.com")
        token_b = self._register_and_get_token(client, "userb@example.com")

        create_resp = client.post(
            "/v1/sessions",
            json={"title": "A's Session"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        session_id = create_resp.json()["session_id"]

        # User B tries to access User A's session
        get_resp = client.get(
            f"/v1/sessions/{session_id}", headers={"Authorization": f"Bearer {token_b}"}
        )
        assert get_resp.status_code == 403

    def test_anonymous_user_cannot_access_owned_session(self, client):
        token = self._register_and_get_token(client, "owner2@example.com")
        create_resp = client.post(
            "/v1/sessions", json={"title": "Owned"}, headers={"Authorization": f"Bearer {token}"}
        )
        session_id = create_resp.json()["session_id"]

        # No Authorization header at all
        get_resp = client.get(f"/v1/sessions/{session_id}")
        assert get_resp.status_code == 403

    def test_owner_can_delete_own_session(self, client):
        token = self._register_and_get_token(client, "owner3@example.com")
        create_resp = client.post(
            "/v1/sessions",
            json={"title": "To Delete"},
            headers={"Authorization": f"Bearer {token}"},
        )
        session_id = create_resp.json()["session_id"]

        delete_resp = client.delete(
            f"/v1/sessions/{session_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert delete_resp.status_code == 200

    def test_non_owner_cannot_delete_session(self, client):
        token_a = self._register_and_get_token(client, "userc@example.com")
        token_b = self._register_and_get_token(client, "userd@example.com")

        create_resp = client.post(
            "/v1/sessions",
            json={"title": "Protected"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        session_id = create_resp.json()["session_id"]

        delete_resp = client.delete(
            f"/v1/sessions/{session_id}", headers={"Authorization": f"Bearer {token_b}"}
        )
        assert delete_resp.status_code == 403

    def test_anonymous_session_remains_accessible_to_anyone_backward_compat(self, client):
        """Sessions created WITHOUT auth (no token provided) have no
        owner, and per the documented backward-compatibility policy,
        remain accessible to anyone - this is what keeps pre-existing
        (pre-auth) data usable."""
        create_resp = client.post("/v1/sessions", json={"title": "Anonymous Session"})
        session_id = create_resp.json()["session_id"]

        # No auth header at all - should still work, since this session has no owner
        get_resp = client.get(f"/v1/sessions/{session_id}")
        assert get_resp.status_code == 200

    def test_index_ownership_enforced_same_as_sessions(self, client):
        token_a = self._register_and_get_token(client, "indexowner@example.com")
        token_b = self._register_and_get_token(client, "indexother@example.com")

        create_resp = client.post(
            "/v1/indexes",
            json={"name": "Owned Index"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        index_id = create_resp.json()["index_id"]

        # Owner can access
        own_resp = client.get(
            f"/v1/indexes/{index_id}", headers={"Authorization": f"Bearer {token_a}"}
        )
        assert own_resp.status_code == 200

        # Other user cannot
        other_resp = client.get(
            f"/v1/indexes/{index_id}", headers={"Authorization": f"Bearer {token_b}"}
        )
        assert other_resp.status_code == 403
