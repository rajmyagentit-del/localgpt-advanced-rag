"""
FastAPI backend (Improvement #9 - migrated from raw http.server).

WHY THIS MIGRATION MATTERS: the previous backend/server.py routed
requests with a chain of `if path.startswith(...)` string checks,
manually read Content-Length + decoded JSON by hand in every handler,
and had no automatic request validation or API documentation. This
version gets, for free: automatic request/response validation via
Pydantic models, interactive OpenAPI docs at /docs, native async
support, and a real router instead of string matching.

All routes are versioned under /v1 (Improvement #16) - this means a
future /v2 can be introduced later without breaking existing clients,
which the previous unversioned API had no way to do cleanly.

Business logic is reused, not rewritten: this file is a thin routing
layer over the same database.py, ollama_client.py, validation.py,
rate_limiter.py, and the newly-extracted chat_service.py - the parts
that actually matter for correctness were already built and tested
earlier in this project's history.

Run with: uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.auth import (
    check_ownership,
    create_access_token,
    get_current_user,
    get_current_user_optional,
    hash_password,
    verify_password,
)
from backend.chat_service import ChatService
from backend.database import ChatDatabase, generate_session_title
from backend.ollama_client import OllamaClient
from backend.rate_limiter import chat_rate_limiter, upload_rate_limiter
from backend.validation import (
    MAX_TOTAL_UPLOAD_BYTES,
    is_valid_id,
    validate_chat_message,
)
from rag_system.config import settings

logger = logging.getLogger(__name__)

db = ChatDatabase()
ollama_client = OllamaClient()
chat_service = ChatService(ollama_client)

try:
    from rag_system.main import PIPELINE_CONFIGS
    RAG_SYSTEM_AVAILABLE = True
except ImportError as e:
    PIPELINE_CONFIGS = {}
    RAG_SYSTEM_AVAILABLE = False
    logger.warning(f"⚠️ RAG system modules not available: {e}")

app = FastAPI(
    title="LocalGPT API",
    version="1.0.0",
    description="RAG system backend API. See /docs for interactive documentation.",
)


# --- Pydantic request/response models ---

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    model: str = "llama3.2:latest"


class ChatRequest(BaseModel):
    message: str
    model: str = "llama3.2:latest"
    conversation_history: list[dict] = Field(default_factory=list)


class SessionChatRequest(BaseModel):
    message: str
    force_rag: bool = False
    compose_sub_answers: bool | None = None
    query_decompose: bool | None = None
    ai_rerank: bool | None = None
    context_expand: bool | None = None
    verify: bool | None = None
    retrieval_k: int | None = None
    context_window_size: int | None = None
    reranker_top_k: int | None = None
    search_type: str | None = None
    dense_weight: float | None = None
    provence_prune: bool | None = None
    provence_threshold: float | None = None


class RenameSessionRequest(BaseModel):
    title: str


class CreateIndexRequest(BaseModel):
    name: str
    description: str | None = None
    metadata: dict = Field(default_factory=dict)


class BuildIndexRequest(BaseModel):
    latechunk: bool = False
    doclingChunk: bool = False
    chunkSize: int = 512
    chunkOverlap: int = 64
    retrievalMode: str = "hybrid"
    windowSize: int = 2
    enableEnrich: bool = True
    embeddingModel: str | None = None
    enrichModel: str | None = None
    batchSizeEmbed: int = 50
    batchSizeEnrich: int = 25
    overviewModel: str | None = None


def _valid_id(value: str, kind: str = "resource") -> str:
    if not is_valid_id(value):
        raise HTTPException(status_code=400, detail=f"Invalid or malformed {kind} ID")
    return value


def _rate_limit(request: Request, limiter):
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
            headers={"Retry-After": str(retry_after)},
        )


@app.get("/v1/health")
def health_check():
    checks = {}
    overall_healthy = True

    try:
        ollama_up = ollama_client.is_ollama_running()
        checks["ollama"] = {"status": "up" if ollama_up else "down", "required": True}
        if not ollama_up:
            overall_healthy = False
    except Exception as e:
        checks["ollama"] = {"status": "error", "required": True, "detail": str(e)}
        overall_healthy = False

    try:
        stats = db.get_stats()
        checks["database"] = {"status": "up", "required": True, "stats": stats}
    except Exception as e:
        checks["database"] = {"status": "error", "required": True, "detail": str(e)}
        overall_healthy = False

    try:
        checks["lancedb"] = {
            "status": "up" if os.path.isdir(settings.lancedb_path) else "not_initialized",
            "required": False,
            "path": settings.lancedb_path,
        }
    except Exception as e:
        checks["lancedb"] = {"status": "error", "required": False, "detail": str(e)}

    status_code = 200 if overall_healthy else 503
    body = {"status": "healthy" if overall_healthy else "unhealthy", "checks": checks}
    return JSONResponse(content=body, status_code=status_code)


# --- Auth (Improvement #10) ---

@app.post("/v1/auth/register", status_code=201)
def register(body: RegisterRequest):
    email = body.email.strip().lower()
    if "@" not in email or len(email) < 5:
        raise HTTPException(status_code=400, detail="Please provide a valid email address")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if db.get_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    password_hash, salt = hash_password(body.password)
    user_id = db.create_user(email, password_hash, salt)
    token = create_access_token(user_id, email)
    return {"access_token": token, "token_type": "bearer", "user_id": user_id, "email": email}


@app.post("/v1/auth/login")
def login(body: LoginRequest):
    email = body.email.strip().lower()
    user = db.get_user_by_email(email)
    # Deliberately identical error for "no such user" and "wrong password" -
    # distinguishing them lets an attacker enumerate valid email addresses.
    invalid_credentials = HTTPException(status_code=401, detail="Invalid email or password")
    if not user:
        raise invalid_credentials
    if not verify_password(body.password, user["password_hash"], user["password_salt"]):
        raise invalid_credentials

    token = create_access_token(user["id"], user["email"])
    return {"access_token": token, "token_type": "bearer", "user_id": user["id"], "email": user["email"]}


@app.post("/v1/chat")
def chat(body: ChatRequest, request: Request):
    _rate_limit(request, chat_rate_limiter)

    is_valid, error = validate_chat_message(body.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    if not ollama_client.is_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama is not running. Please start Ollama first.")

    response = ollama_client.chat(body.message, body.model, body.conversation_history)
    return {
        "response": response,
        "model": body.model,
        "message_count": len(body.conversation_history) + 1,
    }


@app.get("/v1/models")
def get_models():
    generation_models: list[str] = []
    embedding_models: list[str] = []

    if ollama_client.is_ollama_running():
        all_ollama_models = ollama_client.list_models()
        ollama_embedding_models = [
            m for m in all_ollama_models if any(k in m for k in ["embed", "bge", "embedding", "text"])
        ]
        ollama_generation_models = [m for m in all_ollama_models if m not in ollama_embedding_models]
        generation_models.extend(ollama_generation_models)
        embedding_models.extend(ollama_embedding_models)

    embedding_models.extend([
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-8B",
    ])

    generation_models.sort()
    embedding_models.sort()
    return {"generation_models": generation_models, "embedding_models": embedding_models}


@app.get("/v1/sessions")
def get_sessions():
    sessions = db.get_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@app.post("/v1/sessions/cleanup")
def cleanup_sessions():
    cleanup_count = db.cleanup_empty_sessions()
    return {"message": f"Cleaned up {cleanup_count} empty sessions", "cleanup_count": cleanup_count}


@app.post("/v1/sessions", status_code=201)
def create_session(body: CreateSessionRequest, current_user: dict | None = Depends(get_current_user_optional)):
    user_id = current_user["user_id"] if current_user else None
    session_id = db.create_session(body.title, body.model, user_id=user_id)
    session = db.get_session(session_id)
    return {"session": session, "session_id": session_id}


@app.get("/v1/sessions/{session_id}")
def get_session(session_id: str, current_user: dict | None = Depends(get_current_user_optional)):
    session_id = _valid_id(session_id, "session")
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    check_ownership(session.get("user_id"), current_user)
    messages = db.get_messages(session_id)
    return {"session": session, "messages": messages}


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str, current_user: dict | None = Depends(get_current_user_optional)):
    session_id = _valid_id(session_id, "session")
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    check_ownership(session.get("user_id"), current_user)
    deleted = db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": deleted}


@app.put("/v1/sessions/{session_id}/rename")
def rename_session(session_id: str, body: RenameSessionRequest, current_user: dict | None = Depends(get_current_user_optional)):
    session_id = _valid_id(session_id, "session")
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    check_ownership(session.get("user_id"), current_user)
    new_title = body.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    db.update_session_title(session_id, new_title)
    return {"session": db.get_session(session_id)}


@app.get("/v1/sessions/{session_id}/documents")
def get_session_documents(session_id: str):
    session_id = _valid_id(session_id, "session")
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    docs = db.get_documents_for_session(session_id)
    filenames = [
        os.path.basename(p).split("_", 1)[-1] if "_" in os.path.basename(p) else os.path.basename(p)
        for p in docs
    ]
    return {"session": session, "files": filenames, "file_count": len(docs)}


@app.post("/v1/sessions/{session_id}/messages")
def session_chat(session_id: str, body: SessionChatRequest, request: Request, current_user: dict | None = Depends(get_current_user_optional)):
    session_id = _valid_id(session_id, "session")
    _rate_limit(request, chat_rate_limiter)

    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    check_ownership(session.get("user_id"), current_user)

    is_valid, error = validate_chat_message(body.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    if session["message_count"] == 0:
        title = generate_session_title(body.message)
        db.update_session_title(session_id, title)

    db.add_message(session_id, body.message, "user")

    idx_ids = db.get_indexes_for_session(session_id)
    use_rag = True if body.force_rag else chat_service.should_use_rag(body.message, idx_ids)

    if use_rag:
        logger.debug(f"🔍 Using RAG pipeline for document query: '{body.message[:50]}...'")
        response_text, source_docs = chat_service.handle_rag_query(
            session_id, body.message, body.model_dump(), idx_ids
        )
    else:
        logger.debug(f"⚡ Using direct LLM for general query: '{body.message[:50]}...'")
        response_text, source_docs = chat_service.handle_direct_llm_query(
            session_id, body.message, session, db
        )

    db.add_message(session_id, response_text, "assistant")
    updated_session = db.get_session(session_id)

    return {
        "response": response_text,
        "session": updated_session,
        "source_documents": source_docs,
        "used_rag": use_rag,
    }


@app.post("/v1/sessions/{session_id}/upload")
async def upload_files_to_session(session_id: str, request: Request, files: list[UploadFile]):
    session_id = _valid_id(session_id, "session")
    _rate_limit(request, upload_rate_limiter)

    content_length = int(request.headers.get("content-length", 0) or 0)
    if content_length > MAX_TOTAL_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds maximum total size of {MAX_TOTAL_UPLOAD_BYTES // (1024*1024)}MB",
        )

    uploaded_files = []
    upload_dir = "shared_uploads"
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        if not file.filename:
            continue
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        absolute_file_path = os.path.abspath(file_path)
        db.add_document_to_session(session_id, absolute_file_path)
        uploaded_files.append({"filename": file.filename, "stored_path": absolute_file_path})

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    return {"message": f"Successfully uploaded {len(uploaded_files)} files.", "uploaded_files": uploaded_files}


@app.post("/v1/sessions/{session_id}/index")
def index_session_documents(session_id: str):
    session_id = _valid_id(session_id, "session")
    logger.info(f"🔥 Received request to index documents for session {session_id[:8]}...")

    file_paths = db.get_documents_for_session(session_id)
    if not file_paths:
        return {"message": "No documents to index for this session."}

    logger.info(f"Found {len(file_paths)} documents to index. Sending to RAG API...")

    import requests
    try:
        rag_response = requests.post(
            "http://localhost:8001/index", json={"file_paths": file_paths, "session_id": session_id}
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Exception during indexing: {e}")
        raise HTTPException(status_code=502, detail=f"Could not reach the RAG API: {e}")

    if rag_response.status_code == 200:
        logger.info("✅ RAG API successfully indexed documents.")
        try:
            db.update_index_metadata(session_id, {"session_linked": True, "retrieval_mode": "hybrid"})
        except Exception as e:
            logger.warning(f"⚠️ Failed to update index metadata for session index: {e}")
        return rag_response.json()

    logger.error(f"❌ RAG API indexing failed ({rag_response.status_code}): {rag_response.text}")
    raise HTTPException(status_code=500, detail=f"Indexing failed: {rag_response.text}")


@app.post("/v1/sessions/{session_id}/pdf-upload", deprecated=True)
def pdf_upload_deprecated(session_id: str):
    """DEPRECATED: use POST /v1/sessions/{session_id}/upload instead."""
    return JSONResponse(
        content={
            "warning": "This upload method is deprecated. Use the new file upload and indexing flow.",
            "message": "No action taken.",
        },
        status_code=410,
    )


@app.get("/v1/sessions/{session_id}/indexes")
def get_session_indexes(session_id: str):
    session_id = _valid_id(session_id, "session")
    idx_ids = db.get_indexes_for_session(session_id)
    indexes = []
    for idx_id in idx_ids:
        idx = db.get_index(idx_id)
        if idx:
            if not idx.get("metadata") or len(idx["metadata"]) == 0:
                logger.debug(f"🔍 Attempting to infer metadata for index {idx_id[:8]}...")
                inferred = db.inspect_and_populate_index_metadata(idx_id)
                if inferred:
                    idx = db.get_index(idx_id)
            indexes.append(idx)
    return {"indexes": indexes, "total": len(indexes)}


@app.post("/v1/sessions/{session_id}/indexes/{index_id}")
def link_index_to_session(session_id: str, index_id: str):
    session_id = _valid_id(session_id, "session")
    index_id = _valid_id(index_id, "index")
    db.link_index_to_session(session_id, index_id)
    return {"message": "Index linked to session"}


@app.get("/v1/indexes")
def get_indexes():
    data = db.list_indexes()
    return {"indexes": data, "total": len(data)}


@app.post("/v1/indexes", status_code=201)
def create_index(body: CreateIndexRequest, current_user: dict | None = Depends(get_current_user_optional)):
    if not body.name:
        raise HTTPException(status_code=400, detail="Name required")

    metadata = body.metadata
    if RAG_SYSTEM_AVAILABLE and PIPELINE_CONFIGS.get("default"):
        complete_metadata = {
            "status": "created",
            "metadata_source": "rag_system_config",
            "created_at": datetime.now().isoformat(),
            "chunk_size": 512,
            "chunk_overlap": 64,
            "retrieval_mode": "hybrid",
            "window_size": 5,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "enrich_model": "qwen3:0.6b",
            "overview_model": "qwen3:0.6b",
            "enable_enrich": True,
            "latechunk": True,
            "docling_chunk": True,
            "note": "Default configuration from RAG system",
        }
        complete_metadata.update(metadata)
        metadata = complete_metadata

    idx_id = db.create_index(body.name, body.description, metadata, user_id=current_user["user_id"] if current_user else None)
    return {"index_id": idx_id}


@app.get("/v1/indexes/{index_id}")
def get_index(index_id: str, current_user: dict | None = Depends(get_current_user_optional)):
    index_id = _valid_id(index_id, "index")
    data = db.get_index(index_id)
    if not data:
        raise HTTPException(status_code=404, detail="Index not found")
    check_ownership(data.get("user_id"), current_user)
    return data


@app.delete("/v1/indexes/{index_id}")
def delete_index(index_id: str, current_user: dict | None = Depends(get_current_user_optional)):
    index_id = _valid_id(index_id, "index")
    existing = db.get_index(index_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Index not found")
    check_ownership(existing.get("user_id"), current_user)
    deleted = db.delete_index(index_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Index not found")
    return {"message": "Index deleted successfully", "index_id": index_id}


@app.post("/v1/indexes/{index_id}/upload")
async def upload_files_to_index(index_id: str, request: Request, files: list[UploadFile]):
    index_id = _valid_id(index_id, "index")
    _rate_limit(request, upload_rate_limiter)

    content_length = int(request.headers.get("content-length", 0) or 0)
    if content_length > MAX_TOTAL_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds maximum total size of {MAX_TOTAL_UPLOAD_BYTES // (1024*1024)}MB",
        )

    uploaded_files = []
    upload_dir = "shared_uploads"
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        if not file.filename:
            continue
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        absolute_path = os.path.abspath(file_path)
        db.add_document_to_index(index_id, file.filename, absolute_path)
        uploaded_files.append({"filename": file.filename, "stored_path": absolute_path})

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    return {"message": f"Uploaded {len(uploaded_files)} files", "uploaded_files": uploaded_files}


@app.post("/v1/indexes/{index_id}/build")
def build_index(index_id: str, body: BuildIndexRequest = BuildIndexRequest()):
    index_id = _valid_id(index_id, "index")
    index = db.get_index(index_id)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    file_paths = [d["stored_path"] for d in index.get("documents", [])]
    if not file_paths:
        raise HTTPException(status_code=400, detail="No documents to index")

    table_name = index.get("vector_table_name")
    payload: dict[str, Any] = {
        "file_paths": file_paths,
        "session_id": index_id,
        "table_name": table_name,
        "chunk_size": body.chunkSize,
        "chunk_overlap": body.chunkOverlap,
        "retrieval_mode": body.retrievalMode,
        "window_size": body.windowSize,
        "enable_enrich": body.enableEnrich,
        "batch_size_embed": body.batchSizeEmbed,
        "batch_size_enrich": body.batchSizeEnrich,
    }
    if body.latechunk:
        payload["enable_latechunk"] = True
    if body.doclingChunk:
        payload["enable_docling_chunk"] = True
    if body.embeddingModel:
        payload["embedding_model"] = body.embeddingModel
    if body.enrichModel:
        payload["enrich_model"] = body.enrichModel
    if body.overviewModel:
        payload["overview_model_name"] = body.overviewModel

    import requests
    try:
        rag_resp = requests.post("http://localhost:8001/index", json=payload)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the RAG API: {e}")

    if rag_resp.status_code == 200:
        meta_updates = {
            "chunk_size": body.chunkSize,
            "chunk_overlap": body.chunkOverlap,
            "retrieval_mode": body.retrievalMode,
            "window_size": body.windowSize,
            "enable_enrich": body.enableEnrich,
            "latechunk": body.latechunk,
            "docling_chunk": body.doclingChunk,
        }
        if body.embeddingModel:
            meta_updates["embedding_model"] = body.embeddingModel
        if body.enrichModel:
            meta_updates["enrich_model"] = body.enrichModel
        if body.overviewModel:
            meta_updates["overview_model"] = body.overviewModel
        try:
            db.update_index_metadata(index_id, meta_updates)
        except Exception as e:
            logger.warning(f"⚠️ Failed to update index metadata: {e}")
        return {"response": rag_resp.json(), **meta_updates}

    try:
        err_json = rag_resp.json()
    except Exception:
        err_json = {}
    err_text = err_json.get("error") if isinstance(err_json, dict) else rag_resp.text
    if err_text and "already exists" in err_text:
        return {"message": "Index already built – skipping rebuild.", "note": err_text}

    raise HTTPException(status_code=500, detail=f"RAG indexing failed: {rag_resp.text}")
