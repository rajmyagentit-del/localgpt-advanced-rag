import json
import logging
import uuid
from datetime import datetime, timedelta

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from backend.db_schema import metadata as db_metadata

logger = logging.getLogger(__name__)


class ChatDatabase:
    def __init__(self, db_path: str | None = None, database_url: str | None = None):
        """
        Improvement #20: supports both SQLite (default, zero-config) and
        PostgreSQL (via database_url or the DATABASE_URL env var / Settings).

        db_path: legacy parameter, kept for backward compatibility with
                 existing callers/tests - a plain filesystem path,
                 translated into a sqlite:/// URL.
        database_url: an explicit SQLAlchemy connection URL (e.g.
                 "postgresql://user:pass@host:5432/dbname"). Takes
                 precedence over db_path and the Settings-derived default.
        """
        self.engine: Engine = create_engine(self._resolve_database_url(db_path, database_url))
        self.init_database()

    @staticmethod
    def _resolve_database_url(db_path: str | None, database_url: str | None) -> str:
        if database_url:
            return database_url

        if db_path is not None:
            return f"sqlite:///{db_path}"

        # No explicit path/URL given - check Settings for a configured
        # DATABASE_URL (e.g. PostgreSQL in production), otherwise fall
        # back to the original auto-detected SQLite path so existing
        # zero-config local/Docker behavior is unchanged.
        from rag_system.config import settings

        if settings.database_url:
            return settings.database_url

        import os

        if os.path.exists("/app"):  # Docker environment
            resolved_path = "/app/backend/chat_data.db"
        else:  # Local development environment
            resolved_path = "backend/chat_data.db"
        return f"sqlite:///{resolved_path}"

    def init_database(self):
        """
        Create any missing tables (dialect-agnostic, via the shared
        SQLAlchemy schema in backend/db_schema.py - see that file for
        why Core rather than the full ORM). For real production schema
        evolution beyond initial table creation, use Alembic
        (`alembic upgrade head`) - see alembic/ and the README's
        Database Migrations section. This method's ADD-COLUMN-if-missing
        fallback below exists only to keep pre-Alembic SQLite databases
        (created before this improvement existed) working without a
        manual migration step.
        """
        db_metadata.create_all(self.engine)

        # Backward compatibility: a database created by the OLD raw-SQL
        # init_database() (before this migration) has `sessions` and
        # `indexes` tables but without the `user_id` column added later
        # for Improvement #10. create_all() only creates MISSING tables,
        # it never alters existing ones - so for that specific pre-existing
        # case we still need an explicit, idempotent ADD COLUMN check.
        # New databases created via create_all() above already have
        # user_id as part of the schema and this is a no-op for them.
        self._add_column_if_missing("sessions", "user_id", "VARCHAR")
        self._add_column_if_missing("indexes", "user_id", "VARCHAR")

        logger.info("✅ Database initialized successfully")

    def _add_column_if_missing(self, table_name: str, column: str, col_type: str):
        """Dialect-agnostic ADD COLUMN IF NOT EXISTS, via SQLAlchemy's
        inspector (works the same way against SQLite and PostgreSQL)."""
        inspector = inspect(self.engine)
        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        if column not in existing_columns:
            with self.engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {col_type}"))

    # --- Users (Improvement #10) ---

    def create_user(self, email: str, password_hash: str, password_salt: str) -> str:
        """Create a new user account. Raises sqlalchemy.exc.IntegrityError
        if the email is already registered (UNIQUE constraint)."""
        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO users (id, email, password_hash, password_salt, created_at)
                VALUES (:id, :email, :password_hash, :password_salt, :created_at)
            """),
                {
                    "id": user_id,
                    "email": email.lower().strip(),
                    "password_hash": password_hash,
                    "password_salt": password_salt,
                    "created_at": now,
                },
            )
        logger.info(f"👤 Created new user account: {email}")
        return user_id

    def get_user_by_email(self, email: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM users WHERE email = :email"), {"email": email.lower().strip()}
            ).fetchone()
        return dict(row._mapping) if row else None

    def get_user_by_id(self, user_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM users WHERE id = :id"), {"id": user_id}
            ).fetchone()
        return dict(row._mapping) if row else None

    def create_session(self, title: str, model: str, user_id: str | None = None) -> str:
        """Create a new chat session, optionally owned by a user"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO sessions (id, title, created_at, updated_at, model_used, user_id)
                VALUES (:id, :title, :created_at, :updated_at, :model_used, :user_id)
            """),
                {
                    "id": session_id,
                    "title": title,
                    "created_at": now,
                    "updated_at": now,
                    "model_used": model,
                    "user_id": user_id,
                },
            )

        logger.info(f"📝 Created new session: {session_id[:8]}... - {title}")
        return session_id

    def get_sessions(self, limit: int = 50) -> list[dict]:
        """Get all chat sessions, ordered by most recent"""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                SELECT id, title, created_at, updated_at, model_used, message_count
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT :limit
            """),
                {"limit": limit},
            ).fetchall()
        return [dict(row._mapping) for row in rows]

    def get_session(self, session_id: str) -> dict | None:
        """Get a specific session"""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("""
                SELECT id, title, created_at, updated_at, model_used, message_count, user_id
                FROM sessions
                WHERE id = :id
            """),
                {"id": session_id},
            ).fetchone()
        return dict(row._mapping) if row else None

    def add_message(
        self, session_id: str, content: str, sender: str, metadata: dict | None = None
    ) -> str:
        """Add a message to a session"""
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO messages (id, session_id, content, sender, timestamp, metadata)
                VALUES (:id, :session_id, :content, :sender, :timestamp, :metadata)
            """),
                {
                    "id": message_id,
                    "session_id": session_id,
                    "content": content,
                    "sender": sender,
                    "timestamp": now,
                    "metadata": metadata_json,
                },
            )

            conn.execute(
                text("""
                UPDATE sessions
                SET updated_at = :updated_at,
                    message_count = message_count + 1
                WHERE id = :id
            """),
                {"updated_at": now, "id": session_id},
            )

        return message_id

    def get_messages(self, session_id: str, limit: int = 100) -> list[dict]:
        """Get all messages for a session"""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                SELECT id, content, sender, timestamp, metadata
                FROM messages
                WHERE session_id = :session_id
                ORDER BY timestamp ASC
                LIMIT :limit
            """),
                {"session_id": session_id, "limit": limit},
            ).fetchall()

        messages = []
        for row in rows:
            message = dict(row._mapping)
            message["metadata"] = json.loads(message["metadata"])
            messages.append(message)
        return messages

    def get_conversation_history(self, session_id: str) -> list[dict]:
        """Get conversation history in the format expected by Ollama"""
        messages = self.get_messages(session_id)

        history = []
        for msg in messages:
            history.append({"role": msg["sender"], "content": msg["content"]})

        return history

    def update_session_title(self, session_id: str, title: str):
        """Update session title"""
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE sessions
                SET title = :title, updated_at = :updated_at
                WHERE id = :id
            """),
                {"title": title, "updated_at": datetime.now().isoformat(), "id": session_id},
            )

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        with self.engine.begin() as conn:
            result = conn.execute(text("DELETE FROM sessions WHERE id = :id"), {"id": session_id})
            deleted = result.rowcount > 0

        if deleted:
            logger.info(f"🗑️ Deleted session: {session_id[:8]}...")

        return deleted

    def cleanup_empty_sessions(self) -> int:
        """Remove sessions with no messages"""
        with self.engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT s.id FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE m.id IS NULL
            """)).fetchall()
            empty_sessions = [row[0] for row in rows]

            deleted_count = 0
            for session_id in empty_sessions:
                result = conn.execute(
                    text("DELETE FROM sessions WHERE id = :id"), {"id": session_id}
                )
                if result.rowcount > 0:
                    deleted_count += 1
                    logger.info(f"🗑️ Cleaned up empty session: {session_id[:8]}...")

        if deleted_count > 0:
            logger.info(f"✨ Cleaned up {deleted_count} empty sessions")

        return deleted_count

    def get_stats(self) -> dict:
        """Get database statistics"""
        with self.engine.connect() as conn:
            session_count = conn.execute(text("SELECT COUNT(*) FROM sessions")).scalar()
            message_count = conn.execute(text("SELECT COUNT(*) FROM messages")).scalar()
            most_used_model = conn.execute(text("""
                SELECT model_used, COUNT(*) as count
                FROM sessions
                GROUP BY model_used
                ORDER BY count DESC
                LIMIT 1
            """)).fetchone()

        return {
            "total_sessions": session_count,
            "total_messages": message_count,
            "most_used_model": most_used_model[0] if most_used_model else None,
        }

    def add_document_to_session(self, session_id: str, file_path: str) -> int | None:
        """Adds a document file path to a session."""
        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    "INSERT INTO session_documents (session_id, file_path) VALUES (:session_id, :file_path)"
                ),
                {"session_id": session_id, "file_path": file_path},
            )
            # lastrowid works reliably for SQLite but is not guaranteed
            # across DBAPI backends (SQLAlchemy's own docs note this).
            # No current caller actually uses the returned doc_id (both
            # call sites in app.py/server.py discard it) - a graceful
            # None on backends where it's unavailable is an honest
            # degradation, not a silent correctness issue.
            try:
                doc_id = result.lastrowid
            except Exception:
                doc_id = None
        logger.info(f"📄 Added document '{file_path}' to session {session_id[:8]}...")
        return doc_id

    def get_documents_for_session(self, session_id: str) -> list[str]:
        """Retrieves all document file paths for a given session."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT file_path FROM session_documents WHERE session_id = :session_id"),
                {"session_id": session_id},
            ).fetchall()
        return [row[0] for row in rows]

    # -------- Index helpers ---------

    def create_index(
        self,
        name: str,
        description: str | None = None,
        metadata: dict | None = None,
        user_id: str | None = None,
    ) -> str:
        idx_id = str(uuid.uuid4())
        created = datetime.now().isoformat()
        vector_table = f"text_pages_{idx_id}"
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                INSERT INTO indexes (id, name, description, created_at, updated_at, vector_table_name, metadata, user_id)
                VALUES (:id, :name, :description, :created_at, :updated_at, :vector_table_name, :metadata, :user_id)
            """),
                {
                    "id": idx_id,
                    "name": name,
                    "description": description,
                    "created_at": created,
                    "updated_at": created,
                    "vector_table_name": vector_table,
                    "metadata": json.dumps(metadata or {}),
                    "user_id": user_id,
                },
            )
        logger.info(f"📂 Created new index '{name}' ({idx_id[:8]})")
        return idx_id

    def get_index(self, index_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM indexes WHERE id = :id"), {"id": index_id}
            ).fetchone()
            if not row:
                return None
            idx = dict(row._mapping)
            idx["metadata"] = json.loads(idx["metadata"] or "{}")
            doc_rows = conn.execute(
                text(
                    "SELECT original_filename, stored_path FROM index_documents WHERE index_id = :id"
                ),
                {"id": index_id},
            ).fetchall()
        idx["documents"] = [{"filename": r[0], "stored_path": r[1]} for r in doc_rows]
        return idx

    def list_indexes(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM indexes")).fetchall()
            res = []
            for r in rows:
                item = dict(r._mapping)
                item["metadata"] = json.loads(item["metadata"] or "{}")
                doc_rows = conn.execute(
                    text(
                        "SELECT original_filename, stored_path FROM index_documents WHERE index_id = :id"
                    ),
                    {"id": item["id"]},
                ).fetchall()
                item["documents"] = [{"filename": d[0], "stored_path": d[1]} for d in doc_rows]
                res.append(item)
        return res

    def add_document_to_index(self, index_id: str, filename: str, stored_path: str):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO index_documents (index_id, original_filename, stored_path) VALUES (:index_id, :filename, :stored_path)"
                ),
                {"index_id": index_id, "filename": filename, "stored_path": stored_path},
            )

    def link_index_to_session(self, session_id: str, index_id: str):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO session_indexes (session_id, index_id, linked_at) VALUES (:session_id, :index_id, :linked_at)"
                ),
                {
                    "session_id": session_id,
                    "index_id": index_id,
                    "linked_at": datetime.now().isoformat(),
                },
            )

    def get_indexes_for_session(self, session_id: str) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT index_id FROM session_indexes WHERE session_id = :session_id ORDER BY linked_at"
                ),
                {"session_id": session_id},
            ).fetchall()
        return [r[0] for r in rows]

    def delete_index(self, index_id: str) -> bool:
        """Delete an index and its related records (documents, session links). Returns True if deleted."""
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT vector_table_name FROM indexes WHERE id = :id"), {"id": index_id}
            ).fetchone()
            vector_table_name = row[0] if row else None

            # Remove child rows first due to foreign-key constraints
            conn.execute(text("DELETE FROM index_documents WHERE index_id = :id"), {"id": index_id})
            conn.execute(text("DELETE FROM session_indexes WHERE index_id = :id"), {"id": index_id})
            result = conn.execute(text("DELETE FROM indexes WHERE id = :id"), {"id": index_id})
            deleted = result.rowcount > 0

        if deleted:
            logger.info(f"🗑️ Deleted index {index_id[:8]}... and related records")
            # Optional: attempt to drop LanceDB table if available
            if vector_table_name:
                try:
                    from rag_system.config import settings
                    from rag_system.indexing.embedders import LanceDBManager

                    db_path = settings.lancedb_path
                    ldb = LanceDBManager(db_path)
                    db = ldb.db
                    if hasattr(db, "list_tables") and vector_table_name in db.list_tables().tables:
                        db.drop_table(vector_table_name)
                        logger.info(f"🚮 Dropped LanceDB table '{vector_table_name}'")
                except Exception as e:
                    logger.warning(f"⚠️ Could not drop LanceDB table '{vector_table_name}': {e}")
        return deleted

    def update_index_metadata(self, index_id: str, updates: dict):
        """Merge new key/values into an index's metadata JSON column."""
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT metadata FROM indexes WHERE id = :id"), {"id": index_id}
            ).fetchone()
            if row is None:
                raise ValueError("Index not found")
            existing = json.loads(row[0] or "{}")
            existing.update(updates)
            conn.execute(
                text(
                    "UPDATE indexes SET metadata = :metadata, updated_at = :updated_at WHERE id = :id"
                ),
                {
                    "metadata": json.dumps(existing),
                    "updated_at": datetime.now().isoformat(),
                    "id": index_id,
                },
            )

    def inspect_and_populate_index_metadata(self, index_id: str) -> dict:
        """
        Inspect LanceDB table to extract metadata for older indexes.
        Returns the inferred metadata or empty dict if inspection fails.
        """
        try:
            # Get index info
            index_info = self.get_index(index_id)
            if not index_info:
                return {}

            # Check if metadata is already populated
            if index_info.get("metadata") and len(index_info["metadata"]) > 0:
                return index_info["metadata"]

            # Try to inspect the LanceDB table
            vector_table_name = index_info.get("vector_table_name")
            if not vector_table_name:
                return {}

            try:
                # Try to import the RAG system modules
                try:
                    from rag_system.config import settings
                    from rag_system.indexing.embedders import LanceDBManager

                    # Use the same path as the system
                    db_path = settings.lancedb_path
                    ldb = LanceDBManager(db_path)

                    # Check if table exists
                    if (
                        not hasattr(ldb.db, "list_tables")
                        or vector_table_name not in ldb.db.list_tables().tables
                    ):
                        # Table doesn't exist - this means the index was never properly built
                        inferred_metadata = {
                            "status": "incomplete",
                            "issue": "Vector table not found - index may not have been built properly",
                            "vector_table_expected": vector_table_name,
                            "available_tables": (
                                list(ldb.db.list_tables().tables)
                                if hasattr(ldb.db, "list_tables")
                                else []
                            ),
                            "metadata_inferred_at": datetime.now().isoformat(),
                            "metadata_source": "lancedb_inspection",
                        }
                        self.update_index_metadata(index_id, inferred_metadata)
                        logger.warning(
                            f"⚠️ Index {index_id[:8]}... appears incomplete - vector table missing"
                        )
                        return inferred_metadata

                    # Get table and inspect schema/data
                    table = ldb.db.open_table(vector_table_name)

                    # Get a sample record to inspect - use correct LanceDB API
                    try:
                        # Try to get sample data using proper LanceDB methods
                        sample_df = table.to_pandas()
                        if len(sample_df) == 0:
                            inferred_metadata = {
                                "status": "empty",
                                "issue": "Vector table exists but contains no data",
                                "metadata_inferred_at": datetime.now().isoformat(),
                                "metadata_source": "lancedb_inspection",
                            }
                            self.update_index_metadata(index_id, inferred_metadata)
                            return inferred_metadata

                        # Take only first row for inspection
                        sample_df = sample_df.head(1)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not read data from table {vector_table_name}: {e}"
                        )
                        return {}

                    # Infer metadata from table structure
                    inferred_metadata = {
                        "status": "functional",
                        "total_chunks": len(table.to_pandas()),  # Get total count
                    }

                    # Check vector dimensions
                    if "vector" in sample_df.columns:
                        vector_data = sample_df["vector"].iloc[0]
                        if isinstance(vector_data, list):
                            inferred_metadata["vector_dimensions"] = len(vector_data)

                            # Try to infer embedding model from vector dimensions
                            dim_to_model = {
                                384: "BAAI/bge-small-en-v1.5 (or similar)",
                                512: "sentence-transformers/all-MiniLM-L6-v2 (or similar)",
                                768: "BAAI/bge-base-en-v1.5 (or similar)",
                                1024: "Qwen/Qwen3-Embedding-0.6B (or similar)",
                                1536: "text-embedding-ada-002 (or similar)",
                            }
                            if len(vector_data) in dim_to_model:
                                inferred_metadata["embedding_model_inferred"] = dim_to_model[
                                    len(vector_data)
                                ]

                    # Try to parse metadata from sample record
                    if "metadata" in sample_df.columns:
                        try:
                            sample_metadata = json.loads(sample_df["metadata"].iloc[0])
                            # Look for common metadata fields that might give us clues
                            if "document_id" in sample_metadata:
                                inferred_metadata["has_document_structure"] = True
                            if "chunk_index" in sample_metadata:
                                inferred_metadata["has_chunk_indexing"] = True
                            if "original_text" in sample_metadata:
                                inferred_metadata["has_contextual_enrichment"] = True
                                inferred_metadata["retrieval_mode_inferred"] = (
                                    "hybrid (contextual enrichment detected)"
                                )

                            # Check for chunk size patterns
                            if "text" in sample_df.columns:
                                text_length = len(sample_df["text"].iloc[0])
                                if text_length > 0:
                                    inferred_metadata["sample_chunk_length"] = text_length
                                    # Rough chunk size estimation
                                    estimated_tokens = (
                                        text_length // 4
                                    )  # rough estimate: 4 chars per token
                                    if estimated_tokens < 300:
                                        inferred_metadata["chunk_size_inferred"] = (
                                            "256 tokens (estimated)"
                                        )
                                    elif estimated_tokens < 600:
                                        inferred_metadata["chunk_size_inferred"] = (
                                            "512 tokens (estimated)"
                                        )
                                    else:
                                        inferred_metadata["chunk_size_inferred"] = (
                                            "1024+ tokens (estimated)"
                                        )

                        except (json.JSONDecodeError, KeyError):
                            pass

                    # Check if FTS index exists
                    try:
                        indices = table.list_indices()
                        fts_exists = any("fts" in idx.name.lower() for idx in indices)
                        if fts_exists:
                            inferred_metadata["has_fts_index"] = True
                            inferred_metadata["retrieval_mode_inferred"] = "hybrid (FTS + vector)"
                        else:
                            inferred_metadata["retrieval_mode_inferred"] = "vector-only"
                    except Exception:
                        pass

                    # Add inspection timestamp
                    inferred_metadata["metadata_inferred_at"] = datetime.now().isoformat()
                    inferred_metadata["metadata_source"] = "lancedb_inspection"

                    # Update the database with inferred metadata
                    if inferred_metadata:
                        self.update_index_metadata(index_id, inferred_metadata)
                        logger.debug(
                            f"🔍 Inferred metadata for index {index_id[:8]}...: {len(inferred_metadata)} fields"
                        )

                    return inferred_metadata

                except ImportError as import_error:
                    # RAG system modules not available - provide basic fallback metadata
                    logger.error(
                        f"⚠️ RAG system modules not available for inspection: {import_error}"
                    )

                    # Check if this is actually a legacy index by looking at creation date
                    created_at = index_info.get("created_at", "")
                    is_recent = False
                    if created_at:
                        try:
                            created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            # Consider indexes created in the last 30 days as "recent"
                            is_recent = created_date > datetime.now().replace(
                                tzinfo=created_date.tzinfo
                            ) - timedelta(days=30)
                        except Exception:
                            pass

                    # Provide basic fallback metadata with better status detection
                    if is_recent:
                        status = "functional"
                        issue = "Detailed configuration inspection requires RAG system modules, but index appears functional"
                    else:
                        status = "legacy"
                        issue = "This index was created before metadata tracking was implemented. Configuration details are not available."

                    fallback_metadata = {
                        "status": status,
                        "issue": issue,
                        "metadata_inferred_at": datetime.now().isoformat(),
                        "metadata_source": "fallback_inspection",
                        "documents_count": len(index_info.get("documents", [])),
                        "created_at": index_info.get("created_at", "unknown"),
                        "inspection_limitation": "Backend server cannot access full RAG system modules for detailed inspection",
                    }

                    # Try to infer some basic info from the vector table name
                    if vector_table_name:
                        fallback_metadata["vector_table_name"] = vector_table_name
                        fallback_metadata["note"] = (
                            "Vector table exists but detailed inspection requires RAG system modules"
                        )

                    self.update_index_metadata(index_id, fallback_metadata)
                    status_msg = "recent but limited inspection" if is_recent else "legacy"
                    logger.info(
                        f"📝 Added fallback metadata for {status_msg} index {index_id[:8]}..."
                    )
                    return fallback_metadata

            except Exception as e:
                logger.warning(
                    f"⚠️ Could not inspect LanceDB table for index {index_id[:8]}...: {e}"
                )
                return {}

        except Exception as e:
            logger.error(f"⚠️ Failed to inspect index metadata for {index_id[:8]}...: {e}")
            return {}


def generate_session_title(first_message: str, max_length: int = 50) -> str:
    """Generate a session title from the first message"""
    # Clean up the message
    title = first_message.strip()

    # Remove common prefixes
    prefixes = ["hey", "hi", "hello", "can you", "please", "i want", "i need"]
    title_lower = title.lower()
    for prefix in prefixes:
        if title_lower.startswith(prefix):
            title = title[len(prefix) :].strip()
            break

    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]

    # Truncate if too long
    if len(title) > max_length:
        title = title[:max_length].strip() + "..."

    # Fallback
    if not title or len(title) < 3:
        title = "New Chat"

    return title


# Global database instance
if __name__ == "__main__":
    # Manual smoke test - only runs a real ChatDatabase instance when
    # this file is executed directly (`python -m backend.database`), not
    # on every import. A previous version of this file instantiated
    # `db = ChatDatabase()` unconditionally at module level, which meant
    # simply IMPORTING this module had side effects (created/opened a
    # real database connection and schema) - this silently broke
    # Alembic's autogenerate schema comparison (Improvement #20), which
    # imports this module and got a database that already had the
    # target schema applied, making every migration look empty.
    db = ChatDatabase()
    # Test the database
    logger.info("🧪 Testing database...")

    # Create a test session
    session_id = db.create_session("Test Chat", "llama3.2:latest")

    # Add some messages
    db.add_message(session_id, "Hello!", "user")
    db.add_message(session_id, "Hi there! How can I help you?", "assistant")

    # Get messages
    messages = db.get_messages(session_id)
    logger.info(f"📨 Messages: {len(messages)}")

    # Get sessions
    sessions = db.get_sessions()
    logger.info(f"📋 Sessions: {len(sessions)}")

    # Get stats
    stats = db.get_stats()
    logger.info(f"📊 Stats: {stats}")

    logger.info("✅ Database test completed!")
