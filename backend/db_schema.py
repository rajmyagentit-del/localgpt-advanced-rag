"""
SQLAlchemy Core schema for backend/database.py (Improvement #20 - dual
SQLite/PostgreSQL support).

WHY SQLALCHEMY CORE (not the full ORM): the existing database.py is
written as parameterized raw SQL against sqlite3 directly. Rewriting to
the full ORM (declarative models, sessions, relationships) would be a
much larger, riskier rewrite of ~20 already-tested methods. SQLAlchemy
Core gives the two things that actually matter for this migration -
(1) a single schema definition Alembic can generate migrations from,
and (2) a database-agnostic Engine/text() execution layer - without
forcing a full ORM rewrite of business logic that already works and is
already covered by 38+ existing tests.

WHY THESE TABLE DEFINITIONS MUST MATCH THE ORIGINAL SCHEMA EXACTLY:
this is the source of truth Alembic's autogenerate compares the live
database against - any mismatch here would generate an incorrect
migration.
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", String, primary_key=True),
    Column("email", String, unique=True, nullable=False),
    Column("password_hash", String, nullable=False),
    Column("password_salt", String, nullable=False),
    Column("created_at", String, nullable=False),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("model_used", String, nullable=False),
    Column("message_count", Integer, server_default="0"),
    Column("user_id", String, nullable=True),
    Index("idx_sessions_updated_at", "updated_at"),
)

messages = Table(
    "messages",
    metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("content", Text, nullable=False),
    Column("sender", String, nullable=False),
    Column("timestamp", String, nullable=False),
    Column("metadata", Text, server_default="{}"),
    CheckConstraint("sender IN ('user', 'assistant')", name="ck_messages_sender"),
    Index("idx_messages_session_id", "session_id"),
    Index("idx_messages_timestamp", "timestamp"),
)

session_documents = Table(
    "session_documents",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
    Column("file_path", Text, nullable=False),
    Column("indexed", Integer, server_default="0"),
    Index("idx_session_documents_session_id", "session_id"),
)

indexes = Table(
    "indexes",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, unique=True),
    Column("description", Text),
    Column("created_at", String),
    Column("updated_at", String),
    Column("vector_table_name", String),
    Column("metadata", Text),
    Column("user_id", String, nullable=True),
)

index_documents = Table(
    "index_documents",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("index_id", String, ForeignKey("indexes.id")),
    Column("original_filename", Text),
    Column("stored_path", Text),
)

session_indexes = Table(
    "session_indexes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, ForeignKey("sessions.id")),
    Column("index_id", String, ForeignKey("indexes.id")),
    Column("linked_at", String),
)
