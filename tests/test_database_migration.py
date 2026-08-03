"""
Tests for the dual SQLite/PostgreSQL database support (Improvement #20).

Most of ChatDatabase's actual CRUD behavior is already exercised
end-to-end through tests/test_app.py and tests/test_auth.py (38+ tests
against a real temporary SQLite database via the FastAPI app) - this
file specifically covers the NEW pieces this migration added: URL
resolution priority, and that the schema compiles correctly for both
dialects.

Honest limitation: PostgreSQL itself can't be live-tested in this
environment (no server available) - the PostgreSQL-specific assertions
here verify DDL compilation (that SQLAlchemy translates our schema
correctly for that dialect, e.g. SERIAL instead of AUTOINCREMENT), not
an actual connection/query against a live Postgres instance.
"""

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from backend.database import ChatDatabase
from backend.db_schema import metadata


class TestDatabaseUrlResolution:
    def test_explicit_database_url_takes_highest_priority(self, monkeypatch):
        import rag_system.config as config_module

        monkeypatch.setattr(config_module.settings, "database_url", "postgresql://ignored/ignored")
        url = ChatDatabase._resolve_database_url(
            db_path="/some/path.db", database_url="postgresql://explicit/db"
        )
        assert url == "postgresql://explicit/db"

    def test_db_path_translates_to_sqlite_url_when_no_explicit_url(self):
        url = ChatDatabase._resolve_database_url(db_path="/some/path.db", database_url=None)
        assert url == "sqlite:////some/path.db"

    def test_falls_back_to_settings_database_url_when_nothing_explicit_given(self, monkeypatch):
        import rag_system.config as config_module

        monkeypatch.setattr(config_module.settings, "database_url", "postgresql://from-settings/db")
        url = ChatDatabase._resolve_database_url(db_path=None, database_url=None)
        assert url == "postgresql://from-settings/db"

    def test_falls_back_to_default_sqlite_path_when_nothing_configured(self, monkeypatch):
        import rag_system.config as config_module

        monkeypatch.setattr(config_module.settings, "database_url", "")
        url = ChatDatabase._resolve_database_url(db_path=None, database_url=None)
        assert url.startswith("sqlite:///")
        assert "chat_data.db" in url


class TestSchemaCompilesForBothDialects:
    """Verifies the schema (backend/db_schema.py) produces valid DDL for
    both dialects this project supports - real compilation, not a guess."""

    def test_schema_creates_successfully_against_real_sqlite(self):
        from sqlalchemy import create_engine

        engine = create_engine("sqlite:///:memory:")
        metadata.create_all(engine)  # raises if anything is wrong
        from sqlalchemy import inspect

        table_names = set(inspect(engine).get_table_names())
        expected = {
            "users",
            "sessions",
            "messages",
            "session_documents",
            "indexes",
            "index_documents",
            "session_indexes",
        }
        assert expected.issubset(table_names)

    @pytest.mark.parametrize(
        "table_name",
        [
            "users",
            "sessions",
            "messages",
            "session_documents",
            "indexes",
            "index_documents",
            "session_indexes",
        ],
    )
    def test_every_table_compiles_to_valid_postgresql_ddl(self, table_name):
        table = metadata.tables[table_name]
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        assert "CREATE TABLE" in ddl
        assert table_name in ddl

    def test_autoincrement_columns_translate_to_postgresql_serial(self):
        """SQLite's AUTOINCREMENT and PostgreSQL's SERIAL are different
        concepts under the hood - this confirms SQLAlchemy handles the
        dialect translation correctly rather than assuming it does."""
        table = metadata.tables["session_documents"]
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        assert "SERIAL" in ddl

    def test_cascade_delete_foreign_keys_preserved_in_postgresql_ddl(self):
        table = metadata.tables["messages"]
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        assert "ON DELETE CASCADE" in ddl
