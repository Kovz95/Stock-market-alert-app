"""Unit tests for document_store module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call
from copy import deepcopy


class TestBuildCacheKey:
    """Tests for _build_cache_key function."""

    @patch("src.data_access.document_store.build_key")
    def test_builds_key_with_namespace(self, mock_build_key):
        """Test that cache key includes document namespace."""
        from src.data_access.document_store import _build_cache_key

        mock_build_key.return_value = "app:document:my_doc"

        result = _build_cache_key("my_doc")

        mock_build_key.assert_called_once_with("document:my_doc")
        assert result == "app:document:my_doc"


class TestFetchDocumentFromDb:
    """Tests for _fetch_document_from_db function."""

    @patch("src.data_access.document_store.db_config")
    def test_returns_payload_when_document_exists(self, mock_db_config):
        """Test fetching an existing document returns its payload."""
        from src.data_access.document_store import _fetch_document_from_db

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ({"key": "value"},)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        result = _fetch_document_from_db("test_doc")

        assert result == {"key": "value"}
        mock_cursor.execute.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_db_config.close_connection.assert_called_once_with(mock_conn)

    @patch("src.data_access.document_store.db_config")
    def test_returns_none_when_document_missing(self, mock_db_config):
        """Test fetching a non-existent document returns None."""
        from src.data_access.document_store import _fetch_document_from_db

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        result = _fetch_document_from_db("missing_doc")

        assert result is None

    @patch("src.data_access.document_store.db_config")
    def test_returns_deep_copy_of_payload(self, mock_db_config):
        """Test that returned payload is a deep copy."""
        from src.data_access.document_store import _fetch_document_from_db

        original_payload = {"nested": {"data": [1, 2, 3]}}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (original_payload,)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        result = _fetch_document_from_db("test_doc")

        result["nested"]["data"].append(4)
        assert original_payload["nested"]["data"] == [1, 2, 3]


class TestWriteDocumentToDb:
    """Tests for _write_document_to_db function."""

    @patch("src.data_access.document_store.Json")
    @patch("src.data_access.document_store.db_config")
    def test_inserts_document_with_upsert(self, mock_db_config, mock_json):
        """Test writing a document performs an upsert."""
        from src.data_access.document_store import _write_document_to_db

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn
        mock_json.return_value = "json_payload"

        payload = {"data": "test"}
        _write_document_to_db("test_doc", payload, source_path="/path/to/file.json")

        mock_json.assert_called_once_with(payload)
        mock_cursor.execute.assert_called_once()
        assert "INSERT INTO app_documents" in mock_cursor.execute.call_args[0][0]
        assert "ON CONFLICT" in mock_cursor.execute.call_args[0][0]
        mock_conn.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_db_config.close_connection.assert_called_once_with(mock_conn)

    @patch("src.data_access.document_store.Json")
    @patch("src.data_access.document_store.db_config")
    def test_writes_without_source_path(self, mock_db_config, mock_json):
        """Test writing a document without a source path."""
        from src.data_access.document_store import _write_document_to_db

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        _write_document_to_db("test_doc", {"data": "test"})

        call_args = mock_cursor.execute.call_args[0][1]
        assert call_args[2] is None  # source_path should be None


class TestLoadDocument:
    """Tests for load_document function."""

    @patch("src.data_access.document_store.set_json")
    @patch("src.data_access.document_store.get_json")
    @patch("src.data_access.document_store._build_cache_key")
    def test_returns_cached_payload_when_available(
        self, mock_build_key, mock_get_json, mock_set_json
    ):
        """Test that cached documents are returned from Redis."""
        from src.data_access.document_store import load_document

        mock_build_key.return_value = "cache:test_doc"
        mock_get_json.return_value = {"payload": {"cached": "data"}}

        result = load_document("test_doc")

        assert result == {"cached": "data"}
        mock_get_json.assert_called_once_with("cache:test_doc")

    @patch("src.data_access.document_store.set_json")
    @patch("src.data_access.document_store._fetch_document_from_db")
    @patch("src.data_access.document_store.get_json")
    @patch("src.data_access.document_store._build_cache_key")
    def test_returns_default_when_document_missing(
        self, mock_build_key, mock_get_json, mock_fetch_db, mock_set_json
    ):
        """Test that default is returned when document doesn't exist."""
        from src.data_access.document_store import load_document

        mock_build_key.return_value = "cache:missing"
        mock_get_json.return_value = None
        mock_fetch_db.return_value = None

        result = load_document("missing", default={"default": "value"})

        assert result == {"default": "value"}

    @patch("src.data_access.document_store.set_json")
    @patch("src.data_access.document_store._fetch_document_from_db")
    @patch("src.data_access.document_store.get_json")
    @patch("src.data_access.document_store._build_cache_key")
    def test_fetches_from_db_and_caches_when_not_cached(
        self, mock_build_key, mock_get_json, mock_fetch_db, mock_set_json
    ):
        """Test that DB results are cached in Redis."""
        from src.data_access.document_store import load_document

        mock_build_key.return_value = "cache:test_doc"
        mock_get_json.return_value = None
        mock_fetch_db.return_value = {"db": "data"}

        result = load_document("test_doc", cache_ttl=600)

        assert result == {"db": "data"}
        mock_set_json.assert_called_once()
        call_kwargs = mock_set_json.call_args
        assert call_kwargs[1]["ttl_seconds"] == 600

    @patch("src.data_access.document_store.set_json", None)
    @patch("src.data_access.document_store._fetch_document_from_db")
    @patch("src.data_access.document_store.get_json")
    @patch("src.data_access.document_store._build_cache_key")
    def test_works_without_set_json_available(
        self, mock_build_key, mock_get_json, mock_fetch_db
    ):
        """Test that load_document works when set_json is not available."""
        from src.data_access.document_store import load_document

        mock_build_key.return_value = "cache:test_doc"
        mock_get_json.return_value = None
        mock_fetch_db.return_value = {"db": "data"}

        result = load_document("test_doc")

        assert result == {"db": "data"}

    @patch("src.data_access.document_store.set_json")
    @patch("src.data_access.document_store.get_json")
    @patch("src.data_access.document_store._build_cache_key")
    def test_returns_deep_copy_of_cached_data(
        self, mock_build_key, mock_get_json, mock_set_json
    ):
        """Test that cached data is returned as a deep copy."""
        from src.data_access.document_store import load_document

        original = {"payload": {"nested": [1, 2, 3]}}
        mock_build_key.return_value = "cache:test_doc"
        mock_get_json.return_value = original

        result = load_document("test_doc")

        result["nested"].append(4)
        assert original["payload"]["nested"] == [1, 2, 3]


class TestSaveDocument:
    """Tests for save_document function."""

    @patch("src.data_access.document_store.set_json")
    @patch("src.data_access.document_store.delete_key")
    @patch("src.data_access.document_store._write_document_to_db")
    @patch("src.data_access.document_store._build_cache_key")
    def test_writes_to_db_and_updates_cache(
        self, mock_build_key, mock_write_db, mock_delete_key, mock_set_json
    ):
        """Test that save_document writes to DB and updates Redis cache."""
        from src.data_access.document_store import save_document

        mock_build_key.return_value = "cache:test_doc"
        payload = {"new": "data"}

        save_document("test_doc", payload, cache_ttl=600)

        mock_write_db.assert_called_once_with(
            "test_doc", payload, source_path=None
        )
        mock_delete_key.assert_called_once_with("cache:test_doc")
        mock_set_json.assert_called_once()

    @patch("src.data_access.document_store.set_json")
    @patch("src.data_access.document_store.delete_key")
    @patch("src.data_access.document_store._write_document_to_db")
    @patch("src.data_access.document_store._build_cache_key")
    def test_includes_fallback_path_as_source_path(
        self, mock_build_key, mock_write_db, mock_delete_key, mock_set_json
    ):
        """Test that fallback_path is passed as source_path to DB."""
        from src.data_access.document_store import save_document

        mock_build_key.return_value = "cache:test_doc"

        save_document("test_doc", {"data": 1}, fallback_path="/path/to/file.json")

        call_kwargs = mock_write_db.call_args
        assert call_kwargs[1]["source_path"] == "/path/to/file.json"


class TestDeleteDocument:
    """Tests for delete_document function."""

    @patch("src.data_access.document_store.delete_key")
    @patch("src.data_access.document_store._build_cache_key")
    @patch("src.data_access.document_store.db_config")
    def test_deletes_from_db_and_cache(
        self, mock_db_config, mock_build_key, mock_delete_key
    ):
        """Test that delete_document removes from both DB and Redis."""
        from src.data_access.document_store import delete_document

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn
        mock_build_key.return_value = "cache:test_doc"

        delete_document("test_doc")

        mock_cursor.execute.assert_called_once()
        assert "DELETE FROM app_documents" in mock_cursor.execute.call_args[0][0]
        mock_conn.commit.assert_called_once()
        mock_delete_key.assert_called_once_with("cache:test_doc")


class TestDocumentExists:
    """Tests for document_exists function."""

    @patch("src.data_access.document_store.db_config")
    def test_returns_true_when_document_exists(self, mock_db_config):
        """Test that document_exists returns True for existing documents."""
        from src.data_access.document_store import document_exists

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        result = document_exists("test_doc")

        assert result is True

    @patch("src.data_access.document_store.db_config")
    def test_returns_false_when_document_missing(self, mock_db_config):
        """Test that document_exists returns False for missing documents."""
        from src.data_access.document_store import document_exists

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        result = document_exists("missing_doc")

        assert result is False


class TestClearCache:
    """Tests for clear_cache function."""

    @patch("src.data_access.document_store.delete_key")
    @patch("src.data_access.document_store._build_cache_key")
    @patch("src.data_access.document_store.get_client")
    def test_clears_specific_document_cache(
        self, mock_get_client, mock_build_key, mock_delete_key
    ):
        """Test clearing cache for a specific document."""
        from src.data_access.document_store import clear_cache

        mock_get_client.return_value = MagicMock()
        mock_build_key.return_value = "cache:test_doc"

        clear_cache("test_doc")

        mock_delete_key.assert_called_once_with("cache:test_doc")

    @patch("src.data_access.document_store.build_key")
    @patch("src.data_access.document_store.get_client")
    def test_clears_all_document_caches_when_no_key(
        self, mock_get_client, mock_build_key
    ):
        """Test clearing all document caches when no key specified."""
        from src.data_access.document_store import clear_cache

        mock_client = MagicMock()
        mock_client.scan_iter.return_value = ["key1", "key2", "key3"]
        mock_get_client.return_value = mock_client
        mock_build_key.return_value = "app:document:*"

        clear_cache()

        mock_client.scan_iter.assert_called_once_with(match="app:document:*")
        assert mock_client.delete.call_count == 3

    @patch("src.data_access.document_store.get_client")
    def test_handles_no_redis_client_gracefully(self, mock_get_client):
        """Test that clear_cache handles missing Redis client."""
        from src.data_access.document_store import clear_cache

        mock_get_client.return_value = None

        # Should not raise an exception
        clear_cache()
        clear_cache("test_doc")

    @patch("src.data_access.document_store.build_key")
    @patch("src.data_access.document_store.get_client")
    def test_handles_redis_errors_gracefully(self, mock_get_client, mock_build_key):
        """Test that clear_cache handles Redis errors gracefully."""
        from src.data_access.document_store import clear_cache

        mock_client = MagicMock()
        mock_client.scan_iter.side_effect = Exception("Redis connection failed")
        mock_get_client.return_value = mock_client
        mock_build_key.return_value = "app:document:*"

        # Should not raise an exception
        clear_cache()
