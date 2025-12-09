"""
Database configuration for hybrid SQLite/PostgreSQL approach.

Local development continues to use the existing SQLite files while production
deployments (ENVIRONMENT=production) switch to PostgreSQL automatically.  The
helper transparently exposes the same connection interface to callers so that
switching environments does not require code changes at the call site.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


class PostgresCursorProxy:
    """Translate SQLite-style parameter placeholders to psycopg2 syntax."""

    def __init__(self, cursor):
        self._cursor = cursor

    def _translate(
        self, query: str, params: Optional[Union[Sequence, Mapping]]
    ) -> Tuple[str, Optional[Union[Sequence, Mapping]]]:
        if params is None:
            return query, None
        if isinstance(params, Mapping):
            return query, params

        if "%s" in query:
            return query, params

        placeholder_count = query.count("?")
        if placeholder_count == 0:
            return query, params

        if _is_sequence(params) and not params:
            return query.replace("?", "%s"), params

        if _is_sequence(params) and _is_sequence(params[0]):
            # executemany-style call; assume uniform parameter width
            query = query.replace("?", "%s")
            return query, params

        if _is_sequence(params):
            if len(params) != placeholder_count:
                logger.debug(
                    "Placeholder mismatch: %s placeholders vs %s params in query %s",
                    placeholder_count,
                    len(params),  # type: ignore[arg-type]
                    query,
                )
            query = query.replace("?", "%s")
        return query, params

    def execute(self, query: str, params: Optional[Union[Sequence, Mapping]] = None):
        query, params = self._translate(query, params)
        self._cursor.execute(query, params)
        return self

    def executemany(
        self, query: str, param_sets: Iterable[Union[Sequence, Mapping]]
    ):
        # Convert iterable to list so we can inspect for translation
        param_list = list(param_sets)
        query, _ = self._translate(query, param_list)
        self._cursor.executemany(query, param_list)
        return self

    def __getattr__(self, item: str) -> Any:
        return getattr(self._cursor, item)

    def __iter__(self):
        return iter(self._cursor)

    def close(self) -> None:
        self._cursor.close()


class PostgresConnectionProxy:
    """Return connections to the pool when closed while mimicking DB-API."""

    def __init__(self, config, pool, raw_conn):
        self._config = config
        self._pool = pool
        self._raw = raw_conn
        self._closed = False
        self._release_cb = getattr(config, "_release_conn_slot", None)
        self._released = False
        self.is_postgres = True  # for downstream checks

    def _ensure_connection(self):
        try:
            if self._closed or self._raw is None:
                raw_conn = self._pool.getconn()
                self._config._configure_postgres_connection(raw_conn)
                self._raw = raw_conn
                self._closed = False
            elif getattr(self._raw, "closed", 0):
                try:
                    self._pool.putconn(self._raw, close=True)
                except Exception:
                    logger.debug("Failed returning closed connection to pool", exc_info=True)
                raw_conn = self._pool.getconn()
                self._config._configure_postgres_connection(raw_conn)
                self._raw = raw_conn
            return self._raw
        except Exception as exc:
            try:
                from psycopg2.pool import PoolError  # type: ignore
            except Exception:
                PoolError = Exception  # type: ignore

            if isinstance(exc, PoolError):
                logger.error("PostgreSQL pool is closed or exhausted; rebuilding")
                with self._config._pg_pool_lock:
                    try:
                        if self._pool:
                            self._pool.closeall()
                    except Exception:
                        logger.debug("Failed closing dead pool", exc_info=True)
                    self._config._pg_pool = None
                    self._pool = None

                fresh = self._config.get_postgresql_connection()
                self._pool = self._config._pg_pool
                self._raw = fresh._raw
                self._closed = False
                return self._raw
            raise

    def cursor(self, *args, **kwargs) -> PostgresCursorProxy:
        raw = self._ensure_connection()
        return PostgresCursorProxy(raw.cursor(*args, **kwargs))

    def execute(self, query: str, params: Optional[Union[Sequence, Mapping]] = None):
        cursor = self.cursor()
        cursor.execute(query, params)
        return cursor

    def executemany(
        self, query: str, param_sets: Iterable[Union[Sequence, Mapping]]
    ):
        cursor = self.cursor()
        cursor.executemany(query, param_sets)
        return cursor

    def commit(self) -> None:
        self._ensure_connection().commit()

    def rollback(self) -> None:
        self._ensure_connection().rollback()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._raw.rollback()
        except Exception:
            pass
        self._pool.putconn(self._raw)
        self._closed = True
        self._raw = None
        if self._release_cb and not self._released:
            try:
                self._release_cb()
            finally:
                self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc:
            try:
                self.rollback()
            except Exception:
                logger.exception("Failed to rollback PostgreSQL connection", exc_info=exc)
        else:
            try:
                self.commit()
            except Exception:
                logger.exception("Failed to commit PostgreSQL connection")
                raise
        self.close()

    def __getattr__(self, item: str) -> Any:
        if item in {"_config", "_pool", "_raw", "_closed", "is_postgres"}:
            return super().__getattribute__(item)
        raw = self._ensure_connection()
        return getattr(raw, item)


class DatabaseConfig:
    """Database configuration and connection management."""

    def __init__(self) -> None:
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.is_production = self.environment == "production"

        default_url = "postgresql://stockalertapp:Buddha4life@localhost:5432/stockalertapp"
        self.database_url = os.getenv("DATABASE_URL", default_url)
        self.db_type = "postgresql"

        self._pg_pool = None
        self._pg_pool_lock = threading.Lock()
        self._pg_pool_min = int(os.getenv("POSTGRES_POOL_MIN", "5"))
        self._pg_pool_max = int(os.getenv("POSTGRES_POOL_MAX", "50"))
        # Limit concurrent checkouts so threads don’t hammer the pool/server.
        self._pg_conn_sema = threading.Semaphore(int(os.getenv("POSTGRES_CONN_LIMIT", "40")))
        self._configured_connections = set()

    # --------------------------------------------------------------------- #
    # SQLite helpers
    # --------------------------------------------------------------------- #
    def _configure_postgres_connection(self, conn) -> None:
        conn_id = id(conn)
        if conn_id in self._configured_connections:
            return
        original_autocommit = conn.autocommit
        try:
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute("SET statement_timeout = '0'")
                cursor.execute("SET lock_timeout = '10s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '60s'")
        finally:
            conn.autocommit = original_autocommit
        self._configured_connections.add(conn_id)

    def get_postgresql_connection(self) -> PostgresConnectionProxy:
        try:
            from psycopg2.pool import ThreadedConnectionPool
        except ImportError as exc:
            logger.error(
                "psycopg2 is required for PostgreSQL connections. "
                "Install with: pip install psycopg2-binary"
            )
            raise

        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not configured for PostgreSQL access.")

        def _create_pool():
            logger.debug(
                "Creating PostgreSQL connection pool (%s-%s)",
                self._pg_pool_min,
                self._pg_pool_max,
            )
            return ThreadedConnectionPool(
                self._pg_pool_min,
                self._pg_pool_max,
                dsn=self.database_url,
            )

        self._pg_conn_sema.acquire()
        try:
            with self._pg_pool_lock:
                if self._pg_pool is None:
                    self._pg_pool = _create_pool()

                try:
                    raw_conn = self._pg_pool.getconn()  # type: ignore[arg-type]
                except Exception as exc:
                    # When the pool is exhausted or otherwise unhealthy, rebuild it
                    try:
                        from psycopg2.pool import PoolError  # type: ignore
                    except Exception:
                        PoolError = Exception  # type: ignore

                    if isinstance(exc, PoolError):
                        logger.error("PostgreSQL pool exhausted; resetting the pool")
                        try:
                            self._pg_pool.closeall()
                        except Exception:
                            logger.debug("Failed to close exhausted pool", exc_info=True)
                        self._pg_pool = _create_pool()
                        raw_conn = self._pg_pool.getconn()  # type: ignore[arg-type]
                    else:
                        raise

            self._configure_postgres_connection(raw_conn)
            logger.debug("PostgreSQL connection checked out from pool")
            return PostgresConnectionProxy(self, self._pg_pool, raw_conn)
        except Exception:
            # Release the semaphore if checkout failed
            self._pg_conn_sema.release()
            raise

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def get_connection(
        self,
        db_path: Optional[str] = None,
        *,
        role: Optional[str] = None,
    ):
        """
        Get database connection from the PostgreSQL pool.

        Args:
            db_path: Unused (retained for backwards compatibility).
            role: Logical identifier retained for compatibility.
        """
        return self.get_postgresql_connection()

    @contextmanager
    def connection(self, db_path: Optional[str] = None, *, role: Optional[str] = None):
        conn = self.get_connection(db_path=db_path, role=role)
        try:
            yield conn
        finally:
            self.close_connection(conn)

    def close_connection(self, conn) -> None:
        try:
            if conn is None:
                return
            if getattr(conn, "is_postgres", False):
                conn.close()
                logger.debug("PostgreSQL connection returned to pool")
                if getattr(self, "_pg_conn_sema", None):
                    try:
                        self._pg_conn_sema.release()
                    except Exception:
                        logger.debug("Semaphore release failed", exc_info=True)
            else:
                # Fallback for unexpected connection types
                conn.close()
        except Exception as exc:
            logger.warning("Error closing connection: %s", exc)

    def execute_with_retry(
        self,
        conn,
        query: str,
        params: Optional[Union[Sequence, Mapping]] = None,
        max_retries: int = 3,
    ):
        """Execute a query with simple retry logic for SQLite lock handling."""
        import time

        def _execute():
            cursor = conn.cursor()
            try:
                if params is not None:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                if query.strip().lower().startswith("select"):
                    result = cursor.fetchall()
                else:
                    conn.commit()
                    result = cursor.rowcount
            finally:
                cursor.close()
            return result

        return _execute()


# Global instance
db_config = DatabaseConfig()
