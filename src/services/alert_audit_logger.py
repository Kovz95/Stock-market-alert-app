#!/usr/bin/env python3
"""
Alert Audit Logger - Comprehensive tracking system for alert evaluations
Provides tabular logging of all alert checks, evaluations, and results
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from src.data_access.db_config import db_config
from src.data_access.alert_repository import list_alerts as repo_list_alerts

# Default log directory - can be overridden by environment variable
LOG_DIR = os.getenv("LOG_DIR", ".")


class AlertAuditLogger:
    """
    Comprehensive logging system for tracking alert evaluations
    """

    def __init__(self, db_path: str = "alert_audit.db", log_file: Optional[str] = None):
        if log_file is None:
            log_file = os.path.join(LOG_DIR, "alert_audit.log")
        self.db_path = db_path
        self.log_file = log_file
        self.use_postgres = db_config.db_type == "postgresql"
        self.setup_database()
        self.setup_file_logging()

    @contextmanager
    def _connection(self):
        if self.use_postgres:
            with db_config.connection(role="alerts") as conn:
                yield conn
        else:
            with db_config.connection(db_path=self.db_path) as conn:
                yield conn

    def _serialize_json(self, data: Optional[Dict[str, Any]]):
        if data is None:
            return None
        if self.use_postgres:
            try:
                from psycopg2.extras import Json

                return Json(data)
            except ImportError:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "psycopg2.extras.Json not available; storing JSON as text fallback"
                )
                return json.dumps(data)
        return json.dumps(data)

    def _bool(self, value: bool) -> Any:
        if self.use_postgres:
            return value
        return int(value)

    def _cutoff_timestamp(self, days: int) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    def _execute(self, query: str, params: Optional[Sequence[Any]] = None):
        """
        Execute a query and return results.

        Returns:
            - For SELECT: list of rows (e.g. from fetchall()).
            - For INSERT/UPDATE/DELETE: row count (int) from cursor.rowcount.
        """
        with self._connection() as conn:
            return db_config.execute_with_retry(conn, query, params)

    def setup_database(self):
        """
        Initialize the audit database.

        In production we assume the PostgreSQL schema has already been applied
        via migration scripts (db/postgres_schema.sql).
        """
        if self.use_postgres:
            logging.getLogger(__name__).debug(
                "Skipping alert_audits SQLite bootstrap while running on PostgreSQL"
            )
            return

        ddl = """
        CREATE TABLE IF NOT EXISTS alert_audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            alert_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            stock_name TEXT,
            exchange TEXT,
            timeframe TEXT,
            action TEXT,
            evaluation_type TEXT NOT NULL,
            price_data_pulled BOOLEAN,
            price_data_source TEXT,
            conditions_evaluated BOOLEAN,
            alert_triggered BOOLEAN,
            trigger_reason TEXT,
            execution_time_ms INTEGER,
            cache_hit BOOLEAN,
            error_message TEXT,
            additional_data TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_alert_id ON alert_audits(alert_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON alert_audits(timestamp);
        CREATE INDEX IF NOT EXISTS idx_ticker ON alert_audits(ticker);
        """

        try:
            with self._connection() as conn:
                conn.executescript(ddl)
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Error setting up audit database: %s", exc, exc_info=True
            )

    def setup_file_logging(self):
        """Setup file-based logging for this module using a dedicated logger and FileHandler."""
        self.file_logger = logging.getLogger(__name__)
        if self.file_logger.handlers:
            return
        handler = logging.FileHandler(self.log_file, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.file_logger.addHandler(handler)
        self.file_logger.setLevel(logging.INFO)

    def log_no_data_available(self, audit_id: str, ticker: str):
        """Log when price data is not available for a ticker"""
        try:
            payload = {
                "data_status": "no_data",
                "ticker": ticker,
            }
            query = """
                UPDATE alert_audits
                SET price_data_pulled = ?,
                    error_message = ?,
                    additional_data = ?
                WHERE id = ?
            """
            params = (
                self._bool(False),
                "No data available from FMP API",
                self._serialize_json(payload),
                audit_id,
            )
            self._execute(query, params)

            self.file_logger.warning(f"No data available for ticker {ticker} (audit_id: {audit_id})")

        except Exception as e:
            self.file_logger.error(f"Failed to log no data available: {e}")

    def log_alert_check_start(self, alert: Dict, evaluation_type: str = "scheduled") -> str | None:
        """
        Log the start of an alert evaluation

        Args:
            alert: Alert dictionary
            evaluation_type: Type of evaluation (scheduled, manual, test)

        Returns:
            Audit ID for tracking this evaluation
        """
        try:
            audit_data : dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_id": alert.get("alert_id", alert.get("id", "unknown")),
                "ticker": alert.get("ticker", alert.get("ticker1", "unknown")),
                "stock_name": alert.get("stock_name", "unknown"),
                "exchange": alert.get("exchange", "unknown"),
                "timeframe": alert.get("timeframe", "unknown"),
                "action": alert.get("action", "unknown"),
                "evaluation_type": evaluation_type,
                "price_data_pulled": False,
                "price_data_source": None,
                "conditions_evaluated": False,
                "alert_triggered": False,
                "trigger_reason": None,
                "execution_time_ms": None,
                "cache_hit": False,
                "error_message": None,
                "additional_data": {
                    "ratio": alert.get("ratio", False),
                    "ticker2": alert.get("ticker2", None),
                    "conditions_count": len(alert.get("conditions", [])),
                    "combination_logic": alert.get("combination_logic", "AND"),
                },
            }

            columns = (
                "timestamp",
                "alert_id",
                "ticker",
                "stock_name",
                "exchange",
                "timeframe",
                "action",
                "evaluation_type",
                "price_data_pulled",
                "price_data_source",
                "conditions_evaluated",
                "alert_triggered",
                "trigger_reason",
                "execution_time_ms",
                "cache_hit",
                "error_message",
                "additional_data",
            )
            placeholders = ", ".join(["?"] * len(columns))
            query = f"""
                INSERT INTO alert_audits (
                    {', '.join(columns)}
                ) VALUES ({placeholders})
            """
            if self.use_postgres:
                query += " RETURNING id"

            params = (
                audit_data["timestamp"],
                audit_data["alert_id"],
                audit_data["ticker"],
                audit_data["stock_name"],
                audit_data["exchange"],
                audit_data["timeframe"],
                audit_data["action"],
                audit_data["evaluation_type"],
                self._bool(audit_data["price_data_pulled"]),
                audit_data["price_data_source"],
                self._bool(audit_data["conditions_evaluated"]),
                self._bool(audit_data["alert_triggered"]),
                audit_data["trigger_reason"],
                audit_data["execution_time_ms"],
                self._bool(audit_data["cache_hit"]),
                audit_data["error_message"],
                self._serialize_json(audit_data["additional_data"]),
            )

            with self._connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(query, params)
                    if self.use_postgres:
                        audit_id_value = cursor.fetchone()[0]
                    else:
                        audit_id_value = cursor.lastrowid
                    conn.commit()
                finally:
                    cursor.close()

            logging.info(
                "Alert check started: %s for %s",
                audit_data["alert_id"],
                audit_data["ticker"],
            )
            return str(audit_id_value)

        except Exception as e:
            logging.error(f"Error logging alert check start: {e}")
            return None

    def log_price_data_pulled(self, audit_id: str, price_source: str, cache_hit: bool = False):
        """
        Log when price data is successfully pulled

        Args:
            audit_id: Audit ID from log_alert_check_start
            price_source: Source of price data (FMP, etc.)
            cache_hit: Whether data came from cache
        """
        try:
            self._execute(
                '''
                UPDATE alert_audits
                SET price_data_pulled = ?, price_data_source = ?, cache_hit = ?
                WHERE id = ?
            ''',
                (self._bool(True), price_source, self._bool(cache_hit), audit_id),
            )
            logging.info(f"Price data pulled for audit {audit_id}: {price_source} (cache: {cache_hit})")

        except Exception as e:
            logging.error(f"Error logging price data pulled: {e}")

    def log_conditions_evaluated(self, audit_id: str, conditions_result: bool, trigger_reason: str | None = None):
        """
        Log when conditions are evaluated

        Args:
            audit_id: Audit ID from log_alert_check_start
            conditions_result: Whether conditions were successfully evaluated
            trigger_reason: Reason for trigger if applicable
        """
        try:
            self._execute(
                '''
                UPDATE alert_audits
                SET conditions_evaluated = ?, alert_triggered = ?, trigger_reason = ?
                WHERE id = ?
            ''',
                (
                    self._bool(True),
                    self._bool(conditions_result),
                    trigger_reason,
                    audit_id,
                ),
            )
            logging.info(f"Conditions evaluated for audit {audit_id}: triggered={conditions_result}, reason={trigger_reason}")

        except Exception as e:
            logging.error(f"Error logging conditions evaluated: {e}")

    def log_error(self, audit_id: str, error_message: str):
        """
        Log errors during alert evaluation

        Args:
            audit_id: Audit ID from log_alert_check_start
            error_message: Error description
        """
        try:
            self._execute(
                '''
                UPDATE alert_audits
                SET error_message = ?
                WHERE id = ?
            ''',
                (error_message, audit_id),
            )
            logging.error(f"Error logged for audit {audit_id}: {error_message}")

        except Exception as e:
            logging.error(f"Error logging error: {e}")

    def log_completion(self, audit_id: str, execution_time_ms: int):
        """
        Log completion of alert evaluation

        Args:
            audit_id: Audit ID from log_alert_check_start
            execution_time_ms: Total execution time in milliseconds
        """
        try:
            self._execute(
                '''
                UPDATE alert_audits
                SET execution_time_ms = ?
                WHERE id = ?
            ''',
                (execution_time_ms, audit_id),
            )
            logging.info(f"Alert evaluation completed for audit {audit_id}: {execution_time_ms}ms")

        except Exception as e:
            logging.error(f"Error logging completion: {e}")

    def get_audit_summary(self, days: int = 7) -> pd.DataFrame:
        """
        Get summary of alert audits for analysis

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with audit summary
        """
        try:
            cutoff = self._cutoff_timestamp(days)
            query = '''
                SELECT
                    alert_id,
                    ticker,
                    stock_name,
                    exchange,
                    timeframe,
                    action,
                    evaluation_type,
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN price_data_pulled THEN 1 ELSE 0 END) as successful_price_pulls,
                    SUM(CASE WHEN conditions_evaluated THEN 1 ELSE 0 END) as successful_evaluations,
                    SUM(CASE WHEN alert_triggered THEN 1 ELSE 0 END) as total_triggers,
                    AVG(execution_time_ms) as avg_execution_time_ms,
                    MAX(timestamp) as last_check,
                    MIN(timestamp) as first_check
                FROM alert_audits
                WHERE timestamp >= ?
                GROUP BY alert_id, ticker, stock_name, exchange, timeframe, action, evaluation_type
                ORDER BY last_check DESC
            '''

            with self._connection() as conn:
                df = pd.read_sql_query(query, conn, params=(cutoff,))
            return df

        except Exception as e:
            logging.error(f"Error getting audit summary: {e}")
            return pd.DataFrame()

    def get_alert_history(self, alert_id: str | None, limit: int = 100) -> pd.DataFrame:
        """
        Get detailed history for a specific alert

        Args:
            alert_id: Alert ID to get history for
            limit: Maximum number of records to return

        Returns:
            DataFrame with alert history
        """
        try:
            query = '''
                SELECT * FROM alert_audits
                WHERE alert_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''

            with self._connection() as conn:
                df = pd.read_sql_query(query, conn, params=(alert_id or "", limit))
            return df

        except Exception as e:
            logging.error(f"Error getting alert history: {e}")
            return pd.DataFrame()

    def get_performance_metrics(self, days: int = 7) -> Dict:
        """
        Get performance metrics for alert system.

        The 'total_errors' / 'error_rate' fields count evaluations that either
        failed to pull price data or had an explicit error (e.g. no data / FMP API).
        So they represent "failed or no-data" evaluations, not only exception messages.
        """
        try:
            cutoff = self._cutoff_timestamp(days)

            total_checks_result = self._execute(
                "SELECT COUNT(*) FROM alert_audits WHERE timestamp >= ?",
                (cutoff,),
            )
            total_checks = total_checks_result[0][0] if total_checks_result else 0

            successful_pulls_result = self._execute(
                "SELECT COUNT(*) FROM alert_audits WHERE timestamp >= ? AND price_data_pulled = ?",
                (cutoff, self._bool(True)),
            )
            successful_pulls = (
                successful_pulls_result[0][0] if successful_pulls_result else 0
            )

            cache_result = self._execute(
                """
                SELECT
                    SUM(CASE WHEN cache_hit = ? THEN 1 ELSE 0 END) as cache_hits,
                    COUNT(*) as total_pulls
                FROM alert_audits
                WHERE timestamp >= ? AND price_data_pulled = ?
                """,
                (self._bool(True), cutoff, self._bool(True)),
            )
            cache_hits, total_pulls = (cache_result[0] if cache_result else (0, 0))
            cache_hit_rate = (cache_hits / total_pulls * 100) if total_pulls else 0

            avg_execution_result = self._execute(
                """
                SELECT AVG(execution_time_ms) FROM alert_audits
                WHERE timestamp >= ? AND execution_time_ms IS NOT NULL
                """,
                (cutoff,),
            )
            if avg_execution_result and avg_execution_result[0][0] is not None:
                avg_execution_time = avg_execution_result[0][0]
            else:
                avg_execution_time = 0

            total_errors_result = self._execute(
                """
                SELECT COUNT(*) FROM alert_audits
                WHERE timestamp >= ?
                AND (
                    price_data_pulled = ?
                    OR error_message LIKE '%No data available%'
                    OR error_message LIKE '%FMP API%'
                )
                """,
                (cutoff, self._bool(False)),
            )
            total_errors = total_errors_result[0][0] if total_errors_result else 0
            error_rate = (total_errors / total_checks * 100) if total_checks else 0

            return {
                'total_checks': total_checks,
                'successful_price_pulls': successful_pulls,
                'success_rate': (successful_pulls / total_checks * 100) if total_checks > 0 else 0,
                'cache_hit_rate': cache_hit_rate,
                'avg_execution_time_ms': avg_execution_time,
                'total_errors': total_errors,
                'error_rate': error_rate,
                'analysis_period_days': days
            }

        except Exception as e:
            logging.error(f"Error getting performance metrics: {e}")
            return {}

    def cleanup_old_records(self, days_to_keep: int = 30):
        """
        Clean up old audit records to prevent database bloat

        Args:
            days_to_keep: Number of days of records to keep (0 to delete all)
        """
        try:
            if days_to_keep == 0:
                # Delete all records when days_to_keep is 0
                result = self._execute('DELETE FROM alert_audits')
                deleted_count = result if isinstance(result, int) else 0
            else:
                # Delete records older than specified days
                cutoff = self._cutoff_timestamp(days_to_keep)
                result = self._execute(
                    '''
                    DELETE FROM alert_audits
                    WHERE timestamp < ?
                    ''',
                    (cutoff,),
                )
                deleted_count = result if isinstance(result, int) else 0

            logging.info(f"Cleaned up {deleted_count} old audit records")
            return deleted_count

        except Exception as e:
            logging.error(f"Error cleaning up old records: {e}")
            return 0

    def get_daily_evaluation_stats(self, date=None):
        """Get daily evaluation statistics comparing expected vs actual evaluations"""
        if date is None:
            date = datetime.now().date()

        # Get all evaluations for the specified date
        date_str = date.strftime('%Y-%m-%d')
        query = """
            SELECT
                COUNT(*) as total_evaluations,
                SUM(CASE WHEN alert_triggered THEN 1 ELSE 0 END) as triggered_alerts,
                SUM(CASE WHEN NOT alert_triggered THEN 1 ELSE 0 END) as non_triggered_alerts,
                SUM(CASE WHEN error_message IS NOT NULL AND error_message != '' THEN 1 ELSE 0 END) as failed_evaluations,
                SUM(CASE WHEN error_message IS NULL OR error_message = '' THEN 1 ELSE 0 END) as successful_evaluations,
                AVG(execution_time_ms) as avg_execution_time,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
                SUM(CASE WHEN NOT cache_hit THEN 1 ELSE 0 END) as cache_misses
            FROM alert_audits
            WHERE DATE(timestamp) = ?
        """

        try:
            result_rows = self._execute(query, (date_str,))
            result = result_rows[0] if result_rows else None
            if result:
                return {
                    'date': date_str,
                    'total_evaluations': result[0] or 0,
                    'triggered_alerts': result[1] or 0,
                    'non_triggered_alerts': result[2] or 0,
                    'failed_evaluations': result[3] or 0,
                    'successful_evaluations': result[4] or 0,
                    'avg_execution_time': round(result[5] or 0, 2),
                    'cache_hits': result[6] or 0,
                    'cache_misses': result[7] or 0
                }
        except Exception as e:
            logging.error(f"Error getting daily evaluation stats: {e}")
            return None

    def get_evaluation_coverage(self, date=None):
        """Get evaluation coverage statistics for a specific date"""
        if date is None:
            date = datetime.now().date()

        # Get unique alerts that should have been evaluated
        # This assumes alerts are checked at least once per day
        date_str = date.strftime('%Y-%m-%d')

        try:
            query_unique = """
                SELECT COUNT(DISTINCT alert_id) as unique_alerts_evaluated
                FROM alert_audits
                WHERE DATE(timestamp) = ?
            """

            unique_rows = self._execute(query_unique, (date_str,))
            unique_alerts_evaluated = unique_rows[0][0] if unique_rows else 0

            query_frequency = """
                SELECT
                    alert_id,
                    COUNT(*) as evaluation_count
                FROM alert_audits
                WHERE DATE(timestamp) = ?
                GROUP BY alert_id
                ORDER BY evaluation_count DESC
            """

            frequency_results = self._execute(query_frequency, (date_str,)) or []

            # Calculate coverage metrics
            total_evaluations = sum(row[1] for row in frequency_results)
            avg_evaluations_per_alert = total_evaluations / max(unique_alerts_evaluated, 1)

            return {
                'date': date_str,
                'unique_alerts_evaluated': unique_alerts_evaluated,
                'total_evaluations': total_evaluations,
                'avg_evaluations_per_alert': round(avg_evaluations_per_alert, 2),
                'evaluation_frequency': frequency_results
            }

        except Exception as e:
            logging.error(f"Error getting evaluation coverage: {e}")
            return None

    def get_expected_daily_evaluations(self):
        """
        Estimate how many evaluations should happen per day based on total alerts.
        Uses a rough heuristic: ~90% daily alerts, ~10% weekly (evaluated once per week).
        """
        try:
            total_alerts = len(repo_list_alerts())

            # For manual checks, we expect 1 evaluation per alert per manual run
            # For scheduled checks, alerts are checked based on exchange closing times
            # Since the user can manually check all alerts at once, we'll show both scenarios

            # Manual check scenario (like when user clicks "Check All Alerts Now")
            manual_evaluations = total_alerts  # One check per alert

            # Scheduled check scenario (based on exchange times, not all at once)
            # Most exchanges close once per day, so daily alerts get checked once
            # Weekly alerts only on Fridays
            scheduled_daily_evaluations = total_alerts * 0.9  # Assume 90% are daily
            scheduled_weekly_evaluations = total_alerts * 0.1 / 5  # 10% weekly, only on Friday
            scheduled_evaluations = scheduled_daily_evaluations + scheduled_weekly_evaluations

            return {
                'total_alerts': total_alerts,
                'manual_check_evaluations': manual_evaluations,
                'scheduled_evaluations_estimate': int(scheduled_evaluations),
                'evaluation_type': 'Per manual check or scheduled run',
                'note': 'Manual checks evaluate all alerts at once'
            }
        except Exception as e:
            logging.error(f"Error calculating expected daily evaluations: {e}")
            return None


# Global instance for easy access
audit_logger = AlertAuditLogger()

# Convenience functions for easy logging
def log_alert_check_start(alert: Dict, evaluation_type: str = "scheduled") -> str | None:
    """Start logging an alert check"""
    return audit_logger.log_alert_check_start(alert, evaluation_type)

def log_price_data_pulled(audit_id: str, price_source: str, cache_hit: bool = False):
    """Log successful price data pull"""
    audit_logger.log_price_data_pulled(audit_id, price_source, cache_hit)

def log_no_data_available(audit_id: str, ticker: str):
    """Log when price data is not available"""
    audit_logger.log_no_data_available(audit_id, ticker)

def log_conditions_evaluated(audit_id: str, conditions_result: bool, trigger_reason: str | None = None):
    """Log conditions evaluation result"""
    audit_logger.log_conditions_evaluated(audit_id, conditions_result, trigger_reason)

def log_error(audit_id: str, error_message: str):
    """Log an error during alert evaluation"""
    audit_logger.log_error(audit_id, error_message)

def log_completion(audit_id: str, execution_time_ms: int):
    """Log completion of alert evaluation"""
    audit_logger.log_completion(audit_id, execution_time_ms)

def get_daily_evaluation_stats(date=None):
    """Get daily evaluation statistics comparing expected vs actual evaluations"""
    return audit_logger.get_daily_evaluation_stats(date)

def get_evaluation_coverage(date=None):
    """Get evaluation coverage statistics for a specific date"""
    return audit_logger.get_evaluation_coverage(date)

def get_expected_daily_evaluations():
    """Get expected daily evaluations based on total alerts"""
    return audit_logger.get_expected_daily_evaluations()

# Add convenience functions for existing methods
def get_audit_summary(days=7):
    return audit_logger.get_audit_summary(days)

def get_alert_history(alert_id: str | None = None, limit: int = 100):
    return audit_logger.get_alert_history(alert_id, limit)

def get_performance_metrics(days=7):
    return audit_logger.get_performance_metrics(days)
