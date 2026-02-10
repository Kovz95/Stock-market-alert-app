"""Bulk alert creation service for optimized multi-alert operations.

This service provides efficient bulk alert creation by:
- Pre-loading futures database once (not per-alert)
- O(1) duplicate detection via hash set
- Parallel payload preparation using ThreadPoolExecutor
- Single database transaction with batch insert
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.config.performance_config import MAX_WORKERS
from src.data_access.alert_repository import bulk_create_alerts, list_alerts
from src.data_access.document_store import load_document

logger = logging.getLogger(__name__)


@dataclass
class BulkAlertResult:
    """Result of bulk alert creation operation."""

    inserted: int = 0
    skipped_duplicates: int = 0
    skipped_missing_data: int = 0
    failed: int = 0
    alert_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total number of alerts processed."""
        return self.inserted + self.skipped_duplicates + self.skipped_missing_data + self.failed


def _normalize_conditions_for_comparison(entry_conditions_list: Any) -> List[Dict[str, Any]]:
    """Normalize conditions for duplicate comparison.

    Args:
        entry_conditions_list: Raw conditions from UI.

    Returns:
        Normalized list of condition dictionaries.
    """
    if isinstance(entry_conditions_list, dict):
        conditions_for_compare = []
        for key, value in entry_conditions_list.items():
            if isinstance(value, dict) and "conditions" in value:
                for idx, condition_str in enumerate(value["conditions"], 1):
                    conditions_for_compare.append({"index": idx, "conditions": condition_str})
        return conditions_for_compare
    return entry_conditions_list if isinstance(entry_conditions_list, list) else []


def _conditions_to_storage_format(entry_conditions_list: Any) -> List[Dict[str, Any]]:
    """Convert conditions to storage format.

    Args:
        entry_conditions_list: Raw conditions from UI.

    Returns:
        Conditions formatted for database storage.
    """
    if isinstance(entry_conditions_list, dict):
        conditions_for_save = []
        for key, value in entry_conditions_list.items():
            if isinstance(value, dict) and "conditions" in value:
                for idx, condition_str in enumerate(value["conditions"], 1):
                    conditions_for_save.append({"index": idx, "conditions": condition_str})
        return conditions_for_save
    return entry_conditions_list if isinstance(entry_conditions_list, list) else []


def _derive_country(country: Optional[str], exchange: str) -> str:
    """Derive country from exchange if not provided.

    Args:
        country: Explicit country or None.
        exchange: Exchange identifier.

    Returns:
        Country string.
    """
    if country:
        return country
    try:
        from src.utils.reference_data import get_country_for_exchange
        return get_country_for_exchange(exchange) or exchange
    except ImportError:
        return exchange


def _compute_alert_signature(
    stock_name: str,
    ticker: str,
    conditions: List[Dict[str, Any]],
    combination_logic: str,
    exchange: str,
    timeframe: str,
    name: str,
) -> str:
    """Compute a hash signature for duplicate detection.

    Args:
        stock_name: Name of the stock.
        ticker: Stock ticker symbol.
        conditions: Normalized conditions list.
        combination_logic: AND/OR logic.
        exchange: Exchange identifier.
        timeframe: Alert timeframe.
        name: Alert name.

    Returns:
        SHA256 hash signature of the alert.
    """
    sig_data = {
        "stock_name": stock_name,
        "ticker": ticker,
        "conditions": conditions,
        "combination_logic": combination_logic,
        "exchange": exchange,
        "timeframe": timeframe,
        "name": name,
    }
    sig_str = json.dumps(sig_data, sort_keys=True)
    return hashlib.sha256(sig_str.encode()).hexdigest()


class BulkAlertService:
    """Service for efficient bulk alert creation.

    Optimizes alert creation by:
    - Loading futures database once upfront
    - Pre-computing existing alert signatures for O(1) duplicate detection
    - Parallelizing payload preparation
    - Using single database transaction for all inserts
    """

    def __init__(self, futures_db: Optional[Dict[str, Any]] = None):
        """Initialize the service.

        Args:
            futures_db: Optional pre-loaded futures database. If None, loads from storage.
        """
        self._futures_db = futures_db
        self._existing_signatures: Optional[Set[str]] = None

    @property
    def futures_db(self) -> Dict[str, Any]:
        """Lazy-load futures database."""
        if self._futures_db is None:
            self._futures_db = load_document(
                "futures_database",
                default={},
                fallback_path="futures_database.json",
            ) or {}
        return self._futures_db

    def _generate_simple_alert_name(
        self, stock_name: str, conditions_dict: Any
    ) -> str:
        """Generate a simple alert name when no template is provided.

        This is a simplified version to avoid circular imports with Add_Alert.py.

        Args:
            stock_name: Name of the stock.
            conditions_dict: Conditions dictionary.

        Returns:
            Generated alert name.
        """
        if not conditions_dict:
            return f"{stock_name} Alert"

        # Extract first condition for naming
        condition_parts = []
        if isinstance(conditions_dict, dict):
            for key, value in conditions_dict.items():
                if isinstance(value, dict) and "conditions" in value:
                    for cond in value["conditions"][:2]:  # First 2 conditions max
                        if cond:
                            # Extract key indicator names
                            cond_lower = cond.lower()
                            if "sma" in cond_lower:
                                condition_parts.append("SMA")
                            elif "ema" in cond_lower:
                                condition_parts.append("EMA")
                            elif "rsi" in cond_lower:
                                condition_parts.append("RSI")
                            elif "macd" in cond_lower:
                                condition_parts.append("MACD")
                            elif "bollinger" in cond_lower or "bb" in cond_lower:
                                condition_parts.append("BB")
                            elif "volume" in cond_lower:
                                condition_parts.append("Volume")
                            elif "price" in cond_lower or "close" in cond_lower:
                                condition_parts.append("Price")
                            break
                    break

        if condition_parts:
            return f"{stock_name} {' '.join(condition_parts[:2])}"
        return f"{stock_name} Alert"

    def _load_existing_signatures(self) -> Set[str]:
        """Load and compute signatures for all existing alerts.

        Returns:
            Set of signature hashes for existing alerts.
        """
        if self._existing_signatures is not None:
            return self._existing_signatures

        existing_alerts = list_alerts()
        signatures = set()

        for alert in existing_alerts:
            if alert.get("ratio", "No") == "Yes":
                # Skip ratio alerts in non-ratio duplicate checking
                continue

            sig = _compute_alert_signature(
                stock_name=alert.get("stock_name", ""),
                ticker=alert.get("ticker", ""),
                conditions=alert.get("conditions", []),
                combination_logic=alert.get("combination_logic", "AND"),
                exchange=alert.get("exchange") or alert.get("country", ""),
                timeframe=alert.get("timeframe", ""),
                name=alert.get("name", ""),
            )
            signatures.add(sig)

        self._existing_signatures = signatures
        logger.debug(f"Loaded {len(signatures)} existing alert signatures")
        return signatures

    def _is_duplicate(
        self,
        stock_name: str,
        ticker: str,
        conditions: List[Dict[str, Any]],
        combination_logic: str,
        exchange: str,
        timeframe: str,
        name: str,
    ) -> bool:
        """Check if an alert is a duplicate using O(1) hash lookup.

        Args:
            stock_name: Name of the stock.
            ticker: Stock ticker symbol.
            conditions: Normalized conditions list.
            combination_logic: AND/OR logic.
            exchange: Exchange identifier.
            timeframe: Alert timeframe.
            name: Alert name.

        Returns:
            True if alert already exists, False otherwise.
        """
        signatures = self._load_existing_signatures()
        sig = _compute_alert_signature(
            stock_name=stock_name,
            ticker=ticker,
            conditions=conditions,
            combination_logic=combination_logic,
            exchange=exchange,
            timeframe=timeframe,
            name=name,
        )
        return sig in signatures

    def _prepare_single_payload(
        self,
        stock_data: Dict[str, Any],
        conditions_dict: Any,
        combination_logic: str,
        timeframe: str,
        action: str,
        alert_name_template: str,
        adjustment_method: Optional[str],
        is_bulk: bool,
    ) -> Optional[Dict[str, Any]]:
        """Prepare a single alert payload.

        Args:
            stock_data: Dictionary with stock information (Name, Symbol, Exchange, Country).
            conditions_dict: Raw conditions from UI.
            combination_logic: AND/OR logic.
            timeframe: Alert timeframe.
            action: Alert action.
            alert_name_template: User-provided alert name or empty for auto-generation.
            adjustment_method: Futures adjustment method.
            is_bulk: Whether this is part of a bulk operation.

        Returns:
            Alert payload dictionary or None if should be skipped.
        """
        stock_name = stock_data.get("Name", "")
        ticker = stock_data.get("Symbol", "")
        exchange = stock_data.get("Exchange", stock_data.get("exchange", "Unknown"))
        country = stock_data.get("Country")

        if not stock_name or not ticker:
            return None

        # Determine alert name
        if not alert_name_template:
            # Auto-generate from conditions - use simple fallback to avoid circular import
            alert_name = self._generate_simple_alert_name(stock_name, conditions_dict)
        elif is_bulk:
            alert_name = f"{alert_name_template} - {stock_name}"
        else:
            alert_name = alert_name_template

        # Normalize and check for duplicates
        conditions_for_compare = _normalize_conditions_for_comparison(conditions_dict)
        conditions_for_save = _conditions_to_storage_format(conditions_dict)

        if self._is_duplicate(
            stock_name=stock_name,
            ticker=ticker,
            conditions=conditions_for_compare,
            combination_logic=combination_logic,
            exchange=exchange,
            timeframe=timeframe,
            name=alert_name,
        ):
            return {"_skipped": "duplicate", "stock_name": stock_name}

        # Check if futures
        is_futures = ticker in self.futures_db or ticker.upper() in self.futures_db

        country_value = _derive_country(country, exchange)

        payload = {
            "name": alert_name,
            "stock_name": stock_name,
            "ticker": ticker,
            "conditions": conditions_for_save,
            "combination_logic": combination_logic,
            "last_triggered": "",
            "action": action,
            "timeframe": timeframe,
            "exchange": exchange,
            "country": country_value,
            "ratio": "No",
            "is_ratio": False,
        }

        if adjustment_method or is_futures:
            payload["adjustment_method"] = adjustment_method

        return payload

    def create_alerts_batch(
        self,
        stocks_data: List[Dict[str, Any]],
        conditions_dict: Any,
        combination_logic: str,
        timeframe: str,
        action: str,
        alert_name_template: str = "",
        adjustment_method: Optional[str] = None,
        dtp_params: Optional[Dict[str, Any]] = None,
        multi_timeframe_params: Optional[Dict[str, Any]] = None,
        mixed_timeframe_params: Optional[Dict[str, Any]] = None,
        max_workers: int = MAX_WORKERS,
    ) -> BulkAlertResult:
        """Create multiple alerts efficiently in a batch.

        Args:
            stocks_data: List of stock dictionaries with Name, Symbol, Exchange, Country.
            conditions_dict: Raw conditions from UI.
            combination_logic: AND/OR logic for conditions.
            timeframe: Alert timeframe.
            action: Alert action.
            alert_name_template: User-provided alert name or empty for auto-generation.
            adjustment_method: Futures adjustment method.
            dtp_params: DTP-specific parameters.
            multi_timeframe_params: Multi-timeframe parameters.
            mixed_timeframe_params: Mixed-timeframe parameters.
            max_workers: Number of parallel workers for payload preparation.

        Returns:
            BulkAlertResult with counts and status.
        """
        result = BulkAlertResult()

        if not stocks_data:
            return result

        is_bulk = len(stocks_data) > 1
        payloads_to_insert = []
        skipped_duplicates = 0
        skipped_missing_data = 0

        # Pre-load existing signatures for O(1) duplicate detection
        self._load_existing_signatures()

        # Prepare payloads in parallel
        with ThreadPoolExecutor(max_workers=min(max_workers, len(stocks_data))) as executor:
            futures = {
                executor.submit(
                    self._prepare_single_payload,
                    stock_data=stock,
                    conditions_dict=conditions_dict,
                    combination_logic=combination_logic,
                    timeframe=timeframe,
                    action=action,
                    alert_name_template=alert_name_template,
                    adjustment_method=adjustment_method,
                    is_bulk=is_bulk,
                ): stock
                for stock in stocks_data
            }

            for future in as_completed(futures):
                try:
                    payload = future.result()
                    if payload is None:
                        skipped_missing_data += 1
                    elif payload.get("_skipped") == "duplicate":
                        skipped_duplicates += 1
                        logger.debug(f"Skipping duplicate alert for {payload.get('stock_name')}")
                    else:
                        # Add optional params
                        if dtp_params:
                            payload["dtp_params"] = dtp_params
                        if multi_timeframe_params:
                            payload["multi_timeframe_params"] = multi_timeframe_params
                        if mixed_timeframe_params:
                            payload["mixed_timeframe_params"] = mixed_timeframe_params
                        payloads_to_insert.append(payload)
                except Exception as e:
                    logger.error(f"Error preparing payload: {e}")
                    result.errors.append(str(e))
                    result.failed += 1

        result.skipped_duplicates = skipped_duplicates
        result.skipped_missing_data = skipped_missing_data

        if not payloads_to_insert:
            logger.info(
                f"No alerts to insert. Duplicates: {skipped_duplicates}, "
                f"Missing data: {skipped_missing_data}"
            )
            return result

        # Perform bulk insert in single transaction
        logger.info(f"Bulk inserting {len(payloads_to_insert)} alerts")
        db_result = bulk_create_alerts(payloads_to_insert)

        result.inserted = db_result["inserted"]
        result.failed += db_result["failed"]
        result.alert_ids = db_result["alert_ids"]
        result.errors.extend(db_result["errors"])

        # Add any DB-level skips (duplicate alert_ids) to our count
        db_skipped = db_result.get("skipped", 0)
        if db_skipped > 0:
            result.skipped_duplicates += db_skipped

        logger.info(
            f"Bulk alert creation complete: "
            f"inserted={result.inserted}, "
            f"skipped_duplicates={result.skipped_duplicates}, "
            f"skipped_missing={result.skipped_missing_data}, "
            f"failed={result.failed}"
        )

        return result
