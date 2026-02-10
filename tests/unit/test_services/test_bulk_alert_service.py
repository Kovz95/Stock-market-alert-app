"""Unit tests for bulk_alert_service module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestBulkAlertResult:
    """Tests for BulkAlertResult dataclass."""

    def test_default_values(self):
        """Test default values are zero/empty."""
        from src.services.bulk_alert_service import BulkAlertResult

        result = BulkAlertResult()

        assert result.inserted == 0
        assert result.skipped_duplicates == 0
        assert result.skipped_missing_data == 0
        assert result.failed == 0
        assert result.alert_ids == []
        assert result.errors == []

    def test_total_processed_property(self):
        """Test total_processed sums all counts."""
        from src.services.bulk_alert_service import BulkAlertResult

        result = BulkAlertResult(
            inserted=5,
            skipped_duplicates=3,
            skipped_missing_data=2,
            failed=1,
        )

        assert result.total_processed == 11


class TestNormalizeConditionsForComparison:
    """Tests for _normalize_conditions_for_comparison function."""

    def test_handles_dict_with_conditions(self):
        """Test normalizing dict format conditions."""
        from src.services.bulk_alert_service import _normalize_conditions_for_comparison

        conditions = {
            "entry_1": {"conditions": ["price > 100", "volume > 1000"]},
        }

        result = _normalize_conditions_for_comparison(conditions)

        assert len(result) == 2
        assert result[0] == {"index": 1, "conditions": "price > 100"}
        assert result[1] == {"index": 2, "conditions": "volume > 1000"}

    def test_handles_list_input(self):
        """Test that list input is returned as-is."""
        from src.services.bulk_alert_service import _normalize_conditions_for_comparison

        conditions = [{"field": "price", "operator": ">", "value": 100}]

        result = _normalize_conditions_for_comparison(conditions)

        assert result == conditions

    def test_handles_empty_dict(self):
        """Test empty dict returns empty list."""
        from src.services.bulk_alert_service import _normalize_conditions_for_comparison

        result = _normalize_conditions_for_comparison({})

        assert result == []

    def test_handles_none(self):
        """Test None returns empty list."""
        from src.services.bulk_alert_service import _normalize_conditions_for_comparison

        result = _normalize_conditions_for_comparison(None)

        assert result == []


class TestComputeAlertSignature:
    """Tests for _compute_alert_signature function."""

    def test_generates_consistent_hash(self):
        """Test same inputs produce same hash."""
        from src.services.bulk_alert_service import _compute_alert_signature

        sig1 = _compute_alert_signature(
            stock_name="Apple",
            ticker="AAPL",
            conditions=[{"index": 1, "conditions": "price > 100"}],
            combination_logic="AND",
            exchange="NASDAQ",
            timeframe="1D",
            name="Test Alert",
        )

        sig2 = _compute_alert_signature(
            stock_name="Apple",
            ticker="AAPL",
            conditions=[{"index": 1, "conditions": "price > 100"}],
            combination_logic="AND",
            exchange="NASDAQ",
            timeframe="1D",
            name="Test Alert",
        )

        assert sig1 == sig2
        assert len(sig1) == 64  # SHA256 hex digest length

    def test_different_inputs_produce_different_hash(self):
        """Test different inputs produce different hash."""
        from src.services.bulk_alert_service import _compute_alert_signature

        sig1 = _compute_alert_signature(
            stock_name="Apple",
            ticker="AAPL",
            conditions=[],
            combination_logic="AND",
            exchange="NASDAQ",
            timeframe="1D",
            name="Test Alert",
        )

        sig2 = _compute_alert_signature(
            stock_name="Google",  # Different stock
            ticker="GOOGL",
            conditions=[],
            combination_logic="AND",
            exchange="NASDAQ",
            timeframe="1D",
            name="Test Alert",
        )

        assert sig1 != sig2


class TestBulkAlertService:
    """Tests for BulkAlertService class."""

    @pytest.fixture
    def mock_futures_db(self):
        """Mock futures database."""
        return {"ES": {"name": "E-mini S&P 500"}, "NQ": {"name": "E-mini Nasdaq"}}

    @pytest.fixture
    def service(self, mock_futures_db):
        """Create service with mocked futures db."""
        from src.services.bulk_alert_service import BulkAlertService
        return BulkAlertService(futures_db=mock_futures_db)

    def test_futures_db_lazy_loads(self):
        """Test futures_db is lazy loaded when not provided."""
        from src.services.bulk_alert_service import BulkAlertService

        with patch("src.services.bulk_alert_service.load_document") as mock_load:
            mock_load.return_value = {"ES": {}}
            service = BulkAlertService()

            # Access property to trigger load
            _ = service.futures_db

            mock_load.assert_called_once()

    def test_generate_simple_alert_name_with_sma(self, service):
        """Test alert name generation includes SMA."""
        conditions = {
            "entry_1": {"conditions": ["sma(20) > close"]}
        }

        name = service._generate_simple_alert_name("Apple Inc", conditions)

        assert "Apple Inc" in name
        assert "SMA" in name

    def test_generate_simple_alert_name_with_rsi(self, service):
        """Test alert name generation includes RSI."""
        conditions = {
            "entry_1": {"conditions": ["rsi(14) > 70"]}
        }

        name = service._generate_simple_alert_name("Apple Inc", conditions)

        assert "Apple Inc" in name
        assert "RSI" in name

    def test_generate_simple_alert_name_fallback(self, service):
        """Test alert name fallback for empty conditions."""
        name = service._generate_simple_alert_name("Apple Inc", {})

        assert name == "Apple Inc Alert"

    @patch("src.services.bulk_alert_service.list_alerts")
    def test_load_existing_signatures(self, mock_list_alerts, service):
        """Test loading existing alert signatures."""
        mock_list_alerts.return_value = [
            {
                "alert_id": "1",
                "stock_name": "Apple",
                "ticker": "AAPL",
                "conditions": [],
                "combination_logic": "AND",
                "exchange": "NASDAQ",
                "timeframe": "1D",
                "name": "Test",
                "ratio": "No",
            }
        ]

        signatures = service._load_existing_signatures()

        assert len(signatures) == 1
        mock_list_alerts.assert_called_once()

    @patch("src.services.bulk_alert_service.list_alerts")
    def test_is_duplicate_returns_true_for_match(self, mock_list_alerts, service):
        """Test duplicate detection returns True for existing alert."""
        mock_list_alerts.return_value = [
            {
                "alert_id": "1",
                "stock_name": "Apple",
                "ticker": "AAPL",
                "conditions": [],
                "combination_logic": "AND",
                "exchange": "NASDAQ",
                "timeframe": "1D",
                "name": "Test",
                "ratio": "No",
            }
        ]

        is_dup = service._is_duplicate(
            stock_name="Apple",
            ticker="AAPL",
            conditions=[],
            combination_logic="AND",
            exchange="NASDAQ",
            timeframe="1D",
            name="Test",
        )

        assert is_dup is True

    @patch("src.services.bulk_alert_service.list_alerts")
    def test_is_duplicate_returns_false_for_new(self, mock_list_alerts, service):
        """Test duplicate detection returns False for new alert."""
        mock_list_alerts.return_value = []

        is_dup = service._is_duplicate(
            stock_name="Apple",
            ticker="AAPL",
            conditions=[],
            combination_logic="AND",
            exchange="NASDAQ",
            timeframe="1D",
            name="Test",
        )

        assert is_dup is False

    @patch("src.services.bulk_alert_service.list_alerts")
    def test_prepare_single_payload_returns_none_for_missing_data(
        self, mock_list_alerts, service
    ):
        """Test payload preparation returns None for missing ticker."""
        mock_list_alerts.return_value = []

        result = service._prepare_single_payload(
            stock_data={"Name": "Apple", "Symbol": ""},  # Missing symbol
            conditions_dict={},
            combination_logic="AND",
            timeframe="1D",
            action="notify",
            alert_name_template="Test",
            adjustment_method=None,
            is_bulk=True,
        )

        assert result is None

    @patch("src.services.bulk_alert_service.list_alerts")
    def test_prepare_single_payload_marks_duplicate(
        self, mock_list_alerts, service
    ):
        """Test payload preparation marks duplicates."""
        mock_list_alerts.return_value = [
            {
                "alert_id": "1",
                "stock_name": "Apple",
                "ticker": "AAPL",
                "conditions": [],
                "combination_logic": "AND",
                "exchange": "NASDAQ",
                "timeframe": "1D",
                "name": "Test - Apple",
                "ratio": "No",
            }
        ]

        result = service._prepare_single_payload(
            stock_data={"Name": "Apple", "Symbol": "AAPL", "Exchange": "NASDAQ"},
            conditions_dict={},
            combination_logic="AND",
            timeframe="1D",
            action="notify",
            alert_name_template="Test",
            adjustment_method=None,
            is_bulk=True,
        )

        assert result["_skipped"] == "duplicate"

    @patch("src.services.bulk_alert_service.bulk_create_alerts")
    @patch("src.services.bulk_alert_service.list_alerts")
    def test_create_alerts_batch_success(
        self, mock_list_alerts, mock_bulk_create, service
    ):
        """Test successful batch alert creation."""
        mock_list_alerts.return_value = []
        mock_bulk_create.return_value = {
            "inserted": 2,
            "skipped": 0,
            "failed": 0,
            "alert_ids": ["id-1", "id-2"],
            "errors": [],
        }

        stocks_data = [
            {"Name": "Apple", "Symbol": "AAPL", "Exchange": "NASDAQ"},
            {"Name": "Google", "Symbol": "GOOGL", "Exchange": "NASDAQ"},
        ]

        result = service.create_alerts_batch(
            stocks_data=stocks_data,
            conditions_dict={},
            combination_logic="AND",
            timeframe="1D",
            action="notify",
            alert_name_template="Test",
        )

        assert result.inserted == 2
        assert result.skipped_duplicates == 0
        assert result.failed == 0
        mock_bulk_create.assert_called_once()

    @patch("src.services.bulk_alert_service.bulk_create_alerts")
    @patch("src.services.bulk_alert_service.list_alerts")
    def test_create_alerts_batch_skips_duplicates(
        self, mock_list_alerts, mock_bulk_create, service
    ):
        """Test batch creation skips duplicates."""
        mock_list_alerts.return_value = [
            {
                "alert_id": "1",
                "stock_name": "Apple",
                "ticker": "AAPL",
                "conditions": [],
                "combination_logic": "AND",
                "exchange": "NASDAQ",
                "timeframe": "1D",
                "name": "Test - Apple",
                "ratio": "No",
            }
        ]
        mock_bulk_create.return_value = {
            "inserted": 1,
            "skipped": 0,
            "failed": 0,
            "alert_ids": ["id-2"],
            "errors": [],
        }

        stocks_data = [
            {"Name": "Apple", "Symbol": "AAPL", "Exchange": "NASDAQ"},  # Duplicate
            {"Name": "Google", "Symbol": "GOOGL", "Exchange": "NASDAQ"},  # New
        ]

        result = service.create_alerts_batch(
            stocks_data=stocks_data,
            conditions_dict={},
            combination_logic="AND",
            timeframe="1D",
            action="notify",
            alert_name_template="Test",
        )

        assert result.inserted == 1
        assert result.skipped_duplicates == 1

    def test_create_alerts_batch_empty_input(self, service):
        """Test batch creation with empty input."""
        result = service.create_alerts_batch(
            stocks_data=[],
            conditions_dict={},
            combination_logic="AND",
            timeframe="1D",
            action="notify",
        )

        assert result.inserted == 0
        assert result.total_processed == 0

    @patch("src.services.bulk_alert_service.list_alerts")
    def test_create_alerts_batch_handles_missing_data(
        self, mock_list_alerts, service
    ):
        """Test batch creation handles stocks with missing data."""
        mock_list_alerts.return_value = []

        stocks_data = [
            {"Name": "Apple", "Symbol": ""},  # Missing symbol
            {"Name": "", "Symbol": "GOOGL"},  # Missing name
        ]

        result = service.create_alerts_batch(
            stocks_data=stocks_data,
            conditions_dict={},
            combination_logic="AND",
            timeframe="1D",
            action="notify",
        )

        assert result.skipped_missing_data == 2
        assert result.inserted == 0

    @patch("src.services.bulk_alert_service.list_alerts")
    def test_prepare_payload_detects_futures(self, mock_list_alerts, service):
        """Test futures detection adds adjustment_method."""
        mock_list_alerts.return_value = []

        result = service._prepare_single_payload(
            stock_data={"Name": "E-mini S&P 500", "Symbol": "ES", "Exchange": "CME"},
            conditions_dict={},
            combination_logic="AND",
            timeframe="1D",
            action="notify",
            alert_name_template="Test",
            adjustment_method="ratio",
            is_bulk=False,
        )

        assert result is not None
        assert "adjustment_method" in result
