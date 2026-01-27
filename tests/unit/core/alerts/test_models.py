"""Unit tests for alert models."""

from src.stock_alert.core.alerts.models import Alert, AlertCondition, RatioAlert


class TestAlertCondition:
    """Test suite for AlertCondition model."""

    def test_create_alert_condition(self):
        """Test creating an alert condition."""
        condition = AlertCondition(index=1, conditions="Close[-1] > 100")
        assert condition.index == 1
        assert condition.conditions == "Close[-1] > 100"


class TestAlert:
    """Test suite for Alert model."""

    def test_create_basic_alert(self):
        """Test creating a basic alert."""
        alert = Alert(
            name="Test Alert",
            stock_name="Apple Inc.",
            ticker="AAPL",
            conditions=[{"index": 1, "conditions": "Close[-1] > 100"}],
            combination_logic="AND",
            action="Buy",
            timeframe="1d",
            exchange="NASDAQ",
            country="US",
        )
        assert alert.name == "Test Alert"
        assert alert.stock_name == "Apple Inc."
        assert alert.ticker == "AAPL"
        assert alert.action == "Buy"
        assert alert.is_ratio is False

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            name="Test Alert",
            stock_name="Apple Inc.",
            ticker="AAPL",
            conditions=[{"index": 1, "conditions": "Close[-1] > 100"}],
            exchange="NASDAQ",
        )
        alert_dict = alert.to_dict()
        assert isinstance(alert_dict, dict)
        assert alert_dict["name"] == "Test Alert"
        assert alert_dict["ticker"] == "AAPL"
        assert alert_dict["is_ratio"] is False

    def test_alert_from_dict(self):
        """Test creating alert from dictionary."""
        alert_data = {
            "alert_id": "123",
            "name": "Test Alert",
            "stock_name": "Apple Inc.",
            "ticker": "AAPL",
            "conditions": [{"index": 1, "conditions": "Close[-1] > 100"}],
            "combination_logic": "AND",
            "action": "Buy",
            "timeframe": "1d",
            "exchange": "NASDAQ",
            "country": "US",
        }
        alert = Alert.from_dict(alert_data)
        assert alert.alert_id == "123"
        assert alert.name == "Test Alert"
        assert alert.ticker == "AAPL"

    def test_alert_with_dtp_params(self):
        """Test alert with DTP parameters."""
        alert = Alert(
            name="Test Alert",
            ticker="AAPL",
            stock_name="Apple",
            dtp_params={"enabled": True, "parameters": {"period": 20}},
        )
        assert alert.dtp_params is not None
        assert alert.dtp_params["enabled"] is True


class TestRatioAlert:
    """Test suite for RatioAlert model."""

    def test_create_ratio_alert(self):
        """Test creating a ratio alert."""
        alert = RatioAlert(
            name="SPY/QQQ Ratio",
            stock_name="SPY/QQQ",
            ticker1="SPY",
            ticker2="QQQ",
            conditions=[{"index": 1, "conditions": "Close[-1] > 1"}],
            exchange="NASDAQ",
        )
        assert alert.ticker1 == "SPY"
        assert alert.ticker2 == "QQQ"
        assert alert.is_ratio is True
        assert alert.ratio == "Yes"

    def test_ratio_alert_to_dict(self):
        """Test converting ratio alert to dictionary."""
        alert = RatioAlert(
            name="SPY/QQQ Ratio",
            stock_name="SPY/QQQ",
            ticker1="SPY",
            ticker2="QQQ",
            conditions=[{"index": 1, "conditions": "Close[-1] > 1"}],
            exchange="NASDAQ",
        )
        alert_dict = alert.to_dict()
        assert alert_dict["ticker1"] == "SPY"
        assert alert_dict["ticker2"] == "QQQ"
        assert alert_dict["ticker"] == "SPY_QQQ"
        assert alert_dict["is_ratio"] is True
        assert alert_dict["ratio"] == "Yes"

    def test_ratio_alert_from_dict(self):
        """Test creating ratio alert from dictionary."""
        alert_data = {
            "alert_id": "456",
            "name": "SPY/QQQ Ratio",
            "stock_name": "SPY/QQQ",
            "ticker1": "SPY",
            "ticker2": "QQQ",
            "conditions": [{"index": 1, "conditions": "Close[-1] > 1"}],
            "exchange": "NASDAQ",
        }
        alert = RatioAlert.from_dict(alert_data)
        assert alert.alert_id == "456"
        assert alert.ticker1 == "SPY"
        assert alert.ticker2 == "QQQ"
        assert alert.is_ratio is True
