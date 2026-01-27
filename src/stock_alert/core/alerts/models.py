"""Alert domain models and data structures."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlertCondition:
    """Represents a single alert condition."""

    index: int
    conditions: str


@dataclass
class DTPParams:
    """Dynamic Time Period parameters for alerts."""

    enabled: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTimeframeParams:
    """Multi-timeframe parameters for alerts."""

    enabled: bool = False
    timeframes: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedTimeframeParams:
    """Mixed timeframe parameters for alerts."""

    enabled: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents a stock alert."""

    alert_id: str | None = None
    name: str = ""
    stock_name: str = ""
    ticker: str = ""
    conditions: list[dict[str, Any]] = field(default_factory=list)
    combination_logic: str = "AND"
    last_triggered: str | None = None
    action: str = "Buy"
    timeframe: str = "1d"
    exchange: str = ""
    country: str = ""
    ratio: str = "No"
    is_ratio: bool = False
    dtp_params: DTPParams | None = None
    multi_timeframe_params: MultiTimeframeParams | None = None
    mixed_timeframe_params: MixedTimeframeParams | None = None
    adjustment_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        result = {
            "name": self.name,
            "stock_name": self.stock_name,
            "ticker": self.ticker,
            "conditions": self.conditions,
            "combination_logic": self.combination_logic,
            "last_triggered": self.last_triggered,
            "action": self.action,
            "timeframe": self.timeframe,
            "exchange": self.exchange,
            "country": self.country,
            "ratio": self.ratio,
            "is_ratio": self.is_ratio,
        }

        if self.alert_id:
            result["alert_id"] = self.alert_id

        if self.dtp_params:
            result["dtp_params"] = self.dtp_params

        if self.multi_timeframe_params:
            result["multi_timeframe_params"] = self.multi_timeframe_params

        if self.mixed_timeframe_params:
            result["mixed_timeframe_params"] = self.mixed_timeframe_params

        if self.adjustment_method:
            result["adjustment_method"] = self.adjustment_method

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alert":
        """Create alert from dictionary."""
        return cls(
            alert_id=data.get("alert_id"),
            name=data.get("name", ""),
            stock_name=data.get("stock_name", ""),
            ticker=data.get("ticker", ""),
            conditions=data.get("conditions", []),
            combination_logic=data.get("combination_logic", "AND"),
            last_triggered=data.get("last_triggered"),
            action=data.get("action", "Buy"),
            timeframe=data.get("timeframe", "1d"),
            exchange=data.get("exchange", ""),
            country=data.get("country", ""),
            ratio=data.get("ratio", "No"),
            is_ratio=data.get("is_ratio", False),
            dtp_params=data.get("dtp_params"),
            multi_timeframe_params=data.get("multi_timeframe_params"),
            mixed_timeframe_params=data.get("mixed_timeframe_params"),
            adjustment_method=data.get("adjustment_method"),
        )


@dataclass
class RatioAlert(Alert):
    """Represents a ratio alert between two stocks."""

    ticker1: str = ""
    ticker2: str = ""
    is_ratio: bool = True
    ratio: str = "Yes"

    def to_dict(self) -> dict[str, Any]:
        """Convert ratio alert to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "ticker1": self.ticker1,
                "ticker2": self.ticker2,
                "ticker": f"{self.ticker1}_{self.ticker2}",
                "is_ratio": True,
                "ratio": "Yes",
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RatioAlert":
        """Create ratio alert from dictionary."""
        return cls(
            alert_id=data.get("alert_id"),
            name=data.get("name", ""),
            stock_name=data.get("stock_name", ""),
            ticker1=data.get("ticker1", ""),
            ticker2=data.get("ticker2", ""),
            conditions=data.get("conditions", []),
            combination_logic=data.get("combination_logic", "AND"),
            last_triggered=data.get("last_triggered"),
            action=data.get("action", "Buy"),
            timeframe=data.get("timeframe", "1d"),
            exchange=data.get("exchange", ""),
            country=data.get("country", ""),
            adjustment_method=data.get("adjustment_method"),
        )
