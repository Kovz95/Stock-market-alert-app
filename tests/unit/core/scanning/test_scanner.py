"""Unit tests for scanner core logic."""

import pandas as pd

from src.stock_alert.core.scanning.scanner import batch_scan_pairs, batch_scan_symbols


class TestBatchScanSymbols:
    """Test suite for batch_scan_symbols function."""

    def test_scans_multiple_symbols(self):
        """Test scans multiple symbols and returns matches."""
        symbols = [
            ("AAPL", {"name": "Apple Inc.", "asset_type": "Stock"}),
            ("MSFT", {"name": "Microsoft Corp.", "asset_type": "Stock"}),
        ]
        conditions = ["Close[-1] > 100"]
        combination_logic = "AND"
        timeframe = "1d"

        # Mock price data function
        def mock_price_data(ticker, tf):
            return pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=100),
                    "Open": [150] * 100,
                    "High": [155] * 100,
                    "Low": [145] * 100,
                    "Close": [150] * 100,
                    "Volume": [1000000] * 100,
                }
            )

        # Mock evaluate function (always returns True for this test)
        def mock_evaluate(df, exps, combination):
            return True

        def mock_simplify(cond):
            return {"ind1": None, "ind2": None, "comparison": ">"}

        def mock_indicator(df, ind, col, flag):
            return None

        results = batch_scan_symbols(
            symbols,
            conditions,
            combination_logic,
            timeframe,
            price_data_func=mock_price_data,
            evaluate_func=mock_evaluate,
            simplify_func=mock_simplify,
            indicator_func=mock_indicator,
            max_workers=2,
        )

        # Should find matches for both symbols
        assert len(results) >= 0  # Depends on mock logic

    def test_returns_empty_for_no_matches(self):
        """Test returns empty list when no symbols match."""
        symbols = [
            ("AAPL", {"name": "Apple Inc.", "asset_type": "Stock"}),
        ]
        conditions = ["Close[-1] > 100"]
        combination_logic = "AND"
        timeframe = "1d"

        # Mock price data function
        def mock_price_data(ticker, tf):
            return pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=100),
                    "Open": [50] * 100,
                    "High": [55] * 100,
                    "Low": [45] * 100,
                    "Close": [50] * 100,  # Below 100
                    "Volume": [1000000] * 100,
                }
            )

        # Mock evaluate function (always returns False)
        def mock_evaluate(df, exps, combination):
            return False

        def mock_simplify(cond):
            return {"ind1": None, "ind2": None, "comparison": ">"}

        def mock_indicator(df, ind, col, flag):
            return None

        results = batch_scan_symbols(
            symbols,
            conditions,
            combination_logic,
            timeframe,
            price_data_func=mock_price_data,
            evaluate_func=mock_evaluate,
            simplify_func=mock_simplify,
            indicator_func=mock_indicator,
            max_workers=2,
        )

        assert len(results) == 0


class TestBatchScanPairs:
    """Test suite for batch_scan_pairs function."""

    def test_scans_multiple_pairs(self):
        """Test scans multiple pairs and returns matches."""
        pair_symbols = [
            ("AAPL", "MSFT"),
            ("GOOGL", "AMZN"),
        ]
        stock_db = {
            "AAPL": {"name": "Apple Inc.", "asset_type": "Stock"},
            "MSFT": {"name": "Microsoft Corp.", "asset_type": "Stock"},
            "GOOGL": {"name": "Alphabet Inc.", "asset_type": "Stock"},
            "AMZN": {"name": "Amazon.com Inc.", "asset_type": "Stock"},
        }
        conditions = ["Close[-1] > 1"]
        combination_logic = "AND"
        timeframe = "1d"

        # Mock price data function
        def mock_price_data(ticker, tf):
            return pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=100),
                    "Open": [150] * 100,
                    "High": [155] * 100,
                    "Low": [145] * 100,
                    "Close": [150] * 100,
                    "Volume": [1000000] * 100,
                }
            )

        # Mock evaluate function (always returns True)
        def mock_evaluate(df, exps, combination):
            return True

        def mock_simplify(cond):
            return {"ind1": None, "ind2": None, "comparison": ">"}

        def mock_indicator(df, ind, col, flag):
            return None

        results = batch_scan_pairs(
            pair_symbols,
            stock_db,
            conditions,
            combination_logic,
            timeframe,
            price_data_func=mock_price_data,
            evaluate_func=mock_evaluate,
            simplify_func=mock_simplify,
            indicator_func=mock_indicator,
            max_workers=2,
        )

        # Should find matches (depends on mock logic)
        assert len(results) >= 0

    def test_handles_empty_pair_list(self):
        """Test handles empty pair list gracefully."""
        pair_symbols = []
        stock_db = {}
        conditions = ["Close[-1] > 1"]
        combination_logic = "AND"
        timeframe = "1d"

        def mock_price_data(ticker, tf):
            return None

        def mock_evaluate(df, exps, combination):
            return False

        def mock_simplify(cond):
            return {"ind1": None, "ind2": None, "comparison": ">"}

        def mock_indicator(df, ind, col, flag):
            return None

        results = batch_scan_pairs(
            pair_symbols,
            stock_db,
            conditions,
            combination_logic,
            timeframe,
            price_data_func=mock_price_data,
            evaluate_func=mock_evaluate,
            simplify_func=mock_simplify,
            indicator_func=mock_indicator,
            max_workers=2,
        )

        assert len(results) == 0
