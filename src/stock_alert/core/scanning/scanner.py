"""Core scanner business logic for scanning stocks and pairs."""

from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

from src.stock_alert.core.scanning.evaluator import extract_condition_values


def get_price_data_from_db(
    ticker: str, timeframe: str, lookback_days: int, db_config_func: Callable
) -> pd.DataFrame | None:
    """
    Get price data from PostgreSQL database.

    Args:
        ticker: Stock ticker symbol
        timeframe: Timeframe (1h, 1d, 1wk)
        lookback_days: Number of days to look back
        db_config_func: Function to get database config

    Returns:
        DataFrame with price data or None if error
    """
    try:
        end_ts = datetime.now()
        start_ts = end_ts - timedelta(days=lookback_days)

        if timeframe == "1h":
            query = """
                SELECT datetime AS date, open, high, low, close, volume
                FROM hourly_prices
                WHERE ticker = %s AND datetime BETWEEN %s AND %s
                ORDER BY datetime ASC
            """
            params = (ticker, start_ts, end_ts)
        elif timeframe == "1d":
            query = """
                SELECT date, open, high, low, close, volume
                FROM daily_prices
                WHERE ticker = %s AND date BETWEEN %s AND %s
                ORDER BY date ASC
            """
            params = (ticker, start_ts, end_ts)
        elif timeframe == "1wk":
            query = """
                SELECT date, open, high, low, close, volume
                FROM weekly_prices
                WHERE ticker = %s AND date BETWEEN %s AND %s
                ORDER BY date ASC
            """
            params = (ticker, start_ts, end_ts)
        else:
            return None

        with db_config_func() as (conn, cursor):
            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return None

            df = pd.DataFrame(
                rows,
                columns=["Date", "Open", "High", "Low", "Close", "Volume"],
            )

            # Convert to proper types
            df["Date"] = pd.to_datetime(df["Date"])
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Handle NaN values
            df = df.dropna(subset=["Close"])

            # Ensure columns are uppercase to match expected format
            df.columns = [col.capitalize() if col != "Date" else "Date" for col in df.columns]

            return df

    except Exception:
        return None


def scan_symbol(
    ticker: str,
    conditions: list[str],
    combination_logic: str,
    timeframe: str,
    stock_info: dict[str, Any],
    price_data_func: Callable,
    evaluate_func: Callable,
    simplify_func: Callable,
    indicator_func: Callable,
) -> dict[str, Any] | None:
    """
    Scan a single symbol for condition match.

    Args:
        ticker: Stock ticker symbol
        conditions: List of condition strings
        combination_logic: AND/OR logic
        timeframe: Timeframe (1h, 1d, 1wk)
        stock_info: Stock metadata dict
        price_data_func: Function to get price data
        evaluate_func: Function to evaluate expression list
        simplify_func: Function to simplify conditions
        indicator_func: Function to calculate indicators

    Returns:
        Dict with scan result or None if no match/error
    """
    try:
        # Get price data
        df = price_data_func(ticker, timeframe)

        if df is None or df.empty:
            return None

        # Ensure we have enough data for indicators
        if len(df) < 50:
            return None

        # Evaluate conditions
        condition_list = [cond.strip() for cond in conditions if cond.strip()]

        if not condition_list:
            return None

        result = evaluate_func(
            df=df, exps=condition_list, combination=combination_logic if combination_logic else "1"
        )

        if result:
            # Get latest price
            latest_price = df.iloc[-1]["Close"]
            cond_values = extract_condition_values(df, condition_list, simplify_func, indicator_func)

            return {
                "ticker": ticker,
                "name": stock_info.get("name", ticker),
                "isin": stock_info.get("isin", ""),
                "exchange": stock_info.get("exchange", ""),
                "country": stock_info.get("country", ""),
                "price": latest_price,
                "asset_type": stock_info.get("asset_type", "Stock"),
                # RBICS 6-level classification (for stocks)
                "rbics_economy": stock_info.get("rbics_economy", ""),
                "rbics_sector": stock_info.get("rbics_sector", ""),
                "rbics_subsector": stock_info.get("rbics_subsector", ""),
                "rbics_industry_group": stock_info.get("rbics_industry_group", ""),
                "rbics_industry": stock_info.get("rbics_industry", ""),
                "rbics_subindustry": stock_info.get("rbics_subindustry", ""),
                # ETF classification
                "etf_issuer": stock_info.get("etf_issuer", ""),
                "etf_asset_class": stock_info.get("etf_asset_class", ""),
                "etf_focus": stock_info.get("etf_focus", ""),
                "etf_niche": stock_info.get("etf_niche", ""),
                "expense_ratio": stock_info.get("expense_ratio", ""),
                "aum": stock_info.get("aum", ""),
                # Condition values for display
                "condition_values": cond_values,
            }

        return None
    except Exception:
        # Silently skip errors
        return None


def scan_pair(
    symbol1: str,
    symbol2: str,
    conditions: list[str],
    combination_logic: str,
    timeframe: str,
    stock_info1: dict[str, Any],
    stock_info2: dict[str, Any],
    price_data_func: Callable,
    evaluate_func: Callable,
    simplify_func: Callable,
    indicator_func: Callable,
) -> dict[str, Any] | None:
    """
    Scan a pair of symbols for condition match.

    Args:
        symbol1: First symbol ticker
        symbol2: Second symbol ticker
        conditions: List of condition strings
        combination_logic: AND/OR logic
        timeframe: Timeframe (1h, 1d, 1wk)
        stock_info1: First stock metadata dict
        stock_info2: Second stock metadata dict
        price_data_func: Function to get price data
        evaluate_func: Function to evaluate expression list
        simplify_func: Function to simplify conditions
        indicator_func: Function to calculate indicators

    Returns:
        Dict with scan result or None if no match/error
    """
    try:
        # Get price data for both symbols
        df1 = price_data_func(symbol1, timeframe)
        df2 = price_data_func(symbol2, timeframe)

        if df1 is None or df1.empty or df2 is None or df2.empty:
            return None

        # Ensure we have enough data
        if len(df1) < 50 or len(df2) < 50:
            return None

        # Align dataframes by date
        df1 = df1.set_index("Date")
        df2 = df2.set_index("Date")

        # Find common dates
        common_dates = df1.index.intersection(df2.index)
        if len(common_dates) < 50:
            return None

        df1 = df1.loc[common_dates].reset_index()
        df2 = df2.loc[common_dates].reset_index()

        # Create combined dataframe with _2 suffix for second symbol
        df_combined = df1.copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df_combined[f"{col}_2"] = df2[col].values

        # Create ratio columns (symbol1/symbol2) as the primary OHLC
        # This way conditions on Close, Open, etc. operate on the ratio
        df_combined["Open"] = df_combined["Open"] / df_combined["Open_2"]
        df_combined["High"] = df_combined["High"] / df_combined["High_2"]
        df_combined["Low"] = df_combined["Low"] / df_combined["Low_2"]
        df_combined["Close"] = df_combined["Close"] / df_combined["Close_2"]
        # Volume doesn't make sense as ratio, keep as sum
        df_combined["Volume"] = df_combined["Volume"] + df_combined["Volume_2"]

        # Evaluate conditions on combined dataframe (now using ratios)
        condition_list = [cond.strip() for cond in conditions if cond.strip()]

        if not condition_list:
            return None

        result = evaluate_func(
            df=df_combined,
            exps=condition_list,
            combination=combination_logic if combination_logic else "1",
        )

        if result:
            # Get latest prices (from original data before ratio calculation)
            latest_price1 = df1.iloc[-1]["Close"]
            latest_price2 = df2.iloc[-1]["Close"]
            ratio = latest_price1 / latest_price2
            cond_values = extract_condition_values(df_combined, condition_list, simplify_func, indicator_func)

            return {
                "pair": f"{symbol1}/{symbol2}",
                "symbol1": symbol1,
                "symbol2": symbol2,
                "name1": stock_info1.get("name", symbol1),
                "name2": stock_info2.get("name", symbol2),
                "price1": latest_price1,
                "price2": latest_price2,
                "ratio": ratio,
                "exchange1": stock_info1.get("exchange", ""),
                "exchange2": stock_info2.get("exchange", ""),
                "country1": stock_info1.get("country", ""),
                "country2": stock_info2.get("country", ""),
                "asset_type1": stock_info1.get("asset_type", "Stock"),
                "asset_type2": stock_info2.get("asset_type", "Stock"),
                "condition_values": cond_values,
            }

        return None
    except Exception:
        # Silently skip errors
        return None


def batch_scan_symbols(
    symbols: list[tuple[str, dict[str, Any]]],
    conditions: list[str],
    combination_logic: str,
    timeframe: str,
    price_data_func: Callable,
    evaluate_func: Callable,
    simplify_func: Callable,
    indicator_func: Callable,
    max_workers: int = 10,
) -> list[dict[str, Any]]:
    """
    Scan multiple symbols in parallel using ThreadPoolExecutor.

    Args:
        symbols: List of (ticker, stock_info) tuples
        conditions: List of condition strings
        combination_logic: AND/OR logic
        timeframe: Timeframe (1h, 1d, 1wk)
        price_data_func: Function to get price data
        evaluate_func: Function to evaluate expression list
        simplify_func: Function to simplify conditions
        indicator_func: Function to calculate indicators
        max_workers: Maximum number of parallel workers

    Returns:
        List of scan results (only matches)
    """
    import concurrent.futures

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                scan_symbol,
                ticker,
                conditions,
                combination_logic,
                timeframe,
                stock_info,
                price_data_func,
                evaluate_func,
                simplify_func,
                indicator_func,
            ): ticker
            for ticker, stock_info in symbols
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results


def batch_scan_pairs(
    pair_symbols: list[tuple[str, str]],
    stock_db: dict[str, dict[str, Any]],
    conditions: list[str],
    combination_logic: str,
    timeframe: str,
    price_data_func: Callable,
    evaluate_func: Callable,
    simplify_func: Callable,
    indicator_func: Callable,
    max_workers: int = 10,
) -> list[dict[str, Any]]:
    """
    Scan multiple pairs in parallel using ThreadPoolExecutor.

    Args:
        pair_symbols: List of (symbol1, symbol2) tuples
        stock_db: Stock metadata database
        conditions: List of condition strings
        combination_logic: AND/OR logic
        timeframe: Timeframe (1h, 1d, 1wk)
        price_data_func: Function to get price data
        evaluate_func: Function to evaluate expression list
        simplify_func: Function to simplify conditions
        indicator_func: Function to calculate indicators
        max_workers: Maximum number of parallel workers

    Returns:
        List of scan results (only matches)
    """
    import concurrent.futures

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                scan_pair,
                sym1,
                sym2,
                conditions,
                combination_logic,
                timeframe,
                stock_db.get(sym1, {}),
                stock_db.get(sym2, {}),
                price_data_func,
                evaluate_func,
                simplify_func,
                indicator_func,
            ): (sym1, sym2)
            for sym1, sym2 in pair_symbols
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    return results
