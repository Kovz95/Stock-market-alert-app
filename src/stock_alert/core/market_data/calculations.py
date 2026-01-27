"""Data calculation and normalization utilities."""

import pandas as pd


def calculate_ratio(df1, df2):
    """
    Calculate ratio between two dataframes.

    Args:
        df1: First dataframe
        df2: Second dataframe

    Returns:
        DataFrame with ratio values
    """
    try:
        t1 = df1.columns.get_level_values("Ticker")[0]
        t2 = df2.columns.get_level_values("Ticker")[0]

        # pull out just the metrics (Volume, VWAP, Open, …) for each ticker
        a = df1.xs(t1, level="Ticker", axis=1)
        b = df2.xs(t2, level="Ticker", axis=1)

        # element-wise ratio
        df3 = a.div(b)
        df3.columns = df3.columns.droplevel("Ticker")
        df3.columns.name = None
        return df3
    except:
        df3 = df1.div(df2)
        df3.columns.name = None
        df3.index.name = None
        return df3


def calculate_cross_exchange_ratio(df1, df2, ticker1, ticker2):
    """
    Calculate ratio between stocks from different exchanges.
    Handles different data formats and normalizes them.

    Args:
        df1: First dataframe
        df2: Second dataframe
        ticker1: First ticker symbol
        ticker2: Second ticker symbol

    Returns:
        DataFrame with ratio values or None if error
    """
    try:
        # Normalize dataframes to have consistent column names
        df1_normalized = normalize_dataframe(df1, ticker1)
        df2_normalized = normalize_dataframe(df2, ticker2)

        # Ensure both dataframes have the same date range
        common_dates = df1_normalized.index.intersection(df2_normalized.index)
        df1_aligned = df1_normalized.loc[common_dates]
        df2_aligned = df2_normalized.loc[common_dates]

        # Calculate ratio
        ratio_df = df1_aligned.div(df2_aligned)

        # Add metadata
        ratio_df.attrs["ticker1"] = ticker1
        ratio_df.attrs["ticker2"] = ticker2
        ratio_df.attrs["ratio_type"] = "cross_exchange"

        return ratio_df

    except Exception as e:
        print(f"Error calculating cross-exchange ratio: {e}")
        return None


def normalize_dataframe(df, ticker):
    """
    Normalize dataframe to have consistent column names regardless of exchange.

    Args:
        df: Input dataframe
        ticker: Ticker symbol

    Returns:
        Normalized dataframe or None if error
    """
    try:
        # Handle different column structures
        if isinstance(df.columns, pd.MultiIndex):
            if "Ticker" in df.columns.names:
                normalized_df = df.xs(ticker, level="Ticker", axis=1)
            else:
                normalized_df = df
        else:
            normalized_df = df

        # Ensure we have the standard OHLCV columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        available_columns = [col for col in normalized_df.columns if col in required_columns]

        if len(available_columns) < 4:  # Need at least OHLC
            raise ValueError(f"Insufficient data columns for {ticker}")

        # Select only the available OHLCV columns
        normalized_df = normalized_df[available_columns]

        # Ensure numeric data
        for col in normalized_df.columns:
            normalized_df[col] = pd.to_numeric(normalized_df[col], errors="coerce")

        # Remove any rows with NaN values
        normalized_df = normalized_df.dropna()

        return normalized_df

    except Exception as e:
        print(f"Error normalizing dataframe for {ticker}: {e}")
        return None
