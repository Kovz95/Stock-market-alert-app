"""Formatting utilities for UI and messages."""

# Try to import streamlit for caching, but don't fail if not available
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def bl_sp(n: int) -> str:
    """
    Returns blank spaces for UI spacing in Streamlit.

    Args:
        n: Number of spaces

    Returns:
        String with blank spaces
    """
    return "\u200e " * (n + 1)


def split_message(message: str, max_length: int) -> list[str]:
    """
    Split a long message into multiple code blocks.

    Args:
        message: Message to split
        max_length: Maximum length per chunk

    Returns:
        List of message chunks wrapped in code blocks
    """
    lines = message.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 < max_length - 6:  # 6 for code block fences
            current_chunk += line + "\n"
        else:
            chunks.append(f"```{current_chunk.strip()}```")
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(f"```{current_chunk.strip()}```")

    return chunks


@st.cache_data(ttl=60) if STREAMLIT_AVAILABLE else lambda func: func
def get_asset_type(ticker: str) -> str:
    """
    Determine the asset type based on ticker symbol.

    Args:
        ticker: Ticker symbol

    Returns:
        'us_stock' or 'international_stock'
    """
    ticker_upper = ticker.upper()

    # International stock patterns (non-US exchanges)
    international_patterns = [
        ".HK",
        ".SI",
        ".TW",
        ".KL",
        ".KS",
        ".NS",
        ".SS",
        ".BK",
        ".JK",
        ".PS",
        ".HM",  # Asian
        ".AS",
        ".L",
        ".PA",
        ".DE",
        ".MI",
        ".SW",
        ".CH",
        ".NL",
        ".ES",
        ".ST",
        ".OL",
        ".CO",
        ".HE",
        ".BR",
        ".IE",
        ".LS",
        ".VI",
        ".WA",
        ".AT",
        ".BD",
        ".PR",
        ".IS",  # European
    ]

    # Check for international stocks
    for pattern in international_patterns:
        if ticker_upper.endswith(pattern):
            return "international_stock"

    # Default to US stock
    return "us_stock"


def is_us_stock(ticker: str) -> bool:
    """
    Check if a ticker is a US stock (no country suffix).

    Args:
        ticker: Ticker symbol

    Returns:
        True if US stock, False otherwise
    """
    ticker_upper = ticker.upper()

    # International stock patterns (non-US exchanges)
    international_patterns = [
        ".HK",
        ".SI",
        ".TW",
        ".KL",
        ".KS",
        ".NS",
        ".SS",
        ".BK",
        ".JK",
        ".PS",
        ".HM",  # Asian
        ".AS",
        ".L",
        ".PA",
        ".DE",
        ".MI",
        ".SW",
        ".CH",
        ".NL",
        ".ES",
        ".ST",
        ".OL",
        ".CO",
        ".HE",
        ".BR",
        ".IE",
        ".LS",
        ".VI",
        ".WA",
        ".AT",
        ".BD",
        ".PR",
        ".IS",  # European
    ]

    # Check for international patterns
    for pattern in international_patterns:
        if ticker_upper.endswith(pattern):
            return False

        if pattern in ticker_upper:
            return False

    return True
