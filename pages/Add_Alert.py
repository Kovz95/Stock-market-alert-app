import os
import time
import streamlit as st
import traceback
from pathlib import Path

st.set_page_config(
    page_title="Add Alert",
    page_icon="+",
    layout="wide",
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json

import pandas as pd
import uuid
from streamlit_tags import st_tags
from src.data_access.document_store import load_document

# Local imports
from src.utils.utils import (
    load_market_data,
    bl_sp,
    predefined_suggestions,
    # All data fetching now uses FMP only
    grab_new_data_universal,
    get_asset_type,
    save_alert,
    calculate_ratio,
    calculate_cross_exchange_ratio,
    save_ratio_alert,
    suggest_alert_update,
    get_stock_alerts_summary
)

# Initialize session state for entry_conditions and entry_combination
if "entry_conditions" not in st.session_state:
    st.session_state.entry_conditions = {}
if "entry_combination" not in st.session_state:
    st.session_state.entry_combination = "AND"

# Helper functions for alert name generation
def extract_parameter(condition, param_name):
    """
    Extract parameter value from condition string

    Args:
        condition: The condition string
        param_name: Name of the parameter to extract

    Returns:
        Parameter value as string or None if not found
    """
    import re

    # Pattern to match parameter_name = value (with spaces)
    pattern = rf"{param_name}\s*=\s*([^,\s)]+)"
    match = re.search(pattern, condition, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # Pattern to match parameter_name = value (without spaces)
    pattern2 = rf"{param_name}=([^,\s)]+)"
    match2 = re.search(pattern2, condition, re.IGNORECASE)

    if match2:
        return match2.group(1).strip()

    # Pattern to match just the value after parameter_name (with space)
    pattern3 = rf"{param_name}\s+([^,\s)]+)"
    match3 = re.search(pattern3, condition, re.IGNORECASE)

    if match3:
        return match3.group(1).strip()

    return None

def extract_price_value(condition):
    """
    Extract price value from price condition

    Args:
        condition: The condition string

    Returns:
        Price value as string or None if not found
    """
    import re

    # Pattern to match price values (numbers with optional decimals)
    # Look for numbers that are likely to be prices (not periods or other parameters)
    pattern = r"(\d+\.?\d*)"
    matches = re.findall(pattern, condition)

    if matches:
        # Return the first number found (usually the price)
        # Filter out common non-price numbers like periods
        for match in matches:
            try:
                value = float(match)
                # Skip common period values (1-200) and focus on likely price values
                if value > 200 or (value < 200 and '.' in match):  # Likely a price
                    return match
            except ValueError:
                continue
        # If no likely price found, return the first number
        return matches[0]

    return None

def extract_volume_value(condition):
    """
    Extract volume value from volume condition

    Args:
        condition: The condition string

    Returns:
        Volume value as string or None if not found
    """
    import re

    # Pattern to match volume multipliers or thresholds
    pattern = r"(\d+\.?\d*)[xX]?"  # Matches numbers with optional 'x' multiplier
    matches = re.findall(pattern, condition)

    if matches:
        # Return the first number found
        return matches[0]

    return None

# Utility: optionally wrap a numeric indicator expression in a z-score
def apply_zscore_indicator(indicator_expr: str, use_zscore: bool, lookback: int) -> str:
    if not use_zscore or not indicator_expr:
        return indicator_expr
    # Skip boolean/comparison expressions
    if any(op in indicator_expr for op in [">", "<", "=", " and ", " or ", ":"]):
        return indicator_expr
    import re
    base = indicator_expr.strip()
    match = re.match(r"(.+)\[(-?\d+)\]$", base)
    if match:
        base = match.group(1)
    try:
        lb = int(lookback)
    except Exception:
        lb = 20
    return f"zscore({base}, lookback={lb})[-1]"

# Bulk alert creation functions
def generate_alert_name_from_conditions(stock_name, conditions_dict, combination_logic=""):
    """
    Generate a descriptive alert name based on the conditions used

    Args:
        stock_name: Name of the stock
        conditions_dict: Dictionary of conditions from session state
        combination_logic: How conditions are combined (AND/OR)

    Returns:
        Descriptive alert name
    """
    if not conditions_dict:
        return f"{stock_name} Alert"

    # Extract conditions and create a descriptive name
    condition_descriptions = []

    for condition_id, conditions in conditions_dict.items():
        for condition in conditions:
            # Clean up the condition for display
            clean_condition = condition.strip()

            # Extract key indicators with parameters from common patterns
            if "sma(" in clean_condition.lower():
                # Extract period parameter
                period_match = extract_parameter(clean_condition, "period")
                if period_match:
                    condition_descriptions.append(f"SMA({period_match})")
                else:
                    condition_descriptions.append("SMA")
            elif "ema(" in clean_condition.lower():
                # Extract period parameter
                period_match = extract_parameter(clean_condition, "period")
                if period_match:
                    condition_descriptions.append(f"EMA({period_match})")
                else:
                    condition_descriptions.append("EMA")
            elif "hma(" in clean_condition.lower():
                # Extract period parameter
                period_match = extract_parameter(clean_condition, "period")
                if period_match:
                    condition_descriptions.append(f"HMA({period_match})")
                else:
                    condition_descriptions.append("HMA")
            elif "rsi(" in clean_condition.lower():
                # Extract period parameter
                period_match = extract_parameter(clean_condition, "period")
                if period_match:
                    condition_descriptions.append(f"RSI({period_match})")
                else:
                    condition_descriptions.append("RSI")
            elif "macd(" in clean_condition.lower():
                # Extract MACD parameters
                fast_match = extract_parameter(clean_condition, "fast_period")
                slow_match = extract_parameter(clean_condition, "slow_period")
                signal_match = extract_parameter(clean_condition, "signal_period")
                type_match = extract_parameter(clean_condition, "type")

                params = []
                if fast_match:
                    params.append(f"fast={fast_match}")
                if slow_match:
                    params.append(f"slow={slow_match}")
                if signal_match:
                    params.append(f"signal={signal_match}")
                if type_match:
                    params.append(f"type={type_match}")

                if params:
                    condition_descriptions.append(f"MACD({','.join(params)})")
                else:
                    condition_descriptions.append("MACD")
            elif "bb(" in clean_condition.lower() or "bollinger" in clean_condition.lower():
                # Extract Bollinger Bands parameters
                period_match = extract_parameter(clean_condition, "period")
                std_dev_match = extract_parameter(clean_condition, "std_dev")
                type_match = extract_parameter(clean_condition, "type")

                params = []
                if period_match:
                    params.append(f"period={period_match}")
                if std_dev_match:
                    params.append(f"std={std_dev_match}")
                if type_match:
                    params.append(f"type={type_match}")

                if params:
                    condition_descriptions.append(f"BB({','.join(params)})")
                else:
                    condition_descriptions.append("Bollinger")
            elif "atr(" in clean_condition.lower():
                # Extract ATR period parameter
                period_match = extract_parameter(clean_condition, "period")
                if period_match:
                    condition_descriptions.append(f"ATR({period_match})")
                else:
                    condition_descriptions.append("ATR")
            elif "cci(" in clean_condition.lower():
                # Extract CCI period parameter
                period_match = extract_parameter(clean_condition, "period")
                if period_match:
                    condition_descriptions.append(f"CCI({period_match})")
                else:
                    condition_descriptions.append("CCI")
            elif "roc(" in clean_condition.lower():
                # Extract ROC period parameter
                period_match = extract_parameter(clean_condition, "period")
                if period_match:
                    condition_descriptions.append(f"ROC({period_match})")
                else:
                    condition_descriptions.append("ROC")
            elif "williamsr(" in clean_condition.lower():
                # Extract WilliamsR period parameter
                period_match = extract_parameter(clean_condition, "period")
                if period_match:
                    condition_descriptions.append(f"WilliamsR({period_match})")
                else:
                    condition_descriptions.append("WilliamsR")
            elif "harsi_flip(" in clean_condition.lower():
                # Extract HARSI parameters
                period_match = extract_parameter(clean_condition, "period")
                smoothing_match = extract_parameter(clean_condition, "smoothing")

                params = []
                if period_match:
                    params.append(f"period={period_match}")
                if smoothing_match:
                    params.append(f"smooth={smoothing_match}")

                if params:
                    condition_descriptions.append(f"HARSI({','.join(params)})")
                else:
                    condition_descriptions.append("HARSI")
            elif "srocst(" in clean_condition.lower():
                # Extract SROCST parameters (simplified)
                condition_descriptions.append("SROCST")
            elif "dtp_" in clean_condition.lower():
                # Extract DTP parameters
                ma_type_match = extract_parameter(clean_condition, "ma_type")
                length_match = extract_parameter(clean_condition, "length")

                params = []
                if ma_type_match:
                    params.append(f"ma={ma_type_match}")
                if length_match:
                    params.append(f"len={length_match}")

                if params:
                    condition_descriptions.append(f"DTP({','.join(params)})")
                else:
                    condition_descriptions.append("DTP")
            elif "donchian_" in clean_condition.lower():
                # Extract Donchian parameters
                length_match = extract_parameter(clean_condition, "length")

                if "donchian_upper" in clean_condition.lower():
                    if length_match:
                        condition_descriptions.append(f"DC_Upper({length_match})")
                    else:
                        condition_descriptions.append("DC_Upper")
                elif "donchian_lower" in clean_condition.lower():
                    if length_match:
                        condition_descriptions.append(f"DC_Lower({length_match})")
                    else:
                        condition_descriptions.append("DC_Lower")
                elif "donchian_basis" in clean_condition.lower():
                    if length_match:
                        condition_descriptions.append(f"DC_Basis({length_match})")
                    else:
                        condition_descriptions.append("DC_Basis")
                elif "donchian_width" in clean_condition.lower():
                    condition_descriptions.append("DC_Width")
                elif "donchian_position" in clean_condition.lower():
                    condition_descriptions.append("DC_Position")
                else:
                    condition_descriptions.append("Donchian")
            elif "price_above" in clean_condition.lower():
                # Extract price value
                price_match = extract_price_value(clean_condition)
                if price_match:
                    condition_descriptions.append(f"PriceAbove({price_match})")
                else:
                    condition_descriptions.append("PriceAbove")
            elif "price_below" in clean_condition.lower():
                # Extract price value
                price_match = extract_price_value(clean_condition)
                if price_match:
                    condition_descriptions.append(f"PriceBelow({price_match})")
                else:
                    condition_descriptions.append("PriceBelow")
            elif "price_equals" in clean_condition.lower():
                # Extract price value
                price_match = extract_price_value(clean_condition)
                if price_match:
                    condition_descriptions.append(f"PriceEquals({price_match})")
                else:
                    condition_descriptions.append("PriceEquals")
            elif "volume" in clean_condition.lower():
                # Extract volume multiplier or threshold
                volume_match = extract_volume_value(clean_condition)
                if volume_match:
                    condition_descriptions.append(f"Volume({volume_match})")
                else:
                    condition_descriptions.append("Volume")
            elif "breakout" in clean_condition.lower():
                condition_descriptions.append("Breakout")
            elif "close" in clean_condition.lower():
                condition_descriptions.append("Close")
            elif "open" in clean_condition.lower():
                condition_descriptions.append("Open")
            elif "high" in clean_condition.lower():
                condition_descriptions.append("High")
            elif "low" in clean_condition.lower():
                condition_descriptions.append("Low")
            else:
                # For custom conditions, take the first meaningful part
                parts = clean_condition.split('(')[0].split('[')[0].strip()
                if parts and len(parts) > 0:
                    # Clean up the part and capitalize
                    clean_part = parts.replace('_', ' ').replace('-', ' ').strip()
                    if clean_part:
                        condition_descriptions.append(clean_part.capitalize())
                    else:
                        condition_descriptions.append("Custom")
                else:
                    condition_descriptions.append("Custom")

    # Remove duplicates while preserving order
    unique_conditions = []
    for condition in condition_descriptions:
        if condition not in unique_conditions:
            unique_conditions.append(condition)

    # Create the alert name
    if len(unique_conditions) == 1:
        condition_part = unique_conditions[0]
    else:
        # Multiple conditions - use combination logic
        if combination_logic == "AND":
            condition_part = f"{'&'.join(unique_conditions)}"
        elif combination_logic == "OR":
            condition_part = f"{'|'.join(unique_conditions)}"
        else:
            condition_part = f"{'&'.join(unique_conditions)}"

    return f"{stock_name} - {condition_part} Alert"








# Indicator Guide and Condition Syntax
def get_indicator_guide():
    """Returns a comprehensive guide of available indicators and their conditions"""
    return {
        "Price-Based Conditions": {
            "description": "Basic price comparison conditions",
            "conditions": {
                "price_above": {
                    "syntax": "price > value",
                    "description": "Current price is above specified value",
                    "example": "price_above: 150.00"
                },
                "price_below": {
                    "syntax": "price < value",
                    "description": "Current price is below specified value",
                    "example": "price_below: 100.00"
                },
                "price_equals": {
                    "syntax": "price == value",
                    "description": "Current price equals specified value",
                    "example": "price_equals: 125.50"
                }
            }
        },
        "Moving Averages": {
            "description": "Trend-following indicators based on moving averages",
            "conditions": {
                "price_above_ma": {
                    "syntax": "price > MA(period)",
                    "description": "Price is above moving average",
                    "example": "price_above_ma: 20 (SMA)"
                },
                "price_below_ma": {
                    "syntax": "price < MA(period)",
                    "description": "Price is below moving average",
                    "example": "price_below_ma: 50 (EMA)"
                },
                "ma_crossover": {
                    "syntax": "fast_MA > slow_MA",
                    "description": "Fast moving average crosses above slow moving average",
                    "example": "ma_crossover: 10 > 20"
                }
            }
        },
        "Volume Indicators": {
            "description": "Volume-based analysis indicators",
            "conditions": {
                "volume_above_average": {
                    "syntax": "volume > avg_volume",
                    "description": "Current volume is above average",
                    "example": "volume_above_average: 1.5x"
                },
                "volume_spike": {
                    "syntax": "volume > threshold",
                    "description": "Volume spike above threshold",
                    "example": "volume_spike: 2.0x average"
                }
            }
        },
        "RSI (Relative Strength Index)": {
            "description": "Momentum oscillator measuring speed and magnitude of price changes",
            "conditions": {
                "rsi_oversold": {
                    "syntax": "RSI < 30",
                    "description": "RSI indicates oversold condition",
                    "example": "rsi_oversold: 30"
                },
                "rsi_overbought": {
                    "syntax": "RSI > 70",
                    "description": "RSI indicates overbought condition",
                    "example": "rsi_overbought: 70"
                },
                "rsi_divergence": {
                    "syntax": "Price makes new high, RSI doesn't",
                    "description": "Bearish divergence between price and RSI",
                    "example": "rsi_divergence: bearish"
                }
            }
        },
        "MACD (Moving Average Convergence Divergence)": {
            "description": "Trend-following momentum indicator",
            "conditions": {
                "macd_bullish_crossover": {
                    "syntax": "MACD line > Signal line",
                    "description": "MACD line crosses above signal line",
                    "example": "macd_bullish_crossover"
                },
                "macd_bearish_crossover": {
                    "syntax": "MACD line < Signal line",
                    "description": "MACD line crosses below signal line",
                    "example": "macd_bearish_crossover"
                },
                "macd_histogram_positive": {
                    "syntax": "MACD histogram > 0",
                    "description": "MACD histogram is positive",
                    "example": "macd_histogram_positive"
                }
            }
        },
        "Bollinger Bands": {
            "description": "Volatility indicator with upper and lower bands",
            "conditions": {
                "price_above_upper_band": {
                    "syntax": "price > upper_band",
                    "description": "Price is above upper Bollinger Band",
                    "example": "price_above_upper_band"
                },
                "price_below_lower_band": {
                    "syntax": "price < lower_band",
                    "description": "Price is below lower Bollinger Band",
                    "example": "price_below_lower_band"
                },
                "bands_squeeze": {
                    "syntax": "band_width < threshold",
                    "description": "Bollinger Bands are squeezed (low volatility)",
                    "example": "bands_squeeze: 0.1"
                }
            }
        },
        "Cross-Exchange Ratio": {
            "description": "Ratio between two stocks from different exchanges",
            "conditions": {
                "ratio_above": {
                    "syntax": "stock1/stock2 > value",
                    "description": "Ratio is above specified value",
                    "example": "ratio_above: 1.5"
                },
                "ratio_below": {
                    "syntax": "stock1/stock2 < value",
                    "description": "Ratio is below specified value",
                    "example": "ratio_below: 0.8"
                },
                "ratio_equals": {
                    "syntax": "stock1/stock2 == value",
                    "description": "Ratio equals specified value",
                    "example": "ratio_equals: 1.0"
                }
            }
        },
        "Multi-Timeframe Analysis": {
            "description": "Compare daily price movements against weekly trends and values",
            "conditions": {
                "daily_price_above_weekly_ma": {
                    "syntax": "daily_price > weekly_MA(period)",
                    "description": "Daily price is above weekly moving average",
                    "example": "daily_price_above_weekly_ma: 20"
                },
                "daily_price_below_weekly_ma": {
                    "syntax": "daily_price < weekly_MA(period)",
                    "description": "Daily price is below weekly moving average",
                    "example": "daily_price_below_weekly_ma: 50"
                },
                "daily_ma_above_weekly_ma": {
                    "syntax": "daily_MA > weekly_MA",
                    "description": "Daily moving average is above weekly moving average",
                    "example": "daily_ma_above_weekly_ma: 10 > 20"
                },
                "daily_price_above_weekly_high": {
                    "syntax": "daily_price > weekly_high",
                    "description": "Daily price breaks above weekly high",
                    "example": "daily_price_above_weekly_high"
                },
                "daily_price_below_weekly_low": {
                    "syntax": "daily_price < weekly_low",
                    "description": "Daily price breaks below weekly low",
                    "example": "daily_price_below_weekly_low"
                },
                "daily_rsi_vs_weekly_rsi": {
                    "syntax": "daily_RSI > weekly_RSI",
                    "description": "Daily RSI is stronger than weekly RSI",
                    "example": "daily_rsi_vs_weekly_rsi: bullish"
                },
                "daily_macd_vs_weekly_macd": {
                    "syntax": "daily_MACD > weekly_MACD",
                    "description": "Daily MACD is stronger than weekly MACD",
                    "example": "daily_macd_vs_weekly_macd: bullish"
                }
            }
        },
        "Mixed Timeframe Conditions": {
            "description": "Combine different indicators using daily AND weekly data",
            "conditions": {
                "daily_rsi_overbought": {
                    "syntax": "rsi[0] > 70",
                    "description": "Daily RSI overbought condition",
                    "example": "rsi[0] > 70"
                },
                "weekly_ma_bullish": {
                    "syntax": "weekly_MA(fast) > weekly_MA(slow)",
                    "description": "Weekly moving average bullish",
                    "example": "weekly_sma(20)[0] > weekly_sma(50)[0]"
                },
                "daily_volume_spike": {
                    "syntax": "volume[0] > volume[1] * 1.5",
                    "description": "Daily volume spike",
                    "example": "volume[0] > volume[1] * 1.5"
                },
                "weekly_new_high": {
                    "syntax": "weekly_high[0] > weekly_high[1]",
                    "description": "Weekly new high",
                    "example": "weekly_high[0] > weekly_high[1]"
                }
            }
        }
    }

def display_indicator_guide():
    """Displays the comprehensive indicator guide in an expandable section"""
    with st.expander("📊 Indicator Guide & Condition Syntax", expanded=False):
        st.markdown("### Available Indicators and Conditions")
        st.markdown("This guide shows all available indicators and their conditions with proper syntax.")

        guide = get_indicator_guide()

        for category, data in guide.items():
            st.markdown(f"#### {category}")
            st.markdown(f"*{data['description']}*")

            # Create columns for better layout
            col1, col2, col3 = st.columns([1, 2, 2])

            with col1:
                st.markdown("**Condition**")
            with col2:
                st.markdown("**Syntax**")
            with col3:
                st.markdown("**Description**")

            for condition, details in data['conditions'].items():
                with col1:
                    st.markdown(f"`{condition}`")
                with col2:
                    # Use syntax if available, otherwise use example format
                    if 'syntax' in details:
                        st.markdown(f"`{details['syntax']}`")
                    elif 'example' in details:
                        # Extract syntax pattern from example
                        example = details['example']
                        if ':' in example:
                            syntax = example.split(':')[0].strip()
                            st.markdown(f"`{syntax}`")
                        else:
                            st.markdown(f"`{condition}`")
                    else:
                        st.markdown(f"`{condition}`")
                with col3:
                    st.markdown(f"{details.get('description', 'No description available')}")
                    if 'example' in details:
                        st.caption(f"Example: {details['example']}")

            st.markdown("---")

    catalog_path = Path(__file__).resolve().parent.parent / "docs" / "add_alert_indicator_catalog.md"
    with st.expander("📄 Dropdown Catalog (Add Alert)", expanded=False):
        if catalog_path.exists():
            try:
                st.markdown(catalog_path.read_text(encoding="utf-8"))
            except Exception:
                st.warning("Could not load the dropdown catalog file.")
        else:
            st.warning("Catalog file not found.")

# Cache market data loading for better performance
@st.cache_data(ttl=60)  # Cache for 1 minute to ensure fresh data
def load_cached_market_data():
    return load_market_data()

# Load market data with caching
market_data = load_cached_market_data()

# Debug: Check if industry columns exist
if 'RBICS_Sector' not in market_data.columns:
    st.warning("RBICS classification columns not found in market data. Available columns: " + str(market_data.columns.tolist()))
    # Fallback to basic data without industry filtering
    filtered_stocks_data = market_data
else:
    st.success(f"✅ Industry data loaded successfully. Found {len(market_data)} stocks with industry classifications.")

# Load industry filters
@st.cache_data(ttl=600)
def load_industry_filters():
    data = load_document(
        "industry_filters",
        default={},
        fallback_path='industry_filters.json',
    )
    return data if isinstance(data, dict) else {}


@st.cache_data(ttl=600)
def load_futures_database():
    data = load_document(
        "futures_database",
        default={},
        fallback_path='futures_database.json',
    )
    return data if isinstance(data, dict) else {}

# Asset Type Selection
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Asset Type")
selected_asset_type = st.sidebar.selectbox(
    "Select asset type:",
    ["All", "Stocks", "ETFs", "Futures"],
    help="Choose the type of asset for your alert"
)

# Initialize filter variables
selected_countries = []
selected_exchanges = []
selected_economies = []
selected_sectors = []
selected_subsectors = []
selected_industry_groups = []
selected_industries = []
selected_subindustries = []
selected_issuers = []
selected_asset_classes = []
selected_focuses = []
selected_niches = []

# Futures Filters
if selected_asset_type == "Futures":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Futures Categories")

    futures_db = load_futures_database()
    if futures_db:
        futures_categories = {}
        for symbol, data in futures_db.items():
            category = data.get('category', 'Other')
            futures_categories.setdefault(category, []).append(symbol)

        selected_futures_categories = st.sidebar.multiselect(
            "Select Futures Categories:",
            list(futures_categories.keys()),
            default=[],
            help="Select futures categories to display"
        )

        available_futures = []
        if selected_futures_categories:
            for category in selected_futures_categories:
                available_futures.extend(futures_categories.get(category, []))
        else:
            available_futures = list(futures_db.keys())

        st.sidebar.info(f"Available: {len(available_futures)} futures contracts")
    else:
        st.sidebar.error("Error loading futures database from PostgreSQL.")
        available_futures = []

# Country and Exchange Filters - available for stocks and ETFs
elif selected_asset_type in ["All", "Stocks", "ETFs"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Location Filters")

    # Country Filter
    if 'Country' in market_data.columns:
        available_countries = sorted(market_data['Country'].dropna().unique())
        selected_countries = st.sidebar.multiselect(
            "Filter by Country:",
            available_countries,
            default=[],
            help="Select countries to filter symbols"
        )

    # Exchange Filter - cascading from country
    if 'Exchange' in market_data.columns:
        if selected_countries:
            available_exchanges = sorted(market_data[market_data['Country'].isin(selected_countries)]['Exchange'].dropna().unique())
        else:
            available_exchanges = sorted(market_data['Exchange'].dropna().unique())

        selected_exchanges = st.sidebar.multiselect(
            "Filter by Exchange:",
            available_exchanges,
            default=[],
            help="Select exchanges to filter symbols"
        )

# Stock Industry Filters
if selected_asset_type in ["All", "Stocks"] and 'RBICS_Sector' in market_data.columns:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏭 Stock Industry Filters")

    # Only show stocks (not ETFs) for stock filters - use Asset_Type field
    if 'Asset_Type' in market_data.columns:
        stocks_only_data = market_data[market_data['Asset_Type'] == 'Stock']
    else:
        # Fallback to old method if Asset_Type not available
        stocks_only_data = market_data[~market_data['ETF_Issuer'].notna()] if 'ETF_Issuer' in market_data.columns else market_data

    # Apply country and exchange filters if selected
    if selected_countries and 'Country' in stocks_only_data.columns:
        stocks_only_data = stocks_only_data[stocks_only_data['Country'].isin(selected_countries)]
    if selected_exchanges and 'Exchange' in stocks_only_data.columns:
        stocks_only_data = stocks_only_data[stocks_only_data['Exchange'].isin(selected_exchanges)]

    # Economy filter
    available_economies = stocks_only_data['RBICS_Economy'].dropna().unique() if 'RBICS_Economy' in stocks_only_data.columns else []
    selected_economies = st.sidebar.multiselect(
        "Filter by Economy:",
        sorted(available_economies),
        default=[],
        help="Select economies to filter stocks"
    )

    # Sector filter - cascading from economy
    if selected_economies and 'RBICS_Economy' in stocks_only_data.columns:
        # Filter sectors based on selected economies
        available_sectors = stocks_only_data[stocks_only_data['RBICS_Economy'].isin(selected_economies)]['RBICS_Sector'].dropna().unique()
        available_sectors = sorted(available_sectors)
    else:
        available_sectors = stocks_only_data['RBICS_Sector'].dropna().unique() if 'RBICS_Sector' in stocks_only_data.columns else []

    selected_sectors = st.sidebar.multiselect(
        "Filter by Sector:",
        sorted(available_sectors),
        default=[],
        help="Select sectors to filter stocks"
    )

    # Subsector filter - cascading from sector
    if selected_sectors and 'RBICS_Sector' in stocks_only_data.columns:
        # Filter subsectors based on selected sectors
        available_subsectors = stocks_only_data[stocks_only_data['RBICS_Sector'].isin(selected_sectors)]['RBICS_Subsector'].dropna().unique() if 'RBICS_Subsector' in stocks_only_data.columns else []
        available_subsectors = sorted(available_subsectors)
    else:
        available_subsectors = stocks_only_data['RBICS_Subsector'].dropna().unique() if 'RBICS_Subsector' in stocks_only_data.columns else []

    selected_subsectors = st.sidebar.multiselect(
        "Filter by Subsector:",
        sorted(available_subsectors),
        default=[],
        help="Select subsectors to filter stocks"
    )

    # Industry Group filter - cascading from subsector
    if selected_subsectors and 'RBICS_Subsector' in stocks_only_data.columns:
        # Filter industry groups based on selected subsectors
        available_industry_groups = stocks_only_data[stocks_only_data['RBICS_Subsector'].isin(selected_subsectors)]['RBICS_Industry_Group'].dropna().unique() if 'RBICS_Industry_Group' in stocks_only_data.columns else []
        available_industry_groups = sorted(available_industry_groups)
    else:
        available_industry_groups = stocks_only_data['RBICS_Industry_Group'].dropna().unique() if 'RBICS_Industry_Group' in stocks_only_data.columns else []

    selected_industry_groups = st.sidebar.multiselect(
        "Filter by Industry Group:",
        sorted(available_industry_groups),
        default=[],
        help="Select industry groups to filter stocks"
    )

    # Industry filter - cascading from industry group
    if selected_industry_groups and 'RBICS_Industry_Group' in stocks_only_data.columns:
        # Filter industries based on selected industry groups
        available_industries = stocks_only_data[stocks_only_data['RBICS_Industry_Group'].isin(selected_industry_groups)]['RBICS_Industry'].dropna().unique() if 'RBICS_Industry' in stocks_only_data.columns else []
        available_industries = sorted(available_industries)
    else:
        available_industries = stocks_only_data['RBICS_Industry'].dropna().unique() if 'RBICS_Industry' in stocks_only_data.columns else []

    selected_industries = st.sidebar.multiselect(
        "Filter by Industry:",
        sorted(available_industries),
        default=[],
        help="Select industries to filter stocks"
    )

    # Subindustry filter - cascading from industry
    if selected_industries and 'RBICS_Industry' in stocks_only_data.columns:
        # Filter subindustries based on selected industries
        available_subindustries = stocks_only_data[stocks_only_data['RBICS_Industry'].isin(selected_industries)]['RBICS_Subindustry'].dropna().unique() if 'RBICS_Subindustry' in stocks_only_data.columns else []
        available_subindustries = sorted(available_subindustries)
    else:
        available_subindustries = stocks_only_data['RBICS_Subindustry'].dropna().unique() if 'RBICS_Subindustry' in stocks_only_data.columns else []

    selected_subindustries = st.sidebar.multiselect(
        "Filter by Subindustry:",
        sorted(available_subindustries),
        default=[],
        help="Select subindustries to filter stocks"
    )

# ETF Filters
if selected_asset_type in ["All", "ETFs"] and 'ETF_Issuer' in market_data.columns:
    st.sidebar.markdown("---")
    st.sidebar.subheader("💼 ETF Filters")

    # Only show ETFs - use Asset_Type field
    if 'Asset_Type' in market_data.columns:
        etfs_only_data = market_data[market_data['Asset_Type'] == 'ETF']
    else:
        # Fallback to old method if Asset_Type not available
        etfs_only_data = market_data[market_data['ETF_Issuer'].notna()] if 'ETF_Issuer' in market_data.columns else market_data

    # Apply country and exchange filters if selected
    if selected_countries and 'Country' in etfs_only_data.columns:
        etfs_only_data = etfs_only_data[etfs_only_data['Country'].isin(selected_countries)]
    if selected_exchanges and 'Exchange' in etfs_only_data.columns:
        etfs_only_data = etfs_only_data[etfs_only_data['Exchange'].isin(selected_exchanges)]

    # ETF Issuer filter
    available_issuers = etfs_only_data['ETF_Issuer'].dropna().unique() if 'ETF_Issuer' in etfs_only_data.columns else []
    selected_issuers = st.sidebar.multiselect(
        "Filter by ETF Issuer:",
        sorted(available_issuers),
        default=[],
        help="Select ETF issuers/providers"
    )

    # Asset Class filter
    if 'Asset_Class' in etfs_only_data.columns:
        available_asset_classes = etfs_only_data['Asset_Class'].dropna().unique()
        selected_asset_classes = st.sidebar.multiselect(
            "Filter by Asset Class:",
            sorted(available_asset_classes),
            default=[],
            help="Select asset classes"
        )
    else:
        selected_asset_classes = []

    # ETF Focus filter
    if 'ETF_Focus' in etfs_only_data.columns:
        if selected_asset_classes:
            available_focuses = etfs_only_data[etfs_only_data['Asset_Class'].isin(selected_asset_classes)]['ETF_Focus'].dropna().unique()
        else:
            available_focuses = etfs_only_data['ETF_Focus'].dropna().unique()
        selected_focuses = st.sidebar.multiselect(
            "Filter by ETF Focus:",
            sorted(available_focuses),
            default=[],
            help="Select ETF focus areas"
        )
    else:
        selected_focuses = []

    # ETF Niche filter
    if 'ETF_Niche' in etfs_only_data.columns:
        if selected_focuses:
            available_niches = etfs_only_data[etfs_only_data['ETF_Focus'].isin(selected_focuses)]['ETF_Niche'].dropna().unique()
        else:
            available_niches = etfs_only_data['ETF_Niche'].dropna().unique()
        selected_niches = st.sidebar.multiselect(
            "Filter by ETF Niche:",
            sorted(available_niches),
            default=[],
            help="Select ETF niche categories"
        )
    else:
        selected_niches = []

# Apply filters based on selected asset type
if selected_asset_type == "Futures":
    # Load futures from JSON database instead of using IB connection
    futures_db = load_futures_database()
    futures_list = []
    if futures_db:
        for symbol, data in futures_db.items():
            futures_list.append({
                'Name': data.get('long_name', data.get('name', symbol)),
                'Symbol': symbol,
                'Exchange': data.get('exchange', 'Futures'),
                'Country': data.get('country', 'UNITED STATES'),
                'Asset_Type': 'Future'
            })
    else:
        st.error("Error loading futures database from PostgreSQL.")

    filtered_stocks_data = pd.DataFrame(futures_list)

elif selected_asset_type == "All":
    # Start with all data
    filtered_stocks_data = market_data.copy()

    # Apply location filters first
    if selected_countries and 'Country' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[filtered_stocks_data['Country'].isin(selected_countries)]
    if selected_exchanges and 'Exchange' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[filtered_stocks_data['Exchange'].isin(selected_exchanges)]

elif selected_asset_type == "Stocks":
    # Start with stocks only - use Asset_Type field
    if 'Asset_Type' in market_data.columns:
        filtered_stocks_data = market_data[market_data['Asset_Type'] == 'Stock']
    else:
        # Fallback to old method
        filtered_stocks_data = market_data[~market_data['ETF_Issuer'].notna()] if 'ETF_Issuer' in market_data.columns else market_data

    # Apply location filters
    if selected_countries and 'Country' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[filtered_stocks_data['Country'].isin(selected_countries)]
    if selected_exchanges and 'Exchange' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[filtered_stocks_data['Exchange'].isin(selected_exchanges)]

    if selected_economies and 'RBICS_Economy' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['RBICS_Economy'].isin(selected_economies)
        ]

    if selected_sectors and 'RBICS_Sector' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['RBICS_Sector'].isin(selected_sectors)
        ]

    if selected_subsectors and 'RBICS_Subsector' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['RBICS_Subsector'].isin(selected_subsectors)
        ]

    if selected_industry_groups and 'RBICS_Industry_Group' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['RBICS_Industry_Group'].isin(selected_industry_groups)
        ]

    if selected_industries and 'RBICS_Industry' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['RBICS_Industry'].isin(selected_industries)
        ]

    if selected_subindustries and 'RBICS_Subindustry' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['RBICS_Subindustry'].isin(selected_subindustries)
        ]

    if any([selected_economies, selected_sectors, selected_subsectors, selected_industry_groups, selected_industries, selected_subindustries]):
        st.sidebar.info(f"Showing {len(filtered_stocks_data)} stocks after filtering")
    else:
        st.sidebar.info(f"Showing all {len(filtered_stocks_data)} stocks")

elif selected_asset_type == "ETFs":
    # Start with ETFs only - use Asset_Type field
    if 'Asset_Type' in market_data.columns:
        filtered_stocks_data = market_data[market_data['Asset_Type'] == 'ETF']
    else:
        # Fallback to old method
        filtered_stocks_data = market_data[market_data['ETF_Issuer'].notna()] if 'ETF_Issuer' in market_data.columns else market_data

    # Apply location filters
    if selected_countries and 'Country' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[filtered_stocks_data['Country'].isin(selected_countries)]
    if selected_exchanges and 'Exchange' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[filtered_stocks_data['Exchange'].isin(selected_exchanges)]

    if selected_issuers and 'ETF_Issuer' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['ETF_Issuer'].isin(selected_issuers)
        ]

    if selected_asset_classes and 'Asset_Class' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['Asset_Class'].isin(selected_asset_classes)
        ]

    if selected_focuses and 'ETF_Focus' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['ETF_Focus'].isin(selected_focuses)
        ]

    if selected_niches and 'ETF_Niche' in filtered_stocks_data.columns:
        filtered_stocks_data = filtered_stocks_data[
            filtered_stocks_data['ETF_Niche'].isin(selected_niches)
        ]

    if any([selected_issuers, selected_asset_classes, selected_focuses, selected_niches]):
        st.sidebar.info(f"Showing {len(filtered_stocks_data)} ETFs after filtering")
    else:
        st.sidebar.info(f"Showing all {len(filtered_stocks_data)} ETFs")

else:
    filtered_stocks_data = market_data

# Main page structure with tabs for individual and bulk alert creation
st.title("➕ Add Alert")

# Display symbol count based on current filters
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    if selected_asset_type == "Stocks":
        st.metric(
            "Available Stocks",
            f"{len(filtered_stocks_data):,}",
            delta=f"{len(filtered_stocks_data) - 5183:,} from filters" if len(filtered_stocks_data) != 5183 else None
        )
    elif selected_asset_type == "ETFs":
        st.metric(
            "Available ETFs",
            f"{len(filtered_stocks_data):,}",
            delta=f"{len(filtered_stocks_data) - 1712:,} from filters" if len(filtered_stocks_data) != 1712 else None
        )
    elif selected_asset_type == "Futures":
        st.metric(
            "Available Futures",
            f"{len(filtered_stocks_data):,}",
            delta=None
        )
    elif selected_asset_type == "All":
        total_stocks = len(market_data[market_data['Asset_Type'] == 'Stock']) if 'Asset_Type' in market_data.columns else 5183
        total_etfs = len(market_data[market_data['Asset_Type'] == 'ETF']) if 'Asset_Type' in market_data.columns else 1712
        all_futures_count = len(load_futures_database())
        total_all = total_stocks + total_etfs + all_futures_count
        st.metric(
            "All Asset Types",
            f"{total_all:,}",
            delta=f"{total_stocks:,} Stocks, {total_etfs:,} ETFs, {all_futures_count} Futures"
        )
with col2:
    # Show active filters count
    active_filters = []
    if selected_asset_type == "Stocks":
        if selected_economies: active_filters.append(f"{len(selected_economies)} economies")
        if selected_sectors: active_filters.append(f"{len(selected_sectors)} sectors")
        if selected_subsectors: active_filters.append(f"{len(selected_subsectors)} subsectors")
        if selected_industry_groups: active_filters.append(f"{len(selected_industry_groups)} groups")
        if selected_industries: active_filters.append(f"{len(selected_industries)} industries")
        if selected_subindustries: active_filters.append(f"{len(selected_subindustries)} subindustries")
    elif selected_asset_type == "ETFs":
        if selected_issuers: active_filters.append(f"{len(selected_issuers)} issuers")
        if selected_asset_classes: active_filters.append(f"{len(selected_asset_classes)} asset classes")
        if selected_focuses: active_filters.append(f"{len(selected_focuses)} focuses")
        if selected_niches: active_filters.append(f"{len(selected_niches)} niches")
    elif selected_asset_type == "Futures":
        if 'selected_futures_categories' in locals() and selected_futures_categories:
            active_filters.append(f"{len(selected_futures_categories)} categories")

    if active_filters:
        st.info("🔍 **Active Filters:** " + ", ".join(active_filters))
    else:
        st.info("🔍 **No filters applied** - showing all available symbols")

with col3:
    # Clear filters button
    if any([selected_economies, selected_sectors, selected_subsectors, selected_industry_groups, selected_industries, selected_subindustries,
            selected_issuers, selected_asset_classes, selected_focuses, selected_niches]):
        if st.button("🗑️ Clear Filters", use_container_width=True):
            st.rerun()

# Inform users about duplicate handling
st.info("""
💡 **Smart Duplicate Detection**: The system now allows you to create multiple alerts for the same stock with the same conditions,
as long as they have different names. If similar alerts already exist, you'll see suggestions to help you decide whether to
create a new one or update an existing one.
""")

# Removed Quick Bulk Operations section

# Create tab for individual alert creation
tab1 = st.tabs(["🎯 Individual Alert"])[0]

with tab1:
    st.subheader("Create Individual Alert")

    # Initialize selected_stocks for this tab
    selected_stocks = []

    # Display the comprehensive indicator guide
    display_indicator_guide()

    # Quick Reference for Common Conditions
    with st.expander("⚡ Quick Reference - Common Conditions", expanded=False):
        st.markdown("### Most Commonly Used Conditions")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Price Conditions**")
            st.markdown("- `price_above: 150.00` - Price > $150")
            st.markdown("- `price_below: 100.00` - Price < $100")
            st.markdown("- `price_equals: 125.50` - Price = $125.50")

            st.markdown("**Moving Averages**")
            st.markdown("- `price_above_ma: 20` - Price above 20-period SMA")
            st.markdown("- `price_below_ma: 50` - Price below 50-period EMA")
            st.markdown("- `ma_crossover: 10 > 20` - Fast MA crosses above slow MA")

        with col2:
            st.markdown("**RSI Conditions**")
            st.markdown("- `rsi_oversold: 30` - RSI < 30 (oversold)")
            st.markdown("- `rsi_overbought: 70` - RSI > 70 (overbought)")

            st.markdown("**MACD Conditions**")
            st.markdown("- `macd_bullish_crossover` - MACD line > Signal line")
            st.markdown("- `macd_bearish_crossover` - MACD line < Signal line")

            st.markdown("**Multi-Timeframe**")
            st.markdown("- `daily_price_above_weekly_ma: 20` - Daily price > Weekly MA")
            st.markdown("- `daily_price_above_weekly_high` - Daily price > Weekly high")
            st.markdown("- `daily_rsi_vs_weekly_rsi: bullish` - Daily RSI > Weekly RSI")

        st.markdown("---")
        st.markdown("**💡 Tip**: Use these conditions in the 'Add Condition' section below. You can combine multiple conditions with AND/OR logic.")

    # Available Indicators Overview
    with st.expander("🎯 Available Indicators Overview", expanded=False):
        st.markdown("### What Each Indicator Does")

        indicators_info = {
            "📈 **Price-Based**": "Basic price comparisons (above, below, equals)",
            "📊 **Moving Averages**": "Trend following (SMA, EMA, crossovers)",
            "📈 **RSI**": "Momentum oscillator (oversold/overbought)",
            "📊 **MACD**": "Trend and momentum (crossovers, histogram)",
            "📈 **Bollinger Bands**": "Volatility and price levels",
            "📊 **Donchian Channels**": "Highest high/lowest low breakout system",
            "⚖️ **Cross-Exchange Ratio**": "Ratio between different market stocks",
            "🔄 **Multi-Timeframe Analysis**": "Compare daily vs weekly trends and values"
        }

        for indicator, description in indicators_info.items():
            st.markdown(f"**{indicator}**: {description}")

        st.markdown("---")
        st.markdown("**🔧 Advanced Features**:")
        st.markdown("- Combine multiple conditions with AND/OR logic")
        st.markdown("- Set custom parameters for each indicator")
        st.markdown("- Use different timeframes (daily, weekly)")
        st.markdown("- Cross-exchange ratio alerts for global markets")
        st.markdown("- Multi-timeframe analysis (daily vs weekly comparisons)")

    #ask for the name of the alert
    alert_name = st.text_input("Enter the name of the alert (optional - will use stock name + conditions if not provided):")

    # Show alert name preview if conditions are added and no custom name provided
    if st.session_state.entry_conditions and not alert_name:
        st.markdown("### 📝 Alert Name Preview")
        st.markdown("**Example alert names that will be generated:**")

        # Get a sample stock name for preview
        sample_stock = "Apple Inc."  # Default sample
        if selected_stocks and len(selected_stocks) == 1:
            sample_stock = selected_stocks[0]
        elif selected_stocks and len(selected_stocks) > 1:
            sample_stock = selected_stocks[0]  # Use first selected stock for preview

        preview_name = generate_alert_name_from_conditions(
            sample_stock,
            st.session_state.entry_conditions,
            st.session_state.entry_combination
        )

        st.info(f"**Example:** `{preview_name}`")
        st.markdown("💡 *This naming pattern will be applied to all selected stocks. You can override this by entering a custom alert name above.*")

    # Exchange filtering now handled by sidebar filter - no need for exchange schedule loading

    # Select whether to buy or sell
    action = st.selectbox("Select Action:", ["Buy", "Sell"], key="individual_action")

    # Use the asset type selected in the sidebar
    asset_type = selected_asset_type

    # Display selected asset type
    st.info(f"📈 Creating alert for: **{asset_type}**")

    ratio = st.selectbox("Ratio of 2 assets?", ["No", "Yes"], key="individual_ratio")

    # Only show single asset selection if ratio is "No"
    if ratio == "No":
        if asset_type in ["Stocks", "ETFs", "All", "Futures"]:
            # Check if we have any data after filters
            if filtered_stocks_data.empty:
                st.warning(f"⚠️ No {asset_type.lower()} found matching the current filters. Please adjust your filters in the sidebar.")
                selected_stocks = []
            else:
                # Simplified logic - no redundant exchange selector
                # Show info about available stocks
                asset_type_display = "stocks and ETFs" if asset_type == "All" else asset_type.lower()
                st.info(f"📊 Found {len(filtered_stocks_data)} {asset_type_display} matching your filters")

                # Special handling for creating alerts on ALL assets
                # Check if no specific filters are applied
                no_filters_applied = (
                    selected_asset_type == "All" and
                    not selected_countries and
                    not selected_exchanges and
                    not selected_economies and
                    not selected_sectors and
                    not selected_subsectors and
                    not selected_industry_groups and
                    not selected_industries and
                    not selected_subindustries and
                    not selected_issuers and
                    not selected_asset_classes and
                    not selected_focuses and
                    not selected_niches
                )

                # Checkbox to apply to all stocks - default behavior based on context
                if no_filters_applied and len(filtered_stocks_data) > 1000:
                    # For truly ALL stocks with no filters, default to unchecked due to volume
                    default_apply_all = False
                    st.warning(f"⚠️ You have {len(filtered_stocks_data)} {asset_type_display} with no filters. Consider adding filters to reduce the number.")
                else:
                    # For filtered results or smaller sets, use intelligent default
                    default_apply_all = True if len(filtered_stocks_data) <= 100 else False

                apply_to_all = st.checkbox(
                    f"Apply to ALL {len(filtered_stocks_data)} {asset_type_display}",
                    value=default_apply_all,
                    key="apply_to_all_stocks",
                    help="Check this to create alerts for all stocks/ETFs matching your current filters"
                )

                if apply_to_all:
                    selected_stocks = filtered_stocks_data["Name"].tolist()
                    st.success(f"✅ Alert will be applied to ALL {len(selected_stocks)} {asset_type_display}")
                    if len(selected_stocks) > 100:
                        st.warning(f"⚠️ **Note:** Creating alerts for {len(selected_stocks)} {asset_type_display} may take a very long time. Consider using more specific filters in the sidebar.")
                    if len(selected_stocks) > 500:
                        st.error(f"⚠️ **WARNING:** Creating {len(selected_stocks)} alerts is likely to fail. Please use filters to reduce the number.")
                else:
                    # Manual selection of specific stocks
                    # Create display names that include ticker symbols and exchange info
                    filtered_stocks_display = []
                    for _, row in filtered_stocks_data.iterrows():
                        exchange_name = row.get('Exchange', row.get('Country', 'Unknown'))
                        filtered_stocks_display.append(f"{row['Name']} ({row['Symbol']}) - {exchange_name}")

                    selected_stocks_display = st.multiselect(
                        f"Select {asset_type_display} ({len(filtered_stocks_display)} available):",
                        filtered_stocks_display,
                        key="individual_securities"
                    )

                    # Convert display names back to stock names for processing
                    selected_stocks = []
                    for display_name in selected_stocks_display:
                        # Extract the stock name (everything before the ticker in parentheses)
                        stock_name = display_name.split(" (")[0]
                        selected_stocks.append(stock_name)

                    if selected_stocks:
                        st.info(f"✅ Alert will be applied to {len(selected_stocks)} selected {asset_type_display}")

    if ratio == "Yes":
        st.info("📊 **Ratio Alert**: Create alerts based on the ratio between two assets")

        # Simple asset type selection for ratio
        ratio_asset_type = st.selectbox(
            "Select Asset Type for Ratio:",
            ["Stocks", "ETFs", "Futures", "Mixed (Stock/ETF/Future)"],
            help="Choose whether to create a ratio between stocks, ETFs, futures, or mix different types"
        )

        # Initialize variables for ratio calculation
        first_stock_name = None
        first_ticker = None
        second_stock_name = None
        second_ticker = None
        first_country_code = None
        second_country_code = None

        # Create two columns for asset selection
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("First Asset")

            # Filter data based on asset type
            if ratio_asset_type == "Stocks":
                # Use stocks from filtered_stocks_data
                ratio_data = filtered_stocks_data[filtered_stocks_data['Asset_Type'] == 'Stock'] if 'Asset_Type' in filtered_stocks_data.columns else filtered_stocks_data[~filtered_stocks_data['ETF_Issuer'].notna()]
            else:
                # Use ETFs from filtered_stocks_data
                ratio_data = filtered_stocks_data[filtered_stocks_data['Asset_Type'] == 'ETF'] if 'Asset_Type' in filtered_stocks_data.columns else filtered_stocks_data[filtered_stocks_data['ETF_Issuer'].notna()]

            if ratio_asset_type in ["Stocks", "ETFs"]:
                # Get unique exchanges from the filtered data
                if 'Exchange' in ratio_data.columns:
                    available_exchanges = sorted(ratio_data['Exchange'].dropna().unique())

                    if not available_exchanges:
                        st.warning(f"⚠️ No {ratio_asset_type.lower()} found in the database.")
                        first_exchange = None
                    else:
                        first_exchange = st.selectbox(
                            "Select First Exchange:",
                            available_exchanges,
                            key="first_exchange_ratio"
                        )

                        # Filter assets for selected exchange
                        first_exchange_data = ratio_data[ratio_data['Exchange'] == first_exchange]

                        if first_exchange_data.empty:
                            st.warning(f"⚠️ No {ratio_asset_type.lower()} found in {first_exchange}.")
                            first_stock = None
                            first_stock_name = None
                            first_ticker = None
                        else:
                            # Create display names with symbol
                            first_stocks_display = [f"{row['Name']} ({row['Symbol']})" for _, row in first_exchange_data.iterrows()]
                            first_stock = st.selectbox(
                                f"Select First {ratio_asset_type[:-1]}:",
                                sorted(first_stocks_display),
                                key="first_stock_ratio"
                            )

                            # Extract ticker and name
                            if first_stock:
                                first_ticker = first_stock.split(" (")[1].rstrip(")")
                                first_stock_name = first_stock.split(" (")[0]
                                # Get country code for exchange
                                first_country_code = first_exchange_data.iloc[0]['Country'] if 'Country' in first_exchange_data.columns else None
                            else:
                                first_stock_name = None
                                first_ticker = None
                                first_country_code = None
                else:
                    st.error("Exchange information not available in database.")
                    first_stock_name = None
                    first_ticker = None
                    first_country_code = None

        with col2:
            st.subheader("Second Asset")

            # Second asset selection - similar to first
            if ratio_asset_type in ["Stocks", "ETFs"]:
                # Get unique exchanges from the filtered data
                if 'Exchange' in ratio_data.columns:
                    available_exchanges = sorted(ratio_data['Exchange'].dropna().unique())

                    if not available_exchanges:
                        st.warning(f"⚠️ No {ratio_asset_type.lower()} found in the database.")
                        second_exchange = None
                    else:
                        second_exchange = st.selectbox(
                            "Select Second Exchange:",
                            available_exchanges,
                            key="second_exchange_ratio"
                        )

                        # Filter assets for selected exchange
                        second_exchange_data = ratio_data[ratio_data['Exchange'] == second_exchange]

                        if second_exchange_data.empty:
                            st.warning(f"⚠️ No {ratio_asset_type.lower()} found in {second_exchange}.")
                            second_stock = None
                            second_stock_name = None
                            second_ticker = None
                        else:
                            # Create display names with symbol
                            second_stocks_display = [f"{row['Name']} ({row['Symbol']})" for _, row in second_exchange_data.iterrows()]
                            second_stock = st.selectbox(
                                f"Select Second {ratio_asset_type[:-1]}:",
                                sorted(second_stocks_display),
                                key="second_stock_ratio"
                            )

                            # Extract ticker and name
                            if second_stock:
                                second_ticker = second_stock.split(" (")[1].rstrip(")")
                                second_stock_name = second_stock.split(" (")[0]
                                # Get country code for exchange
                                second_country_code = second_exchange_data.iloc[0]['Country'] if 'Country' in second_exchange_data.columns else None
                            else:
                                second_stock_name = None
                                second_ticker = None
                                second_country_code = None
                else:
                    st.error("Exchange information not available in database.")
                    second_stock_name = None
                    second_ticker = None
                    second_country_code = None

        # Display selected assets only if both are selected
        if first_stock_name and second_stock_name and first_ticker and second_ticker:
            st.success(f"📊 **Ratio Alert**: {first_stock_name} ({first_ticker}) / {second_stock_name} ({second_ticker})")

            # Check if assets are from different exchanges
            if ratio_asset_type in ["Stocks", "ETFs"]:
                if first_country_code != second_country_code:
                    st.info("🌍 **Cross-Exchange Alert**: Assets are from different exchanges. This alert will be checked at each exchange's closing time.")
        else:
            st.warning("⚠️ Please select both assets to create a ratio alert.")

    # Section: Define Entry Conditions
    st.subheader("Entry Conditions")

    # Timeframe Selection with Multi-Timeframe Comparison
    st.markdown("### 📅 Timeframe Configuration")

    # Basic timeframe selection
    timeframe = st.selectbox(
        f"{bl_sp(1)}Select the required Timeframe",
        ["1h", "1d", "1wk"],
        index=1,  # Default to 1d (daily)
        key="timeframe_select",
        help="Hourly (1h), Daily (1d), or Weekly (1wk) timeframe"
    )

    # Multi-timeframe comparison options
    col1, col2 = st.columns(2)

    with col1:
        enable_multi_timeframe = st.checkbox(
            "🔄 Enable Multi-Timeframe Comparison",
            help="Compare daily price movements against weekly trends/values"
        )

    with col2:
        enable_mixed_timeframe = st.checkbox(
            "🎯 Enable Mixed Timeframe Conditions",
            help="Combine different indicators using daily AND weekly data"
        )

    if enable_multi_timeframe:
        st.markdown("#### 📊 Multi-Timeframe Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Primary Timeframe** (for current price)")
            primary_timeframe = st.selectbox(
                "Primary Timeframe:",
                ["1d", "1wk"],
                index=0 if timeframe == "1d" else 1,
                key="multi_tf_primary"
            )

        with col2:
            st.markdown("**Comparison Timeframe** (for historical reference)")
            comparison_timeframe = st.selectbox(
                "Comparison Timeframe:",
                ["1d", "1wk"],
                index=1 if timeframe == "1d" else 0,
                key="multi_tf_comparison"
            )

        # Configure multi-timeframe options
        if primary_timeframe == comparison_timeframe:
            st.warning("⚠️ Primary and comparison timeframes should be different for meaningful comparisons")

        multi_timeframe_info = st.info(
            f"📊 **Multi-Timeframe Analysis Enabled**\n\n"
            f"• **Primary**: {primary_timeframe} (for current values)\n"
            f"• **Comparison**: {comparison_timeframe} (for historical reference)\n\n"
            f"You can now use both timeframes in your conditions:\n"
            f"• `Close[-1]` uses {primary_timeframe} data\n"
            f"• `Close_weekly[-1]` uses {comparison_timeframe} data"
        )

    if enable_mixed_timeframe:
        st.markdown("#### 🎯 Mixed Timeframe Conditions")
        st.markdown("**Combine different indicators using daily AND weekly data**")

        mixed_timeframe_conditions = {
            "Daily Indicators": {
                "rsi[0] > 70": "Daily RSI overbought",
                "rsi[0] < 30": "Daily RSI oversold",
                "macd[0] > macd[1]": "Daily MACD bullish crossover",
                "volume[0] > volume[1] * 1.5": "Daily volume spike"
            },
            "Weekly Indicators": {
                "rsi_weekly[0] > 70": "Weekly RSI overbought",
                "rsi_weekly[0] < 30": "Weekly RSI oversold",
                "macd_weekly[0] > macd_weekly[1]": "Weekly MACD bullish crossover",
                "sma_weekly[0] > sma_weekly[1]": "Weekly SMA trending up"
            }
        }

        st.info(
            "💡 **Example Mixed Conditions:**\n\n"
            "• `rsi[0] < 30 AND rsi_weekly[0] > 50` - Daily oversold but weekly bullish\n"
            "• `Close[0] > sma[0] AND Close_weekly[0] > sma_weekly[0]` - Above SMA on both timeframes\n"
            "• `volume[0] > volume_avg * 2 AND macd_weekly[0] > 0` - Daily volume spike with weekly MACD positive"
        )

    # Futures Adjustment Method (only show for futures)
    adjustment_method = "none"  # Default value
    if selected_asset_type == "Futures" or (selected_asset_type == "All" and any("Futures" in str(row.get('Asset_Type', '')) for _, row in filtered_stocks_data.iterrows())):
        st.markdown("### 🔧 Futures Contract Adjustment")

        col1, col2 = st.columns([2, 1])
        with col1:
            adjustment_method = st.selectbox(
                "Select Price Adjustment Method:",
                ["none", "panama", "ratio"],
                format_func=lambda x: {
                    "none": "No Adjustment (Raw Prices)",
                    "panama": "Panama Method (Additive/Back-adjust)",
                    "ratio": "Ratio Method (Multiplicative/Proportional)"
                }[x],
                index=1,  # Default to Panama
                key="futures_adjustment_method",
                help="Choose how to adjust futures prices for contract rolls"
            )

        with col2:
            with st.expander("ℹ️ Learn More", expanded=False):
                st.markdown("""
                **Adjustment Methods Explained:**

                • **No Adjustment**: Uses raw contract prices as-is

                • **Panama (Additive)**:
                  - Preserves absolute price differences
                  - Best for indicators based on price levels
                  - Adjusts historical prices by adding/subtracting differences at roll points

                • **Ratio (Multiplicative)**:
                  - Preserves percentage returns
                  - Best for percentage-based analysis
                  - Adjusts historical prices by multiplying by ratios at roll points
                """)

    # Condition Builder Section
    st.markdown("### 📝 Build Your Alert Conditions")

    # Initialize session state for all conditions
    if 'all_conditions' not in st.session_state:
        st.session_state.all_conditions = []
    if 'condition_logic' not in st.session_state:
        st.session_state.condition_logic = "AND"

    # Single condition builder
    st.markdown("#### Add Conditions")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Indicator category selection
        indicator_category = st.selectbox(
            "Select Indicator Category:",
            ["", "Price Data", "Moving Averages", "RSI", "MACD", "Bollinger Bands",
             "Volume", "ATR", "CCI", "Williams %R", "ROC", "EWO", "MA Z-Score", "HARSI", "OBV MACD",
             "SAR", "SuperTrend", "Trend Magic", "Ichimoku Cloud", "Kalman ROC Stoch", "Pivot S/R", "Donchian Channels", "Custom"],
            key="indicator_category"
        )

        # Indicator selection based on category
        indicator = ""
        if indicator_category == "Price Data":
            price_type = st.selectbox(
                "Select Price Type:",
                ["", "Price Comparison", "Price Data Points"],
                key="price_type"
            )
            if price_type == "Price Comparison":
                price_condition = st.selectbox(
                    "Select Price Condition:",
                    ["", "price_above", "price_below", "price_equals"],
                    key="price_condition_type"
                )
                if price_condition:
                    price_value = st.number_input(
                        "Price Value:",
                        min_value=0.01, max_value=100000.0, value=100.0, step=0.01,
                        key="price_value"
                    )
                    indicator = f"{price_condition}: {price_value}"
            elif price_type == "Price Data Points":
                indicator = st.selectbox(
                    "Select Price Data:",
                    ["", "Close[-1]", "Open[-1]", "High[-1]", "Low[-1]",
                     "Close[-2]", "Open[-2]", "High[-2]", "Low[-2]",
                     "Close[0]", "Open[0]", "High[0]", "Low[0]"],
                    key="price_indicator"
                )
        elif indicator_category == "Moving Averages":
            ma_condition_type = st.selectbox(
                "Select MA Condition Type:",
                ["", "Price vs MA", "MA Crossover", "MA Value"],
                key="ma_condition_type"
            )
            if ma_condition_type == "Price vs MA":
                ma_type = st.selectbox(
                    "Select MA Type:",
                    ["SMA", "EMA", "HMA", "FRAMA", "KAMA"],
                    key="ma_type"
                )
                ma_period = st.number_input(
                    "Period:",
                    min_value=1, max_value=500, value=20,
                    key="ma_period"
                )
                price_ma_condition = st.selectbox(
                    "Condition:",
                    ["", "price_above_ma", "price_below_ma"],
                    key="price_ma_condition"
                )
                if price_ma_condition:
                    indicator = f"{price_ma_condition}: {ma_period} ({ma_type})"
            elif ma_condition_type == "MA Crossover":
                fast_period = st.number_input(
                    "Fast MA Period:",
                    min_value=1, max_value=200, value=10,
                    key="fast_ma_period"
                )
                slow_period = st.number_input(
                    "Slow MA Period:",
                    min_value=2, max_value=500, value=20,
                    key="slow_ma_period"
                )
                ma_type = st.selectbox(
                    "MA Type:",
                    ["SMA", "EMA", "HMA", "FRAMA", "KAMA"],
                    key="ma_crossover_type"
                )
                crossover_direction = st.selectbox(
                    "Crossover Direction:",
                    ["", "ma_crossover", "ma_crossunder"],
                    key="crossover_direction"
                )
                if crossover_direction:
                    indicator = f"{crossover_direction}: {fast_period} > {slow_period} ({ma_type})"
            elif ma_condition_type == "MA Value":
                ma_type = st.selectbox(
                    "Select MA Type:",
                    ["SMA", "EMA", "HMA", "FRAMA", "KAMA"],
                    key="ma_type_value"
                )
                if ma_type == "FRAMA":
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        frama_len = st.number_input(
                            "Length:",
                            min_value=1, max_value=200, value=16,
                            key="frama_len"
                        )
                    with col_b:
                        frama_fc = st.number_input(
                            "Fast Constant:",
                            min_value=1, max_value=300, value=1,
                            key="frama_fc"
                        )
                    with col_c:
                        frama_sc = st.number_input(
                            "Slow Constant:",
                            min_value=1, max_value=300, value=198,
                            key="frama_sc"
                        )
                    indicator = f"FRAMA(df, length={frama_len}, FC={frama_fc}, SC={frama_sc})[-1]"
                elif ma_type == "KAMA":
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        kama_len = st.number_input(
                            "Length:",
                            min_value=2, max_value=200, value=21,
                            key="kama_len"
                        )
                    with col_b:
                        kama_fast = st.number_input(
                            "Fast End:",
                            min_value=0.01, max_value=1.0, value=0.666, step=0.001,
                            key="kama_fast"
                        )
                    with col_c:
                        kama_slow = st.number_input(
                            "Slow End:",
                            min_value=0.01, max_value=1.0, value=0.0645, step=0.0001,
                            key="kama_slow"
                        )
                    indicator = f"KAMA(df, length={kama_len}, fast_end={kama_fast}, slow_end={kama_slow})[-1]"
                else:
                    ma_period = st.number_input(
                        "Period:",
                        min_value=1, max_value=500, value=20,
                        key="ma_period_value"
                    )

                    ma_input_source = st.selectbox(
                        "Input Source:",
                        [
                            "Close (default)",
                            "Open",
                            "High",
                            "Low",
                            "EWO (Elliott Wave Oscillator)",
                            "RSI",
                            "MACD (Line)",
                            "MACD (Signal)",
                            "MACD (Histogram)"
                        ],
                        key="ma_input_source"
                    )

                    ma_input_expr = ""
                    if ma_input_source == "Close (default)":
                        ma_input_expr = ""
                    elif ma_input_source in ["Open", "High", "Low", "Close"]:
                        ma_input_expr = f"'{ma_input_source}'"
                    elif ma_input_source == "EWO (Elliott Wave Oscillator)":
                        ewo_sma1 = st.number_input(
                            "EWO Fast SMA Period:",
                            min_value=1, max_value=100, value=5,
                            key="ma_input_ewo_sma1"
                        )
                        ewo_sma2 = st.number_input(
                            "EWO Slow SMA Period:",
                            min_value=1, max_value=200, value=35,
                            key="ma_input_ewo_sma2"
                        )
                        ewo_source = st.selectbox(
                            "EWO Source:",
                            ["Close", "Open", "High", "Low"],
                            index=0,
                            key="ma_input_ewo_source"
                        )
                        ewo_use_percent = st.checkbox(
                            "EWO as % of price",
                            value=True,
                            key="ma_input_ewo_use_percent"
                        )
                        ma_input_expr = f"EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})"
                    elif ma_input_source == "RSI":
                        ma_rsi_period = st.number_input(
                            "RSI Period (for input):",
                            min_value=2, max_value=100, value=14,
                            key="ma_input_rsi_period"
                        )
                        ma_input_expr = f"rsi({ma_rsi_period})"
                    elif ma_input_source.startswith("MACD"):
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            ma_macd_fast = st.number_input("MACD Fast:", min_value=5, max_value=50, value=12, key="ma_input_macd_fast")
                        with col_m2:
                            ma_macd_slow = st.number_input("MACD Slow:", min_value=10, max_value=100, value=26, key="ma_input_macd_slow")
                        with col_m3:
                            ma_macd_signal = st.number_input("MACD Signal:", min_value=5, max_value=50, value=9, key="ma_input_macd_signal")

                        macd_type = "line"
                        if "Signal" in ma_input_source:
                            macd_type = "signal"
                        elif "Histogram" in ma_input_source:
                            macd_type = "histogram"
                        ma_input_expr = f"macd(fast_period={ma_macd_fast}, slow_period={ma_macd_slow}, signal_period={ma_macd_signal}, type='{macd_type}')"

                    if ma_input_expr:
                        indicator = f"{ma_type.lower()}(period={ma_period}, input={ma_input_expr})[-1]"
                    else:
                        indicator = f"{ma_type.lower()}({ma_period})[-1]"
        elif indicator_category == "RSI":
            rsi_condition_type = st.selectbox(
                "Select RSI Condition:",
                ["", "RSI Levels", "RSI Value"],
                key="rsi_condition_type"
            )
            if rsi_condition_type == "RSI Levels":
                rsi_period = st.number_input(
                    "RSI Period:",
                    min_value=2, max_value=100, value=14,
                    key="rsi_period_levels"
                )
                rsi_level = st.selectbox(
                    "RSI Condition:",
                    ["", "rsi_oversold", "rsi_overbought", "rsi_neutral"],
                    key="rsi_level"
                )
                if rsi_level == "rsi_oversold":
                    oversold_level = st.number_input(
                        "Oversold Level:",
                        min_value=10, max_value=40, value=30,
                        key="oversold_level"
                    )
                    indicator = f"rsi({rsi_period})[-1] < {oversold_level}"
                elif rsi_level == "rsi_overbought":
                    overbought_level = st.number_input(
                        "Overbought Level:",
                        min_value=60, max_value=90, value=70,
                        key="overbought_level"
                    )
                    indicator = f"rsi({rsi_period})[-1] > {overbought_level}"
                elif rsi_level == "rsi_neutral":
                    indicator = f"rsi({rsi_period})[-1] > 30 and rsi({rsi_period})[-1] < 70"
            elif rsi_condition_type == "RSI Value":
                rsi_period = st.number_input(
                    "RSI Period:",
                    min_value=2, max_value=100, value=14,
                    key="rsi_period"
                )
                indicator = f"rsi({rsi_period})[-1]"
        elif indicator_category == "MACD":
            macd_condition_type = st.selectbox(
                "Select MACD Condition:",
                ["", "MACD Crossovers", "MACD Values"],
                key="macd_condition_type"
            )
            if macd_condition_type == "MACD Crossovers":
                macd_crossover = st.selectbox(
                    "Select Crossover:",
                    ["", "macd_bullish_crossover", "macd_bearish_crossover"],
                    key="macd_crossover"
                )
                if macd_crossover:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fast_period = st.number_input("Fast Period:", min_value=5, max_value=50, value=12, key="macd_fast")
                    with col2:
                        slow_period = st.number_input("Slow Period:", min_value=10, max_value=100, value=26, key="macd_slow")
                    with col3:
                        signal_period = st.number_input("Signal Period:", min_value=5, max_value=50, value=9, key="macd_signal")
                    if macd_crossover == "macd_bullish_crossover":
                        indicator = f"macd(fast_period = {fast_period}, slow_period = {slow_period}, signal_period = {signal_period}, type = 'line')[-1] > macd(fast_period = {fast_period}, slow_period = {slow_period}, signal_period = {signal_period}, type = 'signal')[-1]"
                    else:
                        indicator = f"macd(fast_period = {fast_period}, slow_period = {slow_period}, signal_period = {signal_period}, type = 'line')[-1] < macd(fast_period = {fast_period}, slow_period = {slow_period}, signal_period = {signal_period}, type = 'signal')[-1]"
            elif macd_condition_type == "MACD Values":
                indicator = st.selectbox(
                    "Select MACD Component:",
                    ["", "macd[-1]", "macd_signal[-1]", "macd_histogram[-1]",
                     "macd[0]", "macd_signal[0]", "macd_histogram[0]"],
                    key="macd_indicator"
                )
        elif indicator_category == "Bollinger Bands":
            bb_period = st.number_input(
                "BB Period:",
                min_value=5, max_value=100, value=20,
                key="bb_period"
            )
            bb_std = st.number_input(
                "Standard Deviations:",
                min_value=0.5, max_value=5.0, value=2.0, step=0.5,
                key="bb_std"
            )
            bb_component = st.selectbox(
                "Select Band:",
                ["upper", "middle", "lower", "width"],
                key="bb_component"
            )
            indicator = f"bb_{bb_component}({bb_period},{bb_std})[-1]"
        elif indicator_category == "Volume":
            volume_type = st.selectbox(
                "Select Volume Type:",
                ["", "Volume Conditions", "Volume Data"],
                key="volume_type"
            )
            if volume_type == "Volume Conditions":
                volume_condition = st.selectbox(
                    "Select Volume Condition:",
                    ["", "volume_above_average", "volume_spike", "volume_below_average"],
                    key="volume_condition_type"
                )
                if volume_condition:
                    if volume_condition in ["volume_above_average", "volume_spike"]:
                        volume_multiplier = st.number_input(
                            "Volume Multiplier:",
                            min_value=1.1, max_value=10.0, value=1.5, step=0.1,
                            key="volume_multiplier",
                            help="Volume must be X times the average"
                        )
                        indicator = f"{volume_condition}: {volume_multiplier}x"
                    elif volume_condition == "volume_below_average":
                        volume_fraction = st.number_input(
                            "Volume Fraction:",
                            min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                            key="volume_fraction",
                            help="Volume must be less than X times the average"
                        )
                        indicator = f"{volume_condition}: {volume_fraction}x"
            elif volume_type == "Volume Data":
                indicator = st.selectbox(
                    "Select Volume Data:",
                    ["", "volume[-1]", "volume[0]", "volume_avg(20)[-1]",
                     "volume[-1] / volume_avg(20)[-1]"],
                    key="volume_indicator"
                )
        elif indicator_category == "ATR":
            atr_period = st.number_input(
                "ATR Period:",
                min_value=1, max_value=100, value=14,
                key="atr_period"
            )
            indicator = f"atr({atr_period})[-1]"
        elif indicator_category == "CCI":
            cci_period = st.number_input(
                "CCI Period:",
                min_value=5, max_value=100, value=20,
                key="cci_period"
            )
            indicator = f"cci({cci_period})[-1]"
        elif indicator_category == "Williams %R":
            willr_period = st.number_input(
                "Williams %R Period:",
                min_value=5, max_value=100, value=14,
                key="willr_period"
            )
            indicator = f"willr({willr_period})[-1]"
        elif indicator_category == "ROC":
            roc_period = st.number_input(
                "ROC Period:",
                min_value=1, max_value=100, value=12,
                key="roc_period"
            )
            indicator = f"roc({roc_period})[-1]"

        elif indicator_category == "EWO":
            st.markdown("**Elliott Wave Oscillator (EWO)**")

            ewo_sma1 = st.number_input(
                "Fast SMA Period:",
                min_value=1, max_value=100, value=5,
                key="ewo_sma1",
                help="Fast SMA period (default 5)"
            )

            ewo_sma2 = st.number_input(
                "Slow SMA Period:",
                min_value=1, max_value=200, value=35,
                key="ewo_sma2",
                help="Slow SMA period (default 35)"
            )

            ewo_source = st.selectbox(
                "Source:",
                ["Close", "Open", "High", "Low"],
                index=0,
                key="ewo_source"
            )

            ewo_use_percent = st.checkbox(
                "Show as percentage of current price",
                value=True,
                key="ewo_use_percent",
                help="If checked, shows difference as percent; otherwise absolute value"
            )

            ewo_condition_type = st.selectbox(
                "Condition Type:",
                ["", "EWO Levels", "EWO Value"],
                key="ewo_condition_type"
            )

            if ewo_condition_type == "EWO Levels":
                ewo_level_condition = st.selectbox(
                    "EWO Condition:",
                    ["Above Zero", "Below Zero", "Crossover Above Zero", "Crossover Below Zero"],
                    key="ewo_level_condition"
                )

                if ewo_level_condition == "Above Zero":
                    indicator = f"EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] > 0"
                elif ewo_level_condition == "Below Zero":
                    indicator = f"EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] < 0"
                elif ewo_level_condition == "Crossover Above Zero":
                    indicator = f"(EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] > 0) & (EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-2] <= 0)"
                elif ewo_level_condition == "Crossover Below Zero":
                    indicator = f"(EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] < 0) & (EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-2] >= 0)"

            elif ewo_condition_type == "EWO Value":
                ewo_operator = st.selectbox(
                    "Operator:",
                    [">", "<", ">=", "<=", "=="],
                    key="ewo_operator"
                )

                ewo_value = st.number_input(
                    "Value:",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key="ewo_value"
                )

                indicator = f"EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] {ewo_operator} {ewo_value}"

        elif indicator_category == "MA Z-Score":
            st.markdown("**Price vs MA Z-Score**")

            ma_type = st.selectbox(
                "MA Type:",
                ["SMA", "EMA", "HMA"],
                index=0,
                key="ma_zs_alert_ma_type",
                help="Choose SMA, EMA, or HMA as the baseline moving average"
            )

            ma_period = st.number_input(
                "MA Period:",
                min_value=1, max_value=500, value=20,
                key="ma_zs_alert_period"
            )

            spread_mean_window = st.number_input(
                "Spread Mean Window (lookback for average spread):",
                min_value=1, max_value=500, value=ma_period,
                key="ma_zs_alert_mean_window"
            )

            spread_std_window = st.number_input(
                "Spread Std Dev Window:",
                min_value=1, max_value=500, value=spread_mean_window,
                key="ma_zs_alert_std_window"
            )

            price_col = st.selectbox(
                "Price Column:",
                ["Close", "Open", "High", "Low"],
                index=0,
                key="ma_zs_alert_price_col"
            )

            use_percent = st.checkbox(
                "Use percent spread ((Price - MA)/MA * 100)",
                value=True,
                key="ma_zs_alert_use_percent",
                help="If unchecked, uses absolute price minus moving average"
            )

            operator = st.selectbox(
                "Z-Score Condition:",
                [">", ">=", "<", "<="],
                index=0,
                key="ma_zs_alert_operator"
            )

            threshold = st.number_input(
                "Z-Score Threshold:",
                value=2.0,
                step=0.1,
                format="%.2f",
                key="ma_zs_alert_threshold"
            )

            zs_expr = (
                f"MA_SPREAD_ZSCORE(price_col='{price_col}', ma_length={ma_period}, "
                f"spread_mean_window={spread_mean_window}, spread_std_window={spread_std_window}, "
                f"ma_type='{ma_type}', use_percent={use_percent}, output='zscore')[-1]"
            )
            indicator = f"{zs_expr} {operator} {threshold}"

        elif indicator_category == "HARSI":
            harsi_type = st.selectbox(
                "Select HARSI Type:",
                ["", "HARSI_FLIP", "HARSI"],
                key="harsi_type",
                help="HARSI_FLIP returns flip signals (0=no change, 1=red to green, 2=green to red)"
            )
            if harsi_type:
                harsi_period = st.number_input(
                    "HARSI Period:",
                    min_value=5, max_value=100, value=14,
                    key="harsi_period"
                )
                harsi_smoothing = st.number_input(
                    "HARSI Smoothing:",
                    min_value=1, max_value=20, value=1,
                    key="harsi_smoothing"
                )
                if harsi_type == "HARSI_FLIP":
                    # HARSI_FLIP returns transition codes: 0=no change, 1=green to red, 2=red to green
                    indicator_options = st.selectbox(
                        "Select HARSI_FLIP Condition:",
                        ["",
                         f"HARSI_Flip(period = {harsi_period}, smoothing = {harsi_smoothing})[-1] == 1",  # Green to Red (Sell signal)
                         f"HARSI_Flip(period = {harsi_period}, smoothing = {harsi_smoothing})[-1] == 2",  # Red to Green (Buy signal)
                         f"HARSI_Flip(period = {harsi_period}, smoothing = {harsi_smoothing})[-1] > 0"],  # Any flip
                        key="harsi_flip_option",
                        help="1 = Green to Red (bearish), 2 = Red to Green (bullish), >0 = Any flip"
                    )
                    indicator = indicator_options if indicator_options else f"HARSI_Flip(period = {harsi_period}, smoothing = {harsi_smoothing})[-1]"
                else:
                    indicator = f"HARSI(period = {harsi_period}, smoothing = {harsi_smoothing})[-1]"

        elif indicator_category == "OBV MACD":
            st.info("OBV MACD: MACD based on OBV with various moving average types")

            obv_macd_type = st.selectbox(
                "Select OBV MACD Type:",
                ["", "OBV_MACD (Value)", "OBV_MACD_SIGNAL (Direction)"],
                key="obv_macd_type",
                help="OBV_MACD returns the indicator value, OBV_MACD_SIGNAL returns trend direction (1=bullish, -1=bearish)"
            )

            if obv_macd_type:
                col1, col2 = st.columns(2)

                with col1:
                    window_len = st.number_input("Window Length:", min_value=5, max_value=100, value=28, key="obv_window")
                    v_len = st.number_input("Volume Smoothing:", min_value=5, max_value=50, value=14, key="obv_v_len")
                    obv_len = st.number_input("OBV EMA Length:", min_value=1, max_value=50, value=1, key="obv_len")
                    ma_type = st.selectbox("MA Type:",
                        ['DEMA', 'EMA', 'TEMA', 'TDEMA', 'TTEMA', 'AVG', 'THMA', 'ZLEMA', 'ZLDEMA', 'ZLTEMA', 'DZLEMA', 'TZLEMA', 'LLEMA', 'NMA'],
                        index=0, key="obv_ma_type")

                with col2:
                    ma_len = st.number_input("MA Length:", min_value=1, max_value=100, value=9, key="obv_ma_len")
                    slow_len = st.number_input("MACD Slow Length:", min_value=10, max_value=100, value=26, key="obv_slow")
                    slope_len = st.number_input("Slope Length:", min_value=1, max_value=20, value=2, key="obv_slope")

                    if "SIGNAL" in obv_macd_type:
                        p_val = st.number_input("Channel Sensitivity (p):", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="obv_p")

                # Build indicator string
                if obv_macd_type == "OBV_MACD (Value)":
                    indicator = f"OBV_MACD(window_len={window_len}, v_len={v_len}, obv_len={obv_len}, ma_type='{ma_type}', ma_len={ma_len}, slow_len={slow_len}, slope_len={slope_len})[-1]"
                else:  # OBV_MACD_SIGNAL
                    indicator = f"OBV_MACD_SIGNAL(window_len={window_len}, v_len={v_len}, obv_len={obv_len}, ma_type='{ma_type}', ma_len={ma_len}, slow_len={slow_len}, slope_len={slope_len}, p={p_val})[-1]"

        elif indicator_category == "SAR":
            sar_accel = st.number_input(
                "SAR Acceleration:",
                min_value=0.01, max_value=0.2, value=0.02, step=0.01,
                key="sar_accel"
            )
            sar_max = st.number_input(
                "SAR Max Acceleration:",
                min_value=0.1, max_value=0.5, value=0.2, step=0.05,
                key="sar_max"
            )
            indicator = f"sar({sar_accel},{sar_max})[-1]"
        elif indicator_category == "SuperTrend":
            st.markdown("**SuperTrend Indicator**")
            supertrend_type = st.selectbox(
                "Select Condition Type:",
                ["", "Trend Direction", "Price vs SuperTrend", "Trend Change"],
                key="supertrend_type"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st_period = st.number_input(
                    "ATR Period:",
                    min_value=1, max_value=100, value=10,
                    key="st_period"
                )
            with col_b:
                st_mult = st.number_input(
                    "ATR Multiplier:",
                    min_value=0.1, max_value=10.0, value=3.0, step=0.1,
                    key="st_mult"
                )

            col_c, col_d = st.columns(2)
            with col_c:
                st_use_hl2 = st.checkbox(
                    "Use (H+L)/2 as Source",
                    value=True,
                    key="st_use_hl2",
                    help="If unchecked, uses Close price"
                )
            with col_d:
                st_use_atr = st.checkbox(
                    "Use Built-in ATR",
                    value=True,
                    key="st_use_atr",
                    help="Alternative ATR calculation method"
                )

            if supertrend_type == "Trend Direction":
                trend_condition = st.selectbox(
                    "Trend Condition:",
                    ["", "Uptrend", "Downtrend"],
                    key="st_trend_condition"
                )
                if trend_condition == "Uptrend":
                    indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] == 1"
                elif trend_condition == "Downtrend":
                    indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] == -1"

            elif supertrend_type == "Price vs SuperTrend":
                price_condition = st.selectbox(
                    "Price Condition:",
                    ["", "Price above SuperTrend", "Price below SuperTrend"],
                    key="st_price_condition"
                )
                if price_condition == "Price above SuperTrend":
                    indicator = f"Close[-1] > SUPERTREND_UPPER(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1]"
                elif price_condition == "Price below SuperTrend":
                    indicator = f"Close[-1] < SUPERTREND_LOWER(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1]"

            elif supertrend_type == "Trend Change":
                change_condition = st.selectbox(
                    "Change Direction:",
                    ["", "Changed to Uptrend (Buy Signal)", "Changed to Downtrend (Sell Signal)", "Any Change"],
                    key="st_change_condition"
                )
                if change_condition == "Changed to Uptrend (Buy Signal)":
                    indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] == 1 and SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-2] == -1"
                elif change_condition == "Changed to Downtrend (Sell Signal)":
                    indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] == -1 and SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-2] == 1"
                elif change_condition == "Any Change":
                    indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] != SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-2]"
        elif indicator_category == "Trend Magic":
            st.markdown("**Trend Magic Indicator**")
            tm_type = st.selectbox(
                "Select Condition Type:",
                ["", "Trend Direction", "Price vs Trend Magic", "Trend Crossover"],
                key="tm_type"
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                tm_cci_period = st.number_input(
                    "CCI Period:",
                    min_value=5, max_value=100, value=20,
                    key="tm_cci_period"
                )
            with col_b:
                tm_atr_mult = st.number_input(
                    "ATR Multiplier:",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key="tm_atr_mult"
                )
            with col_c:
                tm_atr_period = st.number_input(
                    "ATR Period:",
                    min_value=1, max_value=50, value=5,
                    key="tm_atr_period"
                )

            if tm_type == "Trend Direction":
                trend_condition = st.selectbox(
                    "Trend Condition:",
                    ["", "Bullish (CCI >= 0)", "Bearish (CCI < 0)"],
                    key="tm_trend_condition"
                )
                if trend_condition == "Bullish (CCI >= 0)":
                    indicator = f"TREND_MAGIC_SIGNAL(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] == 1"
                elif trend_condition == "Bearish (CCI < 0)":
                    indicator = f"TREND_MAGIC_SIGNAL(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] == -1"

            elif tm_type == "Price vs Trend Magic":
                price_condition = st.selectbox(
                    "Price Condition:",
                    ["", "Price above Trend Magic", "Price below Trend Magic", "Price crossed Trend Magic"],
                    key="tm_price_condition"
                )
                if price_condition == "Price above Trend Magic":
                    indicator = f"Close[-1] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1]"
                elif price_condition == "Price below Trend Magic":
                    indicator = f"Close[-1] < TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1]"
                elif price_condition == "Price crossed Trend Magic":
                    indicator = f"(Close[-1] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] and Close[-2] <= TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2]) or (Close[-1] < TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] and Close[-2] >= TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2])"

            elif tm_type == "Trend Crossover":
                cross_condition = st.selectbox(
                    "Crossover Type:",
                    ["", "Buy Signal (Low crosses above)", "Sell Signal (High crosses below)", "Any Cross"],
                    key="tm_cross_condition"
                )
                if cross_condition == "Buy Signal (Low crosses above)":
                    indicator = f"Low[-1] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] and Low[-2] <= TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2]"
                elif cross_condition == "Sell Signal (High crosses below)":
                    indicator = f"High[-1] < TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] and High[-2] >= TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2]"
                elif cross_condition == "Any Cross":
                    indicator = f"((Close[-1] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1]) != (Close[-2] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2]))"
        elif indicator_category == "Ichimoku Cloud":
            st.markdown("**Ichimoku Cloud Indicator**")
            ichimoku_type = st.selectbox(
                "Select Condition Type:",
                ["", "Price vs Cloud", "Line Crossovers", "Cloud Color", "Individual Lines", "Lagging Span"],
                key="ichimoku_type"
            )

            # Parameters
            col_a, col_b = st.columns(2)
            with col_a:
                ich_conversion = st.number_input(
                    "Conversion Line Period:",
                    min_value=1, max_value=100, value=9,
                    key="ich_conversion"
                )
                ich_base = st.number_input(
                    "Base Line Period:",
                    min_value=1, max_value=100, value=26,
                    key="ich_base"
                )
            with col_b:
                ich_span_b = st.number_input(
                    "Span B Period:",
                    min_value=1, max_value=200, value=52,
                    key="ich_span_b"
                )
                ich_displacement = st.number_input(
                    "Displacement:",
                    min_value=1, max_value=100, value=26,
                    key="ich_displacement"
                )

            if ichimoku_type == "Price vs Cloud":
                price_cloud = st.selectbox(
                    "Condition:",
                    ["", "Price above cloud", "Price below cloud", "Price in cloud",
                     "Price entered cloud (from above)", "Price entered cloud (from below)", "Price entered cloud (any direction)",
                     "Price crossed above cloud", "Price crossed below cloud"],
                    key="ich_price_cloud"
                )
                if price_cloud == "Price above cloud":
                    indicator = f"Close[-1] > ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]"
                elif price_cloud == "Price below cloud":
                    indicator = f"Close[-1] < ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]"
                elif price_cloud == "Price in cloud":
                    indicator = f"(Close[-1] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-1] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1])"
                elif price_cloud == "Price entered cloud (from above)":
                    # Was above cloud, now inside cloud
                    indicator = f"(Close[-2] > ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2]) and (Close[-1] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-1] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1])"
                elif price_cloud == "Price entered cloud (from below)":
                    # Was below cloud, now inside cloud
                    indicator = f"(Close[-2] < ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2]) and (Close[-1] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-1] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1])"
                elif price_cloud == "Price entered cloud (any direction)":
                    # Was outside cloud (either above or below), now inside cloud
                    indicator = f"((Close[-2] > ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2]) or (Close[-2] < ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2])) and (Close[-1] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-1] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1])"
                elif price_cloud == "Price crossed above cloud":
                    indicator = f"(Close[-1] > ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-2] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2])"
                elif price_cloud == "Price crossed below cloud":
                    indicator = f"(Close[-1] < ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-2] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2])"

            elif ichimoku_type == "Line Crossovers":
                line_cross = st.selectbox(
                    "Crossover Type:",
                    ["", "Conversion crosses above Base (TK Cross Bull)", "Conversion crosses below Base (TK Cross Bear)",
                     "Price crosses above Conversion", "Price crosses below Conversion",
                     "Price crosses above Base", "Price crosses below Base"],
                    key="ich_line_cross"
                )
                if line_cross == "Conversion crosses above Base (TK Cross Bull)":
                    indicator = f"(ICHIMOKU_CONVERSION(periods={ich_conversion})[-1] > ICHIMOKU_BASE(periods={ich_base})[-1]) and (ICHIMOKU_CONVERSION(periods={ich_conversion})[-2] <= ICHIMOKU_BASE(periods={ich_base})[-2])"
                elif line_cross == "Conversion crosses below Base (TK Cross Bear)":
                    indicator = f"(ICHIMOKU_CONVERSION(periods={ich_conversion})[-1] < ICHIMOKU_BASE(periods={ich_base})[-1]) and (ICHIMOKU_CONVERSION(periods={ich_conversion})[-2] >= ICHIMOKU_BASE(periods={ich_base})[-2])"
                elif line_cross == "Price crosses above Conversion":
                    indicator = f"(Close[-1] > ICHIMOKU_CONVERSION(periods={ich_conversion})[-1]) and (Close[-2] <= ICHIMOKU_CONVERSION(periods={ich_conversion})[-2])"
                elif line_cross == "Price crosses below Conversion":
                    indicator = f"(Close[-1] < ICHIMOKU_CONVERSION(periods={ich_conversion})[-1]) and (Close[-2] >= ICHIMOKU_CONVERSION(periods={ich_conversion})[-2])"
                elif line_cross == "Price crosses above Base":
                    indicator = f"(Close[-1] > ICHIMOKU_BASE(periods={ich_base})[-1]) and (Close[-2] <= ICHIMOKU_BASE(periods={ich_base})[-2])"
                elif line_cross == "Price crosses below Base":
                    indicator = f"(Close[-1] < ICHIMOKU_BASE(periods={ich_base})[-1]) and (Close[-2] >= ICHIMOKU_BASE(periods={ich_base})[-2])"

            elif ichimoku_type == "Cloud Color":
                cloud_color = st.selectbox(
                    "Cloud Condition:",
                    ["", "Bullish cloud (green)", "Bearish cloud (red)", "Cloud color changed to bullish", "Cloud color changed to bearish"],
                    key="ich_cloud_color"
                )
                if cloud_color == "Bullish cloud (green)":
                    indicator = f"ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1] == 1"
                elif cloud_color == "Bearish cloud (red)":
                    indicator = f"ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1] == -1"
                elif cloud_color == "Cloud color changed to bullish":
                    indicator = f"(ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1] == 1) and (ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2] != 1)"
                elif cloud_color == "Cloud color changed to bearish":
                    indicator = f"(ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1] == -1) and (ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2] != -1)"

            elif ichimoku_type == "Individual Lines":
                line_condition = st.selectbox(
                    "Line Condition:",
                    ["", "Price above Conversion Line", "Price below Conversion Line",
                     "Price above Base Line", "Price below Base Line",
                     "Conversion Line above Base Line", "Conversion Line below Base Line"],
                    key="ich_line_condition"
                )
                if line_condition == "Price above Conversion Line":
                    indicator = f"Close[-1] > ICHIMOKU_CONVERSION(periods={ich_conversion})[-1]"
                elif line_condition == "Price below Conversion Line":
                    indicator = f"Close[-1] < ICHIMOKU_CONVERSION(periods={ich_conversion})[-1]"
                elif line_condition == "Price above Base Line":
                    indicator = f"Close[-1] > ICHIMOKU_BASE(periods={ich_base})[-1]"
                elif line_condition == "Price below Base Line":
                    indicator = f"Close[-1] < ICHIMOKU_BASE(periods={ich_base})[-1]"
                elif line_condition == "Conversion Line above Base Line":
                    indicator = f"ICHIMOKU_CONVERSION(periods={ich_conversion})[-1] > ICHIMOKU_BASE(periods={ich_base})[-1]"
                elif line_condition == "Conversion Line below Base Line":
                    indicator = f"ICHIMOKU_CONVERSION(periods={ich_conversion})[-1] < ICHIMOKU_BASE(periods={ich_base})[-1]"

            elif ichimoku_type == "Lagging Span":
                lagging_condition = st.selectbox(
                    "Lagging Span Condition:",
                    ["", "Lagging Span above price (26 periods ago)", "Lagging Span below price (26 periods ago)",
                     "Lagging Span crossed above price", "Lagging Span crossed below price"],
                    key="ich_lagging"
                )
                if lagging_condition == "Lagging Span above price (26 periods ago)":
                    indicator = f"ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-1] > Close[-{ich_displacement}-1]"
                elif lagging_condition == "Lagging Span below price (26 periods ago)":
                    indicator = f"ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-1] < Close[-{ich_displacement}-1]"
                elif lagging_condition == "Lagging Span crossed above price":
                    indicator = f"(ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-1] > Close[-{ich_displacement}-1]) and (ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-2] <= Close[-{ich_displacement}-2])"
                elif lagging_condition == "Lagging Span crossed below price":
                    indicator = f"(ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-1] < Close[-{ich_displacement}-1]) and (ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-2] >= Close[-{ich_displacement}-2])"
        elif indicator_category == "Kalman ROC Stoch":
            st.markdown("**Kalman Smoothed ROC & Stochastic Indicator**")
            st.info("Advanced momentum oscillator combining Kalman-filtered ROC with Stochastic, smoothed by various MA types")

            krs_type = st.selectbox(
                "Select Condition Type:",
                ["", "Direction", "Crossovers", "Levels", "Value"],
                key="krs_type"
            )

            # Parameters in expandable section
            with st.expander("Parameters", expanded=True):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.markdown("**Smoothing**")
                    krs_ma_type = st.selectbox(
                        "MA Type:",
                        ['TEMA', 'EMA', 'DEMA', 'WMA', 'VWMA', 'SMA', 'SMMA', 'HMA', 'LSMA', 'PEMA'],
                        key="krs_ma_type"
                    )
                    krs_smooth_len = st.number_input(
                        "Smoothing Length:",
                        min_value=1, max_value=50, value=12,
                        key="krs_smooth_len"
                    )
                    if krs_ma_type == 'LSMA':
                        krs_lsma_off = st.number_input(
                            "LSMA Offset:",
                            min_value=0, max_value=10, value=0,
                            key="krs_lsma_off"
                        )
                    else:
                        krs_lsma_off = 0

                with col_b:
                    st.markdown("**Kalman Filter**")
                    krs_kal_src = st.selectbox(
                        "Source:",
                        ['Close', 'Open', 'High', 'Low', 'HL2', 'HLC3', 'OHLC4'],
                        key="krs_kal_src"
                    )
                    krs_sharp = st.number_input(
                        "Sharpness:",
                        min_value=1.0, max_value=100.0, value=25.0, step=1.0,
                        key="krs_sharp"
                    )
                    krs_k_period = st.number_input(
                        "Filter Period:",
                        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                        key="krs_k_period"
                    )

                with col_c:
                    st.markdown("**ROC & Stochastic**")
                    krs_roc_len = st.number_input(
                        "ROC Length:",
                        min_value=1, max_value=50, value=9,
                        key="krs_roc_len"
                    )
                    krs_stoch_len = st.number_input(
                        "Stoch %K Length:",
                        min_value=1, max_value=50, value=14,
                        key="krs_stoch_len"
                    )
                    krs_smooth_k = st.number_input(
                        "%K Smooth:",
                        min_value=1, max_value=20, value=1,
                        key="krs_smooth_k"
                    )
                    krs_smooth_d = st.number_input(
                        "%D Smooth:",
                        min_value=1, max_value=20, value=3,
                        key="krs_smooth_d"
                    )

            # Build parameter string for function calls
            krs_params = f"ma_type='{krs_ma_type}', lsma_off={krs_lsma_off}, smooth_len={krs_smooth_len}, kal_src='{krs_kal_src}', sharp={krs_sharp}, k_period={krs_k_period}, roc_len={krs_roc_len}, stoch_len={krs_stoch_len}, smooth_k={krs_smooth_k}, smooth_d={krs_smooth_d}"

            if krs_type == "Direction":
                direction = st.selectbox(
                    "Direction:",
                    ["", "Uptrend (White)", "Downtrend (Blue)"],
                    key="krs_direction"
                )
                if direction == "Uptrend (White)":
                    indicator = f"KALMAN_ROC_STOCH_SIGNAL(df, {krs_params})[-1] == 1"
                elif direction == "Downtrend (Blue)":
                    indicator = f"KALMAN_ROC_STOCH_SIGNAL(df, {krs_params})[-1] == -1"

            elif krs_type == "Crossovers":
                crossover = st.selectbox(
                    "Crossover Type:",
                    ["", "Bullish Crossover (Buy)", "Bearish Crossunder (Sell)", "Any Cross"],
                    key="krs_crossover"
                )
                if crossover == "Bullish Crossover (Buy)":
                    indicator = f"KALMAN_ROC_STOCH_CROSSOVER(df, {krs_params})[-1] == 1"
                elif crossover == "Bearish Crossunder (Sell)":
                    indicator = f"KALMAN_ROC_STOCH_CROSSOVER(df, {krs_params})[-1] == -1"
                elif crossover == "Any Cross":
                    indicator = f"KALMAN_ROC_STOCH_CROSSOVER(df, {krs_params})[-1] != 0"

            elif krs_type == "Levels":
                level_condition = st.selectbox(
                    "Level Condition:",
                    ["", "Above 60 (Overbought)", "Below 10 (Oversold)",
                     "Above 50", "Below 50", "Between 10 and 60"],
                    key="krs_level"
                )
                if level_condition == "Above 60 (Overbought)":
                    indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1] > 60"
                elif level_condition == "Below 10 (Oversold)":
                    indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1] < 10"
                elif level_condition == "Above 50":
                    indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1] > 50"
                elif level_condition == "Below 50":
                    indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1] < 50"
                elif level_condition == "Between 10 and 60":
                    indicator = f"(KALMAN_ROC_STOCH(df, {krs_params})[-1] > 10) and (KALMAN_ROC_STOCH(df, {krs_params})[-1] < 60)"

            elif krs_type == "Value":
                st.info("Get the raw indicator value to compare with custom thresholds")
                indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1]"
        elif indicator_category == "Pivot S/R":
            st.info("Pivot Support/Resistance detector finds horizontal levels from pivot highs/lows.\n" +
                   "It checks for both proximity and crossovers automatically.\n" +
                   "Levels are stored and maintained on a rolling 3-year basis.")

            pivot_signal = st.selectbox(
                "Alert When:",
                ["",
                 "Any Signal (Proximity or Crossover)",
                 "Near Support",
                 "Near Resistance",
                 "Near Any Level",
                 "Bullish Crossover",
                 "Bearish Crossover",
                 "Any Crossover",
                 "Broke Strong Support (3+ touches)",
                 "Broke Strong Resistance (3+ touches)",
                 "Custom Signal Value"],
                key="pivot_signal",
                help="Signal values: 3=Broke strong resist, 2=Bullish cross, 1=Near support, -1=Near resist, -2=Bearish cross, -3=Broke strong support"
            )

            # Parameters
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Pivot Detection Parameters:**")
                left_bars = st.number_input(
                    "Left Bars:",
                    min_value=2, max_value=120, value=5,
                    key="pivot_left_bars",
                    help="Number of bars to the left of pivot for validation"
                )
                right_bars = st.number_input(
                    "Right Bars:",
                    min_value=2, max_value=120, value=5,
                    key="pivot_right_bars",
                    help="Number of bars to the right of pivot for validation"
                )

            with col_b:
                st.write("**Alert & Level Management:**")
                proximity_threshold = st.number_input(
                    "Proximity Threshold (%):",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                    key="pivot_proximity",
                    help="Alert when price is within X% of a level"
                )
                buffer_percent = st.number_input(
                    "Buffer Between Levels (%):",
                    min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                    key="pivot_buffer",
                    help="Minimum % separation between levels to avoid clustering"
                )

            # Generate condition based on selection
            detector_func = f"PIVOT_SR(df, ticker, left_bars={left_bars}, right_bars={right_bars}, proximity_threshold={proximity_threshold}, buffer_percent={buffer_percent})"

            if pivot_signal == "Any Signal (Proximity or Crossover)":
                indicator = f"{detector_func} != 0"
            elif pivot_signal == "Near Support":
                indicator = f"{detector_func} == 1"
            elif pivot_signal == "Near Resistance":
                indicator = f"{detector_func} == -1"
            elif pivot_signal == "Near Any Level":
                indicator = f"abs({detector_func}) == 1"
            elif pivot_signal == "Bullish Crossover":
                indicator = f"{detector_func} == 2"
            elif pivot_signal == "Bearish Crossover":
                indicator = f"{detector_func} == -2"
            elif pivot_signal == "Any Crossover":
                indicator = f"abs({detector_func}) == 2"
            elif pivot_signal == "Broke Strong Support (3+ touches)":
                indicator = f"{detector_func} == -3"
            elif pivot_signal == "Broke Strong Resistance (3+ touches)":
                indicator = f"{detector_func} == 3"
            elif pivot_signal == "Custom Signal Value":
                custom_value = st.number_input(
                    "Signal Value to Match:",
                    min_value=-3, max_value=3, value=0,
                    key="pivot_custom_value",
                    help="-3=Broke strong support, -2=Bearish cross, -1=Near resist, 0=None, 1=Near support, 2=Bullish cross, 3=Broke strong resist"
                )
                if custom_value != 0:
                    indicator = f"{detector_func} == {custom_value}"
                else:
                    indicator = ""
            else:
                indicator = ""
        elif indicator_category == "Donchian Channels":
            donchian_type = st.selectbox(
                "Select Donchian Type:",
                ["", "Channel Lines", "Channel Breakout", "Channel Position", "Channel Width"],
                key="donchian_type"
            )

            if donchian_type:
                donchian_length = st.number_input(
                    "Donchian Period:",
                    min_value=5, max_value=200, value=20,
                    key="donchian_length",
                    help="Number of periods for highest high and lowest low"
                )
                donchian_offset = st.number_input(
                    "Offset:",
                    min_value=-50, max_value=50, value=0,
                    key="donchian_offset",
                    help="Positive offset looks back, negative offset looks forward"
                )

                if donchian_type == "Channel Lines":
                    line_type = st.selectbox(
                        "Select Line:",
                        ["", "Upper Band", "Lower Band", "Basis (Middle)", "Price vs Upper", "Price vs Lower", "Price vs Basis"],
                        key="donchian_line"
                    )
                    if line_type == "Upper Band":
                        indicator = f"DONCHIAN_UPPER(df, {donchian_length}, {donchian_offset})[-1]"
                    elif line_type == "Lower Band":
                        indicator = f"DONCHIAN_LOWER(df, {donchian_length}, {donchian_offset})[-1]"
                    elif line_type == "Basis (Middle)":
                        indicator = f"DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-1]"
                    elif line_type == "Price vs Upper":
                        indicator = f"Close[-1] > DONCHIAN_UPPER(df, {donchian_length}, {donchian_offset})[-1]"
                    elif line_type == "Price vs Lower":
                        indicator = f"Close[-1] < DONCHIAN_LOWER(df, {donchian_length}, {donchian_offset})[-1]"
                    elif line_type == "Price vs Basis":
                        indicator = f"Close[-1] > DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-1]"

                elif donchian_type == "Channel Breakout":
                    breakout_type = st.selectbox(
                        "Select Breakout:",
                        ["", "Upper Band Breakout", "Lower Band Breakout", "Basis Cross Up", "Basis Cross Down"],
                        key="donchian_breakout"
                    )
                    if breakout_type == "Upper Band Breakout":
                        indicator = f"(Close[-1] > DONCHIAN_UPPER(df, {donchian_length}, {donchian_offset})[-1]) and (Close[-2] <= DONCHIAN_UPPER(df, {donchian_length}, {donchian_offset})[-2])"
                    elif breakout_type == "Lower Band Breakout":
                        indicator = f"(Close[-1] < DONCHIAN_LOWER(df, {donchian_length}, {donchian_offset})[-1]) and (Close[-2] >= DONCHIAN_LOWER(df, {donchian_length}, {donchian_offset})[-2])"
                    elif breakout_type == "Basis Cross Up":
                        indicator = f"(Close[-1] > DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-1]) and (Close[-2] <= DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-2])"
                    elif breakout_type == "Basis Cross Down":
                        indicator = f"(Close[-1] < DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-1]) and (Close[-2] >= DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-2])"

                elif donchian_type == "Channel Position":
                    position_type = st.selectbox(
                        "Select Position Check:",
                        ["", "Position Value", "Near Upper Band", "Near Lower Band", "Near Middle"],
                        key="donchian_position"
                    )
                    if position_type == "Position Value":
                        indicator = f"DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1]"
                        st.info("Position value: 0 = at lower band, 0.5 = at middle, 1 = at upper band")
                    elif position_type == "Near Upper Band":
                        indicator = f"DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1] > 0.8"
                    elif position_type == "Near Lower Band":
                        indicator = f"DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1] < 0.2"
                    elif position_type == "Near Middle":
                        indicator = f"(DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1] > 0.4) and (DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1] < 0.6)"

                elif donchian_type == "Channel Width":
                    width_type = st.selectbox(
                        "Select Width Check:",
                        ["", "Width Value", "Width Expanding", "Width Contracting"],
                        key="donchian_width"
                    )
                    if width_type == "Width Value":
                        indicator = f"DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-1]"
                    elif width_type == "Width Expanding":
                        indicator = f"DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-1] > DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-2]"
                    elif width_type == "Width Contracting":
                        indicator = f"DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-1] < DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-2]"
        elif indicator_category == "Custom":
            st.info("Enter any custom condition in the text field below. Examples:\n" +
                   "• Close[-1] > 150.00\n" +
                   "• rsi(14)[-1] < 30\n" +
                   "• sma(20)[-1] > sma(50)[-1]\n" +
                   "• HARSI_Flip(period = 14, smoothing = 1)[-1] == 2")
            indicator = ""

    with col2:
        # Condition input with pre-filled indicator
        if indicator:
            use_zscore = st.checkbox(
                "Transform to Z-score (rolling)",
                value=False,
                key="zscore_option_alert",
                help="Use the z-score of this indicator over a rolling lookback window"
            )
            if use_zscore:
                zscore_lookback = st.number_input(
                    "Z-score Lookback:",
                    min_value=5, max_value=500, value=20,
                    key="zscore_lookback_alert"
                )
            else:
                zscore_lookback = 20

            indicator = apply_zscore_indicator(indicator, use_zscore, zscore_lookback)

            condition = st.text_input(
                "Complete the condition:",
                value=indicator,
                key="condition_input",
                help="Complete the condition (e.g., 'Close[-1] > 150' or 'rsi(14)[-1] < 30')"
            )
        else:
            condition = st.text_input(
                "Enter custom condition:",
                key="custom_condition_input",
                placeholder="e.g., Close[-1] > sma(20)[-1]",
                help="Enter any valid condition using the indicator syntax"
            )

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        if st.button("➕ Add Condition", key="add_condition"):
            if condition and condition.strip():
                st.session_state.all_conditions.append(condition.strip())
                st.success(f"✅ Added: {condition.strip()}")
                st.rerun()

    # Logic operator selection
    if len(st.session_state.all_conditions) > 0:
        st.markdown("#### Condition Logic")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.session_state.condition_logic = st.radio(
                "Combine conditions with:",
                ["AND", "OR"],
                index=0 if st.session_state.condition_logic == "AND" else 1,
                key="logic_operator",
                help="AND: All conditions must be true\nOR: At least one condition must be true"
            )
        with col2:
            st.info(f"Conditions will be combined using **{st.session_state.condition_logic}** logic")

    # Display current conditions
    if st.session_state.all_conditions:
        st.markdown("#### Current Conditions")

        # Clear all button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear All", key="clear_all"):
                st.session_state.all_conditions = []
                st.rerun()

        # Display conditions with logic
        for i, cond in enumerate(st.session_state.all_conditions):
            col1, col2 = st.columns([4, 1])
            with col1:
                if i == 0:
                    st.markdown(f"{i+1}. `{cond}`")
                else:
                    st.markdown(f"{st.session_state.condition_logic} {i+1}. `{cond}`")
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.all_conditions.pop(i)
                    st.rerun()

        # Show combined condition preview
        if len(st.session_state.all_conditions) > 1:
            st.markdown("**Combined Condition:**")
            combined = f" {st.session_state.condition_logic} ".join(st.session_state.all_conditions)
            st.code(combined, language="python")
    else:
        st.info("No conditions added yet. Use the dropdowns above to build your alert conditions.")

    # For backward compatibility, set the conditions as entry conditions
    # The backend will handle these as a single condition set
    entry_conditions_list = st.session_state.all_conditions
    exit_conditions_list = []  # No separate exit conditions in this mode

    # Submit button
    if st.button("Add Alert", type="primary"):
        # Validate inputs
        if not entry_conditions_list:
            st.error("Please enter at least one entry condition")
        elif ratio == "No" and (not selected_stocks or len(selected_stocks) == 0):
            st.error("Please select at least one stock/asset. Check the 'Apply to ALL' checkbox above to create alerts for all filtered stocks.")
        elif ratio == "Yes" and (not first_stock_name or not second_stock_name):
            st.error("Please select both assets for the ratio alert")
        else:
            # Import exchange mapping
            from src.utils.reference_data import get_country_for_exchange

            # Format conditions for saving
            conditions_dict = {
                "condition_1": {
                    "conditions": entry_conditions_list,
                    "combination_logic": st.session_state.get('condition_logic', 'AND')
                }
            }

            # Save alerts
            if ratio == "Yes":
                # Ratio alert
                try:
                    # Determine exchanges for ratio assets
                    first_exchange = first_exchange if 'first_exchange' in locals() else "Unknown"

                    # Get country from exchange
                    country = get_country_for_exchange(first_exchange)

                    # Generate alert name if not provided
                    if not alert_name:
                        alert_name = f"{first_stock_name}/{second_stock_name} Ratio Alert"

                    futures_db = load_futures_database()
                    is_futures_ratio = (first_ticker in futures_db) or (second_ticker in futures_db)

                    save_ratio_alert(
                        name=alert_name,
                        entry_conditions_list=conditions_dict,
                        combination_logic=st.session_state.get('condition_logic', 'AND'),
                        ticker1=first_ticker,
                        ticker2=second_ticker,
                        stock_name=f"{first_stock_name}/{second_stock_name}",
                        exchange=first_exchange,
                        timeframe=timeframe,
                        last_triggered="",
                        action=action,
                        ratio="Yes",
                        country=country,
                        adjustment_method=adjustment_method if is_futures_ratio else None
                    )
                    st.success(f"✅ Ratio alert created successfully for {first_stock_name}/{second_stock_name}")
                except Exception as e:
                    st.error(f"❌ Error creating ratio alert: {str(e)}")
            else:
                # Individual stock alerts (can be multiple)
                success_count = 0
                error_count = 0

                for stock_name in selected_stocks:
                    try:
                        # Get the stock data from filtered_stocks_data
                        stock_data = filtered_stocks_data[filtered_stocks_data['Name'] == stock_name]

                        if not stock_data.empty:
                            ticker = stock_data.iloc[0]['Symbol']
                            exchange = stock_data.iloc[0].get('Exchange', stock_data.iloc[0].get('exchange', 'Unknown'))
                            country = stock_data.iloc[0].get('Country', get_country_for_exchange(exchange))

                            # Generate alert name if not provided
                            if not alert_name:
                                current_alert_name = generate_alert_name_from_conditions(
                                    stock_name,
                                    st.session_state.entry_conditions,
                                    st.session_state.get('condition_logic', 'AND')
                                )
                            else:
                                # For bulk alerts, append stock name to user-provided name
                                if len(selected_stocks) > 1:
                                    current_alert_name = f"{alert_name} - {stock_name}"
                                else:
                                    current_alert_name = alert_name

                            futures_db = load_futures_database()
                            is_futures_alert = ticker in futures_db

                            save_alert(
                                name=current_alert_name,
                                entry_conditions_list=conditions_dict,
                                combination_logic=st.session_state.get('condition_logic', 'AND'),
                                ticker=ticker,
                                stock_name=stock_name,
                                exchange=exchange,
                                timeframe=timeframe,
                                last_triggered="",
                                action=action,
                                ratio="No",
                                country=country,
                                adjustment_method=adjustment_method if is_futures_alert else None
                            )
                            success_count += 1
                        else:
                            error_count += 1
                            st.warning(f"⚠️ Could not find data for {stock_name}")
                    except Exception as e:
                        error_count += 1
                        st.error(f"❌ Error creating alert for {stock_name}: {str(e)}")

                if success_count > 0:
                    st.success(f"✅ Successfully created {success_count} alert(s)")
                if error_count > 0:
                    st.warning(f"⚠️ Failed to create {error_count} alert(s)")
