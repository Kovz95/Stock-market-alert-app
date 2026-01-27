"""Stock filtering utilities for market scanner."""

from typing import Any


def count_filtered_symbols(
    stock_db: dict[str, dict[str, Any]],
    *,
    selected_portfolio: str,
    portfolio_symbols: set[str],
    asset_type_filter: list[str],
    selected_countries: list[str],
    selected_exchanges: list[str],
    selected_economies: list[str],
    selected_sectors: list[str],
    selected_subsectors: list[str],
    selected_industry_groups: list[str],
    selected_industries: list[str],
    selected_subindustries: list[str],
) -> int:
    """
    Count how many symbols match the current filter criteria.

    Args:
        stock_db: Stock metadata database
        selected_portfolio: Portfolio name or "All"
        portfolio_symbols: Set of symbols in selected portfolio
        asset_type_filter: List of asset types to include
        selected_countries: List of countries to filter
        selected_exchanges: List of exchanges to filter
        selected_economies: List of RBICS economies to filter
        selected_sectors: List of RBICS sectors to filter
        selected_subsectors: List of RBICS subsectors to filter
        selected_industry_groups: List of RBICS industry groups to filter
        selected_industries: List of RBICS industries to filter
        selected_subindustries: List of RBICS subindustries to filter

    Returns:
        Count of symbols matching all filters
    """
    count = 0
    for ticker, info in stock_db.items():
        if symbol_matches_filters(
            ticker,
            info,
            selected_portfolio=selected_portfolio,
            portfolio_symbols=portfolio_symbols,
            asset_type_filter=asset_type_filter,
            selected_countries=selected_countries,
            selected_exchanges=selected_exchanges,
            selected_economies=selected_economies,
            selected_sectors=selected_sectors,
            selected_subsectors=selected_subsectors,
            selected_industry_groups=selected_industry_groups,
            selected_industries=selected_industries,
            selected_subindustries=selected_subindustries,
        ):
            count += 1
    return count


def symbol_matches_filters(
    ticker: str,
    info: dict[str, Any],
    *,
    selected_portfolio: str,
    portfolio_symbols: set[str],
    asset_type_filter: list[str],
    selected_countries: list[str],
    selected_exchanges: list[str],
    selected_economies: list[str],
    selected_sectors: list[str],
    selected_subsectors: list[str],
    selected_industry_groups: list[str],
    selected_industries: list[str],
    selected_subindustries: list[str],
) -> bool:
    """
    Check if a symbol matches all filter criteria.

    Args:
        ticker: Symbol ticker
        info: Symbol metadata
        (same filter args as count_filtered_symbols)

    Returns:
        True if symbol matches all filters, False otherwise
    """
    # Portfolio filter
    if selected_portfolio != "All" and ticker not in portfolio_symbols:
        return False

    # Asset type filter
    if asset_type_filter and info.get("asset_type") not in asset_type_filter:
        return False

    # Country filter
    if selected_countries and info.get("country") not in selected_countries:
        return False

    # Exchange filter
    if selected_exchanges and info.get("exchange") not in selected_exchanges:
        return False

    # RBICS filters (only for stocks)
    if info.get("asset_type") == "Stock":
        if selected_economies and info.get("rbics_economy") not in selected_economies:
            return False
        if selected_sectors and info.get("rbics_sector") not in selected_sectors:
            return False
        if selected_subsectors and info.get("rbics_subsector") not in selected_subsectors:
            return False
        if selected_industry_groups and info.get("rbics_industry_group") not in selected_industry_groups:
            return False
        if selected_industries and info.get("rbics_industry") not in selected_industries:
            return False
        if selected_subindustries and info.get("rbics_subindustry") not in selected_subindustries:
            return False

    return True


def get_filtered_symbols(
    stock_db: dict[str, dict[str, Any]],
    *,
    selected_portfolio: str,
    portfolio_symbols: set[str],
    asset_type_filter: list[str],
    selected_countries: list[str],
    selected_exchanges: list[str],
    selected_economies: list[str],
    selected_sectors: list[str],
    selected_subsectors: list[str],
    selected_industry_groups: list[str],
    selected_industries: list[str],
    selected_subindustries: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Get all symbols that match the filter criteria.

    Args:
        stock_db: Stock metadata database
        (same filter args as count_filtered_symbols)

    Returns:
        List of (ticker, info) tuples for symbols matching filters
    """
    filtered = []
    for ticker, info in stock_db.items():
        if symbol_matches_filters(
            ticker,
            info,
            selected_portfolio=selected_portfolio,
            portfolio_symbols=portfolio_symbols,
            asset_type_filter=asset_type_filter,
            selected_countries=selected_countries,
            selected_exchanges=selected_exchanges,
            selected_economies=selected_economies,
            selected_sectors=selected_sectors,
            selected_subsectors=selected_subsectors,
            selected_industry_groups=selected_industry_groups,
            selected_industries=selected_industries,
            selected_subindustries=selected_subindustries,
        ):
            filtered.append((ticker, info))
    return filtered


def extract_unique_values_from_stock_db(
    stock_db: dict[str, dict[str, Any]], field: str, asset_type: str | None = None
) -> list[str]:
    """
    Extract unique values for a field from stock database.

    Args:
        stock_db: Stock metadata database
        field: Field name to extract
        asset_type: Optional asset type filter

    Returns:
        Sorted list of unique non-empty values
    """
    values = set()
    for info in stock_db.values():
        if asset_type and info.get("asset_type") != asset_type:
            continue
        value = info.get(field)
        if value:
            values.add(value)
    return sorted(values)
