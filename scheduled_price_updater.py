"""
Shim loader for scheduled_price_updater.

This auto-generated file loads the preserved bytecode so imports keep
working until the original source is restored.
"""

from __future__ import annotations

import importlib
import marshal
from pathlib import Path
from types import CodeType
from typing import Any, Dict

try:
    from exchange_country_mapping import EXCHANGE_COUNTRY_MAP as EXCHANGE_TO_COUNTRY
except Exception:
    EXCHANGE_TO_COUNTRY = {}

# Local fallback so price updates still work if the mapping file cannot be imported.
FALLBACK_EXCHANGE_COUNTRY_MAP = {
    # Asia-Pacific
    "TOKYO": "Japan",
    "TAIWAN": "Taiwan",
    "HONG KONG": "Hong Kong",
    "SINGAPORE": "Singapore",
    "MALAYSIA": "Malaysia",
    "INDONESIA": "Indonesia",
    "THAILAND": "Thailand",
    "ASX": "Australia",
    "OMX NORDIC ICELAND": "Iceland",
    "OMX NORDIC STOCKHOLM": "Sweden",
    "OMX NORDIC HELSINKI": "Finland",
    "OMX NORDIC COPENHAGEN": "Denmark",

    # India
    "BSE INDIA": "India",
    "NSE INDIA": "India",

    # Europe
    "LONDON": "United Kingdom",
    "XETRA": "Germany",
    "FRANKFURT": "Germany",
    "EURONEXT AMSTERDAM": "Netherlands",
    "EURONEXT BRUSSELS": "Belgium",
    "EURONEXT DUBLIN": "Ireland",
    "EURONEXT LISBON": "Portugal",
    "EURONEXT PARIS": "France",
    "MILAN": "Italy",
    "MADRID": "Spain",
    "SPAIN": "Spain",
    "VIENNA": "Austria",
    "WARSAW": "Poland",
    "ATHENS": "Greece",
    "PRAGUE": "Czech Republic",
    "BUDAPEST": "Hungary",
    "OSLO": "Norway",
    "SIX SWISS": "Switzerland",

    # Americas
    "NASDAQ": "United States",
    "NYSE": "United States",
    "NYSE AMERICAN": "United States",
    "NYSE ARCA": "United States",
    "CBOE BZX": "United States",
    "TORONTO": "Canada",
    "SANTIAGO": "Chile",
    "BUENOS AIRES": "Argentina",
    "MEXICO": "Mexico",
    "COLOMBIA": "Colombia",
    "SAO PAULO": "Brazil",

    # Middle East / Africa
    "ISTANBUL": "Turkey",
    "JSE": "South Africa",
}

_LEGACY_BYTECODE = Path(__file__).parent / "__pycache__/scheduled_price_updater_legacy.cpython-311.pyc"
_PYC_HEADER_SIZE = 16
_BOOTSTRAPPED = False


def _load_legacy_code() -> CodeType:
    if not _LEGACY_BYTECODE.exists():
        raise FileNotFoundError(
            f"Missing legacy bytecode for scheduled_price_updater at {_LEGACY_BYTECODE}."
        )
    with _LEGACY_BYTECODE.open("rb") as fh:
        fh.read(_PYC_HEADER_SIZE)
        return marshal.load(fh)


def _bootstrap(namespace: Dict[str, Any]) -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    code = _load_legacy_code()
    exec(code, namespace)
    _BOOTSTRAPPED = True


_bootstrap(globals())


def _get_exchange_mapping():
    """
    Return the exchange->country map, re-importing if the initial import failed.
    This avoids NameError if the module is loaded from a different working dir.
    """
    if EXCHANGE_TO_COUNTRY:
        return EXCHANGE_TO_COUNTRY
    try:
        mod = importlib.import_module("exchange_country_mapping")
        return getattr(mod, "EXCHANGE_COUNTRY_MAP", {}) or {}
    except Exception:
        return FALLBACK_EXCHANGE_COUNTRY_MAP


# Patch the legacy __init__ to avoid reading exchange_country_mapping.py from CWD.
# The compiled bytecode tries to exec the mapping file via a relative path, which
# fails when the scheduler runs outside the repo root. We already import the map
# above, so rebuild __init__ without the file read.
def _patched_init(self):
    self.collector = OptimizedDailyPriceCollector()
    self.db = DailyPriceDatabase()
    self.monitor = PriceUpdateMonitor()
    self.exchange_mapping = _get_exchange_mapping()
    self.metadata = fetch_stock_metadata_map()


ScheduledPriceUpdater.__init__ = _patched_init
