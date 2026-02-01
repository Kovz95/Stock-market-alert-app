"""
Country code to full name mapping for display and lookup.
"""

# Country code to full name mapping for better display
COUNTRY_CODE_TO_NAME = {
    "US": "United States",
    "CA": "Canada",
    "UK": "United Kingdom",
    "JP": "Japan",
    "DE": "Germany",
    "FR": "France",
    "AU": "Australia",
    "CH": "Switzerland",
    "NL": "Netherlands",
    "IT": "Italy",
    "ES": "Spain",
    "SE": "Sweden",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "BE": "Belgium",
    "IE": "Ireland",
    "PT": "Portugal",
    "AT": "Austria",
    "PL": "Poland",
    "GR": "Greece",
    "HU": "Hungary",
    "CZ": "Czech Republic",
    "TR": "Turkey",
    "MX": "Mexico",
    "HK": "Hong Kong",
    "SG": "Singapore",
    "TW": "Taiwan",
    "MY": "Malaysia",
    "KR": "South Korea",
    "IN": "India",
    "CN": "China",
    "TH": "Thailand",
    "ID": "Indonesia",
    "PH": "Philippines",
    "VN": "Vietnam",
}

# Reverse mapping for lookup (only using primary codes)
COUNTRY_NAME_TO_CODE = {v: k for k, v in COUNTRY_CODE_TO_NAME.items()}


def get_country_display_name(country_code: str) -> str:
    """Convert country code to full display name."""
    return COUNTRY_CODE_TO_NAME.get(country_code, country_code)


def get_country_code(country_name: str) -> str:
    """Convert country name to code."""
    return COUNTRY_NAME_TO_CODE.get(country_name, country_name)
