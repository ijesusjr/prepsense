"""
core/regions.py
----------------
Country → regional neighbours mapping for HAVEN regional risk fetcher.

Used by:
    api/main.py   — to derive region_countries when location changes
    app.py        — to check if selected country is supported and show warning

Adding a new country: add it to REGION_NEIGHBOURS with its neighbours list.
The app will automatically support it in both the API and the UI warning.
"""

# ---------------------------------------------------------------------------
# Neighbours map
# ---------------------------------------------------------------------------

REGION_NEIGHBOURS = {
    # Iberian
    "Spain":          ["Spain", "France", "Portugal", "Morocco", "Algeria"],
    "Portugal":       ["Portugal", "Spain", "Morocco"],

    # Western Europe
    "France":         ["France", "Spain", "Belgium", "Luxembourg", "Germany",
                       "Switzerland", "Italy"],
    "Belgium":        ["Belgium", "Netherlands", "Luxembourg", "France", "Germany"],
    "Netherlands":    ["Netherlands", "Belgium", "Germany"],
    "Luxembourg":     ["Luxembourg", "Belgium", "France", "Germany"],

    # Central Europe
    "Germany":        ["Germany", "France", "Belgium", "Netherlands", "Luxembourg",
                       "Denmark", "Poland", "Czech Republic", "Austria", "Switzerland"],
    "Austria":        ["Austria", "Germany", "Czech Republic", "Slovakia", "Hungary",
                       "Slovenia", "Italy", "Switzerland"],
    "Switzerland":    ["Switzerland", "France", "Germany", "Austria", "Italy"],
    "Czech Republic": ["Czech Republic", "Germany", "Poland", "Slovakia", "Austria"],
    "Slovakia":       ["Slovakia", "Czech Republic", "Poland", "Ukraine", "Hungary",
                       "Austria"],
    "Hungary":        ["Hungary", "Austria", "Slovakia", "Ukraine", "Romania",
                       "Serbia", "Croatia", "Slovenia"],
    "Poland":         ["Poland", "Germany", "Czech Republic", "Slovakia", "Ukraine",
                       "Lithuania"],

    # Nordic
    "Sweden":         ["Sweden", "Norway", "Finland", "Denmark"],
    "Norway":         ["Norway", "Sweden", "Finland"],
    "Finland":        ["Finland", "Sweden", "Norway", "Estonia"],
    "Denmark":        ["Denmark", "Germany", "Sweden", "Norway"],

    # Baltic
    "Estonia":        ["Estonia", "Latvia", "Finland"],
    "Latvia":         ["Latvia", "Estonia", "Lithuania"],
    "Lithuania":      ["Lithuania", "Latvia", "Poland"],

    # Southern Europe
    "Italy":          ["Italy", "France", "Switzerland", "Austria", "Slovenia"],
    "Greece":         ["Greece", "Bulgaria", "Albania", "North Macedonia", "Turkey"],
    "Croatia":        ["Croatia", "Slovenia", "Hungary", "Serbia", "Bosnia"],
    "Slovenia":       ["Slovenia", "Italy", "Austria", "Hungary", "Croatia"],

    # Southeast Europe
    "Romania":        ["Romania", "Hungary", "Ukraine", "Moldova", "Bulgaria",
                       "Serbia"],
    "Bulgaria":       ["Bulgaria", "Romania", "Serbia", "North Macedonia", "Greece",
                       "Turkey"],

    # Islands / other EU
    "Cyprus":         ["Cyprus", "Greece", "Turkey"],
    "Malta":          ["Malta", "Italy", "Tunisia"],
    "Ireland":        ["Ireland", "United Kingdom"],

    # Non-EU commonly used
    "United Kingdom": ["United Kingdom", "Ireland", "France", "Netherlands", "Belgium"],
}

# Fallback when country is not in the map
DEFAULT_REGION = ["Spain", "France", "Portugal", "Morocco", "Algeria"]

# Flat set of supported country names — used for the UI warning check
SUPPORTED_COUNTRIES = set(REGION_NEIGHBOURS.keys())


def get_region(country: str) -> list:
    """
    Return the neighbours list for a given country.
    Falls back to DEFAULT_REGION if the country is not supported.

    Args:
        country: Country name as returned by Nominatim reverse geocoding.

    Returns:
        List of country names to monitor for regional risk.
    """
    return REGION_NEIGHBOURS.get(country, DEFAULT_REGION)


def is_supported(country: str) -> bool:
    """Return True if the country has a defined neighbours list."""
    return country in REGION_NEIGHBOURS