# utils/config.py

import os

# Top-level folder where your data is stored
DATA_ROOT = os.path.join("data")

# Brands to include in the dashboard
# BRANDS = ["Swedbank", "Citadele", "Luminor", "SEB", "Artea"]
BRANDS = ["EPSO-G", "Exergi", "Gijos", "Helen", "HOFOR", "Ignitis", "Kauno Energija", "LTG", "LTOU", "Teltonika", "Via Lietuva"]

# Mapping from file name variations to canonical brand names
# This helps normalize brand names extracted from file names
BRAND_NAME_MAPPING = {
    # Canonical names (identity mapping)
    "EPSO-G": "EPSO-G",
    "Exergi": "Exergi",
    "Gijos": "Gijos",
    "Helen": "Helen",
    "HOFOR": "HOFOR",
    "Ignitis": "Ignitis",
    "Kauno Energija": "Kauno Energija",
    "LTG": "LTG",
    "LTOU": "LTOU",
    "Teltonika": "Teltonika",
    "Via Lietuva": "Via Lietuva",
    # File name variations (lowercase with hyphens)
    "epso-g": "EPSO-G",
    "exergi": "Exergi",
    "stockholm-exergi": "Exergi",
    "helen": "Helen",
    "helen-oy": "Helen",
    "hofor": "HOFOR",
    "hofor-a-s": "HOFOR",
    "ignitis": "Ignitis",
    "ignitis grupė": "Ignitis",
    "ignitis grupe": "Ignitis",
    "ignitis-grupe": "Ignitis",
    "kauno-energija": "Kauno Energija",
    "kauno energija": "Kauno Energija",
    "ltg": "LTG",
    "ab-lietuvos-gelezinkeliai": "LTG",
    "lietuvos geležinkeliai": "LTG",
    "ltou": "LTOU",
    "lithuanian-airports": "LTOU",
    "vilnius airport": "LTOU",
    "teltonika": "Teltonika",
    "via-lietuva": "Via Lietuva",
    "via lietuva": "Via Lietuva",
    # Additional variations from ads data
    "stockholm exergi ab": "Exergi",
    "stockholm exergi": "Exergi",
    # Gijos variations from source files
    "gijos": "Gijos",
    "gijos_compos_analysis": "Gijos",
    "gijos_compos": "Gijos",
    "miestogijos": "Gijos",
    "miestosgijos": "Gijos",
    "miesto gijos": "Gijos",
    "miestos gijos": "Gijos",
    "miesto-gijos": "Gijos",
    "miestos-gijos": "Gijos",
    "ads_scraping_miestosgijos": "Gijos",
}

BRAND_COLORS = {
    "EPSO-G": "#18958F",  # Blue
    "Exergi": "#3066BE",  # Teal/Green
    "Gijos": "#3D348B",  # Fresh green
    "Helen": "#71216C",  # Orange
    "HOFOR": "#A40E4C",  # Turquoise
    "Ignitis": "#FE5F55",  # Yellow
    "Kauno Energija": "#F9A051",  # Mint Green
    "LTG": "#F4E04D",  # Coral
    "LTOU": "#18958F",  # Purple
    "Teltonika": "#3066BE",  # Pink
    "Via Lietuva": "#71216C",  # Light Blue
}

# Global UI color palette
PRIMARY_ACCENT_COLOR = "#00C35F"
POSITIVE_HIGHLIGHT_COLOR = "#0E34A0"
NEGATIVE_HIGHLIGHT_COLOR = "#F25757"
NEUTRAL_HIGHLIGHT_COLOR = "#6B7280"