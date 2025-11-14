# utils/config.py

import os

# Top-level folder where your data is stored
DATA_ROOT = os.path.join("data")

# Brands to include in the dashboard
# BRANDS = ["Swedbank", "Citadele", "Luminor", "SEB", "Artea"]
BRANDS = ["EPSO-G", "Exergi", "Helen", "HOFOR", "Ignitis", "Kauno Energija", "LTG", "LTOU", "Teltonika", "Via Lietuva"]

# Mapping from file name variations to canonical brand names
# This helps normalize brand names extracted from file names
BRAND_NAME_MAPPING = {
    # Canonical names (identity mapping)
    "EPSO-G": "EPSO-G",
    "Exergi": "Exergi",
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
}

BRAND_COLORS = {
    "EPSO-G": "#4083B3",  # Blue
    "Exergi": "#2FB375",  # Teal/Green
    "Helen": "#FF6B35",  # Orange
    "HOFOR": "#4ECDC4",  # Turquoise
    "Ignitis": "#FFE66D",  # Yellow
    "Kauno Energija": "#95E1D3",  # Mint Green
    "LTG": "#F38181",  # Coral
    "LTOU": "#AA96DA",  # Purple
    "Teltonika": "#FCBAD3",  # Pink
    "Via Lietuva": "#A8D8EA",  # Light Blue
}