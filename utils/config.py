# utils/config.py

import os

# Top-level folder where your data is stored
DATA_ROOT = os.path.join("data")

# Brands to include in the dashboard
# BRANDS = ["Swedbank", "Citadele", "Luminor", "SEB", "Artea"]
BRANDS = ["PUMA", "Nike", "Adidas"]

BRAND_NAME_MAPPING = {
    "PUMA": "PUMA",
    "Nike": "Nike",
    "Adidas": "Adidas"
}

BRAND_COLORS = {
    "PUMA": "#4083B3",  # Plotly Blue
    "Nike":      "#2FB375",  # Teal/Green
    "Adidas":    "#FF0E0E",  # Plotly Orange-Red
}