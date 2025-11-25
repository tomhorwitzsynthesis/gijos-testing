import streamlit as st
import pandas as pd
import os
import glob
import re
import unicodedata
from utils.config import BRANDS, BRAND_NAME_MAPPING, DATA_ROOT, PRIMARY_ACCENT_COLOR

SOCIAL_COMPOS_DIR = os.path.join(DATA_ROOT, "social_media", "compos")
SOCIAL_COMPOS_SUMMARY_PATH = os.path.join(SOCIAL_COMPOS_DIR, "compos_summary.xlsx")

ARCHETYPE_DISPLAY_ORDER = [
    "The Technologist",
    "The Optimiser",
    "The Globe-trotter",
    "The Accelerator",
    "The Value Seeker",
    "The Expert",
    "The Guardian",
    "The Futurist",
    "The Simplifier",
    "The Personaliser",
    "The Principled",
    "The Collaborator",
    "The Mentor",
    "The Nurturer",
    "The People's Champion",
    "The Eco Warrior",
]

_ARCHETYPE_CANONICAL_MAP = {
    "technologist": "The Technologist",
    "optimizer": "The Optimiser",
    "optimiser": "The Optimiser",
    "jet setter": "The Globe-trotter",
    "jetsetter": "The Globe-trotter",
    "globe trotter": "The Globe-trotter",
    "accelerator": "The Accelerator",
    "value seeker": "The Value Seeker",
    "valueseeker": "The Value Seeker",
    "value s": "The Value Seeker",
    "expert": "The Expert",
    "guardian": "The Guardian",
    "futurist": "The Futurist",
    "simplifier": "The Simplifier",
    "personalizer": "The Personaliser",
    "personaliser": "The Personaliser",
    "principled": "The Principled",
    "collaborator": "The Collaborator",
    "mentor": "The Mentor",
    "nurturer": "The Nurturer",
    "peoples champion": "The People's Champion",
    "people champion": "The People's Champion",
    "people s champion": "The People's Champion",
    "eco warrior": "The Eco Warrior",
    "ecowarrior": "The Eco Warrior",
}


def _canonicalize_archetype(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    text = unicodedata.normalize("NFKD", name)
    text = text.replace("", "'").replace("'", "'")
    
    # Check if this looks like letters separated by spaces (e.g., "T H E   C O L L A B O R A T O R")
    # If so, remove all spaces to reconstruct the word
    text_no_spaces = re.sub(r'\s+', '', text)
    if len(text_no_spaces) >= 5 and text_no_spaces.isalpha() and len(text.split()) > len(text_no_spaces) * 0.5:
        # Looks like spaced letters - use the space-removed version
        text = text_no_spaces.lower()
    else:
        text = text.lower().strip()
    
    if text.startswith("the"):
        text = re.sub(r'^the\s*', '', text)
    text = text.replace("-", " ")
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z ]+", " ", text)
    text = " ".join(text.split())
    if not text:
        return None
    mapped = _ARCHETYPE_CANONICAL_MAP.get(text)
    if mapped:
        return mapped
    display = " ".join(word.capitalize() for word in text.split())
    return f"The {display}" if display else None


def _green_gradient(pct: float) -> tuple[str, str]:
    """Return background and text colors based on percentage intensity."""
    try:
        value = float(pct)
    except (TypeError, ValueError):
        value = 0.0
    value = max(0.0, min(100.0, value))
    base_rgb = (245, 255, 249)  # light mint
    peak_hex = PRIMARY_ACCENT_COLOR.lstrip("#")
    peak_rgb = tuple(int(peak_hex[i : i + 2], 16) for i in (0, 2, 4))
    ratio = value / 100.0
    r = round(base_rgb[0] + (peak_rgb[0] - base_rgb[0]) * ratio)
    g = round(base_rgb[1] + (peak_rgb[1] - base_rgb[1]) * ratio)
    b = round(base_rgb[2] + (peak_rgb[2] - base_rgb[2]) * ratio)
    bg_hex = f"#{r:02X}{g:02X}{b:02X}"
    text_color = "#1F2933" if value < 60 else "#FFFFFF"
    return bg_hex, text_color


def _load_archetype_stats_from_summary():
    """Load archetype statistics from compos summary file (preferred)."""
    if not os.path.exists(SOCIAL_COMPOS_SUMMARY_PATH):
        return {}
    df = pd.read_excel(SOCIAL_COMPOS_SUMMARY_PATH)
    stats = {}
    for col in df.columns:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        counts = values.value_counts()
        canonical_counts = {}
        for archetype, count in counts.items():
            archetype_str = str(archetype)
            # Try direct match first (in case it's already in correct format)
            if archetype_str in ARCHETYPE_DISPLAY_ORDER:
                canonical = archetype_str
            else:
                canonical = _canonicalize_archetype(archetype_str)
            if not canonical:
                continue
            canonical_counts[canonical] = canonical_counts.get(canonical, 0) + int(count)
        total = int(sum(canonical_counts.values()))
        if total == 0:
            continue
        stats[col] = {
            "total": total,
            "counts": canonical_counts,
        }
    return stats


def _load_archetype_stats_from_compos():
    """Load archetype statistics from individual compos files (fallback)."""
    stats = {}
    if not os.path.isdir(SOCIAL_COMPOS_DIR):
        return stats
    for path in glob.glob(os.path.join(SOCIAL_COMPOS_DIR, "*.xlsx")):
        fname = os.path.basename(path)
        if fname.startswith("~$"):
            continue
        brand_display = fname.replace("_compos_analysis.xlsx", "").replace(".xlsx", "").strip()
        # Normalize brand name from file name to canonical name
        normalized_brand = BRAND_NAME_MAPPING.get(brand_display, brand_display)
        # Only process if it's one of our current brands
        if normalized_brand not in BRANDS:
            continue
        try:
            try:
                df_comp = pd.read_excel(path, sheet_name="Raw Data")
            except Exception:
                df_comp = pd.read_excel(path)
            if 'Top Archetype' not in df_comp.columns:
                continue
            values = df_comp['Top Archetype'].dropna()
            if len(values) == 0:
                continue
            counts = values.value_counts()
            canonical_counts = {}
            for archetype, count in counts.items():
                archetype_str = str(archetype)
                # Try direct match first (in case it's already in correct format)
                if archetype_str in ARCHETYPE_DISPLAY_ORDER:
                    canonical = archetype_str
                else:
                    canonical = _canonicalize_archetype(archetype_str)
                if not canonical:
                    continue
                canonical_counts[canonical] = canonical_counts.get(canonical, 0) + int(count)
            total = int(sum(canonical_counts.values()))
            if total == 0:
                continue
            stats[normalized_brand] = {
                "total": total,
                "counts": canonical_counts,
            }
        except Exception:
            pass
    return stats


def _render_archetype_matrix(counts_dict, total_count, item_label="posts"):
    counts_dict = counts_dict or {}
    total = total_count if total_count and total_count > 0 else 0
    card_base_style = (
        "border:1px solid #CDE7D8; border-radius:10px; padding:12px; "
        "margin-bottom:12px; text-align:center;"
    )
    for row_start in range(0, len(ARCHETYPE_DISPLAY_ORDER), 4):
        cols = st.columns(4)
        for idx, archetype in enumerate(ARCHETYPE_DISPLAY_ORDER[row_start:row_start + 4]):
            pct_value = 0.0
            count_value = int(counts_dict.get(archetype, 0))
            if total > 0:
                pct_value = (count_value / total) * 100
            bg_hex, text_color = _green_gradient(pct_value)
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="{card_base_style} background-color:{bg_hex}; color:{text_color};">
                        <h5 style="margin:0;">{archetype}</h5>
                        <p style="margin:6px 0 0; font-size:1.1em; font-weight:600;">{pct_value:.1f}%</p>
                        <p style="margin:0; font-size:0.85em; color:{text_color}; opacity:0.85;">{count_value} {item_label}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render():
    st.markdown("### Archetype Coverage Matrix")
    
    # Try summary file first, then fall back to individual compos files
    archetype_stats = _load_archetype_stats_from_summary()
    if not archetype_stats:
        archetype_stats = _load_archetype_stats_from_compos()
    
    if not archetype_stats:
        st.info("No archetype data available. Ensure compos files are in data/social_media/compos or compos_summary.xlsx exists.")
        return
    
    # Calculate overall counts
    overall_counts = {}
    overall_total = 0
    for info in archetype_stats.values():
        counts = info.get("counts", {})
        for archetype, count in counts.items():
            count_int = int(count)
            overall_counts[archetype] = overall_counts.get(archetype, 0) + count_int
            overall_total += count_int
    
    # Create tabs for overall and each company
    matrix_labels = ["Overall"] + list(archetype_stats.keys())
    matrix_tabs = st.tabs(matrix_labels)
    
    with matrix_tabs[0]:
        st.subheader("Overall Archetype Distribution")
        _render_archetype_matrix(overall_counts, overall_total, "posts")
    
    for idx, (company, info) in enumerate(archetype_stats.items()):
        with matrix_tabs[idx + 1]:
            st.subheader(f"{company} Archetype Distribution")
            _render_archetype_matrix(info.get("counts", {}), info.get("total", 0), "posts")
    
    st.markdown(
        'Read more about brand archetypes here: [Brandtypes](https://www.comp-os.com/brandtypes)'
    )

