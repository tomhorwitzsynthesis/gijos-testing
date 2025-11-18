import streamlit as st
import json
import html
import os
import re
import unicodedata
from pathlib import Path
from utils.config import BRAND_NAME_MAPPING, DATA_ROOT, BRAND_COLORS

BRAND_ORDER = list(BRAND_COLORS.keys())

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
    "jet-setter": "The Globe-trotter",
    "jetsetter": "The Globe-trotter",
    "globe trotter": "The Globe-trotter",
    "globe-trotter": "The Globe-trotter",
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
    "people's champion": "The People's Champion",
    "eco warrior": "The Eco Warrior",
    "ecowarrior": "The Eco Warrior",
}


def _canonicalize_archetype(name: str) -> str | None:
    """Canonicalize archetype name to match display order."""
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


def _normalize_company_name(company: str) -> str:
    """Normalize company name using BRAND_NAME_MAPPING."""
    if not isinstance(company, str):
        return ""
    company_str = str(company).strip()
    
    # Use case-insensitive lookup
    brand_lookup = {k.lower(): v for k, v in BRAND_NAME_MAPPING.items()}
    company_lower = company_str.lower()
    
    # Try case-insensitive match first
    if company_lower in brand_lookup:
        return brand_lookup[company_lower]
    
    # Try exact match
    if company_str in BRAND_NAME_MAPPING:
        return BRAND_NAME_MAPPING[company_str]
    
    # Try to find in BRAND_ORDER
    for brand in BRAND_ORDER:
        if brand.lower() == company_lower:
            return brand
    
    # Return original if no match found
    return company_str


@st.cache_data(ttl=0)
def _load_archetypes_data():
    """Load archetypes from archetypes.json file."""
    archetypes_path = Path(DATA_ROOT) / "employer_branding" / "archetypes.json"
    
    if not archetypes_path.exists():
        return {}
    
    try:
        with open(archetypes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalize company names and archetypes
        normalized_data = {}
        for company, archetypes in data.items():
            normalized_company = _normalize_company_name(company)
            if not normalized_company:
                # Try to use the company name as-is if normalization fails
                normalized_company = company
            
            canonical_archetypes = []
            for archetype in archetypes:
                archetype_str = str(archetype).strip()
                # First, check if it's already in the correct format
                if archetype_str in ARCHETYPE_DISPLAY_ORDER:
                    canonical_archetypes.append(archetype_str)
                else:
                    # Try canonicalization
                    canonical = _canonicalize_archetype(archetype_str)
                    if canonical:
                        canonical_archetypes.append(canonical)
            
            if canonical_archetypes:
                normalized_data[normalized_company] = canonical_archetypes
        
        return normalized_data
    except Exception as e:
        st.error(f"Error loading archetypes.json: {e}")
        return {}


def _render_archetype_matrix_binary(archetypes_set):
    """Render archetype matrix with binary green/white coloring (no percentages)."""
    card_base_style = (
        "border:1px solid #CDE7D8; border-radius:10px; padding:12px; "
        "margin-bottom:12px; text-align:center;"
    )
    
    for row_start in range(0, len(ARCHETYPE_DISPLAY_ORDER), 4):
        cols = st.columns(4)
        for idx, archetype in enumerate(ARCHETYPE_DISPLAY_ORDER[row_start:row_start + 4]):
            is_present = archetype in archetypes_set
            
            # Green if present, white if not
            if is_present:
                bg_hex = "#1FB375"  # Strong green
                text_color = "#FFFFFF"
            else:
                bg_hex = "#FFFFFF"
                text_color = "#1F2933"
            
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="{card_base_style} background-color:{bg_hex}; color:{text_color};">
                        <h5 style="margin:0;">{archetype}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_archetype_matrix_comparison(archetype_companies_map, total_companies):
    """Render comparison matrix with gradient based on number of companies, with hover tooltips."""
    card_base_style = (
        "border:1px solid #CDE7D8; border-radius:10px; padding:12px; "
        "margin-bottom:12px; text-align:center; position:relative;"
    )
    
    for row_start in range(0, len(ARCHETYPE_DISPLAY_ORDER), 4):
        cols = st.columns(4)
        for idx, archetype in enumerate(ARCHETYPE_DISPLAY_ORDER[row_start:row_start + 4]):
            companies = archetype_companies_map.get(archetype, [])
            count = len(companies)
            
            # Calculate gradient intensity (0-100%)
            if total_companies > 0:
                intensity = (count / total_companies) * 100
            else:
                intensity = 0
            
            # Generate gradient color
            base_rgb = (245, 255, 249)  # light mint
            peak_rgb = (31, 179, 117)   # strong green
            ratio = intensity / 100.0
            r = round(base_rgb[0] + (peak_rgb[0] - base_rgb[0]) * ratio)
            g = round(base_rgb[1] + (peak_rgb[1] - base_rgb[1]) * ratio)
            b = round(base_rgb[2] + (peak_rgb[2] - base_rgb[2]) * ratio)
            bg_hex = f"#{r:02X}{g:02X}{b:02X}"
            text_color = "#1F2933" if intensity < 60 else "#FFFFFF"
            
            # Create tooltip text (escape HTML for safety)
            companies_text = ", ".join(companies) if companies else "None"
            tooltip_text = html.escape(f"{count} company{'ies' if count != 1 else ''}: {companies_text}")
            
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="{card_base_style} background-color:{bg_hex}; color:{text_color}; cursor:pointer;"
                         title="{tooltip_text}">
                        <h5 style="margin:0;">{archetype}</h5>
                        <p style="margin:6px 0 0; font-size:0.9em; opacity:0.85;">{count}/{total_companies}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render():
    st.markdown("### Employer Branding Archetype Matrix")
    
    archetypes_data = _load_archetypes_data()
    
    if not archetypes_data:
        st.info("No archetype data available. Ensure archetypes.json exists in data/employer_branding/ folder.")
        return
    
    # Calculate comparison data (which companies have which archetypes)
    archetype_companies_map = {}
    for company, archetypes in archetypes_data.items():
        for archetype in archetypes:
            if archetype not in archetype_companies_map:
                archetype_companies_map[archetype] = []
            archetype_companies_map[archetype].append(company)
    
    total_companies = len(archetypes_data)
    
    # Get company names and order them (prefer BRAND_ORDER if available)
    companies = list(archetypes_data.keys())
    ordered_companies = [c for c in BRAND_ORDER if c in companies]
    extra_companies = sorted([c for c in companies if c not in BRAND_ORDER])
    
    # Create tab labels - add comparison tab first
    tab_labels = ["🔍 Comparison"] + ordered_companies + extra_companies
    
    tabs = st.tabs(tab_labels)
    
    # Render comparison tab
    with tabs[0]:
        st.subheader("Cross-Company Archetype Comparison")
        st.markdown("Archetypes are colored based on how many companies have them. Hover over each archetype to see which companies.")
        _render_archetype_matrix_comparison(archetype_companies_map, total_companies)
    
    # Render company tabs
    company_list = ordered_companies + extra_companies
    for idx, company in enumerate(company_list):
        with tabs[idx + 1]:
            st.subheader(f"{company} Archetypes")
            archetypes = archetypes_data.get(company, [])
            archetypes_set = set(archetypes)
            _render_archetype_matrix_binary(archetypes_set)
    
    st.markdown(
        'Read more about brand archetypes here: [Brandtypes](https://www.comp-os.com/brandtypes)'
    )

