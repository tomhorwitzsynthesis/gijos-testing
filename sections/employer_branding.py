import os
import json
import html
import streamlit as st
import pandas as pd
import glob
import re
import unicodedata
from pathlib import Path
from utils.config import DATA_ROOT, BRAND_COLORS, BRAND_NAME_MAPPING

BRAND_ORDER = list(BRAND_COLORS.keys())

SOCIAL_COMPOS_DIR = os.path.join(DATA_ROOT, "social_media", "compos")
EMPLOYER_BRANDING_POSTS_PATH = os.path.join(DATA_ROOT, "social_media", "employer_branding", "employer_branding_posts.xlsx")

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
    
    # Check if this looks like letters separated by spaces
    text_no_spaces = re.sub(r'\s+', '', text)
    if len(text_no_spaces) >= 5 and text_no_spaces.isalpha() and len(text.split()) > len(text_no_spaces) * 0.5:
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
    peak_rgb = (31, 179, 117)   # strong green
    ratio = value / 100.0
    r = round(base_rgb[0] + (peak_rgb[0] - base_rgb[0]) * ratio)
    g = round(base_rgb[1] + (peak_rgb[1] - base_rgb[1]) * ratio)
    b = round(base_rgb[2] + (peak_rgb[2] - base_rgb[2]) * ratio)
    bg_hex = f"#{r:02X}{g:02X}{b:02X}"
    text_color = "#1F2933" if value < 60 else "#FFFFFF"
    return bg_hex, text_color


def _render_archetype_matrix(counts_dict, total_count, item_label="posts"):
    """Render archetype matrix with green gradient cards."""
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


@st.cache_data(ttl=0)
def _load_employer_branding_archetypes():
    """Load archetype data for employer branding posts only (employer_branding == 1)."""
    # First, load employer branding posts to get the list of posts with employer_branding == 1
    if not os.path.exists(EMPLOYER_BRANDING_POSTS_PATH):
        return {}
    
    try:
        eb_df = pd.read_excel(EMPLOYER_BRANDING_POSTS_PATH)
        
        # Check for employer_branding column
        if 'employer_branding' not in eb_df.columns:
            return {}
        
        # Filter for employer_branding == 1
        eb_df = eb_df[eb_df['employer_branding'] == 1].copy()
        
        if eb_df.empty:
            return {}
        
        # Get company column
        company_col = None
        for col in ['user_id', 'pageName', 'company', 'brand']:
            if col in eb_df.columns:
                company_col = col
                break
        
        if company_col is None:
            return {}
        
        # Get URL column for matching (could be 'url' or similar)
        url_col = None
        for col in ['url', 'post_url', 'link']:
            if col in eb_df.columns:
                url_col = col
                break
        
        # Create a set of URLs for employer branding posts (if URL matching is possible)
        eb_urls = set()
        if url_col:
            eb_urls = set(eb_df[url_col].dropna().astype(str).str.strip())
        
        # Now load compos files and match with employer branding posts
        stats = {}
        if not os.path.isdir(SOCIAL_COMPOS_DIR):
            return stats
        
        for path in glob.glob(os.path.join(SOCIAL_COMPOS_DIR, "*.xlsx")):
            fname = os.path.basename(path)
            if fname.startswith("~$") or fname == "compos_summary.xlsx":
                continue
            
            brand_display = fname.replace("_compos_analysis.xlsx", "").replace(".xlsx", "").strip()
            normalized_brand = BRAND_NAME_MAPPING.get(brand_display, brand_display)
            
            # Only process if it's one of our current brands
            if normalized_brand not in BRAND_ORDER:
                continue
            
            try:
                try:
                    df_comp = pd.read_excel(path, sheet_name="Raw Data")
                except Exception:
                    df_comp = pd.read_excel(path)
                
                if 'Top Archetype' not in df_comp.columns:
                    continue
                
                # Filter compos data to only employer branding posts
                # Try to match by URL if available
                df_comp_filtered = None
                if url_col and 'url' in df_comp.columns:
                    # Match by URL
                    df_comp['url_str'] = df_comp['url'].astype(str).str.strip()
                    df_comp_filtered = df_comp[df_comp['url_str'].isin(eb_urls)].copy()
                    if 'url_str' in df_comp_filtered.columns:
                        df_comp_filtered = df_comp_filtered.drop('url_str', axis=1)
                else:
                    # If no URL column in compos, try to match by other identifiers
                    # Check if compos has employer_branding column
                    if 'employer_branding' in df_comp.columns:
                        df_comp_filtered = df_comp[df_comp['employer_branding'] == 1].copy()
                    else:
                        # Can't match without URL or employer_branding column, skip this company
                        continue
                
                if df_comp_filtered is None or df_comp_filtered.empty:
                    continue
                
                values = df_comp_filtered['Top Archetype'].dropna()
                if len(values) == 0:
                    continue
                
                counts = values.value_counts()
                canonical_counts = {}
                for archetype, count in counts.items():
                    archetype_str = str(archetype)
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
            except Exception as e:
                # Silently skip errors
                continue
        
        return stats
    except Exception as e:
        st.error(f"Error loading employer branding archetypes: {e}")
        return {}


# Theme display names mapping
THEME_DISPLAY_NAMES = {
    "value_proposition": "Value Proposition",
    "purpose_impact": "Purpose & Impact",
    "target_talent_profile": "Target Talent Profile",
    "company_culture": "Company Culture",
    "learning_possibilities": "Learning Possibilities"
}


@st.cache_data(ttl=0)
def _load_employer_branding_data():
    """Load all employer branding JSON files and return dict of company -> themes data."""
    employer_branding_dir = Path(DATA_ROOT) / "employer_branding"
    
    if not employer_branding_dir.exists():
        return {}
    
    data_by_company = {}
    
    # Get all JSON files (excluding comparison.json)
    json_files = [f for f in employer_branding_dir.glob("*.json") if f.name != "comparison.json"]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            company_name = data.get('company', '')
            if not company_name:
                continue
            
            # Normalize company name to match dashboard naming
            company_normalized = _normalize_company_name(company_name)
            
            themes = data.get('themes', {})
            if themes:
                data_by_company[company_normalized] = themes
        except Exception as e:
            st.error(f"Error loading {json_file.name}: {e}")
            continue
    
    return data_by_company


@st.cache_data(ttl=0)
def _load_comparison_data():
    """Load comparison.json file and return the comparison data."""
    comparison_path = Path(DATA_ROOT) / "employer_branding" / "comparison.json"
    
    if not comparison_path.exists():
        return None
    
    try:
        with open(comparison_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading comparison.json: {e}")
        return None


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


def _render_comparison_tab(comparison_data):
    """Render the comparison tab showing commonalities and differences across companies."""
    if not comparison_data or 'themes' not in comparison_data:
        st.info("No comparison data available.")
        return
    
    st.subheader("Cross-Company Comparison")
    st.markdown("Analysis of commonalities and differences across all companies.")
    
    themes = comparison_data.get('themes', {})
    
    for theme_key, theme_data in themes.items():
        if not isinstance(theme_data, dict):
            continue
        
        # Get display name for theme
        theme_display_name = THEME_DISPLAY_NAMES.get(theme_key, theme_key.replace('_', ' ').title())
        
        st.markdown(f"### {theme_display_name}")
        
        # Common section
        common = theme_data.get('common', {})
        if common:
            st.markdown("#### Common Across Companies")
            
            common_tags = common.get('tags', [])
            if common_tags:
                cols = st.columns(3)
                for i, tag in enumerate(common_tags[:3]):
                    with cols[i]:
                        escaped_tag = html.escape(str(tag))
                        st.markdown(
                            f"""
                            <div style="background-color:#2FB375; color:white; padding:12px 16px; border-radius:20px; text-align:center; font-weight:500; font-size:0.9em; margin-bottom:15px;">
                                {escaped_tag}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            common_summary = common.get('summary', '')
            if common_summary:
                escaped_summary = html.escape(str(common_summary))
                st.markdown(
                    f"""
                    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; border-left:4px solid #2FB375; margin-bottom:25px;">
                        <p style="margin:0; color:#333; line-height:1.6; font-size:0.95em;">{escaped_summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Differences section
        differences = theme_data.get('differences', {})
        if differences:
            st.markdown("#### Key Differences")
            
            diff_tags = differences.get('tags', [])
            if diff_tags:
                cols = st.columns(3)
                for i, tag in enumerate(diff_tags[:3]):
                    with cols[i]:
                        escaped_tag = html.escape(str(tag))
                        st.markdown(
                            f"""
                            <div style="background-color:#FFA726; color:white; padding:12px 16px; border-radius:20px; text-align:center; font-weight:500; font-size:0.9em; margin-bottom:15px;">
                                {escaped_tag}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            diff_summary = differences.get('summary', '')
            if diff_summary:
                escaped_summary = html.escape(str(diff_summary))
                st.markdown(
                    f"""
                    <div style="background-color:#FFF3E0; padding:15px; border-radius:10px; border-left:4px solid #FFA726; margin-bottom:25px;">
                        <p style="margin:0; color:#333; line-height:1.6; font-size:0.95em;">{escaped_summary}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Add spacing between themes
        st.markdown("<br>", unsafe_allow_html=True)


def render():
    """Render employer branding overview with tabs per company showing themes with tags and summaries."""
    data_by_company = _load_employer_branding_data()
    comparison_data = _load_comparison_data()
    
    if not data_by_company and not comparison_data:
        st.info("No employer branding data available.")
        return
    
    st.markdown("### Employer Branding Analysis")
    
    # Get company names and order them (prefer BRAND_ORDER if available)
    companies = list(data_by_company.keys())
    ordered_companies = [c for c in BRAND_ORDER if c in companies]
    extra_companies = sorted([c for c in companies if c not in BRAND_ORDER])
    
    # Create tab labels - add comparison tab first if available
    tab_labels = []
    if comparison_data:
        tab_labels.append("🔍 Comparison")
    tab_labels.extend(ordered_companies + extra_companies)
    
    if not tab_labels:
        st.info("No data available.")
        return
    
    tabs = st.tabs(tab_labels)
    
    # Render comparison tab if available
    tab_idx = 0
    if comparison_data:
        with tabs[tab_idx]:
            _render_comparison_tab(comparison_data)
        tab_idx += 1
    
    # Render company tabs
    company_list = ordered_companies + extra_companies
    for idx, company in enumerate(company_list):
        with tabs[tab_idx + idx]:
            themes = data_by_company.get(company, {})
            
            if not themes:
                st.info(f"No employer branding data available for {company}.")
                continue
            
            st.subheader(f"{company} Employer Branding")
            
            # Display archetype matrix at the top
            archetype_stats = _load_employer_branding_archetypes()
            company_archetype_data = archetype_stats.get(company)
            
            if company_archetype_data:
                st.markdown("#### Archetype Distribution (Employer Branding Posts Only)")
                _render_archetype_matrix(
                    company_archetype_data.get("counts", {}),
                    company_archetype_data.get("total", 0),
                    "posts"
                )
                st.markdown("---")
            
            # Display each theme
            for theme_key, theme_data in themes.items():
                if not isinstance(theme_data, dict):
                    continue
                
                tags = theme_data.get('tags', [])
                summary = theme_data.get('summary', '')
                
                # Get display name for theme
                theme_display_name = THEME_DISPLAY_NAMES.get(theme_key, theme_key.replace('_', ' ').title())
                
                # Theme header
                st.markdown(f"#### {theme_display_name}")
                
                # Display 3 tags in green bubbles
                if tags:
                    # Create columns for the 3 tags
                    cols = st.columns(3)
                    for i, tag in enumerate(tags[:3]):  # Ensure max 3 tags
                        with cols[i]:
                            # Escape HTML to prevent injection
                            escaped_tag = html.escape(str(tag))
                            st.markdown(
                                f"""
                                <div style="background-color:#2FB375; color:white; padding:12px 16px; border-radius:20px; text-align:center; font-weight:500; font-size:0.9em; margin-bottom:15px;">
                                    {escaped_tag}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                # Display summary text below
                if summary:
                    # Escape HTML to prevent injection
                    escaped_summary = html.escape(str(summary))
                    st.markdown(
                        f"""
                        <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; border-left:4px solid #2FB375; margin-bottom:25px;">
                            <p style="margin:0; color:#333; line-height:1.6; font-size:0.95em;">{escaped_summary}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Add spacing between themes
                st.markdown("<br>", unsafe_allow_html=True)

