# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import glob
import ast

# Configure wide layout
st.set_page_config(layout="wide")

# --- Paths & helpers ---
DATA_DIR = "data_LP_ads"
ADS_DIR = os.path.join(DATA_DIR, "ads")
COMPOS_DIR = os.path.join(DATA_DIR, "compos")
CREATIVITY_DIR = os.path.join(DATA_DIR, "creativity")
KEY_ADVANTAGES_DIR = os.path.join(DATA_DIR, "key_advantages")


def find_ads_file():
    # Prefer explicitly known file, else fallback to first xlsx in ads dir
    preferred = os.path.join(ADS_DIR, "ads_scraping_LP.xlsx")
    if os.path.exists(preferred):
        return preferred
    candidates = sorted(glob.glob(os.path.join(ADS_DIR, "*.xlsx")))
    return candidates[0] if candidates else None


def load_key_advantages():
    """Load key advantages data from key_advantages.xlsx"""
    path = os.path.join(KEY_ADVANTAGES_DIR, "key_advantages.xlsx")
    if not os.path.exists(path):
        return {}
    
    advantages = {}
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(path)
        for sheet_name in excel_file.sheet_names:
            # Convert sheet name back to brand name (replace underscores with spaces)
            brand_name = sheet_name.replace("_", " ")
            df_adv = pd.read_excel(path, sheet_name=sheet_name)
            
            # Normalize column names
            df_adv.columns = [str(col).lower().strip() for col in df_adv.columns]
            
            # Ensure required columns exist
            required_cols = ['advantage_id', 'title', 'evidence_list', 'example_ad_index', 'example_quote']
            for col in required_cols:
                if col not in df_adv.columns:
                    df_adv[col] = ""
            
            advantages[brand_name] = df_adv[required_cols]
    except Exception as e:
        st.error(f"Error loading key advantages: {e}")
    
    return advantages


# Clear cache for key advantages
@st.cache_data(ttl=0)
def load_key_advantages_cached():
    return load_key_advantages()


def load_key_advantages_summary():
    """Load key advantages summary from key_advantages_summary.xlsx"""
    path = os.path.join(KEY_ADVANTAGES_DIR, "key_advantages_summary.xlsx")
    if not os.path.exists(path):
        return ""
    
    try:
        df_summary = pd.read_excel(path)
        # Look for 'Summary' column (case insensitive)
        summary_col = None
        for col in df_summary.columns:
            if 'summary' in str(col).lower():
                summary_col = col
                break
        
        if summary_col and len(df_summary) > 0:
            return str(df_summary[summary_col].iloc[0])
        else:
            return ""
    except Exception as e:
        st.error(f"Error loading key advantages summary: {e}")
        return ""


def normalize_brand(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # Drop location suffixes like "Brand | City"
    base = name.split("|")[0].strip()
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in base).split())


# Load and preprocess data
ads_path = find_ads_file()
if not ads_path:
    st.error("No ads Excel file found in data/ads. Please add one.")
    st.stop()

df = pd.read_excel(ads_path)

# Parse and normalize
df['startDateFormatted'] = pd.to_datetime(df.get('startDateFormatted'), errors='coerce')
df['endDateFormatted'] = pd.to_datetime(df.get('endDateFormatted'), errors='coerce')
df['duration_days'] = (df['endDateFormatted'] - df['startDateFormatted']).dt.days
# Reach column in source
reach_col = 'ad_details/aaa_info/eu_total_reach' if 'ad_details/aaa_info/eu_total_reach' in df.columns else 'reach'
df['reach'] = pd.to_numeric(df[reach_col], errors='coerce')
df['platforms'] = df.get('publisherPlatform')
df['brand'] = df.get('pageName')
df['isActive'] = df.get('isActive', False).astype(bool) if 'isActive' in df.columns else False

# Canonicalize brand names for consistency across charts

def canonicalize_brand(name: str) -> str:
    n = normalize_brand(name)
    mapping = {
        'dpd lietuva': 'DPD Lietuva',
        'lp express': 'LP EXPRESS',
        'omniva lietuva': 'Omniva Lietuva',
        'smartpost lietuva': 'SmartPosti Lietuva',
        'smartposti lietuva': 'SmartPosti Lietuva',
        'venipak lietuva': 'Venipak Lietuva',
    }
    return mapping.get(n, name if isinstance(name, str) else "")

# Apply canonicalization
if 'brand' in df.columns:
    df['brand'] = df['brand'].apply(canonicalize_brand)

# Filter for last 6 months (Feb–Jul 2025)
end_date = pd.Timestamp("2025-07-31", tz="UTC")
start_date = end_date - relativedelta(months=6)
mid_date = start_date + relativedelta(months=3)
df_filtered = df[(df['startDateFormatted'] >= start_date) & (df['startDateFormatted'] <= end_date)].copy()

# Summary calculation function
def summary(df, value_col):
    curr = df[(df['startDateFormatted'] >= mid_date) & (df['startDateFormatted'] <= end_date)]
    prev = df[(df['startDateFormatted'] >= start_date) & (df['startDateFormatted'] < mid_date)]

    curr_agg = curr.groupby('brand')[value_col].sum()
    prev_agg = prev.groupby('brand')[value_col].sum()

    rank_curr = curr_agg.sort_values(ascending=False).rank(ascending=False, method="min")
    rank_prev = prev_agg.sort_values(ascending=False).rank(ascending=False, method="min")

    result = pd.DataFrame({
        'current': curr_agg,
        'previous': prev_agg,
        'change_pct': ((curr_agg - prev_agg) / prev_agg.replace(0, 1)) * 100,
        'rank_now': rank_curr,
        'rank_prev': rank_prev,
        'rank_change': rank_curr - rank_prev
    }).fillna(0)

    return result

# Header
st.title("Ad Intelligence Dashboard")
st.caption("Monitoring ad activity and reach over the past 6 months (Feb–Jul 2025)")

# --- OVERVIEW SECTION ---
st.header("Overview")

# 1. Ad Volume Share by Month (with Reach and Count tabs)
st.subheader("Ad Volume Share by Month")

# Hardcoded color mapping for consistent colors across all charts
color_map = {
    "DPD Lietuva": "#1f77b4",      # Blue
    "LP EXPRESS": "#ff7f0e",       # Orange
    "Omniva Lietuva": "#2ca02c",   # Green
    "SmartPosti Lietuva": "#d62728", # Red
    "Venipak Lietuva": "#9467bd"   # Purple
}

# Get all brands from data, but use hardcoded colors when available
all_brands = sorted(df_filtered['brand'].unique())

month_labels = ["6 months", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
months = ["all", 2, 3, 4, 5, 6, 7]
month_tabs = st.tabs(month_labels)

for i, month in enumerate(months):
    with month_tabs[i]:
        sub_tabs = st.tabs(["Number of Ads", "Reach"])

        if month == "all":
            month_data = df_filtered
        else:
            month_data = df_filtered[df_filtered['startDateFormatted'].dt.month == month]
        
        with sub_tabs[0]:  # Number of Ads
            ad_counts = month_data['brand'].value_counts().reset_index()
            ad_counts.columns = ['brand', 'count']
            ad_counts = ad_counts.sort_values('brand')
            # Create a color column based on brand
            ad_counts['color'] = ad_counts['brand'].map(color_map)
            fig = px.pie(ad_counts, values='count', names='brand', color='color', title=f'Ad Count Share – {"All 6 months" if month == "all" else f"{month:02d}/2025"}')
            st.plotly_chart(fig, use_container_width=True, key=f"pie_ads_{month}")

        with sub_tabs[1]:  # Reach
            reach_totals = month_data.groupby('brand')['reach'].sum().reset_index()
            reach_totals = reach_totals.sort_values('brand')
            # Create a color column based on brand
            reach_totals['color'] = reach_totals['brand'].map(color_map)
            fig = px.pie(reach_totals, values='reach', names='brand', color='color', title=f'Reach Share – {"All 6 months" if month == "all" else f"{month:02d}/2025"}')
            st.plotly_chart(fig, use_container_width=True, key=f"pie_reach_{month}")

# 2. Summary Cards (Reach, Brand Strength, Creativity)
reach_stats = summary(df_filtered, 'reach')
df_filtered['ad_count'] = 1
ads_stats = summary(df_filtered, 'ad_count')
duration_stats = summary(df_filtered, 'duration_days')
from pandas.tseries.offsets import MonthEnd

# Define current and previous month ranges
curr_month_start = pd.Timestamp("2025-07-01", tz="UTC")
curr_month_end = curr_month_start + MonthEnd(1)

prev_month_start = curr_month_start - MonthEnd(1)
prev_month_end = curr_month_start - pd.Timedelta(days=1)

# Ads active during those months
active_curr = df_filtered[
    (df_filtered['startDateFormatted'] <= curr_month_end) &
    (df_filtered['endDateFormatted'] >= curr_month_start)
]
active_prev = df_filtered[
    (df_filtered['startDateFormatted'] <= prev_month_end) &
    (df_filtered['endDateFormatted'] >= prev_month_start)
]

active_stats = pd.DataFrame(index=df_filtered['brand'].unique())
active_stats['current'] = active_curr.groupby('brand').size()
active_stats['previous'] = active_prev.groupby('brand').size()
active_stats = active_stats.fillna(0)
active_stats['change_pct'] = ((active_stats['current'] - active_stats['previous']) / active_stats['previous'].replace(0, 1)) * 100

active_stats['rank_now'] = active_stats['current'].rank(ascending=False, method="min")
active_stats['rank_prev'] = active_stats['previous'].rank(ascending=False, method="min")
active_stats['rank_change'] = active_stats['rank_now'] - active_stats['rank_prev']

# --- Load Brand Strength (COMPOS) ---

def load_brand_strength():
    strength = {}
    if not os.path.isdir(COMPOS_DIR):
        return strength
    for path in glob.glob(os.path.join(COMPOS_DIR, "*_compos_analysis.xlsx")):
        fname = os.path.basename(path)
        if fname.startswith("~$"):
            continue  # skip temp/lock files
        brand_display = fname.replace("_compos_analysis.xlsx", "").strip()
        try:
            df_comp = pd.read_excel(path, sheet_name="Raw Data")
            if 'Top Archetype' in df_comp.columns and len(df_comp) > 0:
                vc = df_comp['Top Archetype'].dropna().value_counts()
                pct = float((vc.max() / vc.sum()) * 100) if vc.sum() > 0 else 0.0
                strength[brand_display] = pct
        except Exception:
            # ignore bad files
            pass
    return strength


brand_strength = load_brand_strength()

# Compute rank and mean deltas for brand strength
if brand_strength:
    bs_df = pd.DataFrame({
        'brand': list(brand_strength.keys()),
        'strength': list(brand_strength.values())
    })
    bs_df['rank'] = bs_df['strength'].rank(ascending=False, method='min')
    bs_mean = bs_df['strength'].mean() if len(bs_df) else 0
    bs_df['delta_vs_mean_pct'] = ((bs_df['strength'] - bs_mean) / (bs_mean if bs_mean != 0 else 1)) * 100
else:
    bs_df = pd.DataFrame(columns=['brand', 'strength', 'rank', 'delta_vs_mean_pct'])
    bs_mean = 0

# --- Load Creativity ---

@st.cache_data(ttl=0)  # Disable caching to force reload
def load_creativity():
    path = os.path.join(CREATIVITY_DIR, "creativity_ranking.xlsx")
    if not os.path.exists(path):
        return pd.DataFrame(columns=['brand', 'rank', 'originality_score', 'justification', 'examples'])
    try:
        df_cre = pd.read_excel(path, sheet_name="Overall Ranking")
        # Normalize column names to lowercase for robustness
        df_cre = df_cre.rename(columns={c: c.lower() for c in df_cre.columns})
        # Ensure required columns exist
        required = ['brand', 'rank', 'originality_score']
        for col in required:
            if col not in df_cre.columns:
                df_cre[col] = None
        # Optional justification and examples columns
        if 'justification' not in df_cre.columns:
            df_cre['justification'] = ""
        if 'examples' not in df_cre.columns:
            df_cre['examples'] = ""
        # Keep only the required columns in desired order
        return df_cre[['brand', 'rank', 'originality_score', 'justification', 'examples']]
    except Exception:
        return pd.DataFrame(columns=['brand', 'rank', 'originality_score', 'justification', 'examples'])


creativity_df = load_creativity()

# Compute creativity delta vs mean for card display
if not creativity_df.empty:
    # Coerce types
    creativity_df['rank'] = pd.to_numeric(creativity_df['rank'], errors='coerce')
    creativity_df['originality_score'] = pd.to_numeric(creativity_df['originality_score'], errors='coerce')
    cre_mean = creativity_df['originality_score'].mean()
    denom = cre_mean if cre_mean != 0 else 1
    creativity_df['delta_vs_mean_pct'] = ((creativity_df['originality_score'] - denom) / denom) * 100
else:
    creativity_df['delta_vs_mean_pct'] = []

# --- Brand tabs & metric cards ---

def format_metric_card(label, val, pct=None, rank_now=None, total_ranks=None):
    # No arrows; color percentages and rank text only
    rank_color = "gray"
    if rank_now is not None and total_ranks:
        if int(rank_now) == 1:
            rank_color = "green"
        elif int(rank_now) == int(total_ranks):
            rank_color = "red"
        else:
            rank_color = "gray"

    pct_color = None
    if pct is not None:
        pct_color = "green" if pct > 0 else "red" if pct < 0 else "gray"

    pct_html = f'<p style="margin:0; color:{pct_color};">Δ {pct:.1f}%</p>' if pct is not None else ''
    rank_html = f'<p style="margin:0; color:{rank_color};">Rank {int(rank_now)}</p>' if rank_now is not None else ''

    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px;">
            <h5 style="margin:0;">{label}</h5>
            <h3 style="margin:5px 0;">{val}</h3>
            {pct_html}
            {rank_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Prepare brand alignment between sources
ads_brands = sorted({b for b in df_filtered['brand'].dropna().unique()})
ads_brand_norm_map = {}
for b in ads_brands:
    ads_brand_norm_map.setdefault(normalize_brand(b), b)

# Build canonical brand list for tabs: prefer COMPOS brands (usually exactly the 5 companies)
if len(bs_df) > 0:
    brand_tab_names = list(bs_df['brand'].sort_values().unique())
else:
    brand_tab_names = ads_brands

st.subheader("Brand Summary")
if not brand_tab_names:
    st.info("No brands available to display.")
else:
    brand_tabs = st.tabs(brand_tab_names)
    reach_mean_current = reach_stats['current'].mean() if len(reach_stats) else 0
    reach_total_ranks = int(reach_stats['rank_now'].max()) if len(reach_stats) else None
    bs_total_ranks = len(bs_df) if len(bs_df) else None
    creativity_total_ranks = creativity_df['brand'].nunique() if len(creativity_df) else None
    # Precompute reach rank display (keep rank) without arrows
    for i, brand_disp in enumerate(brand_tab_names):
        with brand_tabs[i]:
            # Resolve brand name in ads stats
            brand_norm = normalize_brand(brand_disp)
            ads_brand_key = ads_brand_norm_map.get(brand_norm)

            col1, col2, col3 = st.columns(3)

            # Reach card: value + Δ vs mean; keep rank, no arrows
            with col1:
                if ads_brand_key in reach_stats.index:
                    r = reach_stats.loc[ads_brand_key]
                    # Use full 6-month reach total instead of just current 3 months
                    full_6month_reach = df_filtered[df_filtered['brand'] == ads_brand_key]['reach'].sum()
                    current_val = int(full_6month_reach)
                    # Calculate mean for full 6-month period
                    full_6month_mean = df_filtered.groupby('brand')['reach'].sum().mean()
                    denom = full_6month_mean if full_6month_mean != 0 else 1
                    delta_mean_pct = ((full_6month_reach - denom) / denom) * 100
                    # Calculate rank for full 6-month period
                    full_6month_ranks = df_filtered.groupby('brand')['reach'].sum().rank(ascending=False, method="min")
                    full_6month_rank = full_6month_ranks.get(ads_brand_key, 0)
                    format_metric_card(
                        label="Reach (6 months)",
                        val=f"{current_val:,}",
                        pct=delta_mean_pct,
                        rank_now=full_6month_rank,
                        total_ranks=len(full_6month_ranks)
                    )
                else:
                    format_metric_card("Reach (6 months)", "0", pct=0)

            # Brand Strength card: value + rank among brands + Δ vs mean
            with col2:
                row = bs_df[bs_df['brand'] == brand_disp]
                if not row.empty:
                    strength = float(row['strength'].iloc[0])
                    rank_bs = int(row['rank'].iloc[0])
                    delta_bs = float(row['delta_vs_mean_pct'].iloc[0])
                    format_metric_card(
                        label="Brand Strength",
                        val=f"{strength:.1f}%",
                        pct=delta_bs,
                        rank_now=rank_bs,
                        total_ranks=bs_total_ranks
                    )
                else:
                    format_metric_card("Brand Strength", "N/A")

            # Creativity card: originality score + rank
            with col3:
                cre_row = creativity_df[creativity_df['brand'].astype(str).str.lower() == brand_disp.lower()]
                if not cre_row.empty:
                    score = cre_row['originality_score'].iloc[0]
                    rank_cre = int(cre_row['rank'].iloc[0])
                    delta_cre = float(cre_row['delta_vs_mean_pct'].iloc[0])
                    format_metric_card(
                        label="Creativity",
                        val=f"{score:.2f}",
                        pct=delta_cre,
                        rank_now=rank_cre,
                        total_ranks=creativity_total_ranks
                    )
                else:
                    format_metric_card("Creativity", "N/A")

            # Add creativity justification below the cards
            if not creativity_df.empty:
                cre_row = creativity_df[creativity_df['brand'].astype(str).str.lower() == brand_disp.lower()]
                if not cre_row.empty:
                    just_text = str(cre_row['justification'].iloc[0]) if pd.notna(cre_row['justification'].iloc[0]) else ""
                    examples_text = str(cre_row['examples'].iloc[0]) if pd.notna(cre_row['examples'].iloc[0]) else ""
                    
                    if just_text or examples_text:
                        st.markdown("### Creativity Analysis")
                        st.markdown(f"""
                        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px;">
                            <h5 style="margin:0;">{brand_disp} — Rank {rank_cre} — Score {score:.2f}</h5>
                            {f'<p style="margin:8px 0 0; color:#444;">{just_text}</p>' if just_text else ''}
                            {f'<p style="margin:8px 0 0; color:#444;">Examples: {examples_text}</p>' if examples_text else ''}
                        </div>
                        """, unsafe_allow_html=True)






# --- TOP ARCHETYPES SECTION ---

st.header("Top Archetypes by Company")

def load_top_archetypes():
    """Load top 3 archetypes for each company from compos analysis files"""
    archetypes_data = {}
    if not os.path.isdir(COMPOS_DIR):
        return archetypes_data
    
    for path in glob.glob(os.path.join(COMPOS_DIR, "*_compos_analysis.xlsx")):
        fname = os.path.basename(path)
        if fname.startswith("~$"):
            continue  # skip temp/lock files
        brand_display = fname.replace("_compos_analysis.xlsx", "").strip()
        try:
            df_comp = pd.read_excel(path, sheet_name="Raw Data")
            if 'Top Archetype' in df_comp.columns and len(df_comp) > 0:
                # Get value counts of archetypes
                vc = df_comp['Top Archetype'].dropna().value_counts()
                total_ads = len(df_comp['Top Archetype'].dropna())
                
                # Get top 3 archetypes with their counts and percentages
                top_3 = vc.head(3)
                archetypes_list = []
                for archetype, count in top_3.items():
                    percentage = (count / total_ads) * 100 if total_ads > 0 else 0
                    archetypes_list.append({
                        'archetype': archetype,
                        'percentage': percentage,
                        'count': count
                    })
                
                archetypes_data[brand_display] = archetypes_list
        except Exception as e:
            # ignore bad files
            pass
    return archetypes_data

archetypes_data = load_top_archetypes()

if archetypes_data:
    # Create tabs for each company
    company_tabs = st.tabs(list(archetypes_data.keys()))
    
    for i, (company, archetypes) in enumerate(archetypes_data.items()):
        with company_tabs[i]:
            st.subheader(f"{company} - Top 3 Archetypes")
            
            # Create 3 columns for the 3 archetype cards
            col1, col2, col3 = st.columns(3)
            
            for j, archetype_info in enumerate(archetypes):
                # Determine which column to use
                if j == 0:
                    col = col1
                elif j == 1:
                    col = col2
                else:
                    col = col3
                
                with col:
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:10px; margin-bottom:10px; text-align:center;">
                        <h4 style="margin:0; color:#333;">{archetype_info['archetype']}</h4>
                        <h2 style="margin:5px 0; color:#333; font-size:2.5em;">{archetype_info['percentage']:.1f}%</h2>
                        <p style="margin:0; color:#666; font-size:0.9em;">{archetype_info['count']} ads</p>
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.info("No archetype data available. Please ensure compos analysis files are in the data/compos directory.")

# --- IN DEPTH SECTION ---

st.header("In-Depth View")

# Start Date Distribution
st.subheader("Ad Start Date Distribution")
st.markdown("This chart shows the number of ads launched each day per brand to help identify campaign patterns and bursts.")
hist = px.histogram(df_filtered, x='startDateFormatted', color='brand', nbins=60, barmode='overlay')
hist.update_layout(
    xaxis_title="",
    yaxis_title="Number of Ads"
)
st.plotly_chart(hist, use_container_width=True)

# Volume Trends
st.subheader("Volume Trends")
tabs = st.tabs(["Total", "FACEBOOK", "INSTAGRAM", "MESSENGER", "THREADS", "AUDIENCE_NETWORK"])
platforms = ["FACEBOOK", "INSTAGRAM", "MESSENGER", "THREADS", "AUDIENCE_NETWORK"]

# Total tab (weekly)
with tabs[0]:
    st.markdown("#### Reach (Total)")
    reach_trend = df_filtered.groupby([pd.Grouper(key='startDateFormatted', freq='W'), 'brand'])['reach'].sum().reset_index()
    fig = px.line(reach_trend, x='startDateFormatted', y='reach', color='brand')
    st.plotly_chart(fig, use_container_width=True, key="total_reach")

    st.markdown("#### New Ads (Total)")
    ads_trend = df_filtered.groupby([pd.Grouper(key='startDateFormatted', freq='W'), 'brand']).size().reset_index(name='ads')
    fig = px.line(ads_trend, x='startDateFormatted', y='ads', color='brand')
    st.plotly_chart(fig, use_container_width=True, key="total_ads")

# Fix platform list parsing

def parse_platforms(val):
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        elif isinstance(val, list):
            return val
    except Exception:
        return []
    return []


exploded = df_filtered.copy()
exploded['platforms'] = exploded['platforms'].apply(parse_platforms)
exploded = exploded.explode('platforms')

for i, platform in enumerate(platforms):
    with tabs[i + 1]:
        st.markdown(f"#### Reach – {platform}")
        pf = exploded[exploded['platforms'] == platform]

        if pf.empty:
            st.warning(f"No data available for {platform}.")
        else:
            pf_reach = pf.groupby([pd.Grouper(key='startDateFormatted', freq='W'), 'brand'])['reach'].sum().reset_index()
            fig = px.line(pf_reach, x='startDateFormatted', y='reach', color='brand')
            st.plotly_chart(fig, use_container_width=True, key=f"reach_{platform}")

            st.markdown(f"#### New Ads – {platform}")
            pf_ads = pf.groupby([pd.Grouper(key='startDateFormatted', freq='W'), 'brand']).size().reset_index(name='ads')
            fig = px.line(pf_ads, x='startDateFormatted', y='ads', color='brand')
            st.plotly_chart(fig, use_container_width=True, key=f"ads_{platform}")

# --- KEY ADVANTAGES SECTION ---

st.header("Key Advantages")

key_advantages_data = load_key_advantages_cached()

if not key_advantages_data:
    st.info("No key advantages data loaded. Please ensure key_advantages.xlsx is in the data/key_advantages directory.")
else:
    if not brand_tab_names:
        st.info("No brands available to display key advantages for.")
    else:
        brand_tabs = st.tabs(brand_tab_names)
        for i, brand_disp in enumerate(brand_tab_names):
            with brand_tabs[i]:
                st.subheader(f"{brand_disp} Key Advantages")
                brand_data = key_advantages_data.get(brand_disp)
                if brand_data.empty:
                    st.info(f"No specific key advantages found for {brand_disp}.")
                else:
                    # Group by title and evidence, combine multiple examples
                    grouped_data = brand_data.groupby(['title', 'evidence_list']).agg({
                        'example_quote': lambda x: list(x)
                    }).reset_index()

                    # Create 2 columns for the advantages
                    col1, col2 = st.columns(2)
                    for idx, (_, row) in enumerate(grouped_data.iterrows()):
                        # Alternate between columns
                        with col1 if idx % 2 == 0 else col2:
                            # Format examples as a list
                            examples_html = ""
                            for i, example in enumerate(row['example_quote']):
                                if example and str(example).strip():
                                    examples_html += f"<li style='margin:4px 0; color:#444;'>{example}</li>"
                            
                            if examples_html:
                                examples_html = f"<ul style='margin:4px 0 0; padding-left:20px;'>{examples_html}</ul>"
                            else:
                                examples_html = "<p style='margin:4px 0 0; color:#444;'>No examples available</p>"

                            st.markdown(f"""
                            <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px; max-height:800px; overflow-y:auto;">
                                <h5 style="margin:0; word-wrap:break-word;">{row['title']}</h5>
                                <p style="margin:8px 0 0; font-weight:bold; color:#333;">Evidence:</p>
                                <p style="margin:4px 0 8px; color:#444; word-wrap:break-word; white-space:pre-wrap;">{row['evidence_list']}</p>
                                <p style="margin:8px 0 0; font-weight:bold; color:#333;">Examples:</p>
                                {examples_html}
                            </div>
                            """, unsafe_allow_html=True)

    # Display the key advantages summary table
    st.subheader("Key Advantages Summary")
    
    # Hardcoded summary data
    summary_data = {
        "Company": [
            "DPD Lietuva",
            "LP EXPRESS", 
            "Omniva Lietuva",
            "SmartPosti Lietuva",
            "Venipak Lietuva"
        ],
        "Key Positioning": [
            "Positions itself as an environmentally responsible carrier, stressing sustainability achievements and awards.",
            "Promotes practical delivery perks, frequent discounts (including student offers), and ease of shipping.",
            "Highlights reliability of service, competitive pricing, and user-friendly delivery processes.",
            "Markets technological innovation in delivery, secure indoor parcel storage, and cost-effective pricing.",
            "Emphasizes rapid delivery times, a large and convenient pickup network, and engagement with local communities."
        ]
    }
    
    # Create and display the table
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df.set_index('Company'))



